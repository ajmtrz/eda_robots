import threading
from numba import njit, prange, float64, int64
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import onnxruntime as rt
from functools import lru_cache

# Configuración thread-safe de ONNX Runtime
_onnx_configured = False
_onnx_lock = threading.RLock()

def _configure_onnx_runtime():
    """Configuración thread-safe de ONNX Runtime"""
    global _onnx_configured
    with _onnx_lock:
        if not _onnx_configured:
            rt.set_default_logger_severity(4)
            _onnx_configured = True

def clear_onnx_cache():
    """Limpia la caché de sesiones ONNX (útil para gestión de memoria)"""
    with _session_lock:
        _session_cache.clear()

# Thread-safe session cache
_session_cache = {}
_session_lock = threading.RLock()

# Thread-safe plotting
_plot_lock = threading.RLock()

# Constantes financieras corregidas
RISK_FREE_RATE = 0.02    # Tasa libre de riesgo anual (2%)

def _safe_plot(equity_curve, title="Strategy Performance", score=None):
    """Thread-safe plotting function"""
    with _plot_lock:
        plt.figure(figsize=(10, 6))
        plt.plot(equity_curve, label='Equity Curve', linewidth=1.5)
        if score is not None:
            plt.title(f"{title} - Score: {score:.3f}")
        else:
            plt.title(title)
        plt.xlabel("Time")
        plt.ylabel("Cumulative P&L")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()
        plt.close()

@njit(cache=True, fastmath=True)
def process_data(close, labels, metalabels, meta_thr=0.5):
    last_deal  = 2
    last_price = 0.0
    report, chart = [0.0], [0.0]

    for i in range(len(close)):
        pred, pr, pred_meta = labels[i], close[i], metalabels[i]

        # ── abrir posición
        if last_deal == 2 and pred_meta > meta_thr:
            last_price = pr
            last_deal  = 0 if pred < 0.5 else 1
            continue

        # ── cerrar por señal opuesta
        if last_deal == 0 and pred > 0.5:
            last_deal = 2
            profit = (pr - last_price)
            report.append(report[-1] + profit)
            chart.append(chart[-1] + profit)
            continue

        if last_deal == 1 and pred < 0.5:
            last_deal = 2
            profit = (last_price - pr)
            report.append(report[-1] + profit)
            chart.append(chart[-1] + (pr - last_price))
            continue

    # Cierre forzoso al final si sigue abierta
    if last_deal == 0:
        profit = close[-1] - last_price
        report.append(report[-1] + profit)
        chart.append(chart[-1] + profit)
    elif last_deal == 1:
        profit = last_price - close[-1]
        report.append(report[-1] + profit)
        chart.append(chart[-1] + (close[-1] - last_price))

    return np.array(report), np.array(chart)

@njit(cache=True, fastmath=True)
def process_data_one_direction(close, main_labels, meta_labels, direction_int):
    last_deal  = 2            # 2 = flat, 1 = position open
    last_price = 0.0
    report = [0.0]
    chart  = [0.0]
    long_side = (direction_int == 0)  # 0=buy, 1=sell
    min_prob  = 0.5

    for i in range(close.size):
        pred_main = main_labels[i]
        pr        = close[i]
        pred_meta = meta_labels[i]

        # ── abrir posición ───────────────────────────────
        if last_deal == 2 and pred_meta > min_prob and pred_main > min_prob:
            last_deal  = 1
            last_price = pr
            continue

        # ── cerrar posición ──────────────────────────────
        if last_deal == 1 and pred_main < min_prob:
            last_deal = 2
            profit = (pr - last_price) if long_side else (last_price - pr)
            report.append(report[-1] + profit)
            chart.append(chart[-1]  + profit)

    # Cierre forzoso al final si sigue abierta
    if last_deal == 1:
        profit = (close[-1] - last_price) if long_side else (last_price - close[-1])
        report.append(report[-1] + profit)
        chart.append(chart[-1]  + profit)

    return np.asarray(report, dtype=np.float64), np.asarray(chart, dtype=np.float64)


# ───────────────────────────────────────────────────────────────────
# 2)  Wrappers del tester
# ───────────────────────────────────────────────────────────────────

def get_periods_per_year(timeframe: str) -> float:
    """
    Calcula períodos por año basado en el timeframe.
    Asume mercado XAUUSD: ~120 horas de trading por semana, 52 semanas/año.
    
    Args:
        timeframe: 'M5', 'M15', 'M30', 'H1', 'H4', 'D1'
    
    Returns:
        float: Número de períodos por año para ese timeframe
    """
    # Mapeo de timeframes a períodos por año (ajustado para XAUUSD)
    if timeframe == 'M5':
        return 74880.0    # 120h/sem * 60min/h / 5min * 52sem = 74,880
    elif timeframe == 'M15':
        return 24960.0    # 120h/sem * 60min/h / 15min * 52sem = 24,960
    elif timeframe == 'M30':
        return 12480.0    # 120h/sem * 60min/h / 30min * 52sem = 12,480
    elif timeframe == 'H1':
        return 6240.0     # 120h/sem * 52sem = 6,240
    elif timeframe == 'H4':
        return 1560.0     # 30 períodos/sem * 52sem = 1,560
    elif timeframe == 'D1':
        return 260.0      # 5 días/sem * 52sem = 260
    else:
        return 6240.0     # Default a H1 si timeframe no reconocido

def tester(
        dataset: pd.DataFrame,
        model_main: object,
        model_meta: object,
        model_main_cols: list[str],
        model_meta_cols: list[str],
        direction: str = 'both',
        plot: bool = False,
        prd: str = '',
        timeframe: str = 'H1',
        print_metrics: bool = False) -> float:

    """Evalúa una estrategia para una o ambas direcciones.

    Parameters
    ----------
    dataset : pd.DataFrame
        DataFrame con las columnas de cierre y etiquetas.
    model_main_cols : list[str]
        Lista de nombres de columnas para el modelo principal.
    model_meta_cols : list[str]
        Lista de nombres de columnas para el meta-modelo.
    model_main : object
        Modelo principal entrenado con ``predict_proba``.
    model_meta : object
        Meta-modelo entrenado con ``predict_proba``.
    direction : str, optional
        ``'buy'``, ``'sell'`` o ``'both'``. Por defecto ``'both'``.
    plot : bool, optional
        Si ``True`` muestra la curva de equity.  Por defecto ``False``.
    prd : str, optional
        Etiqueta del periodo a mostrar en el gráfico.
    timeframe : str, optional
        Timeframe de los datos para cálculos de anualización. Por defecto ``'H1'``.
    print_metrics : bool, optional
        Si ``True`` imprime métricas detalladas para debugging. Por defecto ``False``.

    Returns
    -------
    float
        Puntuación de la estrategia según :func:`evaluate_report`.
    """
    # Convertir timeframe a períodos por año fuera de funciones jiteadas
    periods_per_year = get_periods_per_year(timeframe)
    
    # Preparación de datos
    ds_main = dataset[model_main_cols].to_numpy()
    ds_meta = dataset[model_meta_cols].to_numpy()
    close = dataset['close'].to_numpy()

    # Calcular probabilidades usando ambos modelos (sin binarizar)
    main = _predict_one(model_main, ds_main)
    meta = _predict_one(model_meta, ds_meta)

    # Asegurar contigüidad en memoria
    close = np.ascontiguousarray(close)
    main = np.ascontiguousarray(main)
    meta = np.ascontiguousarray(meta)

    if direction == 'both':
        rpt, _ = process_data(close, main, meta)
    else:
        direction_map = {'buy': 0, 'sell': 1}
        direction_int = direction_map.get(direction, 0)
        rpt, _ = process_data_one_direction(close, main, meta, direction_int)

    if rpt.size < 2:
        return -1.0

    
    if print_metrics:
        # Recalcular con métricas completas para debugging
        score, metrics_tuple = evaluate_report(rpt, periods_per_year)
        metrics_dict = metrics_tuple_to_dict(score, metrics_tuple, periods_per_year)
        title = f"Tester {direction.upper()}"
        if prd:
            title += f" {prd}"
        print_detailed_metrics(metrics_dict, title)
    else:
        score = evaluate_report(rpt, periods_per_year)[0]

    if plot:
        title = f"Period: {prd}" if prd else "Strategy Performance"
        _safe_plot(rpt, title=title, score=score)

    return score

# ────────── helpers ──────────────────────────────────────────────────────────
@njit(cache=True, fastmath=True)
def _signed_r2(eq):
    n  = eq.size
    t  = np.arange(n, dtype=np.float64)
    xm = t.mean(); ym = eq.mean()
    cov   = ((t-xm)*(eq-ym)).sum()
    var_t = ((t-xm)**2).sum()
    var_y = ((eq-ym)**2).sum()
    if var_t == 0.0 or var_y == 0.0:
        return 0.0
    slope = cov / var_t
    r2    = (cov*cov)/(var_t*var_y)
    return np.sign(slope) * r2        # ∈[-1,1]

@njit(cache=True, fastmath=True)
def _max_dd_and_gap(eq):
    peak = eq[0]
    mdd  = 0.0
    gap  = 0
    last_gap = 0
    for x in eq:
        if x >= peak:
            peak = x
            gap  = 0
        else:
            d = peak - x
            if d > mdd:
                mdd = d
            gap += 1
            last_gap = gap
    return mdd, last_gap

@njit(cache=True, fastmath=True)
def _sharpe(ret, ppy=6240.0, rf=0.0):
    if ret.size < 2:
        return 0.0
    ex = ret.mean() - rf/ppy
    sd = ret.std() + 1e-12
    return np.sqrt(ppy) * ex / sd

@njit(cache=True, fastmath=True)
def _sortino(ret, ppy=6240.0, rf=0.0):
    target = rf/ppy
    ex     = ret.mean() - target
    if ex <= 0.0:
        return 0.0
    downside = ret[ret < target]
    if downside.size == 0:
        return 10.0
    dd_std = downside.std() + 1e-12
    return np.sqrt(ppy) * ex / dd_std

@njit(cache=True, fastmath=True)
def _calmar(total_ret, max_dd, years):
    if abs(max_dd) < 1e-12 or years <= 1e-12:
        return 0.0
    if abs(1.0 + total_ret) < 1e-12:
        return 0.0
    ann_ret = (1.0 + total_ret) ** (1.0/years) - 1.0
    return ann_ret / abs(max_dd)

@njit(cache=True, fastmath=True)
def _deflated_sharpe(sr, skew, kurt, n_obs):
    if n_obs < 2:
        return 0.0
    var_sr = (1.0 + sr*sr/2.0) / (n_obs-1.0)
    var_sr = max(var_sr, 1e-12)
    z      = 1.645                      # ≈ 95 %
    return sr * (1.0 - z*np.sqrt(var_sr))

# ────────── evaluación ───────────────────────────────────────────────────────
@njit(cache=True, fastmath=True)
def evaluate_report(eq: np.ndarray, ppy: float = 6240.0):
    """Devuelve (score, metrics_tuple) - Sistema de scoring más realista."""
    # 0) sanidad mínima
    if eq.size < 300 or not np.isfinite(eq).all():
        return (-1.0, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))

    # Protección adicional para curvas con valores muy pequeños o negativos
    eq_min = np.min(eq)
    if eq_min <= 0.0:
        # Desplazar la curva para que todos los valores sean positivos
        eq = eq - eq_min + 1.0
    
    # Cálculo de retornos con protección mejorada
    eq_prev = eq[:-1]
    eq_prev_safe = np.maximum(np.abs(eq_prev), 1e-6)  # Protección más fuerte
    ret = np.diff(eq) / eq_prev_safe
    
    n_trades = np.count_nonzero(ret)
    if n_trades < 250:
        return (-1.0, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))

    # 1) métricas básicas
    sr      = _sharpe(ret, ppy)
    sortino = _sortino(ret, ppy)
    r2      = _signed_r2(eq)
    mdd, gap = _max_dd_and_gap(eq)

    # retornos totales
    eq_range   = max(abs(eq[-1]), abs(eq[0]), 1.0)
    total_ret  = (eq[-1]-eq[0]) / eq_range
    years      = eq.size / ppy
    calmar     = _calmar(total_ret, mdd/eq_range, years)

    # distribución
    m  = ret.mean()
    s  = ret.std() + 1e-12
    skew = np.mean(((ret-m)/s)**3)
    kurt = np.mean(((ret-m)/s)**4)
    
    # Volatilidad anualizada para penalización extra
    vol_annual = s * np.sqrt(ppy)

    # 2) normalizaciones [0,1] - CORREGIDAS Y MÁS ESTRICTAS
    r2_n    = max(0.0, min(1.0, (r2 + 1.0) / 2.0))  # R² normalizado [-1,1] -> [0,1]
    
    # Sharpe y Sortino: más estrictos y realistas
    sr_n    = 1.0 / (1.0 + np.exp(-max(sr, -10.0)/1.0))      # Más estricto
    sortino_n = 1.0 / (1.0 + np.exp(-max(sortino, -10.0)/1.0))  # Más estricto
    
    # Drawdown: MUCHO más estricto
    mdd_rel = mdd/eq_range
    mdd_n   = np.exp(-15.0 * mdd_rel)  # Penalización exponencial severa
    
    # Gap/stagnation: más estricto
    gap_n   = np.exp(-gap/50.0)       # Penalizar estancamiento duramente
    
    # Calmar: corregido
    calmar_n = 1.0 / (1.0 + np.exp(-max(calmar, -10.0)))
    
    # Volatilidad: penalización directa por alta volatilidad
    vol_penalty = 1.0 / (1.0 + vol_annual/0.3)  # Penalizar vol > 30% anual
    
    # Distribución
    skew_n  = 1.0/(1.0+np.exp(-skew))
    kurt_n  = 1.0/(1.0+0.25*(kurt-3.0)**2)
    activ   = 1.0/(1.0+np.exp(-(n_trades-300.0)/30.0))
    agil_n  = gap_n

    # 3) Score final: priorizar riesgo-retorno pero premiar curvas lineales

    # Componentes principales con pesos más realistas
    risk_adj = (sr_n * sortino_n * calmar_n) ** (1.0/3.0)  # Riesgo-retorno (peso dominante)
    quality  = (mdd_n ** 2.0 * gap_n * vol_penalty) ** (1.0/4.0)  # Control de riesgo MÁS estricto en DD

    # Tendencia: solo se premia si la pendiente es positiva
    trend_lin = max(0.0, r2)             # [-1,1] -> [0,1] (negativa = 0)

    # Score: riesgo-retorno domina, drawdown penaliza fuertemente,
    # y se incentiva la linealidad creciente de la curva
    core_score = (risk_adj ** 3.0 * quality ** 2.0 * trend_lin * activ) ** (1.0/6.0)
    
    # Media geométrica final para suavizar
    score = core_score ** 0.6  # Permitir más diferenciación
    
    # Penalización inteligente por retornos negativos
    if total_ret <= 0.0:
        # Si tiene buen Sharpe/Sortino, ser menos severo
        risk_quality = (sr_n * sortino_n) ** 0.5
        base_penalty = 0.3 + 0.4 * risk_quality  # [0.3, 0.7] basado en calidad
        ret_penalty = 1.0 / (1.0 + abs(total_ret) * 1.5)  # Gradual por retorno
        penalty = base_penalty * ret_penalty
        score *= penalty

    # 4) métricas extra para debug/registro
    defl_sr = _deflated_sharpe(sr, skew, kurt, n_trades)
    shape_score = 0.5*(skew_n + kurt_n)

    metrics_tuple = (
        r2, activ,
        mdd, mdd_rel, mdd_n,
        sr, sr_n,
        sortino, sortino_n,
        calmar, calmar_n,
        skew, skew_n,
        kurt, kurt_n,
        agil_n, defl_sr,
        core_score, shape_score,
        total_ret, n_trades, gap
    )

    return score, metrics_tuple

# ────────── conversor ────────────────────────────────────────────────────────
def metrics_tuple_to_dict(
        score: float,
        metrics_tuple: tuple,
        periods_per_year: float
    ) -> dict:
    """Convierte la tupla devuelta por evaluate_report en un dict legible."""
    # Orden y nombres según la tupla metrics_tuple:
    # (r2, activ, mdd, mdd_rel, mdd_n, sr, sr_n, sortino, sortino_n, calmar, calmar_n,
    #  skew, skew_n, kurt, kurt_n, agil_n, defl_sr, core_score, shape_score, total_ret, n_trades, gap)
    return {
        # ✅ óptimo global
        'score':                    score,
        # ── métricas de tendencia/actividad
        'r2':                       metrics_tuple[0],
        'activity':                 metrics_tuple[1],
        # ── drawdown
        'max_drawdown':             metrics_tuple[2],
        'max_drawdown_relative':    metrics_tuple[3],
        'max_drawdown_normalized':  metrics_tuple[4],
        # ── riesgo-retorno
        'sharpe_ratio':             metrics_tuple[5],
        'sharpe_normalized':        metrics_tuple[6],
        'sortino_ratio':            metrics_tuple[7],
        'sortino_normalized':       metrics_tuple[8],
        'calmar_ratio':             metrics_tuple[9],
        'calmar_normalized':        metrics_tuple[10],
        # ── forma de la distribución
        'skewness':                 metrics_tuple[11],
        'skewness_normalized':      metrics_tuple[12],
        'kurtosis':                 metrics_tuple[13],
        'kurtosis_normalized':      metrics_tuple[14],
        # ── agilidad / stagnation
        'agility_normalized':       metrics_tuple[15],
        # ── control de sobre-optimización
        'deflated_sharpe':          metrics_tuple[16],
        # ── componentes internos del score
        'core_score':               metrics_tuple[17],
        'shape_score':              metrics_tuple[18],
        # ── rentabilidad global
        'total_return':             metrics_tuple[19],
        # ── operativa
        'n_trades':                 metrics_tuple[20],
        'last_high_gap':            metrics_tuple[21],
        # ── contexto
        'periods_per_year':         periods_per_year
    }

# ────────── impresor de métricas ─────────────────────────────────────────────
def print_detailed_metrics(metrics: dict, title: str = "Strategy Metrics"):
    """
    Imprime las métricas detalladas en formato de depuración.
    
    Args:
        metrics: Diccionario devuelto por metrics_tuple_to_dict
        title:   Encabezado para el bloque de debug
    """
    # Correspondencia exacta con el diccionario y la tupla
    print(f"🔍 DEBUG: {title} - Score: {metrics['score']:.4f}\n"
          f"  • R²={metrics['r2']:.4f} | Activity={metrics['activity']:.4f}\n"
          f"  • Sharpe={metrics['sharpe_ratio']:.4f} | Sharpe_N={metrics['sharpe_normalized']:.4f}\n"
          f"  • Sortino={metrics['sortino_ratio']:.4f} | Sortino_N={metrics['sortino_normalized']:.4f}\n"
          f"  • Calmar={metrics['calmar_ratio']:.4f} | Calmar_N={metrics['calmar_normalized']:.4f}\n"
          f"  • Max DD={metrics['max_drawdown']:.4f} | Max DD Rel={metrics['max_drawdown_relative']:.4f} | Max DD N={metrics['max_drawdown_normalized']:.4f}\n"
          f"  • Total Ret={metrics['total_return']:.4f} | Trades={metrics['n_trades']}\n"
          f"  • Skew={metrics['skewness']:.4f} | Skew_N={metrics['skewness_normalized']:.4f}\n"
          f"  • Kurt={metrics['kurtosis']:.4f} | Kurt_N={metrics['kurtosis_normalized']:.4f}\n"
          f"  • Agility_N={metrics['agility_normalized']:.4f} | Gap={metrics['last_high_gap']}\n"
          f"  • Core={metrics['core_score']:.4f} | Shape={metrics['shape_score']:.4f} | DSR={metrics['deflated_sharpe']:.4f}\n"
          f"  • Periods/Year={metrics['periods_per_year']:.2f}")

def _ort_session(model_path: str):
    """Thread-safe ONNX session cache"""
    _configure_onnx_runtime()
    
    with _session_lock:
        if model_path in _session_cache:
            return _session_cache[model_path]
        
        # Crear nueva sesión
        sess = rt.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        iname = sess.get_inputs()[0].name
        
        # Limitar el tamaño de la caché (máximo 10 sessions)
        if len(_session_cache) >= 10:
            # Eliminar la primera entrada (FIFO)
            oldest_key = next(iter(_session_cache))
            del _session_cache[oldest_key]
        
        _session_cache[model_path] = (sess, iname)
    return sess, iname

def _predict_onnx(model_path:str, X_3d:np.ndarray) -> np.ndarray:
    n_sim, n_rows, n_feat = X_3d.shape
    sess, iname = _ort_session(model_path)

    raw = sess.run(None, {iname: X_3d.reshape(-1, n_feat).astype(np.float32)})[0]

    # ─── des-ZipMap / distintos formatos de salida ─────────────────────
    if raw.dtype == object:                     # lista de dicts
        prob_pos = np.fromiter((row[b'1'] for row in raw), dtype=np.float32)

    elif raw.ndim == 2:                         # matriz (n,2)
        prob_pos = raw[:, 1].astype(np.float32)

    elif raw.ndim == 1:                         # vector (n,)  → ya es proba+
        prob_pos = raw.astype(np.float32)

    else:
        raise RuntimeError(f"Formato de salida ONNX no soportado: {raw.shape}")

    return prob_pos.reshape(n_sim, n_rows)

def _predict_one(model_any, X_2d: np.ndarray) -> np.ndarray:
    """
    Devuelve la probabilidad de la clase positiva para una sola matriz 2-D.
      · Si 'model_any' es CatBoost -> usa predict_proba.
      · Si es ruta .onnx, bytes, o ModelProto -> usa _predict_onnx.
    Resultado shape: (n_rows,)
    """
    if hasattr(model_any, "predict_proba"):
        return model_any.predict_proba(X_2d)[:, 1]
    else:
        # _predict_onnx espera tensor 3-D: (n_sim, n_rows, n_feat)
        return _predict_onnx(model_any, X_2d[None, :, :])[0]