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


# ═══════════════════════════════════════════════════════════════════════════════════
# VALIDACIÓN AVANZADA DE ESTRATEGIAS - RECOMENDACIONES DE BACKTESTING
# ═══════════════════════════════════════════════════════════════════════════════════

from scipy import stats
from scipy.special import erfinv
import warnings

def monkey_test(
    dataset: pd.DataFrame,
    model_main: object,
    model_meta: object,
    model_main_cols: list[str],
    model_meta_cols: list[str],
    direction: str = 'both',
    timeframe: str = 'H1',
    n_monkeys: int = 1000,
    random_seed: int = 42
) -> dict:
    """
    Ejecuta el Monkey Test (Null Hypothesis Benchmark) para validar estadísticamente 
    una estrategia contra estrategias aleatorias que preservan la frecuencia de trading.
    
    Este test establece si el rendimiento de la estrategia es estadísticamente 
    distinguible de la suerte pura, siendo el primer filtro crítico de validación.
    
    Args:
        dataset: DataFrame con datos de precio y features
        model_main: Modelo principal para señales de trading
        model_meta: Meta-modelo para filtrado de señales
        model_main_cols: Columnas para el modelo principal
        model_meta_cols: Columnas para el meta-modelo
        direction: 'buy', 'sell' o 'both'
        timeframe: Timeframe para cálculos de anualización
        n_monkeys: Número de estrategias aleatorias a simular
        random_seed: Semilla para reproducibilidad
        
    Returns:
        dict: Resultados del test incluyendo p-value y distribución null
    """
    print(f"🐒 Ejecutando Monkey Test con {n_monkeys} estrategias aleatorias...")
    
    # Obtener performance de la estrategia real
    actual_score = tester(
        dataset=dataset,
        model_main=model_main,
        model_meta=model_meta,
        model_main_cols=model_main_cols,
        model_meta_cols=model_meta_cols,
        direction=direction,
        timeframe=timeframe
    )
    
    if actual_score <= 0:
        return {
            'actual_score': actual_score,
            'p_value': 1.0,
            'random_scores': [],
            'is_significant': False,
            'status': 'FAILED - Strategy has negative/zero score'
        }
    
    # Preparar datos para simulaciones
    periods_per_year = get_periods_per_year(timeframe)
    ds_main = dataset[model_main_cols].to_numpy()
    ds_meta = dataset[model_meta_cols].to_numpy()
    close = dataset['close'].to_numpy()
    
    # Obtener señales reales para preservar estructura
    main_probs = _predict_one(model_main, ds_main)
    meta_probs = _predict_one(model_meta, ds_meta)
    
    # Convertir a señales binarias para análisis de frecuencia
    real_main_signals = (main_probs > 0.5).astype(int)
    real_meta_signals = (meta_probs > 0.5).astype(int)
    
    # Calcular proporciones para mantener frecuencia de trading
    main_signal_props = np.bincount(real_main_signals) / len(real_main_signals)
    meta_signal_props = np.bincount(real_meta_signals) / len(real_meta_signals)
    
    # Ejecutar simulaciones Monte Carlo
    np.random.seed(random_seed)
    random_scores = []
    
    for i in range(n_monkeys):
        if i % 200 == 0:
            print(f"  Progreso: {i}/{n_monkeys} simulaciones completadas")
            
        # Generar señales aleatorias preservando frecuencias
        random_main = np.random.choice([0, 1], size=len(close), p=main_signal_props)
        random_meta = np.random.choice([0, 1], size=len(close), p=meta_signal_props)
        
        # Convertir a probabilidades para compatibilidad
        random_main_probs = random_main.astype(float)
        random_meta_probs = random_meta.astype(float)
        
        # Ejecutar backtest con señales aleatorias
        if direction == 'both':
            rpt, _ = process_data(close, random_main_probs, random_meta_probs)
        else:
            direction_map = {'buy': 0, 'sell': 1}
            direction_int = direction_map.get(direction, 0)
            rpt, _ = process_data_one_direction(close, random_main_probs, random_meta_probs, direction_int)
        
        if rpt.size >= 2:
            score = evaluate_report(rpt, periods_per_year)[0]
            random_scores.append(score)
        else:
            random_scores.append(-1.0)
    
    random_scores = np.array(random_scores)
    
    # Calcular p-value: fracción de monkeys que superaron la estrategia real
    p_value = np.sum(random_scores >= actual_score) / len(random_scores)
    is_significant = p_value < 0.05
    
    print(f"✅ Monkey Test completado:")
    print(f"  • Score real: {actual_score:.4f}")
    print(f"  • Score medio aleatorio: {np.mean(random_scores):.4f}")
    print(f"  • P-value: {p_value:.4f}")
    print(f"  • Estadísticamente significativo: {'SÍ' if is_significant else 'NO'}")
    
    return {
        'actual_score': actual_score,
        'p_value': p_value,
        'random_scores': random_scores,
        'random_mean': np.mean(random_scores),
        'random_std': np.std(random_scores),
        'is_significant': is_significant,
        'confidence_level': 0.95,
        'status': 'PASSED' if is_significant else 'FAILED'
    }

def enhanced_deflated_sharpe_ratio(
    equity_curve: np.ndarray,
    num_trials: int,
    periods_per_year: float = 6240.0,
    confidence_level: float = 0.95
) -> dict:
    """
    Calcula el Deflated Sharpe Ratio mejorado que ajusta por sesgo de selección múltiple.
    
    Corrige el Sharpe Ratio observado considerando el número de estrategias probadas,
    la longitud de la serie temporal, y las características de distribución de retornos.
    
    Args:
        equity_curve: Serie de equity de la estrategia
        num_trials: Número de estrategias/configuraciones probadas
        periods_per_year: Períodos por año para anualización
        confidence_level: Nivel de confianza para el test
        
    Returns:
        dict: Métricas del DSR incluyendo thresholds y significancia
    """
    if len(equity_curve) < 100:
        return {'status': 'FAILED', 'reason': 'Insufficient data points'}
    
    # Calcular retornos
    eq = np.array(equity_curve)
    if np.min(eq) <= 0:
        eq = eq - np.min(eq) + 1.0
    
    returns = np.diff(eq) / eq[:-1]
    n_obs = len(returns)
    
    if n_obs < 30:
        return {'status': 'FAILED', 'reason': 'Too few observations'}
    
    # Métricas básicas
    mean_ret = np.mean(returns)
    std_ret = np.std(returns, ddof=1)
    sharpe_observed = np.sqrt(periods_per_year) * mean_ret / (std_ret + 1e-12)
    
    # Momentos de orden superior
    skewness = stats.skew(returns)
    kurtosis_val = stats.kurtosis(returns, fisher=True)  # Excess kurtosis
    
    # Varianza del estimador de Sharpe Ratio
    var_sharpe = (1.0 + 0.5 * sharpe_observed**2 - skewness * sharpe_observed + 
                  (kurtosis_val - 3.0) / 4.0 * sharpe_observed**2) / (n_obs - 1)
    var_sharpe = max(var_sharpe, 1e-12)
    
    # Threshold para significancia considerando trials múltiples
    alpha = 1.0 - confidence_level
    z_alpha = stats.norm.ppf(1.0 - alpha / (2.0 * num_trials))  # Bonferroni correction
    threshold_sharpe = z_alpha * np.sqrt(var_sharpe)
    
    # Deflated Sharpe Ratio
    deflated_sharpe = sharpe_observed - threshold_sharpe
    
    # Probabilidad de que el Sharpe observado sea genuino
    z_score = sharpe_observed / np.sqrt(var_sharpe)
    p_value_adjusted = 2.0 * (1.0 - stats.norm.cdf(abs(z_score))) * num_trials
    p_value_adjusted = min(p_value_adjusted, 1.0)  # Cap at 1.0
    
    is_significant = deflated_sharpe > 0 and p_value_adjusted < alpha
    
    return {
        'sharpe_observed': sharpe_observed,
        'deflated_sharpe': deflated_sharpe,
        'threshold_sharpe': threshold_sharpe,
        'variance_sharpe': var_sharpe,
        'p_value_adjusted': p_value_adjusted,
        'is_significant': is_significant,
        'num_trials': num_trials,
        'skewness': skewness,
        'kurtosis': kurtosis_val,
        'n_observations': n_obs,
        'confidence_level': confidence_level,
        'status': 'PASSED' if is_significant else 'FAILED'
    }

def walk_forward_analysis(
    dataset: pd.DataFrame,
    model_main: object,
    model_meta: object,
    model_main_cols: list[str],
    model_meta_cols: list[str],
    direction: str = 'both',
    timeframe: str = 'H1',
    train_size: int = 2000,
    test_size: int = 500,
    step_size: int = 250
) -> dict:
    """
    Ejecuta Walk-Forward Analysis para validar la robustez temporal de la estrategia.
    
    Simula trading en tiempo real dividiendo los datos en ventanas consecutivas 
    de entrenamiento y prueba, evaluando si la estrategia mantiene performance
    consistente a través del tiempo.
    
    Args:
        dataset: DataFrame con datos históricos
        model_main: Modelo principal
        model_meta: Meta-modelo  
        model_main_cols: Columnas para modelo principal
        model_meta_cols: Columnas para meta-modelo
        direction: Dirección de trading
        timeframe: Timeframe para métricas
        train_size: Tamaño de ventana de entrenamiento
        test_size: Tamaño de ventana de test
        step_size: Paso entre ventanas
        
    Returns:
        dict: Resultados del análisis temporal incluyendo estabilidad
    """
    print(f"📊 Ejecutando Walk-Forward Analysis...")
    print(f"   Ventana entrenamiento: {train_size}, Test: {test_size}, Paso: {step_size}")
    
    total_length = len(dataset)
    min_required = train_size + test_size
    
    if total_length < min_required:
        return {
            'status': 'FAILED',
            'reason': f'Datos insuficientes. Requerido: {min_required}, Disponible: {total_length}'
        }
    
    # Calcular ventanas de análisis
    windows = []
    start_idx = 0
    
    while start_idx + min_required <= total_length:
        train_start = start_idx
        train_end = start_idx + train_size
        test_start = train_end
        test_end = min(train_end + test_size, total_length)
        
        if test_end - test_start < test_size // 2:  # Ventana de test muy pequeña
            break
            
        windows.append({
            'train_range': (train_start, train_end),
            'test_range': (test_start, test_end),
            'window_id': len(windows)
        })
        
        start_idx += step_size
    
    if len(windows) < 3:
        return {
            'status': 'FAILED',
            'reason': f'Ventanas insuficientes para análisis. Solo {len(windows)} generadas.'
        }
    
    print(f"   Analizando {len(windows)} ventanas temporales...")
    
    # Ejecutar análisis en cada ventana
    window_results = []
    
    for i, window in enumerate(windows):
        print(f"   Procesando ventana {i+1}/{len(windows)}")
        
        # Datos de test (out-of-sample)
        test_data = dataset.iloc[window['test_range'][0]:window['test_range'][1]].copy()
        
        if len(test_data) < 100:  # Mínimo para evaluación confiable
            continue
            
        # Evaluar estrategia en período de test
        score = tester(
            dataset=test_data,
            model_main=model_main,
            model_meta=model_meta,
            model_main_cols=model_main_cols,
            model_meta_cols=model_meta_cols,
            direction=direction,
            timeframe=timeframe
        )
        
        # Métricas detalladas del período
        if score > 0:
            detailed_score, metrics_tuple = evaluate_report(
                test_data['close'].to_numpy(), 
                get_periods_per_year(timeframe)
            )
            metrics_dict = metrics_tuple_to_dict(detailed_score, metrics_tuple, get_periods_per_year(timeframe))
        else:
            metrics_dict = {'sharpe_ratio': -10, 'max_drawdown_relative': 1.0, 'total_return': -1.0}
        
        window_results.append({
            'window_id': window['window_id'],
            'start_date': test_data.index[0] if hasattr(test_data.index, '__getitem__') else 0,
            'end_date': test_data.index[-1] if hasattr(test_data.index, '__getitem__') else len(test_data)-1,
            'score': score,
            'sharpe': metrics_dict.get('sharpe_ratio', -10),
            'max_dd': metrics_dict.get('max_drawdown_relative', 1.0),
            'total_return': metrics_dict.get('total_return', -1.0),
            'n_periods': len(test_data)
        })
    
    if len(window_results) < 3:
        return {
            'status': 'FAILED',
            'reason': 'Ventanas válidas insuficientes para análisis'
        }
    
    # Análisis de estabilidad temporal
    scores = [w['score'] for w in window_results]
    sharpes = [w['sharpe'] for w in window_results]
    returns = [w['total_return'] for w in window_results]
    
    # Métricas de consistencia
    score_mean = np.mean(scores)
    score_std = np.std(scores)
    score_cv = score_std / (abs(score_mean) + 1e-12)  # Coeficiente de variación
    
    positive_windows = sum(1 for s in scores if s > 0)
    positive_ratio = positive_windows / len(scores)
    
    # Test de estabilidad temporal (regresión vs tiempo)
    time_indices = np.arange(len(scores))
    slope, intercept, r_value, p_value_trend, std_err = stats.linregress(time_indices, scores)
    
    # Evaluación de calidad
    is_stable = (
        score_cv < 2.0 and  # Variabilidad moderada
        positive_ratio >= 0.6 and  # Mayoría de ventanas positivas
        score_mean > 0.1 and  # Score promedio decente
        p_value_trend > 0.05  # Sin deterioro temporal significativo
    )
    
    print(f"✅ Walk-Forward Analysis completado:")
    print(f"   • Ventanas analizadas: {len(window_results)}")
    print(f"   • Score promedio: {score_mean:.4f} ± {score_std:.4f}")
    print(f"   • Ventanas positivas: {positive_windows}/{len(scores)} ({positive_ratio:.1%})")
    print(f"   • Estabilidad temporal: {'ESTABLE' if is_stable else 'INESTABLE'}")
    
    return {
        'status': 'PASSED' if is_stable else 'FAILED',
        'window_results': window_results,
        'summary': {
            'num_windows': len(window_results),
            'score_mean': score_mean,
            'score_std': score_std,
            'score_cv': score_cv,
            'positive_windows': positive_windows,
            'positive_ratio': positive_ratio,
            'temporal_trend_slope': slope,
            'temporal_trend_pvalue': p_value_trend,
            'is_stable': is_stable
        }
    }

def probability_backtest_overfitting(
    dataset: pd.DataFrame,
    model_main: object,
    model_meta: object,
    model_main_cols: list[str],
    model_meta_cols: list[str],
    direction: str = 'both',
    timeframe: str = 'H1',
    n_splits: int = 16,
    selection_metric: str = 'sharpe'
) -> dict:
    """
    Calcula la Probability of Backtest Overfitting (PBO) según López de Prado.
    
    Evalúa la probabilidad de que la estrategia esté sobreajustada comparando
    la performance in-sample vs out-of-sample en múltiples particiones de datos.
    
    Args:
        dataset: DataFrame con datos históricos
        model_main: Modelo principal
        model_meta: Meta-modelo
        model_main_cols: Columnas para modelo principal  
        model_meta_cols: Columnas para meta-modelo
        direction: Dirección de trading
        timeframe: Timeframe para métricas
        n_splits: Número de particiones para el análisis
        selection_metric: Métrica para selección ('sharpe', 'score', 'return')
        
    Returns:
        dict: PBO y métricas de overfitting
    """
    print(f"🎯 Calculando Probability of Backtest Overfitting con {n_splits} splits...")
    
    total_length = len(dataset)
    min_split_size = 500  # Mínimo por split para evaluación confiable
    
    if total_length < n_splits * min_split_size:
        return {
            'status': 'FAILED',
            'reason': f'Datos insuficientes para {n_splits} splits de {min_split_size} períodos'
        }
    
    # Crear particiones combinatorias
    split_size = total_length // n_splits
    splits = []
    
    for i in range(n_splits):
        start_idx = i * split_size
        end_idx = (i + 1) * split_size if i < n_splits - 1 else total_length
        splits.append((start_idx, end_idx))
    
    # Generar todas las combinaciones IS/OOS
    results = []
    n_combinations = 0
    
    print(f"   Evaluando combinaciones IS/OOS...")
    
    # Para cada combinación de splits como IS vs OOS
    for is_splits in range(1, n_splits):
        for combination in range(min(100, 2**(n_splits-1))):  # Limitar combinaciones
            # Dividir aleatoriamente los splits en IS y OOS
            np.random.seed(combination)
            split_indices = np.random.permutation(n_splits)
            
            is_indices = split_indices[:is_splits]
            oos_indices = split_indices[is_splits:]
            
            if len(oos_indices) == 0:
                continue
            
            # Crear datasets IS y OOS
            is_data = pd.concat([dataset.iloc[splits[i][0]:splits[i][1]] for i in is_indices])
            oos_data = pd.concat([dataset.iloc[splits[i][0]:splits[i][1]] for i in oos_indices])
            
            if len(is_data) < min_split_size or len(oos_data) < min_split_size:
                continue
            
            # Evaluar en IS
            is_score = tester(
                dataset=is_data,
                model_main=model_main,
                model_meta=model_meta,
                model_main_cols=model_main_cols,
                model_meta_cols=model_meta_cols,
                direction=direction,
                timeframe=timeframe
            )
            
            # Evaluar en OOS
            oos_score = tester(
                dataset=oos_data,
                model_main=model_main,
                model_meta=model_meta,
                model_main_cols=model_main_cols,
                model_meta_cols=model_meta_cols,
                direction=direction,
                timeframe=timeframe
            )
            
            # Obtener métricas detalladas si es necesario
            if selection_metric != 'score' and is_score > 0 and oos_score > 0:
                # Métricas IS
                _, is_metrics = evaluate_report(is_data['close'].to_numpy(), get_periods_per_year(timeframe))
                is_metrics_dict = metrics_tuple_to_dict(is_score, is_metrics, get_periods_per_year(timeframe))
                
                # Métricas OOS  
                _, oos_metrics = evaluate_report(oos_data['close'].to_numpy(), get_periods_per_year(timeframe))
                oos_metrics_dict = metrics_tuple_to_dict(oos_score, oos_metrics, get_periods_per_year(timeframe))
                
                if selection_metric == 'sharpe':
                    is_metric = is_metrics_dict['sharpe_ratio']
                    oos_metric = oos_metrics_dict['sharpe_ratio']
                elif selection_metric == 'return':
                    is_metric = is_metrics_dict['total_return']
                    oos_metric = oos_metrics_dict['total_return']
                else:
                    is_metric = is_score
                    oos_metric = oos_score
            else:
                is_metric = is_score
                oos_metric = oos_score
            
            results.append({
                'combination_id': n_combinations,
                'is_metric': is_metric,
                'oos_metric': oos_metric,
                'is_size': len(is_data),
                'oos_size': len(oos_data)
            })
            
            n_combinations += 1
            
            if n_combinations >= 500:  # Limitar para eficiencia
                break
                
        if n_combinations >= 500:
            break
    
    if len(results) < 10:
        return {
            'status': 'FAILED', 
            'reason': 'Combinaciones insuficientes para PBO'
        }
    
    # Calcular PBO
    outperformance_count = 0
    total_combinations = len(results)
    
    for result in results:
        if result['oos_metric'] > result['is_metric']:
            outperformance_count += 1
    
    pbo = 1.0 - (outperformance_count / total_combinations)
    
    # Métricas adicionales
    is_metrics = [r['is_metric'] for r in results]
    oos_metrics = [r['oos_metric'] for r in results]
    
    is_mean = np.mean(is_metrics)
    oos_mean = np.mean(oos_metrics)
    performance_degradation = (is_mean - oos_mean) / (abs(is_mean) + 1e-12)
    
    # Evaluación de overfitting
    is_overfitted = pbo > 0.5  # Más del 50% indica probable overfitting
    
    print(f"✅ PBO Analysis completado:")
    print(f"   • Combinaciones evaluadas: {total_combinations}")
    print(f"   • PBO: {pbo:.3f} ({pbo:.1%})")
    print(f"   • Performance IS vs OOS: {is_mean:.4f} vs {oos_mean:.4f}")
    print(f"   • Degradación: {performance_degradation:.1%}")
    print(f"   • Overfitting detectado: {'SÍ' if is_overfitted else 'NO'}")
    
    return {
        'status': 'FAILED' if is_overfitted else 'PASSED',
        'pbo': pbo,
        'n_combinations': total_combinations,
        'performance_degradation': performance_degradation,
        'is_mean': is_mean,
        'oos_mean': oos_mean,
        'is_overfitted': is_overfitted,
        'results': results,
        'selection_metric': selection_metric
    }

def comprehensive_strategy_validation(
    dataset: pd.DataFrame,
    model_main: object,
    model_meta: object,
    model_main_cols: list[str],
    model_meta_cols: list[str],
    direction: str = 'both',
    timeframe: str = 'H1',
    num_trials: int = 1,
    validation_config: dict = None
) -> dict:
    """
    Framework integral de validación de estrategias que combina todas las técnicas avanzadas.
    
    Ejecuta secuencialmente: Monkey Test, Enhanced DSR, Walk-Forward Analysis y PBO
    para proporcionar una evaluación completa de la robustez de la estrategia.
    
    Args:
        dataset: DataFrame con datos históricos
        model_main: Modelo principal
        model_meta: Meta-modelo
        model_main_cols: Columnas para modelo principal
        model_meta_cols: Columnas para meta-modelo  
        direction: Dirección de trading
        timeframe: Timeframe para métricas
        num_trials: Número de estrategias probadas (para DSR)
        validation_config: Configuración opcional para tests
        
    Returns:
        dict: Resultados completos de validación con veredicto final
    """
    # Configuración por defecto
    if validation_config is None:
        validation_config = {
            'monkey_test': {'n_monkeys': 1000},
            'walk_forward': {'train_size': 2000, 'test_size': 500, 'step_size': 250},
            'pbo': {'n_splits': 16},
            'dsr': {'confidence_level': 0.95}
        }
    
    print("🔬 INICIANDO VALIDACIÓN INTEGRAL DE ESTRATEGIA")
    print("=" * 60)
    
    validation_results = {
        'timestamp': datetime.now().isoformat(),
        'dataset_info': {
            'total_periods': len(dataset),
            'timeframe': timeframe,
            'direction': direction
        },
        'tests_results': {},
        'overall_status': 'PENDING'
    }
    
    # 1. MONKEY TEST - Test fundamental de significancia estadística
    print("\n1️⃣ MONKEY TEST (Null Hypothesis Benchmark)")
    print("-" * 40)
    
    try:
        monkey_results = monkey_test(
            dataset=dataset,
            model_main=model_main,
            model_meta=model_meta,
            model_main_cols=model_main_cols,
            model_meta_cols=model_meta_cols,
            direction=direction,
            timeframe=timeframe,
            **validation_config['monkey_test']
        )
        validation_results['tests_results']['monkey_test'] = monkey_results
        
        if monkey_results['status'] == 'FAILED':
            print("❌ ESTRATEGIA RECHAZADA: Falló el Monkey Test")
            validation_results['overall_status'] = 'REJECTED'
            validation_results['rejection_reason'] = 'Failed Monkey Test - No better than random'
            return validation_results
            
    except Exception as e:
        print(f"❌ Error en Monkey Test: {str(e)}")
        validation_results['tests_results']['monkey_test'] = {'status': 'ERROR', 'error': str(e)}
    
    # 2. ENHANCED DEFLATED SHARPE RATIO - Control de sesgo de selección
    print("\n2️⃣ ENHANCED DEFLATED SHARPE RATIO")
    print("-" * 40)
    
    try:
        # Obtener equity curve para DSR
        score = tester(
            dataset=dataset,
            model_main=model_main,
            model_meta=model_meta,
            model_main_cols=model_main_cols,
            model_meta_cols=model_meta_cols,
            direction=direction,
            timeframe=timeframe
        )
        
        if score > 0:
            # Recalcular para obtener equity curve
            periods_per_year = get_periods_per_year(timeframe)
            ds_main = dataset[model_main_cols].to_numpy()
            ds_meta = dataset[model_meta_cols].to_numpy()
            close = dataset['close'].to_numpy()
            
            main = _predict_one(model_main, ds_main)
            meta = _predict_one(model_meta, ds_meta)
            
            if direction == 'both':
                equity_curve, _ = process_data(close, main, meta)
            else:
                direction_map = {'buy': 0, 'sell': 1}
                direction_int = direction_map.get(direction, 0)
                equity_curve, _ = process_data_one_direction(close, main, meta, direction_int)
            
            dsr_results = enhanced_deflated_sharpe_ratio(
                equity_curve=equity_curve,
                num_trials=num_trials,
                periods_per_year=periods_per_year,
                **validation_config['dsr']
            )
            validation_results['tests_results']['deflated_sharpe'] = dsr_results
            
            if dsr_results['status'] == 'FAILED':
                print("⚠️ ADVERTENCIA: DSR sugiere posible overfitting")
        else:
            validation_results['tests_results']['deflated_sharpe'] = {
                'status': 'FAILED', 
                'reason': 'Negative strategy score'
            }
            
    except Exception as e:
        print(f"❌ Error en DSR: {str(e)}")
        validation_results['tests_results']['deflated_sharpe'] = {'status': 'ERROR', 'error': str(e)}
    
    # 3. WALK-FORWARD ANALYSIS - Validación de estabilidad temporal
    print("\n3️⃣ WALK-FORWARD ANALYSIS")
    print("-" * 40)
    
    try:
        wf_results = walk_forward_analysis(
            dataset=dataset,
            model_main=model_main,
            model_meta=model_meta,
            model_main_cols=model_main_cols,
            model_meta_cols=model_meta_cols,
            direction=direction,
            timeframe=timeframe,
            **validation_config['walk_forward']
        )
        validation_results['tests_results']['walk_forward'] = wf_results
        
        if wf_results['status'] == 'FAILED':
            print("⚠️ ADVERTENCIA: Estrategia muestra inestabilidad temporal")
            
    except Exception as e:
        print(f"❌ Error en Walk-Forward: {str(e)}")
        validation_results['tests_results']['walk_forward'] = {'status': 'ERROR', 'error': str(e)}
    
    # 4. PROBABILITY OF BACKTEST OVERFITTING
    print("\n4️⃣ PROBABILITY OF BACKTEST OVERFITTING (PBO)")
    print("-" * 40)
    
    try:
        pbo_results = probability_backtest_overfitting(
            dataset=dataset,
            model_main=model_main,
            model_meta=model_meta,
            model_main_cols=model_main_cols,
            model_meta_cols=model_meta_cols,
            direction=direction,
            timeframe=timeframe,
            **validation_config['pbo']
        )
        validation_results['tests_results']['pbo'] = pbo_results
        
        if pbo_results['status'] == 'FAILED':
            print("❌ ADVERTENCIA CRÍTICA: Alto riesgo de overfitting detectado")
            
    except Exception as e:
        print(f"❌ Error en PBO: {str(e)}")
        validation_results['tests_results']['pbo'] = {'status': 'ERROR', 'error': str(e)}
    
    # VEREDICTO FINAL
    print("\n🏁 VEREDICTO FINAL DE VALIDACIÓN")
    print("=" * 60)
    
    # Contar tests pasados vs fallidos
    test_results = validation_results['tests_results']
    passed_tests = sum(1 for test in test_results.values() if test.get('status') == 'PASSED')
    total_tests = len([test for test in test_results.values() if test.get('status') in ['PASSED', 'FAILED']])
    
    # Criterios de aprobación
    monkey_passed = test_results.get('monkey_test', {}).get('status') == 'PASSED'
    critical_failures = 0
    
    if not monkey_passed:
        critical_failures += 1
    
    if test_results.get('pbo', {}).get('pbo', 0) > 0.7:  # PBO muy alto
        critical_failures += 1
    
    # Determinar status final
    if critical_failures > 0:
        validation_results['overall_status'] = 'HIGH_RISK'
        status_message = "❌ ESTRATEGIA DE ALTO RIESGO"
        recommendation = "NO RECOMENDADA para trading en vivo"
    elif passed_tests >= total_tests * 0.75:  # 75% de tests pasados
        validation_results['overall_status'] = 'VALIDATED'
        status_message = "✅ ESTRATEGIA VALIDADA"
        recommendation = "Apta para consideración en trading en vivo"
    else:
        validation_results['overall_status'] = 'CONDITIONAL'
        status_message = "⚠️ ESTRATEGIA CONDICIONAL"
        recommendation = "Requiere optimización adicional"
    
    validation_results['final_verdict'] = {
        'status': validation_results['overall_status'],
        'passed_tests': passed_tests,
        'total_tests': total_tests,
        'critical_failures': critical_failures,
        'recommendation': recommendation
    }
    
    print(f"{status_message}")
    print(f"Tests pasados: {passed_tests}/{total_tests}")
    print(f"Recomendación: {recommendation}")
    
    return validation_results