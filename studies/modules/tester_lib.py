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
# Constantes financieras
RISK_FREE_RATE = 0.02    # Tasa libre de riesgo anual (2%)


@njit(cache=True, fastmath=True)
def _signed_r2(eq):
    """R² con signo - versión optimizada que favorece pendientes positivas"""
    n = eq.size
    t = np.arange(n, dtype=np.float64)
    xm = t.mean()
    ym = eq.mean()
    cov = ((t-xm)*(eq-ym)).sum()
    var_t = ((t-xm)**2).sum()
    var_y = ((eq-ym)**2).sum()
    
    if var_t == 0.0 or var_y == 0.0:
        return 0.0
    
    slope = cov / var_t
    r2 = (cov*cov)/(var_t*var_y)
    
    # Bonus significativo para pendientes positivas
    if slope > 0:
        # Potenciar el R² para pendientes positivas
        r2_enhanced = min(1.0, r2 * (1.0 + slope/10.0))  # Bonus por pendiente
        return r2_enhanced
    else:
        # Penalizar fuertemente pendientes negativas
        return -r2


@njit(cache=True, fastmath=True)
def _linearity_bonus(eq):
    """Calcula un bonus específico por linealidad ascendente perfecta"""
    n = eq.size
    if n < 10:
        return 0.0
    
    # Ajuste lineal manual
    t = np.arange(n, dtype=np.float64)
    x_mean = t.mean()
    y_mean = eq.mean()
    
    # Calcular pendiente
    num = np.sum((t - x_mean) * (eq - y_mean))
    den = np.sum((t - x_mean) ** 2)
    
    if den == 0:
        return 0.0
    
    slope = num / den
    
    # Solo bonus para pendientes positivas
    if slope <= 0:
        return 0.0
    
    # Calcular linealidad (R²)
    y_pred = slope * (t - x_mean) + y_mean
    ss_res = np.sum((eq - y_pred) ** 2)
    ss_tot = np.sum((eq - y_mean) ** 2)
    
    if ss_tot == 0:
        return 1.0 if slope > 0 else 0.0
    
    r2 = 1.0 - (ss_res / ss_tot)
    
    # Bonus combinado: linealidad * pendiente normalizada
    # Pendiente ideal: entre 0.1 y 2.0
    slope_normalized = min(1.0, max(0.0, slope / 2.0))
    linear_bonus = r2 * slope_normalized
    
    return max(0.0, min(1.0, linear_bonus))


@njit(cache=True, fastmath=True)
def _consistency_score(eq):
    """Evalúa la consistencia del crecimiento (sin volatilidad excesiva)"""
    if eq.size < 3:
        return 0.0
    
    # Calcular diferencias (returns)
    diffs = np.diff(eq)
    
    # Porcentaje de períodos con crecimiento positivo
    positive_periods = np.sum(diffs > 0) / len(diffs)
    
    # Consistencia de la dirección
    direction_consistency = min(1.0, positive_periods * 1.5)  # Favor hacia crecimiento
    
    # Penalizar volatilidad excesiva relativa al crecimiento promedio
    if len(diffs) > 0:
        mean_growth = np.mean(diffs)
        if mean_growth > 0:
            volatility = np.std(diffs)
            vol_ratio = volatility / (mean_growth + 1e-8)
            vol_penalty = 1.0 / (1.0 + vol_ratio * 2.0)  # Penalizar alta volatilidad
        else:
            vol_penalty = 0.0
    else:
        vol_penalty = 1.0
    
    return direction_consistency * vol_penalty


@njit(cache=True, fastmath=True)
def _slope_reward(eq):
    """Recompensa específica por pendiente ascendente fuerte"""
    n = eq.size
    if n < 2:
        return 0.0
    
    # Pendiente simple: (final - inicial) / tiempo
    total_growth = eq[-1] - eq[0]
    time_span = n - 1
    
    if time_span == 0:
        return 0.0
    
    slope = total_growth / time_span
    
    # Normalizar pendiente (rango ideal: 0.1 a 2.0)
    if slope <= 0:
        return 0.0
    
    # Función sigmoide para recompensar pendientes moderadas a altas
    # Pendientes muy pequeñas obtienen poco reward
    # Pendientes ideales (0.2-1.0) obtienen máximo reward
    # Pendientes muy altas también se recompensan pero menos
    
    if slope < 0.1:
        return slope / 0.1 * 0.3  # Pendientes muy pequeñas: reward mínimo
    elif slope <= 1.0:
        return 0.3 + (slope - 0.1) / 0.9 * 0.7  # Rango ideal: reward lineal
    else:
        # Pendientes altas: reward alto pero decreciente
        excess = slope - 1.0
        return 1.0 * np.exp(-excess * 0.2)  # Decae exponencialmente


@njit(cache=True, fastmath=True)
def evaluate_report(eq: np.ndarray, ppy: float = 6240.0):
    """
    Sistema de scoring optimizado que favorece curvas lineales ascendentes perfectas.
    
    Returns:
        tuple: (score, metrics_tuple) donde score está optimizado para linealidad ascendente
    """
    # Validaciones básicas
    if eq.size < 300 or not np.isfinite(eq).all():
        return (-1.0, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    
    # Protección para valores negativos
    eq_min = np.min(eq)
    if eq_min <= 0.0:
        eq = eq - eq_min + 1.0
    
    # === NUEVAS MÉTRICAS OPTIMIZADAS ===
    
    # 1. R² con bonus por pendiente positiva
    r2 = _signed_r2(eq)
    
    # 2. Bonus específico por linealidad ascendente
    linearity_bonus = _linearity_bonus(eq)
    
    # 3. Consistencia del crecimiento
    consistency = _consistency_score(eq)
    
    # 4. Recompensa por pendiente fuerte
    slope_reward = _slope_reward(eq)
    
    # 5. Retorno total normalizado
    total_return = (eq[-1] - eq[0]) / max(abs(eq[0]), 1.0)
    
    # 6. Penalización por drawdown (simplificada)
    peak = eq[0]
    max_dd = 0.0
    for val in eq:
        if val > peak:
            peak = val
        else:
            dd = (peak - val) / peak if peak > 0 else 0.0
            max_dd = max(max_dd, dd)
    
    # Penalización por drawdown exponencial
    dd_penalty = np.exp(-max_dd * 10.0)  # Fuerte penalización por DD > 10%
    
    # === SCORE OPTIMIZADO ===
    
    # Componentes principales
    linearity_component = (r2 + linearity_bonus) / 2.0  # [0,1]
    growth_component = (slope_reward + consistency) / 2.0          # [0,1]
    
    # Score base: promedio ponderado favoreciendo linealidad
    base_score = (
        linearity_component * 0.5 +  # 50% peso a linealidad
        growth_component * 0.3 +      # 30% peso a crecimiento
        min(1.0, max(0.0, total_return)) * 0.2  # 20% peso a retorno total
    )
    
    # Aplicar penalización por drawdown
    final_score = base_score * dd_penalty
    
    # Bonus adicional para curvas perfectamente lineales ascendentes
    if r2 > 0.98 and slope_reward > 0.5 and max_dd < 0.01:
        final_score = min(1.0, final_score * 1.2)  # Bonus del 20%
    
    # Asegurar rango [0,1]
    final_score = max(0.0, min(1.0, final_score))
    
    # === MÉTRICAS PARA DEBUGGING ===
    metrics_tuple = (
        r2, linearity_bonus, consistency, slope_reward,
        total_return, max_dd, dd_penalty, linearity_component,
        growth_component, base_score, final_score,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0  # Padding
    )
    
    return final_score, metrics_tuple


def metrics_tuple_to_dict(score: float, metrics_tuple: tuple, periods_per_year: float) -> dict:
    """Convierte la tupla de métricas optimizada a diccionario"""
    return {
        'score': score,
        'r2': metrics_tuple[0],
        'linearity_bonus': metrics_tuple[1],
        'consistency': metrics_tuple[2],
        'slope_reward': metrics_tuple[3],
        'total_return': metrics_tuple[4],
        'max_drawdown': metrics_tuple[5],
        'dd_penalty': metrics_tuple[6],
        'linearity_component': metrics_tuple[7],
        'growth_component': metrics_tuple[8],
        'base_score': metrics_tuple[9],
        'final_score': metrics_tuple[10],
        'periods_per_year': periods_per_year
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
          f"  • R²={metrics['r2']:.4f} | Linearity Bonus={metrics['linearity_bonus']:.4f}\n"
          f"  • Consistency={metrics['consistency']:.4f} | Slope Reward={metrics['slope_reward']:.4f}\n"
          f"  • Total Return={metrics['total_return']:.4f} | Max Drawdown={metrics['max_drawdown']:.4f}\n"
          f"  • DD Penalty={metrics['dd_penalty']:.4f}\n"
          f"  • Linearity Comp={metrics['linearity_component']:.4f} | Growth Comp={metrics['growth_component']:.4f}\n"
          f"  • Base Score={metrics['base_score']:.4f} | Final Score={metrics['final_score']:.4f}\n"
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