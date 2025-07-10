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
    
# Thread-safe session cache
_session_cache = {}
_session_lock = threading.RLock()

# Thread-safe plotting
_plot_lock = threading.RLock()

def _safe_plot(equity_curve, score):
    """Thread-safe plotting function"""
    with _plot_lock:
        plt.figure(figsize=(10, 6))
        plt.plot(equity_curve, label='Equity Curve', linewidth=1.5)
        plt.title(f"Score: {score:.6f}")
        plt.xlabel("Trades")
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
    
    # ── NUEVAS VARIABLES PARA TRACKING DE TRADES ──
    trade_profits = []  # Lista de profits individuales
    trade_count = 0     # Contador total de trades

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
            # ── REGISTRAR TRADE ──
            trade_profits.append(profit)
            trade_count += 1
            continue

        if last_deal == 1 and pred < 0.5:
            last_deal = 2
            profit = (last_price - pr)
            report.append(report[-1] + profit)
            chart.append(chart[-1] + (pr - last_price))
            # ── REGISTRAR TRADE ──
            trade_profits.append(profit)
            trade_count += 1
            continue

    # Cierre forzoso al final si sigue abierta
    if last_deal == 0:
        profit = close[-1] - last_price
        report.append(report[-1] + profit)
        chart.append(chart[-1] + profit)
        # ── REGISTRAR TRADE FINAL ──
        trade_profits.append(profit)
        trade_count += 1
    elif last_deal == 1:
        profit = last_price - close[-1]
        report.append(report[-1] + profit)
        chart.append(chart[-1] + (close[-1] - last_price))
        # ── REGISTRAR TRADE FINAL ──
        trade_profits.append(profit)
        trade_count += 1

    # ── CALCULAR ESTADÍSTICAS DE TRADES ──
    if trade_count > 0:
        trade_profits_array = np.array(trade_profits)
        positive_trades = np.sum(trade_profits_array > 0)
        negative_trades = np.sum(trade_profits_array < 0)
        zero_trades = np.sum(trade_profits_array == 0)
        win_rate = positive_trades / trade_count
        avg_positive = np.mean(trade_profits_array[trade_profits_array > 0]) if positive_trades > 0 else 0.0
        avg_negative = np.mean(trade_profits_array[trade_profits_array < 0]) if negative_trades > 0 else 0.0
    else:
        positive_trades = negative_trades = zero_trades = 0
        win_rate = avg_positive = avg_negative = 0.0

    # ── EMPAQUETIZAR ESTADÍSTICAS ──
    trade_stats = np.array([
        trade_count, positive_trades, negative_trades, zero_trades,
        win_rate, avg_positive, avg_negative
    ], dtype=np.float64)

    return np.array(report), np.array(chart), trade_stats

@njit(cache=True, fastmath=True)
def process_data_one_direction(close, main_labels, meta_labels, direction_int):
    last_deal  = 2            # 2 = flat, 1 = position open
    last_price = 0.0
    report = [0.0]
    chart  = [0.0]
    long_side = (direction_int == 0)  # 0=buy, 1=sell
    min_prob  = 0.5
    
    # ── NUEVAS VARIABLES PARA TRACKING DE TRADES ──
    trade_profits = []  # Lista de profits individuales
    trade_count = 0     # Contador total de trades

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
            # ── REGISTRAR TRADE ──
            trade_profits.append(profit)
            trade_count += 1

    # Cierre forzoso al final si sigue abierta
    if last_deal == 1:
        profit = (close[-1] - last_price) if long_side else (last_price - close[-1])
        report.append(report[-1] + profit)
        chart.append(chart[-1]  + profit)
        # ── REGISTRAR TRADE FINAL ──
        trade_profits.append(profit)
        trade_count += 1

    # ── CALCULAR ESTADÍSTICAS DE TRADES ──
    if trade_count > 0:
        trade_profits_array = np.array(trade_profits)
        positive_trades = np.sum(trade_profits_array > 0)
        negative_trades = np.sum(trade_profits_array < 0)
        zero_trades = np.sum(trade_profits_array == 0)
        win_rate = positive_trades / trade_count
        avg_positive = np.mean(trade_profits_array[trade_profits_array > 0]) if positive_trades > 0 else 0.0
        avg_negative = np.mean(trade_profits_array[trade_profits_array < 0]) if negative_trades > 0 else 0.0
    else:
        positive_trades = negative_trades = zero_trades = 0
        win_rate = avg_positive = avg_negative = 0.0

    # ── EMPAQUETIZAR ESTADÍSTICAS ──
    trade_stats = np.array([
        trade_count, positive_trades, negative_trades, zero_trades,
        win_rate, avg_positive, avg_negative
    ], dtype=np.float64)

    return np.asarray(report, dtype=np.float64), np.asarray(chart, dtype=np.float64), trade_stats

def tester(
        dataset: pd.DataFrame,
        model_main: object,
        model_meta: object,
        model_main_cols: list[str],
        model_meta_cols: list[str],
        direction: str = 'both',
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
    timeframe : str, optional
        Timeframe de los datos para cálculos de anualización. Por defecto ``'H1'``.
    print_metrics : bool, optional
        Si ``True`` imprime métricas detalladas para debugging. Por defecto ``False``.

    Returns
    -------
    float
        Puntuación de la estrategia según :func:`evaluate_report`.
    """
    try:
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
            rpt, _, trade_stats = process_data(close, main, meta)
        else:
            direction_map = {'buy': 0, 'sell': 1}
            direction_int = direction_map.get(direction, 0)
            rpt, _, trade_stats = process_data_one_direction(close, main, meta, direction_int)

        score = evaluate_report(rpt)
        # if print_metrics:
        #     _safe_plot(rpt, score)

        return score
    
    except Exception as e:
        print(f"🔍 DEBUG: Error en tester: {e}")
        return -1.0

# ────────── helpers ──────────────────────────────────────────────────────────

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
    
@njit(cache=True, fastmath=True)
def manual_linear_regression(x, y):
    """
    Regresión lineal manual optimizada con numba.
    
    Args:
        x: array 1D de valores independientes
        y: array 1D de valores dependientes
    
    Returns:
        tuple: (r2_signed, slope, intercept)
    """
    n = len(x)
    if n < 2:
        return 0.0, 0.0
    
    # Medias
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    # Calcular numerador y denominador para la pendiente
    numerator = 0.0
    denominator = 0.0
    
    for i in range(n):
        x_diff = x[i] - x_mean
        y_diff = y[i] - y_mean
        numerator += x_diff * y_diff
        denominator += x_diff * x_diff
    
    # Evitar división por cero
    if abs(denominator) < 1e-12:
        return 0.0, 0.0
    
    # Pendiente e intercepto
    slope = numerator / denominator
    intercept = y_mean - slope * x_mean
    
    # Calcular R²
    ss_res = 0.0  # Suma de cuadrados residuales
    ss_tot = 0.0  # Suma de cuadrados totales
    
    for i in range(n):
        y_pred = slope * x[i] + intercept
        y_diff_mean = y[i] - y_mean
        y_diff_pred = y[i] - y_pred
        
        ss_res += y_diff_pred * y_diff_pred
        ss_tot += y_diff_mean * y_diff_mean
    
    # Calcular R²
    if abs(ss_tot) < 1e-12:
        r2 = 0.0
    else:
        r2 = 1.0 - (ss_res / ss_tot)
    
    # Aplicar signo basado en la pendiente
    sign = 1.0 if slope >= 0 else -1.0
    r2_signed = r2 * sign
    
    return r2_signed, slope

@njit(cache=True, fastmath=True)
def evaluate_report(
    equity_curve: np.ndarray,
    min_trades: int = 200,
    rdd_threshold: float = 3.0,
) -> float:
    """
    Devuelve un score de [0, +∞) para curvas de equity.
    - equity_curve: serie acumulada.
    """

    n = equity_curve.size
    if n < 2:
        return -1.0

    returns = np.diff(equity_curve)
    num_trades = returns.size
    if num_trades < 50:
        return -1.0

    # ── Métricas base ─────────────────────────────────────────────
    running_max = np.empty_like(equity_curve)
    running_max[0] = equity_curve[0]
    for i in range(1, equity_curve.size):
        running_max[i] = max(running_max[i - 1], equity_curve[i])
    max_dd = np.max(running_max - equity_curve)
    total_ret = equity_curve[-1] - equity_curve[0]
    if total_ret <= 0:
        return -1.0
    rdd = total_ret / (max_dd + 1e-9)
    if rdd < rdd_threshold:
        rdd *= 0.9
    # log1p para suavizar
    rdd_nl = np.log1p(max(0.0, rdd))

    # ── Ajuste por nº de trades ───────────────────────────────────
    trade_nl = 1 + np.log1p(max(0, num_trades - min_trades))
    
    # ── Linealidad ───────────────────────────────────────────────
    x = np.arange(n, dtype=np.float64)
    y = equity_curve.astype(np.float64)
    r2, slope = manual_linear_regression(x, y)
    # lin_score: tanh para acotar, y log1p para suavizar pendiente
    lin_nl = np.tanh(r2) * np.log1p(max(0.0, slope))
    
    final = rdd_nl * lin_nl * trade_nl

    return final