import threading
from numba import njit
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import onnxruntime as rt
from typing import List, Tuple

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
        
        # Preparación de datos
        ds_main = dataset[model_main_cols].to_numpy()
        ds_meta = dataset[model_meta_cols].to_numpy()
        close = dataset['close'].to_numpy()

        # Calcular probabilidades usando ambos modelos (sin binarizar)
        main = predict_proba_onnx_models([model_main], ds_main)
        meta = predict_proba_onnx_models([model_meta], ds_meta)

        # Como predict_proba_onnx_models devuelve shape (n_models, n_samples), 
        # necesitamos aplanar para obtener shape (n_samples,)
        main = main.flatten()
        meta = meta.flatten()

        # Asegurar contigüidad en memoria
        close = np.ascontiguousarray(close)
        main = np.ascontiguousarray(main)
        meta = np.ascontiguousarray(meta)

        # ── BACKTEST ──────────────────────────────────────────────────
        dir_flag = {"buy": 0, "sell": 1, "both": 2}[direction]
        if dir_flag == 0:
            pb = main
            ps = np.zeros_like(main)
        elif dir_flag == 1:
            pb = np.zeros_like(main)
            ps = main
        else:
            pb = main
            ps = 1.0 - main

        rpt, _, trade_stats = backtest_equivalent(
            close,
            prob_buy   = pb,
            prob_sell  = ps,
            meta_sig   = meta,
            main_thr   = 0.5,
            meta_thr   = 0.5,
            direction  = dir_flag,
            max_orders = 0,
            delay_bars = 1
        )

        periods_per_year = get_periods_per_year(timeframe)
        trade_nl, rdd_nl, r2, slope_nl, calmar_nl = evaluate_report(rpt, periods_per_year=periods_per_year)
        if (trade_nl <= -1.0 and rdd_nl <= -1.0 and r2 <= -1.0 and slope_nl <= -1.0 and calmar_nl <= -1.0):
            return -1.0
        # Asignar pesos para promover sobre todo el ajuste de los datos a la recta (r2)
        # Ejemplo de pesos: r2 0.5, pendiente 0.15, rdd 0.1, calmar 0.15, trade_nl 0.1
        score = (
            0.5 * r2 +
            0.15 * slope_nl +
            0.1 * rdd_nl +
            0.15 * calmar_nl +
            0.1 * trade_nl
        )
        if score < 0.0:
            return -1.0
        if print_metrics:
            print(f"🔍 DEBUG - Métricas de evaluación: trade_nl: {trade_nl}, rdd_nl: {rdd_nl}, r2: {r2}, slope_nl: {slope_nl}, calmar_nl: {calmar_nl}")
            print(f"🔍 DEBUG - Trade stats: n_trades: {trade_stats[0]}, n_positivos: {trade_stats[1]}, n_negativos: {trade_stats[2]}")
            plt.figure(figsize=(10, 6))
            plt.plot(rpt, label='Equity Curve', linewidth=1.5)
            plt.title(f"Score: {score:.6f}")
            plt.xlabel("Trades")
            plt.ylabel("Cumulative P&L")
            plt.legend()
            plt.grid(alpha=0.3)
            plt.show()
            plt.close()

        return score
    
    except Exception as e:
        print(f"🔍 DEBUG: Error en tester: {e}")
        return -1.0

@njit(cache=True, fastmath=True)
def evaluate_report(
    equity_curve: np.ndarray,
    periods_per_year: float = 6240.0,
    min_trades: int = 200,
    rdd_floor: float = 1.0,
    calmar_floor: float = 1.0
) -> tuple[float, float, float, float, float]:
    """
    Devuelve un score escalar para Optuna.
    Premia:
        1. pendiente media positiva y estable
        2. ratio retorno / drawdown alto
        3. número suficiente de trades
        4. buen ajuste lineal (R²)
        5. Calmar Ratio anualizado alto
    Penaliza curvas cortas, con drawdown cero o pendiente negativa.
    """
    n = equity_curve.size
    if n < 2:
        return -1.0, -1.0, -1.0, -1.0, -1.0

    # ---------- nº de trades (normalizado) -----------------------------------
    returns = np.diff(equity_curve)
    n_trades = returns.size
    if n_trades < min_trades:
        return -1.0, -1.0, -1.0, -1.0, -1.0
    # Normalización avanzada: escalar n_trades entre 0 y 1 usando una función logística suave
    # El centro de la transición es min_trades, la pendiente la controla el divisor (ajustable)
    trade_nl = 1.0 / (1.0 + np.exp(-(n_trades - min_trades) / (min_trades * 5.0)))

    # ---------- return / drawdown (normalizado) ------------------------------
    running_max = np.empty_like(equity_curve)
    running_max[0] = equity_curve[0]
    for i in range(1, n):
        running_max[i] = running_max[i - 1] if running_max[i - 1] > equity_curve[i] else equity_curve[i]
    max_dd = np.max(running_max - equity_curve)
    total_ret = equity_curve[-1] - equity_curve[0]
    # Protección contra división por cero en max_dd
    if max_dd == 0.0:
        rdd = 0.0
    else:
        rdd = total_ret / max_dd
    if rdd < rdd_floor:
        return -1.0, -1.0, -1.0, -1.0, -1.0

    # Normalización avanzada de rdd entre 0 y 1 usando función logística suave
    # El centro de la transición es rdd_floor, la pendiente la controla el divisor (ajustable)
    # Esto penaliza valores bajos y suaviza la transición
    rdd_nl = 1.0 / (1.0 + np.exp(-(rdd - rdd_floor) / (rdd_floor * 5.0)))

    # ---------- Calmar Ratio anualizado (normalizado) -------------------------
    # Calmar Ratio = Retorno Anualizado / Drawdown Máximo
    if max_dd == 0.0:
        calmar_ratio = 0.0
    else:
        # Anualizar el retorno: (retorno por trade) * (trades por año)
        annualized_return = (total_ret / n_trades) * periods_per_year
        calmar_ratio = annualized_return / max_dd
    if calmar_ratio < calmar_floor:
        return -1.0, -1.0, -1.0, -1.0, -1.0
    # Normalización avanzada del Calmar Ratio usando función logística suave
    # El centro de la transición es calmar_floor, la pendiente la controla el divisor
    calmar_nl = 1.0 / (1.0 + np.exp(-(calmar_ratio - calmar_floor) / (calmar_floor * 65.0)))

    # ---------- linealidad y pendiente (normalizado) ---------------------------
    x = np.arange(n, dtype=np.float64)
    y = equity_curve.astype(np.float64)
    r2, slope = manual_linear_regression(x, y)
    if slope < 0.0:
        return -1.0, -1.0, -1.0, -1.0, -1.0
    slope_nl = min(1.0, 1.0 / (1.0 + np.exp(-(np.log1p(slope / n) - 0.001) / 0.003)))

    return trade_nl, rdd_nl, r2, slope_nl, calmar_nl

@njit(cache=True, fastmath=True)
def backtest_equivalent(close,
                        prob_buy,          # numpy[float] - P(buy) de la red MAIN
                        prob_sell,         # numpy[float] - P(sell) idem
                        meta_sig,          # numpy[float] - P(clase 1) de la red META
                        main_thr   = 0.5,
                        meta_thr   = 0.5,
                        direction  = 2,    # 0=solo buy, 1=solo sell, 2=both
                        max_orders = 0,    # 0 → ilimitado
                        delay_bars = 1):
    """
    Réplica exacta de la lógica MQL5 para abrir/cerrar posiciones.
    Devuelve:
        report  – equity acumulada por trade   (len = nº_trades + 1)
        chart   – equity tick-a-tick (para dibujar)
        stats   – tupla (n_trades, n_positivos, n_negativos)
    """
    FLAT, LONG, SHORT = 2, 0, 1
    pos_state   = FLAT
    last_price  = 0.0
    last_trade_bar = -delay_bars-1   # para que se pueda abrir en la barra 0
    open_trades = 0                  # número de posiciones vivas (≤ max_orders)

    report = [0.0]                   # equity por trade
    chart  = [0.0]                   # equity tick-a-tick
    trade_profits = []

    for bar in range(close.size):
        pb, ps, pm = prob_buy[bar], prob_sell[bar], meta_sig[bar]
        price      = close[bar]

        # 0) señales elementales
        buy_sig  = pb > main_thr if direction != 1 else False
        sell_sig = ps > main_thr if direction != 0 else False
        meta_ok  = pm > meta_thr

        # 1) CIERRE si la señal desaparece
        if pos_state == LONG and not buy_sig:
            profit = price - last_price
            report.append(report[-1] + profit)
            chart.append(chart[-1]  + profit)
            trade_profits.append(profit)
            pos_state = FLAT
            open_trades -= 1
            last_trade_bar = bar
        elif pos_state == SHORT and not sell_sig:
            profit = last_price - price
            report.append(report[-1] + profit)
            chart.append(chart[-1]  + profit)
            trade_profits.append(profit)
            pos_state = FLAT
            open_trades -= 1
            last_trade_bar = bar

        # 2) Apertura:  meta OK, señal BUY/SELL OK, delay cumplido, cupo OK
        if meta_ok and (bar - last_trade_bar) >= delay_bars \
           and (max_orders == 0 or open_trades < max_orders):

            if buy_sig and pos_state == FLAT:
                pos_state  = LONG
                last_price = price
                open_trades += 1
                last_trade_bar = bar
                continue

            if sell_sig and pos_state == FLAT:
                pos_state  = SHORT
                last_price = price
                open_trades += 1
                last_trade_bar = bar
                continue

        # 3) tick-equity (solo para visual)
        chart.append(chart[-1])

    # 4) Cierre forzoso al final
    if pos_state == LONG:
        profit = close[-1] - last_price
        report.append(report[-1] + profit)
        chart.append(chart[-1]  + profit)
        trade_profits.append(profit)
    elif pos_state == SHORT:
        profit = last_price - close[-1]
        report.append(report[-1] + profit)
        chart.append(chart[-1]  + profit)
        trade_profits.append(profit)

    # Calcular estadísticas de trades
    n_trades = len(trade_profits)
    n_positivos = 0
    n_negativos = 0
    for p in trade_profits:
        if p > 0:
            n_positivos += 1
        elif p < 0:
            n_negativos += 1

    stats = (n_trades, n_positivos, n_negativos)

    return np.asarray(report, dtype=np.float64), \
           np.asarray(chart,  dtype=np.float64), \
           stats

# ────────── helpers ──────────────────────────────────────────────────────────

def get_periods_per_year(timeframe: str) -> float:
    """
    Calcula períodos por año basado en el timeframe.
    Adaptado para criptoactivos: 24/7 los 365 días del año.
    
    Args:
        timeframe: 'M5', 'M15', 'M30', 'H1', 'H4', 'D1'
    
    Returns:
        float: Número de períodos por año para ese timeframe
    """
    # 24 horas * 60 minutos = 1440 minutos por día
    # 365 días por año
    minutos_por_año = 1440 * 365  # 525,600

    if timeframe == 'M5':
        return minutos_por_año / 5      # 105,120
    elif timeframe == 'M15':
        return minutos_por_año / 15     # 35,040
    elif timeframe == 'M30':
        return minutos_por_año / 30     # 17,520
    elif timeframe == 'H1':
        return 24 * 365                 # 8,760
    elif timeframe == 'H4':
        return (24 / 4) * 365           # 2,190
    elif timeframe == 'D1':
        return 365.0                    # 365
    else:
        return 24 * 365                 # Default a H1 si timeframe no reconocido (8,760)
    
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
    
    return r2, slope

# ───── sesión cache – thread-safe ────────────────────────────────────────────
_session_cache: dict[str, Tuple[rt.InferenceSession, str]] = {}
_session_lock = threading.RLock()

def _get_ort_session(model_path: str) -> Tuple[rt.InferenceSession, str]:
    """
    Devuelve (sess, input_name) reutilizando sesiones ya abiertas.
    """
    with _session_lock:
        if model_path in _session_cache:
            return _session_cache[model_path]

        sess = rt.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        input_name = sess.get_inputs()[0].name
        _session_cache[model_path] = (sess, input_name)
        return sess, input_name

def clear_onnx_session_cache():
    """
    Limpia el caché de sesiones ONNX de manera thread-safe.
    Debe ser llamada al finalizar fit_final_models para liberar memoria.
    """
    with _session_lock:
        # Cerrar explícitamente todas las sesiones si es posible
        for session, _ in _session_cache.values():
            try:
                # Las sesiones de ONNX Runtime se liberan automáticamente
                # pero podemos forzar la liberación de referencias
                del session
            except Exception:
                pass  # Ignorar errores al cerrar sesiones
        
        # Limpiar el diccionario del caché
        _session_cache.clear()

# ───── función principal ────────────────────────────────────────────────────
def predict_proba_onnx_models(
    onnx_paths: List[str],
    X: np.ndarray,
) -> np.ndarray:
    """
    Devuelve las probabilidades de clase positiva para varios modelos ONNX.

    Parameters
    ----------
    onnx_paths : list[str]
        Rutas a los ficheros .onnx generados por `export_models_to_ONNX`.
    X : np.ndarray  shape (n_samples, n_features)
        Matriz de características.

    Returns
    -------
    np.ndarray
        Si agg='none' → shape (n_models, n_samples)
        en otro caso  → shape (n_samples,)
    """
    X = X.astype(np.float32, copy=False)
    n_models = len(onnx_paths)
    n_samples = X.shape[0]
    probs = np.empty((n_models, n_samples), dtype=np.float32)

    for k, path in enumerate(onnx_paths):
        sess, inp = _get_ort_session(path)
        raw = sess.run(None, {inp: X})[0]

        # --- detectar formato de salida ---
        if raw.dtype == object:          # listado de dicts {'0':p0,'1':p1}
            probs[k] = np.fromiter((r[b"1"] for r in raw), dtype=np.float32)
        elif raw.ndim == 2:              # matriz (n_samples, 2)
            probs[k] = raw[:, 1]
        elif raw.ndim == 1:              # vector ya binario
            probs[k] = raw
        else:
            raise RuntimeError(f"Salida ONNX inesperada: shape={raw.shape}")

    return probs