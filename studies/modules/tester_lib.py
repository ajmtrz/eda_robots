import threading
from numba import njit
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import onnxruntime as rt
from typing import List, Tuple, Union

def tester(
        dataset: pd.DataFrame,
        model_main: object,
        model_meta: object,
        model_main_cols: list[str],
        model_meta_cols: list[str],
        direction: str = 'both',
        timeframe: str = 'H1',
        print_metrics: bool = False,
        model_main_threshold: float = 0.5,
        model_meta_threshold: float = 0.5,
        model_max_orders: int = 1,
        model_delay_bars: int = 1) -> tuple[float, pd.DataFrame]:
    """
    Evalúa una estrategia para una o ambas direcciones, usando ejecución realista:
    - Las operaciones se abren y cierran al precio 'open' de la barra actual (índice t),
      ya que las features y señales de t son válidas para operar en t.
    - Solo se pasan los arrays estrictamente necesarios a la función jiteada backtest.

    Returns
    -------
    tuple
        (score, dataset_con_labels)
        score: Puntuación de la estrategia según :func:`evaluate_report`.
        dataset_con_labels: DataFrame original con columna "labels" (1.0/0.0) según lógica MQL5.
    """
    try:
        # Preparación de datos
        ds_main = dataset[model_main_cols].to_numpy()
        ds_meta = dataset[model_meta_cols].to_numpy()
        open_ = dataset['open'].to_numpy()

        # Calcular probabilidades usando ambos modelos (sin binarizar)
        main = predict_proba_onnx_models(model_main, ds_main)
        meta = predict_proba_onnx_models(model_meta, ds_meta)

        # Asegurarse de que main y meta sean arrays 1D (n_samples,)
        if main.ndim > 1:
            main = main[0]
        if meta.ndim > 1:
            meta = meta[0]

        # Asegurar contigüidad en memoria
        open_ = np.ascontiguousarray(open_)
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
        # --- Generar columna "labels" al estilo MQL5 ---
        buy_sig = pb > model_main_threshold
        sell_sig = ps > model_main_threshold
        meta_ok = meta > model_meta_threshold
        label_arr = ((buy_sig | sell_sig) & meta_ok).astype(float)
        dataset_with_labels = dataset.copy()
        dataset_with_labels["labels"] = label_arr

        rpt, trade_stats, trade_profits = backtest(
            open_,
            prob_buy   = pb,
            prob_sell  = ps,
            meta_sig   = meta,
            main_thr   = model_main_threshold,
            meta_thr   = model_meta_threshold,
            direction  = dir_flag,
            max_orders = model_max_orders,
            delay_bars = model_delay_bars
        )

        trade_nl, rdd_nl, r2, slope_nl, wf_nl = evaluate_report(rpt, trade_profits=trade_profits)
        if (trade_nl <= -1.0 and rdd_nl <= -1.0 and r2 <= -1.0 and slope_nl <= -1.0 and wf_nl <= -1.0):
            return -1.0, dataset_with_labels
        score = (
                0.12 * r2 +
                0.15 * slope_nl +
                0.24 * rdd_nl +
                0.19 * trade_nl +
                0.30 * wf_nl
        )
        if score < 0.0:
            return -1.0, dataset_with_labels
        if print_metrics:
            print(f"🔍 DEBUG - Main threshold: {model_main_threshold}, Meta threshold: {model_meta_threshold}, Max orders: {model_max_orders}, Delay bars: {model_delay_bars}")
            print(f"🔍 DEBUG - Métricas de evaluación: SCORE: {score}, trade_nl: {trade_nl}, rdd_nl: {rdd_nl}, r2: {r2}, slope_nl: {slope_nl}, wf_nl: {wf_nl}")
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

        return score, dataset_with_labels

    except Exception as e:
        print(f"🔍 DEBUG: Error en tester: {e}")
        return -1.0, dataset_with_labels

@njit(cache=True)
def evaluate_report(
    equity_curve: np.ndarray,
    trade_profits: np.ndarray,
    min_trades: int = 200,
    rdd_floor: float = 1.0,
) -> tuple:
    """
    Devuelve un score escalar para Optuna.
    Premia:
        1. pendiente media positiva y estable
        2. ratio retorno / drawdown alto
        3. número suficiente de trades
        4. buen ajuste lineal (R²)
        5. Consistencia temporal (walk-forward)
    Penaliza curvas cortas, con drawdown cero o pendiente negativa.
    """
    n = equity_curve.size
    if n < 2:
        # Siempre devolver 6 valores para evitar error de tipado en Numba
        return -1.0, -1.0, -1.0, -1.0, -1.0

    # ---------- nº de trades (normalizado) -----------------------------------
    # CORREGIDO: usar trade_profits.size en lugar de equity_curve differences
    n_trades = trade_profits.size
    if n_trades < min_trades:
        return -1.0, -1.0, -1.0, -1.0, -1.0
    trade_nl = 1.0 / (1.0 + np.exp(-(n_trades - min_trades) / (min_trades * 5.0)))

    # ---------- return / drawdown (normalizado) ------------------------------
    running_max = np.empty_like(equity_curve)
    running_max[0] = equity_curve[0]
    for i in range(1, n):
        running_max[i] = running_max[i - 1] if running_max[i - 1] > equity_curve[i] else equity_curve[i]
    max_dd = np.max(running_max - equity_curve)
    total_ret = equity_curve[-1] - equity_curve[0]
    if max_dd == 0.0:
        rdd = 0.0
    else:
        rdd = total_ret / max_dd
    if rdd < rdd_floor:
        return -1.0, -1.0, -1.0, -1.0, -1.0
    rdd_nl = 1.0 / (1.0 + np.exp(-(rdd - rdd_floor) / (rdd_floor * 5.0)))

    # ---------- linealidad y pendiente (normalizado) ---------------------------
    x = np.arange(n, dtype=np.float64)
    y = equity_curve.astype(np.float64)
    r2, slope = manual_linear_regression(x, y)
    if slope < 0.0:
        return -1.0, -1.0, -1.0, -1.0, -1.0
    slope_nl = 1.0 / (1.0 + np.exp(-(np.log1p(slope) / 5.0)))

    # Walk-Forward Analysis - consistencia temporal
    wf_nl = _walk_forward_validation(equity_curve, trade_profits)
    if not np.isfinite(wf_nl):
        return -1.0, -1.0, -1.0, -1.0, -1.0

    return trade_nl, rdd_nl, r2, slope_nl, wf_nl

@njit(cache=True)
def backtest(open_,
            prob_buy,          # numpy[float] - P(buy) de la red MAIN
            prob_sell,         # numpy[float] - P(sell) idem
            meta_sig,          # numpy[float] - P(clase 1) de la red META
            main_thr   = 0.5,
            meta_thr   = 0.5,
            direction  = 2,    # 0=solo buy, 1=solo sell, 2=both
            max_orders = 1,    # 0 → ilimitado
            delay_bars = 1):
    """
    Backtest realista: las operaciones se abren y cierran al precio 'open' de la barra actual (índice t),
    ya que las features y señales de t son válidas para operar en t.
    No se usa 'close' para la ejecución de operaciones.
    """
    LONG, SHORT = 0, 1

    open_positions_type = np.empty(max_orders if max_orders > 0 else open_.size, dtype=np.int64)
    open_positions_price = np.empty_like(open_positions_type, dtype=np.float64)
    open_positions_bar = np.empty_like(open_positions_type, dtype=np.int64)
    n_open = 0

    report = [0.0]
    trade_profits = []

    last_trade_bar = -delay_bars-1

    for bar in range(open_.size):
        pb, ps, pm = prob_buy[bar], prob_sell[bar], meta_sig[bar]
        price      = open_[bar]

        # 0) señales elementales
        buy_sig  = pb > main_thr if direction != 1 else False
        sell_sig = ps > main_thr if direction != 0 else False
        meta_ok  = pm > meta_thr

        # 1) CIERRE: cerrar posiciones cuyo tipo ya no tiene señal
        i = 0
        while i < n_open:
            pos_type = open_positions_type[i]
            if (pos_type == LONG and not buy_sig) or (pos_type == SHORT and not sell_sig):
                if pos_type == LONG:
                    profit = price - open_positions_price[i]
                else:
                    profit = open_positions_price[i] - price
                report.append(report[-1] + profit)
                trade_profits.append(profit)
                if i != n_open - 1:
                    open_positions_type[i] = open_positions_type[n_open - 1]
                    open_positions_price[i] = open_positions_price[n_open - 1]
                    open_positions_bar[i] = open_positions_bar[n_open - 1]
                n_open -= 1
                last_trade_bar = bar
                continue
            i += 1

        # 2) Apertura: meta OK, señal BUY/SELL OK, delay cumplido, cupo OK
        if meta_ok and (bar - last_trade_bar) >= delay_bars:
            trade_opened_this_bar = False
            
            # BUY
            if buy_sig and (max_orders == 0 or n_open < max_orders):
                open_positions_type[n_open] = LONG
                open_positions_price[n_open] = price
                open_positions_bar[n_open] = bar
                n_open += 1
                trade_opened_this_bar = True
                
            # SELL - Check position limit again after potential BUY opening
            if sell_sig and (max_orders == 0 or n_open < max_orders):
                open_positions_type[n_open] = SHORT
                open_positions_price[n_open] = price
                open_positions_bar[n_open] = bar
                n_open += 1
                trade_opened_this_bar = True
                
            # Update last_trade_bar only once per bar, regardless of how many positions opened
            if trade_opened_this_bar:
                last_trade_bar = bar

    # 4) Cierre forzoso al final de todas las posiciones abiertas
    for i in range(n_open):
        pos_type = open_positions_type[i]
        if pos_type == LONG:
            profit = open_[-1] - open_positions_price[i]
        else:
            profit = open_positions_price[i] - open_[-1]
        report.append(report[-1] + profit)
        trade_profits.append(profit)

    n_trades = len(trade_profits)
    n_positivos = 0
    n_negativos = 0
    for p in trade_profits:
        if p > 0:
            n_positivos += 1
        elif p < 0:
            n_negativos += 1

    stats = (n_trades, n_positivos, n_negativos)

    return np.asarray(report, dtype=np.float64), stats, np.asarray(trade_profits, dtype=np.float64)

# ────────── helpers ──────────────────────────────────────────────────────────

@njit(cache=True)
def _walk_forward_validation(eq, trade_profits):
    """
    Calcula un score de consistencia temporal combinando:
    - Proporción de ventanas (walk-forward windows) con equity final > inicial
    - Promedio del ratio de trades ganadores en cada ventana

    El score final es el producto de ambas métricas.

    Args:
        eq (np.ndarray): Curva de equity (acumulada) como array 1D (N+1 puntos).
        trade_profits (np.ndarray): Array 1D de profits por trade (N puntos).

    Returns:
        float: Score combinado (0.0 a 1.0).
    """
    n = eq.size
    n_trades = len(trade_profits)
    window = 5
    step = 1

    if n_trades < window:
        return 0.0

    wins = 0
    total = 0
    win_ratios_sum = 0.0
    win_ratios_count = 0

    for start in range(0, n_trades - window + 1, step):
        end = start + window
        # 1) Rentabilidad de la ventana (equity)
        r = eq[end] - eq[start]  # eq tiene N+1 puntos, así que eq[end] es válido
        if r > 0:
            wins += 1
        total += 1

        # 2) Ratio de ganadoras/perdedoras en la ventana
        trades_in_window = trade_profits[start:end]
        n_window_trades = len(trades_in_window)
        if n_window_trades > 0:
            n_winners = 0
            for p in trades_in_window:
                if p > 0:
                    n_winners += 1
            win_ratio = n_winners / n_window_trades
            win_ratios_sum += win_ratio
            win_ratios_count += 1

    prop_ventanas_rentables = wins / total if total else 0.0
    avg_win_ratio = win_ratios_sum / win_ratios_count if win_ratios_count > 0 else 0.0

    return prop_ventanas_rentables * avg_win_ratio

def get_periods_per_year(timeframe: str) -> float:
    """
    Calcula períodos por año basado en el timeframe.
    Ajustado a 252 semanas, 22 horas por día, 5 días a la semana (horario de cotización del oro).
    
    Args:
        timeframe: 'M5', 'M15', 'M30', 'H1', 'H4', 'D1'
    
    Returns:
        float: Número de períodos por año para ese timeframe
    """
    # Horario de cotización del oro: 22 horas al día, 5 días a la semana, 252 semanas al año
    horas_por_dia = 22
    dias_por_semana = 5
    semanas_por_ano = 252

    horas_por_ano = horas_por_dia * dias_por_semana * semanas_por_ano  # 22*5*252 = 27,720
    minutos_por_ano = horas_por_ano * 60  # 1,663,200

    if timeframe == 'M5':
        return minutos_por_ano / 5
    elif timeframe == 'M15':
        return minutos_por_ano / 15
    elif timeframe == 'M30':
        return minutos_por_ano / 30
    elif timeframe == 'H1':
        return horas_por_ano
    elif timeframe == 'H4':
        return horas_por_ano / 4
    elif timeframe == 'D1':
        return dias_por_semana * semanas_por_ano
    else:
        return horas_por_ano
    
@njit(cache=True)
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
    onnx_paths: Union[str, List[str]],
    X: np.ndarray,
) -> np.ndarray:
    """
    Devuelve las probabilidades de clase positiva para uno o varios modelos ONNX.

    Parameters
    ----------
    onnx_paths : str o list[str]
        Ruta o rutas a los ficheros .onnx generados por `export_models_to_ONNX`.
    X : np.ndarray  shape (n_samples, n_features)
        Matriz de características.

    Returns
    -------
    np.ndarray
        Si se pasa una sola ruta (str) → shape (n_samples,)
        Si se pasa una lista de rutas  → shape (n_models, n_samples)
    """
    X = X.astype(np.float32, copy=False)

    # Caso más común: un solo modelo (str)
    if isinstance(onnx_paths, str):
        sess, inp = _get_ort_session(onnx_paths)
        raw = sess.run(None, {inp: X})[0]
        # --- detectar formato de salida ---
        if raw.dtype == object:  # listado de dicts {'0':p0,'1':p1}
            return np.fromiter((r[b"1"] for r in raw), dtype=np.float32)
        elif raw.ndim == 2:      # matriz (n_samples, 2)
            return raw[:, 1]
        elif raw.ndim == 1:      # vector ya binario
            return raw
        else:
            raise RuntimeError(f"Salida ONNX inesperada: shape={raw.shape}")

    # Varios modelos (lista)
    n_models = len(onnx_paths)
    n_samples = X.shape[0]
    probs = np.empty((n_models, n_samples), dtype=np.float32)

    for k, path in enumerate(onnx_paths):
        sess, inp = _get_ort_session(path)
        raw = sess.run(None, {inp: X})[0]
        if raw.dtype == object:
            probs[k] = np.fromiter((r[b"1"] for r in raw), dtype=np.float32)
        elif raw.ndim == 2:
            probs[k] = raw[:, 1]
        elif raw.ndim == 1:
            probs[k] = raw
        else:
            raise RuntimeError(f"Salida ONNX inesperada: shape={raw.shape}")

    return probs