import threading
from numba import njit
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import onnxruntime as ort
from typing import List, Tuple, Union
from numba import njit
import math

# Suprimir warnings temporalmente (nivel ERROR=3)
ort.set_default_logger_severity(3)

def tester(
        dataset: pd.DataFrame,
        model_main: object,
        model_meta: object,
        model_main_cols: list[str],
        model_meta_cols: list[str],
        direction: str = 'both',
        model_main_threshold: float = 0.5,
        model_meta_threshold: float = 0.5,
        model_max_orders: int = 1,
        model_delay_bars: int = 1,
        evaluate_strategy: bool = True,
        debug: bool = False,
        plot: bool = False
        ) -> tuple[float, pd.DataFrame]:
    """
    Evalúa una estrategia para una o ambas direcciones, usando ejecución realista:
    - Las operaciones se abren y cierran al precio 'open' de la barra actual (índice t),
      ya que las features y señales de t son válidas para operar en t.
    - Solo se pasan los arrays estrictamente necesarios a la función jiteada backtest.

    Parameters
    ----------
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

        # Calcular predicciones usando ambos modelos
        main = predict_proba_onnx_models(model_main, ds_main)
        
        # Modelo meta: siempre clasificación (probabilidades)
        meta = predict_proba_onnx_models(model_meta, ds_meta)

        # Asegurarse de que main y meta sean arrays 1D (n_samples,)
        if main.ndim > 1:
            main = main[0]
        if meta.ndim > 1:
            meta = meta[0]

        dataset['labels_main'] = main.astype(float)
        dataset['labels_meta'] = meta.astype(float)

        # Asegurar contigüidad en memoria
        open_ = np.ascontiguousarray(open_)
        main = np.ascontiguousarray(main)
        meta = np.ascontiguousarray(meta)

        # DEBUG: Información de entrada
        if debug:
            print(f"🔍 DEBUG - tester - Información de entrada:")
            print(f"🔍   direction: {direction}")
            print(f"🔍   model_main_threshold: {model_main_threshold}")
            print(f"🔍   model_meta_threshold: {model_meta_threshold}")
            print(f"🔍   main.shape: {main.shape}, main.dtype: {main.dtype}")
            print(f"🔍   meta.shape: {meta.shape}, meta.dtype: {meta.dtype}")
            print(f"🔍   main.min(): {main.min():.6f}, main.max(): {main.max():.6f}")
            print(f"🔍   meta.min(): {meta.min():.6f}, meta.max(): {meta.max():.6f}")

        # ── BACKTEST ────────────────────────────────────────────────
        # Mapeo para jit
        direction_int = {"buy": 0, "sell": 1, "both": 2}[direction]

        # Backtest simple (sin soporte de datasets disjuntos)
        equity_curve, trade_stats, trade_profits, pos_series, returns_series = backtest(
            open_,
            main_predictions = main,
            meta_predictions = meta,
            main_thr   = model_main_threshold,
            meta_thr   = model_meta_threshold,
            direction_int  = direction_int,
            max_orders = model_max_orders,
            delay_bars = model_delay_bars,
        )

        # DEBUG: Resultados del backtest
        if debug:
            print(f"🔍 DEBUG tester - Backtest: trades={trade_stats[0]}, pos={trade_stats[1]}, neg={trade_stats[2]}, rpt_len={len(equity_curve)}")

        score = -1.0
        if evaluate_strategy:
            trade_nl, rdd_nl, r2, slope_nl, wf_nl = evaluate_report(equity_curve=equity_curve, trade_profits=trade_profits)
            
            if (trade_nl <= -1.0 and rdd_nl <= -1.0 and r2 <= -1.0 and slope_nl <= -1.0 and wf_nl <= -1.0):
                if debug:
                    print(f"🔍 DEBUG tester - TODAS las métricas son -1.0, retornando -1.0")
                return (-1.0, np.array([], dtype=np.float64), np.array([], dtype=np.float64), np.array([], dtype=np.int8))
            
            # Pesos ajustados para promover rectilineidad y desempeño OOS
            # Prioriza: 1) Consistencia temporal (55%), 2) Linealidad (25%), 3) Pendiente (10%), 4) RDD (7%), 5) Trades (3%)
            score = (
                    0.25 * r2 +        # Lineal idad de la curva (R²)
                    0.10 * slope_nl +  # Pendiente positiva moderada
                    0.07 * rdd_nl +    # Ratio retorno/drawdown
                    0.03 * trade_nl +  # Número de trades (muy baja influencia)
                    0.55 * wf_nl       # Consistencia temporal (máxima prioridad)
            )

            if debug:
                print(f"🔍 DEBUG - Tester - Params: thr_main={model_main_threshold}, thr_meta={model_meta_threshold}, max_orders={model_max_orders}, delay_bars={model_delay_bars}")
                print(f"🔍 DEBUG - Tester - Metrics: SCORE={score:.6f}, trade_nl={trade_nl:.6f}, rdd_nl={rdd_nl:.6f}, r2={r2:.6f}, slope_nl={slope_nl:.6f}, wf_nl={wf_nl:.6f}")
                print(f"🔍 DEBUG - Tester - Trades: n={trade_stats[0]}, pos={trade_stats[1]}, neg={trade_stats[2]}")
                if plot:
                    plt.figure(figsize=(10, 6))
                    plt.plot(equity_curve, label='Equity Curve', linewidth=1.5)
                    plt.title(f"Score: {score:.6f}")
                    plt.xlabel("Trades")
                    plt.ylabel("Cumulative P&L")
                    plt.legend()
                    plt.grid(alpha=0.3)
                    plt.show()
                    plt.close()

        return score, np.asarray(equity_curve, dtype=np.float64), returns_series, pos_series

    except Exception as e:
        if debug:
            print(f"⚠️ ERROR - Tester - Error: {e}")
        return (-1.0, np.array([], dtype=np.float64), np.array([], dtype=np.float64), np.array([], dtype=np.int8))

@njit(cache=True)
def evaluate_report(
    equity_curve: np.ndarray,
    trade_profits: np.ndarray
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
        return -1.0, -1.0, -1.0, -1.0, -1.0
    
    # Descartar inmediatamente curvas con equity final <= equity inicial (p. ej., 0.0)
    if equity_curve[-1] <= equity_curve[0]:
        return -1.0, -1.0, -1.0, -1.0, -1.0

    # ---------- nº de trades (normalizado) -----------------------------------
    n_trades = trade_profits.size
    trade_nl = 1.0 / (1.0 + np.exp(-(n_trades - 200) / (200 * 5.0)))

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

    rdd_nl = 1.0 / (1.0 + np.exp(-(rdd - 2.0) / (2.0 * 3.0)))

    # ---------- linealidad y pendiente (normalizado) ---------------------------
    x = np.arange(n, dtype=np.float64)
    y = equity_curve.astype(np.float64)
    r2, slope = manual_linear_regression(x, y)
    if slope < 0.0:
        return -1.0, -1.0, -1.0, -1.0, -1.0

    # Endurecer linealidad: si R² < 0.80 penalizar fuertemente
    if r2 < 0.80:
        r2 = r2 * 0.4  # Penalización más dura para baja linealidad
    else:
        r2 = 0.40 + (r2 - 0.80) * 3.0  # Reescalar 0.80-1.0 a 0.40-1.0

    slope_nl = 1.0 / (1.0 + np.exp(-(np.log1p(slope) / 3.0)))  # Más sensible a pendientes pequeñas

    # Walk-Forward Analysis - consistencia temporal
    wf_nl = _walk_forward_validation(equity_curve, trade_profits)
    if not np.isfinite(wf_nl):
        return -1.0, -1.0, -1.0, -1.0, -1.0

    return trade_nl, rdd_nl, r2, slope_nl, wf_nl

@njit(cache=True)
def backtest(open_,
            main_predictions,    # numpy[float] - Predicciones del modelo MAIN
            meta_predictions,    # numpy[float] - Predicciones del modelo META
            main_thr   = 0.5,
            meta_thr   = 0.5,
            direction_int  = 2,  # 0=buy, 1=sell, 2=both (mapeado)
            max_orders = 1,      # 0 → ilimitado
            delay_bars = 1):
    """
    Backtest realista: las operaciones se abren y cierran al precio 'open' de la barra actual (índice t),
    ya que las features y señales de t son válidas para operar en t.
    No se usa 'close' para la ejecución de operaciones.
    """
    LONG, SHORT = 0, 1

    open_positions_type = np.empty(max_orders if max_orders > 0 else open_.size, dtype=np.int64)
    open_positions_price = np.full_like(open_positions_type, np.nan, dtype=np.float64)  # Inicializar con NaN
    open_positions_bar = np.empty_like(open_positions_type, dtype=np.int64)
    n_open = 0

    report = [0.0]
    trade_profits = []
    pos_series = np.zeros(open_.size, dtype=np.int8)  # -1 short, 0 flat, 1 long
    returns_series = np.zeros(open_.size, dtype=np.float64)

    last_trade_bar = -delay_bars-1

    for bar in range(open_.size):
        main_val = main_predictions[bar]
        meta_val = meta_predictions[bar]
        price = open_[bar]
        
        # Los datos ya están limpios de NaN en tester(), no necesitamos verificar aquí

        # Generación de señales según dirección
        if direction_int == 0:  # Solo BUY
            buy_sig = main_val > main_thr  # P(éxito BUY) > thr
            sell_sig = False
        elif direction_int == 1:  # Solo SELL
            buy_sig = False
            sell_sig = main_val > main_thr  # P(éxito SELL) > thr
        else:  # direction_int == 2 (BOTH)
            # main_val es P(clase=1)=SELL; BUY cuando P(SELL) < (1 - thr), SELL cuando P(SELL) > thr
            buy_sig = (1.0 - main_val) > main_thr
            sell_sig = main_val > main_thr

        # Meta siempre es clasificación
        meta_ok = meta_val > meta_thr

        # 1) CIERRE: cerrar posiciones cuyo tipo ya no tiene señal
        i = 0
        while i < n_open:
            pos_type = open_positions_type[i]
            
            # Determinar si cerrar según tipo y dirección
            must_close = False
            
            # Lógica de cierre según direction y meta
            if direction_int == 0:  # Solo BUY
                must_close = (pos_type == LONG and (not buy_sig or not meta_ok))
            elif direction_int == 1:  # Solo SELL
                must_close = (pos_type == SHORT and (not sell_sig or not meta_ok))
            else:  # direction_int == 2 (BOTH)
                must_close = (pos_type == LONG and (not buy_sig or not meta_ok)) or (pos_type == SHORT and (not sell_sig or not meta_ok))
            
            if must_close:
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
                # Limpiar la posición que se acaba de cerrar
                open_positions_price[n_open - 1] = np.nan
                open_positions_type[n_open - 1] = 0
                open_positions_bar[n_open - 1] = 0
                n_open -= 1
                continue
            i += 1

        # Actualizar serie de posición (antes de abrir nuevas posiciones)
        current_pos = 0
        for j in range(n_open):
            if open_positions_type[j] == LONG:
                current_pos = 1
                break
            elif open_positions_type[j] == SHORT:
                current_pos = -1
                break
        pos_series[bar] = current_pos

        # 2) Apertura: meta OK, señal BUY/SELL OK, delay cumplido, cupo OK
        delay_ok = (bar - last_trade_bar) >= delay_bars
        if meta_ok and delay_ok:
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

    # 3b) Calcular retornos por barra basados en posición previa
    for bar in range(1, open_.size):
        prev_pos = pos_series[bar - 1]
        if prev_pos == 0:
            returns_series[bar] = 0.0
        else:
            prev_price = open_[bar - 1]
            if prev_price == 0.0:
                returns_series[bar] = 0.0
            else:
                returns_series[bar] = prev_pos * ((open_[bar] - prev_price) / prev_price)

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

    return np.asarray(report, dtype=np.float64), stats, np.asarray(trade_profits, dtype=np.float64), pos_series, returns_series

# ────────── helpers ──────────────────────────────────────────────────────────

@njit(cache=True)
def _walk_forward_validation(eq, trade_profits):
    """
    Calcula un score de consistencia temporal mejorado combinando:
    - Proporción de ventanas con equity final > inicial (ventanas más largas)
    - Promedio del ratio de trades ganadores en cada ventana
    - Penalización por volatilidad excesiva entre ventanas
    - Evaluación de múltiples escalas temporales

    El score final penaliza fuertemente las inconsistencias temporales.

    Args:
        eq (np.ndarray): Curva de equity (acumulada) como array 1D (N+1 puntos).
        trade_profits (np.ndarray): Array 1D de profits por trade (N puntos).

    Returns:
        float: Score combinado (0.0 a 1.0) con mayor énfasis en consistencia.
    """
    n = eq.size
    n_trades = len(trade_profits)
    
    # Ventanas más largas para mayor robustez y foco OOS
    base_window = max(15, n_trades // 10)  # Al menos 15, ~10% del total
    
    if n_trades < base_window:
        return 0.0

    # Evaluar múltiples escalas temporales con ventanas monótonamente crecientes
    # Asegurar que cada ventana sea mayor que la anterior y no exceda el límite
    window1 = base_window
    window2 = min(window1 * 2, n_trades // 5)
    window3 = min(window1 * 3, n_trades // 4)
    
    # Filtrar ventanas válidas (debe ser al menos 5 trades por ventana)
    windows = []
    weights = []
    
    if window1 >= 5:
        windows.append(window1)
        weights.append(0.2)
    
    if window2 > window1 and window2 >= 5:
        windows.append(window2)
        weights.append(0.35)
    
    if window3 > window2 and window3 >= 5:
        windows.append(window3)
        weights.append(0.45)
    
    # Si no hay ventanas válidas, usar solo la base
    if len(windows) == 0:
        windows = [base_window]
        weights = [1.0]
    
    # Normalizar pesos si hay menos ventanas de las esperadas
    if len(weights) > 0:
        weight_sum = sum(weights)
        weights = [w / weight_sum for w in weights]
    
    total_score = 0.0
    
    for window, weight in zip(windows, weights):
        if n_trades < window:
            continue
            
        step = max(1, window // 8)  # Solapamiento del 87.5%
        wins = 0
        total = 0
        win_ratios_sum = 0.0
        win_ratios_count = 0
        window_returns = []

        for start in range(0, n_trades - window + 1, step):
            end = start + window
            # 1) Rentabilidad de la ventana (equity)
            r = eq[end] - eq[start]
            window_returns.append(r)
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

        if total == 0:
            continue
            
        prop_ventanas_rentables = wins / total
        avg_win_ratio = win_ratios_sum / win_ratios_count if win_ratios_count > 0 else 0.0
        
        # 3) Penalización por volatilidad entre ventanas
        stability_penalty = 1.0
        if len(window_returns) >= 3:
            # Calcular volatilidad de los retornos de ventanas
            mean_return = sum(window_returns) / len(window_returns)
            variance = 0.0
            for ret in window_returns:
                variance += (ret - mean_return) ** 2
            variance /= len(window_returns)
            
            # Penalizar alta volatilidad (inestabilidad)
            if variance > 0:
                cv = (variance ** 0.5) / (abs(mean_return) + 1e-8)  # Coeficiente de variación
                stability_penalty = 1.0 / (1.0 + cv * 3.5)  # Penalización más fuerte por inestabilidad
        
        # Score para esta ventana
        window_score = prop_ventanas_rentables * avg_win_ratio * stability_penalty
        total_score += window_score * weight

    # Aplicar función sigmoide para mayor discriminación
    # Penalizar aún más los scores bajos
    final_score = total_score ** 2.0
    
    return min(1.0, final_score)

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
        tuple: (r2, slope)
        r2: coeficiente de determinación no firmado
        slope: pendiente de la recta de regresión (equity por barra)
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
_session_cache: dict[str, Tuple[ort.InferenceSession, str]] = {}
_session_lock = threading.RLock()

def _get_ort_session(model_path: str) -> Tuple[ort.InferenceSession, str]:
    """
    Devuelve (sess, input_name) reutilizando sesiones ya abiertas.
    """
    with _session_lock:
        if model_path in _session_cache:
            return _session_cache[model_path]

        sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
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
    debug: bool = False
) -> np.ndarray:
    """
    Devuelve las probabilidades de clase positiva para uno o varios modelos ONNX.
    Maneja el formato zipmap de CatBoost ONNX que incluye labels y probabilidades.

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
        # Manejar entrada vacía de forma segura
        if X is None or X.shape[0] == 0:
            return np.zeros(0, dtype=np.float32)
        sess, inp = _get_ort_session(onnx_paths)
        
        # Ejecutar modelo ONNX
        outputs = sess.run(None, {inp: X})
        
        # Manejar formato zipmap de CatBoost
        if len(outputs) == 2:  # zipmap: [labels, probabilities]
            # outputs[0] = labels, outputs[1] = probabilities dict
            proba_dict = outputs[1]
            if isinstance(proba_dict, list):
                if len(proba_dict) == 0:
                    # Entrada vacía → salida vacía
                    return np.zeros(0, dtype=np.float32)
                # Debug: verificar estructura del zipmap
                if debug:
                    print(f"🔍 DEBUG ONNX - outputs[0] type: {type(outputs[0])}, shape: {np.array(outputs[0]).shape}")
                    print(f"🔍 DEBUG ONNX - outputs[1] type: {type(proba_dict)}, len: {len(proba_dict)}")
                    if len(proba_dict) > 0:
                        print(f"🔍 DEBUG ONNX - proba_dict[0] type: {type(proba_dict[0])}, keys: {list(proba_dict[0].keys())}")
                    print(f"🔍 DEBUG ONNX - proba_dict[0] values: {list(proba_dict[0].values())}")
                    print(f"🔍 DEBUG ONNX - proba_dict[0] items: {list(proba_dict[0].items())}")
                
                # Extraer probabilidad de clase 1 (positiva)
                # Las claves pueden ser bytes b"1" o enteros 1
                result = np.fromiter((p.get(1, p.get(b"1", 0.0)) for p in proba_dict), dtype=np.float32)
                if debug:
                    print(f"🔍 DEBUG ONNX - result shape: {result.shape}, min: {result.min()}, max: {result.max()}")    
                    # Debug adicional: verificar primeros valores
                    if len(result) > 0:
                        print(f"🔍 DEBUG ONNX - primeros 5 valores: {result[:5]}")
                        print(f"🔍 DEBUG ONNX - primeros 5 dicts: {proba_dict[:5]}")
                
                return result
            else:
                raise RuntimeError(f"Formato zipmap inesperado: {type(proba_dict)}")
            
        elif len(outputs) == 1:  # formato simple
            raw = outputs[0]
            if debug:
                print(f"🔍 DEBUG ONNX - formato simple: raw type: {type(raw)}, shape: {raw.shape}, dtype: {raw.dtype}")
            # --- detectar formato de salida ---
            if raw.dtype == object:  # listado de dicts {'0':p0,'1':p1}
                result = np.fromiter((r[b"1"] for r in raw), dtype=np.float32)
                if debug:
                    print(f"🔍 DEBUG ONNX - formato simple object: result shape: {result.shape}, min: {result.min()}, max: {result.max()}")
                return result
            elif raw.ndim == 2:      # matriz (n_samples, 2)
                result = raw[:, 1]
                if debug:
                    print(f"🔍 DEBUG ONNX - formato simple 2D: result shape: {result.shape}, min: {result.min()}, max: {result.max()}")
                return result
            elif raw.ndim == 1:      # vector ya binario
                if debug:
                    print(f"🔍 DEBUG ONNX - formato simple 1D: result shape: {raw.shape}, min: {raw.min()}, max: {raw.max()}")
                return raw
            else:
                raise RuntimeError(f"Salida ONNX inesperada: shape={raw.shape}")
        else:
            raise RuntimeError(f"Número inesperado de outputs ONNX: {len(outputs)}")

    # Varios modelos (lista)
    n_models = len(onnx_paths)
    n_samples = X.shape[0]
    if n_samples == 0:
        # Devolver array vacío con forma (n_models, 0) para consistencia
        return np.empty((n_models, 0), dtype=np.float32)
    probs = np.empty((n_models, n_samples), dtype=np.float32)

    for k, path in enumerate(onnx_paths):
        sess, inp = _get_ort_session(path)
        outputs = sess.run(None, {inp: X})
        
        if len(outputs) == 2:  # zipmap
            proba_dict = outputs[1]
            if isinstance(proba_dict, list):
                if len(proba_dict) == 0:
                    probs[k] = np.zeros(n_samples, dtype=np.float32)
                else:
                    # Las claves pueden ser bytes b"1" o enteros 1
                    probs[k] = np.fromiter((p.get(1, p.get(b"1", 0.0)) for p in proba_dict), dtype=np.float32)
            else:
                raise RuntimeError(f"Formato zipmap inesperado: {type(proba_dict)}")
        elif len(outputs) == 1:  # formato simple
            raw = outputs[0]
            if raw.dtype == object:
                probs[k] = np.fromiter((r[b"1"] for r in raw), dtype=np.float32)
            elif raw.ndim == 2:
                probs[k] = raw[:, 1]
            elif raw.ndim == 1:
                probs[k] = raw
            else:
                raise RuntimeError(f"Salida ONNX inesperada: shape={raw.shape}")
        else:
            raise RuntimeError(f"Número inesperado de outputs ONNX: {len(outputs)}")

    return probs

# ───── Monkey Test (Null Hypothesis Benchmark) ─────────────────────────────
@njit(cache=True)
def _compute_sharpe(returns: np.ndarray) -> float:
    if returns.size < 2:
        return 0.0
    mu = np.nanmean(returns)
    sigma = np.nanstd(returns)
    if sigma <= 0.0:
        return 0.0
    return float(mu / sigma)

@njit(cache=True)
def _sanitize_positions_for_direction_code(pos: np.ndarray, direction_code: int) -> np.ndarray:
    # direction_code: 0=buy-only, 1=sell-only, 2=both
    if pos.size == 0:
        return np.zeros(0, dtype=np.int8)
    out = pos.astype(np.int8)
    if direction_code == 0:
        for i in range(out.size):
            if out[i] < 0:
                out[i] = 0
    elif direction_code == 1:
        for i in range(out.size):
            if out[i] > 0:
                out[i] = 0
    # direction_code==2 no cambia
    return out

@njit(cache=True)
def _estimate_block_size_jit(pos: np.ndarray) -> int:
    # Estima tamaño de bloque como mediana de longitudes de racha con mínimo 5, máximo 64
    n = pos.size
    if n < 10:
        return max(2, n)
    # Primera pasada: contar número de rachas
    last = pos[0]
    cnt = 1
    n_runs = 0
    for i in range(1, n):
        v = pos[i]
        if v == last:
            cnt += 1
        else:
            n_runs += 1
            last = v
            cnt = 1
    n_runs += 1  # última racha
    # Segunda pasada: almacenar longitudes
    runs = np.empty(n_runs, dtype=np.int64)
    last = pos[0]
    cnt = 1
    idx = 0
    for i in range(1, n):
        v = pos[i]
        if v == last:
            cnt += 1
        else:
            runs[idx] = cnt
            idx += 1
            last = v
            cnt = 1
    runs[idx] = cnt
    # Mediana discreta
    runs_sorted = np.sort(runs)
    m = runs_sorted.size
    if m % 2 == 1:
        med = runs_sorted[m // 2]
    else:
        a = runs_sorted[m // 2 - 1]
        b = runs_sorted[m // 2]
        med = (a + b) // 2 if ((a + b) % 2 == 0) else (a + b) // 2
    if med < 5:
        med = 5
    if med > 64:
        med = 64
    return int(med)

@njit(cache=True)
def _block_bootstrap_positions(base_pos: np.ndarray, length: int, block_size: int) -> np.ndarray:
    # Circular block bootstrap: concatena bloques contiguos con wrap-around
    if length <= 0:
        return np.zeros(0, dtype=np.int8)
    n = base_pos.size
    if n == 0:
        return np.zeros(length, dtype=np.int8)
    if block_size <= 0:
        block_size = 1
    if block_size > n:
        block_size = n
    out = np.empty(length, dtype=np.int8)
    i = 0
    while i < length:
        start = np.random.randint(0, n)
        take_total = min(block_size, length - i)
        end = start + take_total
        if end <= n:
            out[i:i+take_total] = base_pos[start:end]
        else:
            first = n - start
            out[i:i+first] = base_pos[start:n]
            out[i+first:i+take_total] = base_pos[0:end - n]
        i += take_total
    return out

@njit(cache=True)
def _circular_shift_positions_jit(pos: np.ndarray, shift: int) -> np.ndarray:
    n = pos.size
    if n == 0:
        return np.zeros(0, dtype=np.int8)
    s = 0 if n == 0 else (shift % n)
    if s == 0:
        return pos.copy()
    out = np.empty(n, dtype=np.int8)
    out[:s] = pos[n - s:]
    out[s:] = pos[:n - s]
    return out

@njit(cache=True)
def _randomize_block_size_jit(base_block: int) -> int:
    # Genera un tamaño de bloque aleatorio alrededor de la mediana observada, acotado [2,64]
    b = base_block
    if b < 2:
        b = 2
    low = int(0.5 * b)
    high = int(1.5 * b)
    if low < 2:
        low = 2
    if high > 64:
        high = 64
    if high < low:
        high = low
    # randint es inclusivo-exclusivo en numba? Aseguramos incluir high sumando 1 si posible
    return int(np.random.randint(low, high + 1))

@njit(cache=True)
def _simulate_returns_from_positions(price_series: np.ndarray,
                                     pos_series_sim: np.ndarray) -> np.ndarray:
    # Calcula retornos por barra usando la posición previa (sin comisiones)
    n = price_series.size
    ret = np.zeros(n, dtype=np.float64)
    prev_pos = 0
    for t in range(1, n):
        prev_price = price_series[t - 1]
        pos = pos_series_sim[t - 1]
        if prev_price != 0.0 and pos != 0:
            ret[t] = pos * ((price_series[t] - prev_price) / prev_price)
        prev_pos = pos
    return ret

def run_monkey_test(actual_returns: np.ndarray,
                    price_series: np.ndarray,
                    pos_series: np.ndarray,
                    direction: str,
                    n_simulations: int = 1000,
                    ) -> dict:
    """
    Monkey Test basado en Sharpe con:
    - Direccionalidad: respeta buy/ sell/ both eliminando estados no permitidos
    - Block bootstrap de posiciones: preserva rachas/autocorrelación de la serie de posiciones
    - p-valor estable: (k+1)/(N+1) y percentile = 100*(1-p)
    """
    try:
        if actual_returns is None or price_series is None or pos_series is None:
            return {'p_value': 1.0, 'is_significant': False, 'percentile': 0.0}
        price_series = price_series.astype(np.float64)
        pos_series = pos_series.astype(np.int8)

        # Endurecer calidad OOS: tamaño mínimo efectivo y nº de entradas mínimo
        #  - Mínimo de barras con exposición distinta de 0
        #  - Mínimo de cambios 0->(!=0) como proxy de nº de trades
        if actual_returns.size < 50 or price_series.size < 50 or pos_series.size < 50:
            return {'p_value': 1.0, 'percentile': 0.0}
        n_exposed = int(np.count_nonzero(pos_series != 0))
        # Contar entradas como transiciones desde 0 a no-cero
        entries = 0
        prev = 0
        for v in pos_series:
            if prev == 0 and v != 0:
                entries += 1
            prev = v
        if n_exposed < 200 or entries < 30:
            return {'p_value': 1.0, 'percentile': 0.0}

        # Sanitizar posiciones según direccionalidad
        direction_code = 2
        if direction == 'buy':
            direction_code = 0
        elif direction == 'sell':
            direction_code = 1
        base_pos = _sanitize_positions_for_direction_code(pos_series, direction_code)
        # Estimar tamaño de bloque y preparar simulaciones
        block_size_med = _estimate_block_size_jit(base_pos)

        # Sharpe real (por barra, sin anualizar)
        sr_actual = _compute_sharpe(actual_returns.astype(np.float64))

        # Simulaciones
        sharpes = np.zeros(int(n_simulations), dtype=np.float64)
        n = price_series.size
        for k in range(int(n_simulations)):
            # Mezclar dos nulos: circular shift (fase) y block bootstrap (estructura)
            if (k & 1) == 0:
                # Circular shift: preserva estructura y frecuencia de rachas; rompe la alineación temporal
                shift = np.random.randint(0, n) if n > 0 else 0
                sim_pos = _circular_shift_positions_jit(base_pos, shift)
            else:
                # Block bootstrap con tamaño de bloque aleatorio alrededor de la mediana observada
                bs = _randomize_block_size_jit(block_size_med)
                sim_pos = _block_bootstrap_positions(base_pos, n, bs)
            sim_ret = _simulate_returns_from_positions(price_series, sim_pos)
            sharpes[k] = _compute_sharpe(sim_ret)

        # p-valor estabilizado y percentil
        count_ge = int(np.sum(sharpes >= sr_actual))
        p_value = (count_ge + 1.0) / (sharpes.size + 1.0)
        percentile = (1.0 - p_value) * 100.0

        return {
            'p_value': float(p_value),
            'percentile': float(percentile)
        }
    except Exception:
        return {'p_value': 1.0, 'percentile': 0.0}

# ───── Robust Changepoint Protocol (Quant Beckman Inspired) ───────────────────
@njit(cache=True)
def _robust_median_mad(x: np.ndarray) -> tuple:
    n = x.size
    if n == 0:
        return 0.0, 1.0
    med = np.median(x)
    abs_dev = np.empty(n, dtype=np.float64)
    for i in range(n):
        abs_dev[i] = abs(x[i] - med)
    mad = np.median(abs_dev)
    # Escala consistente con normal: 1.4826 * MAD
    scale = 1.4826 * mad
    if scale <= 1e-12:
        scale = 1.0
    return med, scale

def _winsorize_by_mad(x: np.ndarray, k: float = 6.0) -> np.ndarray:
    """
    Winsoriza/clipea una serie usando mediana y MAD (escala robusta).
    Esto reduce la influencia de outliers (colas gordas) antes de segmentar.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x
    med, scale = _robust_median_mad(x)
    lo = med - float(k) * float(scale)
    hi = med + float(k) * float(scale)
    return np.clip(x, lo, hi)

def _biweight_mean(x: np.ndarray, K: float = 2.5, max_iter: int = 20, tol: float = 1e-6) -> float:
    """
    Media robusta tipo Tukey biweight (bisquare) con IRLS simple.
    - K controla cuán agresivo es el downweight de outliers.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n == 0:
        return 0.0
    m, s = _robust_median_mad(x)
    if not np.isfinite(s) or s <= 1e-12:
        return float(np.mean(x))
    m = float(m)
    s = float(s)
    K = float(K)
    for _ in range(int(max_iter)):
        u = (x - m) / (K * s)
        mask = np.abs(u) < 1.0
        if not np.any(mask):
            break
        uu = u[mask]
        w = (1.0 - uu * uu)
        w = w * w
        xs = x[mask]
        den = float(np.sum(w))
        if den <= 1e-12:
            break
        m_new = float(np.sum(w * xs) / den)
        if abs(m_new - m) <= tol * s:
            m = m_new
            break
        m = m_new
    return float(m)

def _pelt_mean_l2(x: np.ndarray, penalty: float, min_seg_len: int) -> list[int]:
    """
    Segmentación exacta (DP) para cambios en media con coste L2 (SSE).
    - Más robusto que un PELT con pruning mal calibrado.
    - Complejidad O(n^2): aceptable para series de tamaño típico (cientos / pocos miles).

    Segmentos: [s, t) con t en 0..n, retornando changepoints internos (excluye 0 y n).
    """
    x = np.asarray(x, dtype=np.float64)
    n = int(x.size)
    if n <= 2 * int(min_seg_len):
        return []
    msl = max(2, int(min_seg_len))
    pen = float(penalty)

    # Prefix sums para coste O(1)
    S1 = np.zeros(n + 1, dtype=np.float64)
    S2 = np.zeros(n + 1, dtype=np.float64)
    S1[1:] = np.cumsum(x)
    S2[1:] = np.cumsum(x * x)

    def cost(s: int, t: int) -> float:
        L = t - s
        if L <= 1:
            return 0.0
        sum1 = S1[t] - S1[s]
        sum2 = S2[t] - S2[s]
        return float(sum2 - (sum1 * sum1) / float(L))

    # DP exacto
    F = np.full(n + 1, np.inf, dtype=np.float64)
    prev = np.full(n + 1, -1, dtype=np.int64)
    F[0] = -pen

    for t in range(msl, n + 1):
        best = np.inf
        best_s = -1
        # s debe permitir segmento [s,t) con longitud >= msl
        for s in range(0, t - msl + 1):
            if (t - s) < msl:
                continue
            val = F[s] + cost(s, t) + pen
            if val < best:
                best = val
                best_s = s
        F[t] = best
        prev[t] = best_s

    # Reconstrucción de changepoints
    cps_out: list[int] = []
    t = n
    while t > 0:
        s = int(prev[t])
        if s <= 0:
            break
        cps_out.append(s)
        t = s
    cps_out = sorted(set(cps_out))
    cps_out = [c for c in cps_out if 0 < c < n]
    return cps_out

def _switch_off_from_segments(x: np.ndarray,
                              changepoints: list[int],
                              mean_kind: str = "biweight",
                              K: float = 2.5,
                              rel_drop: float = 0.6,
                              abs_threshold: float = 0.0,
                              min_bad_len: int = 20,
                              min_seg_len: int | None = None) -> dict:
    """
    Regla “switch-off” inspirada en el artículo:
    - segmenta en regímenes
    - estima la calidad por media robusta por régimen
    - dispara si el último régimen es suficientemente largo y muestra degradación
    """
    x = np.asarray(x, dtype=np.float64)
    n = int(x.size)
    if n == 0:
        return {"switch_off": False, "reason": "empty_series"}
    bounds = [0] + [int(c) for c in changepoints if 0 < int(c) < n] + [n]
    bounds = sorted(set(bounds))
    segs = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1) if (bounds[i + 1] - bounds[i]) > 0]
    if len(segs) <= 1:
        # Fallback robusto: aunque no haya changepoints explícitos,
        # podemos evaluar degradación con ventanas rodantes.
        mbl = int(min_bad_len)
        if n < max(2 * mbl, mbl + 5):
            return {"switch_off": False, "reason": "only_one_regime_short", "segments": segs}

        win_prev = int(min_seg_len) if (min_seg_len is not None and int(min_seg_len) >= 5) else mbl
        win_prev = max(5, min(win_prev, n - mbl))
        step = max(1, win_prev // 2)

        last_seg = x[n - mbl:n]
        if mean_kind == "l2":
            last_mean = float(np.mean(last_seg)) if last_seg.size else 0.0
        else:
            last_mean = float(_biweight_mean(last_seg, K=K))

        best_prev_mean = -np.inf
        # barrido de ventanas previas (evita mirar el mismo tramo final)
        end_limit = n - mbl
        for s in range(0, max(1, end_limit - win_prev + 1), step):
            seg = x[s:s + win_prev]
            if seg.size < 5:
                continue
            if mean_kind == "l2":
                m = float(np.mean(seg))
            else:
                m = float(_biweight_mean(seg, K=K))
            if m > best_prev_mean:
                best_prev_mean = m
        if not np.isfinite(best_prev_mean):
            best_prev_mean = float(np.mean(x[:end_limit])) if end_limit > 5 else float(np.mean(x))

        cond_len = mbl >= int(min_bad_len)
        cond_rel = (best_prev_mean > 0.0) and (last_mean <= float(rel_drop) * best_prev_mean)
        cond_abs = (last_mean <= float(abs_threshold))
        switch_off = bool(cond_len and (cond_rel or cond_abs))

        if cond_rel and cond_abs:
            reason = "rolling_relative_drop_and_absolute_failure"
        elif cond_rel:
            reason = "rolling_relative_drop"
        elif cond_abs:
            reason = "rolling_absolute_failure"
        else:
            reason = "rolling_no_degradation"

        return {
            "switch_off": switch_off,
            "reason": reason,
            "segments": segs,
            "means": np.asarray([np.nan], dtype=np.float64),
            "lengths": np.asarray([n], dtype=np.int64),
            "last_mean": float(last_mean),
            "best_prev_mean": float(best_prev_mean),
        }

    means = np.zeros(len(segs), dtype=np.float64)
    lens = np.zeros(len(segs), dtype=np.int64)
    for i, (s, e) in enumerate(segs):
        lens[i] = int(e - s)
        seg = x[s:e]
        if mean_kind == "l2":
            means[i] = float(np.mean(seg)) if seg.size else 0.0
        else:
            means[i] = float(_biweight_mean(seg, K=K))

    last_mean = float(means[-1])
    last_len = int(lens[-1])
    best_prev_mean = float(np.max(means[:-1]))

    cond_len = last_len >= int(min_bad_len)
    cond_rel = (best_prev_mean > 0.0) and (last_mean <= float(rel_drop) * best_prev_mean)
    cond_abs = (last_mean <= float(abs_threshold))
    switch_off = bool(cond_len and (cond_rel or cond_abs))

    if not cond_len:
        reason = f"current_regime_too_short(len={last_len}, min_bad_len={int(min_bad_len)})"
    elif cond_rel and cond_abs:
        reason = "relative_drop_and_absolute_failure"
    elif cond_rel:
        reason = "relative_drop"
    elif cond_abs:
        reason = "absolute_failure"
    else:
        reason = "no_degradation"

    return {
        "switch_off": switch_off,
        "reason": reason,
        "segments": segs,
        "means": means,
        "lengths": lens,
        "last_mean": last_mean,
        "best_prev_mean": best_prev_mean,
    }

@njit(cache=True)
def _ewma(x: np.ndarray, alpha: float) -> np.ndarray:
    n = x.size
    out = np.zeros(n, dtype=np.float64)
    if n == 0:
        return out
    val = x[0]
    out[0] = val
    for i in range(1, n):
        val = alpha * x[i] + (1.0 - alpha) * val
        out[i] = val
    return out

@njit(cache=True)
def _detect_shock(returns: np.ndarray, burn_in: int, z_thr: float, window: int) -> int:
    n = returns.size
    if n <= burn_in + 1:
        return -1
    w = max(4, window)
    for t in range(burn_in, n):
        start = t - w
        if start < 0:
            start = 0
        med, scale = _robust_median_mad(returns[start:t])
        z = (returns[t] - med) / scale
        if z < -z_thr:
            return t
    return -1

@njit(cache=True)
def _detect_erosion(returns: np.ndarray, burn_in: int, erosion_window: int, l_min: int, m_consecutive: int, scale: float) -> int:
    n = returns.size
    if n <= burn_in + l_min + 1:
        return -1
    sc = scale
    if sc <= 1e-12:
        sc = 1.0
    returns_z = returns / sc
    # EWMA de retornos con alpha derivado de la ventana
    span = float(max(2, erosion_window))
    alpha = 2.0 / (span + 1.0)
    ew = _ewma(returns_z, alpha)
    # Gracia: no activar erosión antes de burn_in + l_min
    grace = burn_in + l_min
    consec = 0
    for t in range(grace, n):
        if ew[t] < 0.0:
            consec += 1
        else:
            consec = 0
        if consec >= m_consecutive:
            return t
    return -1

def robust_cp_guard(cumulative_pnl: np.ndarray,
                    params: dict | None = None,
                    debug: bool = False) -> dict:
    """
    Guardia robusta de cambio de régimen inspirada en el protocolo “switch-off”:
      - Segmenta la serie (robusta a outliers) para estimar regímenes.
      - Decide “switch_off” si el último régimen (suficientemente largo) muestra
        degradación relativa y/o fallo absoluto.

    Nota importante: un “shock” aislado NO debería matar automáticamente un sistema
    en presencia de colas gordas; por eso aquí lo tratamos como señal informativa,
    y el criterio principal es `switch_off`.

    Referencia conceptual: “Switch-off: Robust changepoint protocol”:
    https://www.quantbeckman.com/p/with-code-switch-off-robust-changepoint?utm_source=post-email-title&publication_id=3833331&post_id=180103121&utm_campaign=email-post-title&isFreemail=true&r=cx4yj&triedRedirect=true&utm_medium=email
    Args:
        cumulative_pnl: Serie de PnL acumulado (REAL).
        params: {
            'use_series': 'pnl' | 'returns',   # default 'pnl' (usa diff del PnL)
            'burn_in': int,                    # default 30
            'shock_z': float,                  # default 3.0
            'erosion_window': int,             # default 64
            'l_min': int,                      # default 20
            'm_consecutive': int,              # default 3
        }
        debug: logging de diagnóstico si True.
    Returns:
        dict con:
            - kill_shock: bool
            - kill_erosion: bool
            - idx_shock: int | None (índice absoluto)
            - idx_erosion: int | None
            - stable: bool (no se activaron gatillos)
    """
    try:
        if cumulative_pnl is None:
            return {'kill_shock': False, 'kill_erosion': False, 'idx_shock': None, 'idx_erosion': None, 'stable': True}
        pnl = np.asarray(cumulative_pnl, dtype=np.float64)
        n = pnl.size
        if n < 10:
            return {'kill_shock': False, 'kill_erosion': False, 'idx_shock': None, 'idx_erosion': None, 'stable': True}
        # Parámetros
        p = params or {}
        use_series = p.get('use_series', 'pnl')
        burn_in = int(p.get('burn_in', 30))
        shock_z = float(p.get('shock_z', 3.0))
        shock_window = int(p.get('shock_window', burn_in))
        erosion_window = int(p.get('erosion_window', 64))
        l_min = int(p.get('l_min', 20))
        m_consecutive = int(p.get('m_consecutive', 3))
        # Segmentación / switch-off (defaults conservadores)
        winsor_k = float(p.get("winsor_k", 6.0))
        mean_kind = str(p.get("mean_kind", "biweight"))
        biweight_K = float(p.get("biweight_K", 2.5))
        rel_drop = float(p.get("rel_drop", 0.6))
        abs_threshold = float(p.get("abs_threshold", 0.0))
        min_bad_len = int(p.get("min_bad_len", l_min))
        min_seg_len = int(p.get("min_seg_len", max(10, burn_in)))
        # Multiplicador del penalty para PELT:
        # Usamos una escala ~ var * log(n) (clásica para cambios en media con SSE).
        penalty_mult = float(p.get("penalty_mult", 8.0))
        # Serie analizada: retornos por barra
        if use_series == 'returns':
            # Si el usuario desea pasar retornos directamente, asúmelos en pnl
            returns = pnl
        else:
            # Por defecto: diferenciar PnL acumulado
            returns = np.zeros(n, dtype=np.float64)
            returns[1:] = pnl[1:] - pnl[:-1]
        # Si la serie viene de diff(PnL), el primer punto es 0.0; no aporta
        if returns.size >= 2 and abs(float(returns[0])) <= 0.0:
            returns_eff = returns[1:]
        else:
            returns_eff = returns

        # Señales auxiliares (informativas) – NO criterio principal
        _, scale_global = _robust_median_mad(returns_eff[:max(burn_in, 1)])
        idx_shock = _detect_shock(returns_eff, burn_in, shock_z, shock_window)
        idx_erosion = _detect_erosion(returns_eff, burn_in, erosion_window, l_min, m_consecutive, scale_global)
        kill_shock = bool(idx_shock >= 0)
        kill_erosion = bool(idx_erosion >= 0)

        # Switch-off principal: robustificar (winsor) y segmentar
        x_rob = _winsorize_by_mad(returns_eff, k=winsor_k)
        # penalty escala con var*log(n) (más estable que var*min_seg_len)
        v = float(np.var(x_rob)) if x_rob.size > 1 else 0.0
        n_eff = int(x_rob.size)
        base_pen = max(1e-12, penalty_mult * v * float(np.log(max(3, n_eff))))
        penalty = float(p.get("penalty", base_pen))
        cps = _pelt_mean_l2(x_rob, penalty=penalty, min_seg_len=min_seg_len)
        dec = _switch_off_from_segments(
            x_rob,
            changepoints=cps,
            mean_kind=mean_kind,
            K=biweight_K,
            rel_drop=rel_drop,
            abs_threshold=abs_threshold,
            min_bad_len=min_bad_len,
            min_seg_len=min_seg_len,
        )

        switch_off = bool(dec.get("switch_off", False))
        reason = str(dec.get("reason", "stable"))
        if debug:
            print(f"🔍 DEBUG - robust_cp_guard: n_eff={int(returns_eff.size)}, burn_in={burn_in}, shock_z={shock_z}, shock_window={shock_window}, erosion_window={erosion_window}, l_min={l_min}, m={m_consecutive}")
            print(f"🔍 DEBUG - robust_cp_guard: idx_shock={int(idx_shock)}, idx_erosion={int(idx_erosion)}, scale_global={scale_global}")
            print(f"🔍 DEBUG - robust_cp_guard: penalty={penalty:.6g}, min_seg_len={min_seg_len}, rel_drop={rel_drop}, abs_threshold={abs_threshold}, min_bad_len={min_bad_len}")
            if isinstance(dec, dict):
                print(f"🔍 DEBUG - robust_cp_guard: switch_off={switch_off}, reason={reason}")
        return {
            # Señales auxiliares (compatibilidad)
            'kill_shock': bool(kill_shock),
            'kill_erosion': bool(kill_erosion),
            'idx_shock': int(idx_shock) if kill_shock else None,
            'idx_erosion': int(idx_erosion) if kill_erosion else None,
            # Señal principal (nuevo)
            'switch_off': bool(switch_off),
            'switch_reason': reason,
            # estable = no hay degradación estructural detectada
            'stable': not bool(switch_off),
            # detalles (útiles para debug/ajuste)
            'changepoints': cps,
            'decision': dec,
        }
    except Exception as e:
        if debug:
            print(f"⚠️ ERROR - robust_cp_guard: {e}")
        return {'kill_shock': False, 'kill_erosion': False, 'idx_shock': None, 'idx_erosion': None, 'switch_off': False, 'switch_reason': 'error', 'stable': True, 'reason': 'error'}