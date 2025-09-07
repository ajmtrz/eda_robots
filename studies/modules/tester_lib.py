import threading
from numba import njit
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import onnxruntime as ort
from typing import List, Tuple, Union
from numba import njit

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
        rpt, trade_stats, trade_profits, pos_series, returns_series = backtest(
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
            print(f"🔍 DEBUG tester - Backtest: trades={trade_stats[0]}, pos={trade_stats[1]}, neg={trade_stats[2]}, rpt_len={len(rpt)}")

        trade_nl, rdd_nl, r2, slope_nl, wf_nl = evaluate_report(rpt, trade_profits=trade_profits)
        
        if (trade_nl <= -1.0 and rdd_nl <= -1.0 and r2 <= -1.0 and slope_nl <= -1.0 and wf_nl <= -1.0):
            if debug:
                print(f"🔍 DEBUG tester - TODAS las métricas son -1.0, retornando -1.0")
            return (-1.0, np.array([], dtype=np.float64), np.array([], dtype=np.float64), np.array([], dtype=np.int8))
        
        # Pesos ajustados para promover rectilineidad y desempeño OOS
        # Prioriza: 1) Consistencia temporal (55%), 2) Linealidad (25%), 3) Pendiente (10%), 4) RDD (7%), 5) Trades (3%)
        score = (
                0.25 * r2 +        # Linealidad de la curva (R²)
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
                plt.plot(rpt, label='Equity Curve', linewidth=1.5)
                plt.title(f"Score: {score:.6f}")
                plt.xlabel("Trades")
                plt.ylabel("Cumulative P&L")
                plt.legend()
                plt.grid(alpha=0.3)
                plt.show()
                plt.close()

        return score, np.asarray(rpt, dtype=np.float64), returns_series, pos_series

    except Exception as e:
        if debug:
            print(f"⚠️ ERROR - Tester - Error: {e}")
        return (-1.0, np.array([], dtype=np.float64), np.array([], dtype=np.float64), np.array([], dtype=np.int8))

@njit(cache=True)
def evaluate_report(
    equity_curve: np.ndarray,
    trade_profits: np.ndarray,
    min_trades: int = 300,
    rdd_floor: float = 1.5
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

    # ---------- nº de trades (normalizado) -----------------------------------
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

    min_rdd = max(rdd_floor * 1.5, 2.0)  # Al menos 2.0 o 1.5 veces el floor
    if rdd < min_rdd:
        return -1.0, -1.0, -1.0, -1.0, -1.0

    rdd_nl = 1.0 / (1.0 + np.exp(-(rdd - min_rdd) / (min_rdd * 3.0)))

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
        sess, inp = _get_ort_session(onnx_paths)
        
        # Ejecutar modelo ONNX
        outputs = sess.run(None, {inp: X})
        
        # Manejar formato zipmap de CatBoost
        if len(outputs) == 2:  # zipmap: [labels, probabilities]
            # outputs[0] = labels, outputs[1] = probabilities dict
            proba_dict = outputs[1]
            if isinstance(proba_dict, list) and len(proba_dict) > 0:
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
    probs = np.empty((n_models, n_samples), dtype=np.float32)

    for k, path in enumerate(onnx_paths):
        sess, inp = _get_ort_session(path)
        outputs = sess.run(None, {inp: X})
        
        if len(outputs) == 2:  # zipmap
            proba_dict = outputs[1]
            if isinstance(proba_dict, list) and len(proba_dict) > 0:
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

        # Sanitizar posiciones según direccionalidad
        direction_code = 2
        if direction == 'buy':
            direction_code = 0
        elif direction == 'sell':
            direction_code = 1
        base_pos = _sanitize_positions_for_direction_code(pos_series, direction_code)
        # Estimar tamaño de bloque y preparar simulaciones
        block_size = _estimate_block_size_jit(base_pos)

        # Sharpe real (por barra, sin anualizar)
        sr_actual = _compute_sharpe(actual_returns.astype(np.float64))

        # Simulaciones
        sharpes = np.zeros(int(n_simulations), dtype=np.float64)
        for k in range(int(n_simulations)):
            sim_pos = _block_bootstrap_positions(base_pos, price_series.size, block_size)
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