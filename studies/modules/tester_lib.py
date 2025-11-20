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

# ───── BOCPD (Bayesian Online Changepoint Detection) ─────────────────────────
@njit(cache=True)
def _logsumexp_vec(log_vals: np.ndarray) -> float:
    if log_vals.size == 0:
        return float('-inf')
    m = float(np.max(log_vals))
    if not np.isfinite(m):
        return m
    return m + float(np.log(np.sum(np.exp(log_vals - m))))

@njit(cache=True)
def _student_t_logpdf(x: float, mu: float, sigma: float, nu: float) -> float:
    # log t_nu(x | mu, sigma)
    # f = Gamma((nu+1)/2) / (Gamma(nu/2) * sqrt(nu*pi) * sigma) * (1 + ((x-mu)^2)/(nu*sigma^2))^{-(nu+1)/2}
    if sigma <= 0.0 or nu <= 0.0 or not np.isfinite(x) or not np.isfinite(mu):
        return -1e12
    z = (x - mu) / sigma
    log_num = math.lgamma(0.5 * (nu + 1.0))
    log_den = math.lgamma(0.5 * nu) + 0.5 * (np.log(nu) + np.log(np.pi)) + np.log(sigma)
    log_core = -0.5 * (nu + 1.0) * np.log(1.0 + (z * z) / nu)
    return float(log_num - log_den + log_core)

@njit(cache=True)
def _student_t_logpdf_vec(x: float, m_arr: np.ndarray, sigma: np.ndarray, nu: np.ndarray) -> np.ndarray:
    n = m_arr.size
    out = np.empty(n, dtype=np.float64)
    for j in range(n):
        out[j] = _student_t_logpdf(x, m_arr[j], sigma[j], nu[j])
    return out
    
def bocpd_guard(cumulative_pnl: np.ndarray, params: dict | None = None, debug: bool = False) -> dict:
    """
    Guardia de cambios de régimen (BOCPD) sobre PnL acumulado.
    Devuelve arrays por tick y un flag 'regime_stable' (False si dispara kill).
    """
    try:
        if cumulative_pnl is None or len(cumulative_pnl) < 2:
            return {
                'kill_signals_shock': np.array([], dtype=bool),
                'kill_signals_erosion': np.array([], dtype=bool),
                'changepoint_probs': np.array([], dtype=np.float64),
                'exp_run_lengths': np.array([], dtype=np.float64),
                'regime_stable': True
            }
        # Serie estacionaria
        returns = np.diff(cumulative_pnl, prepend=cumulative_pnl[0]).astype(np.float64)
        T = returns.size
        if T < 10:
            return {
                'kill_signals_shock': np.zeros(T, dtype=bool),
                'kill_signals_erosion': np.zeros(T, dtype=bool),
                'changepoint_probs': np.zeros(T, dtype=np.float64),
                'exp_run_lengths': np.arange(T, dtype=np.float64),
                'regime_stable': True
            }
        # Heurísticas automáticas
        burn_in = max(30, int(T * 0.15))
        lam = max(burn_in + 10, int(T / 3.0))
        kill_threshold = 0.5
        l_min = max(15, int(lam * 0.25))
        m_consecutive = max(5, int(l_min * 0.3))
        if params:
            burn_in = int(params.get('burn_in_period', burn_in))
            lam = int(params.get('expected_run_length', lam))
            kill_threshold = float(params.get('kill_threshold', kill_threshold))
            l_min = int(params.get('l_min', l_min))
            m_consecutive = int(params.get('m_consecutive', m_consecutive))
        hazard = 1.0 / max(lam, 1e-6)
        if hazard >= 1.0:
            hazard = 0.999999
        # Priors NIG desde burn-in
        bi = min(burn_in, T)
        burn_data = returns[:bi]
        prior_m0 = float(np.mean(burn_data))
        var_guess = float(np.var(burn_data)) if burn_data.size > 1 else 0.0
        if var_guess <= 0.0 or not np.isfinite(var_guess):
            var_guess = 1e-4
        prior_k0 = 1.0
        prior_a0 = 1.0
        prior_b0 = var_guess * prior_a0
        # Estado inicial
        # Rlog: log P(runlen=r | x_{1:t})
        Rlog = np.array([0.0], dtype=np.float64)  # t=0, r=0
        # Hyperparámetros por run-length (alineados con Rlog)
        m_arr = np.array([prior_m0], dtype=np.float64)
        k_arr = np.array([prior_k0], dtype=np.float64)
        a_arr = np.array([prior_a0], dtype=np.float64)
        b_arr = np.array([prior_b0], dtype=np.float64)
        # Salidas
        cp_probs = np.zeros(T, dtype=np.float64)
        erl_series = np.zeros(T, dtype=np.float64)
        kill_shock = np.zeros(T, dtype=bool)
        kill_erosion = np.zeros(T, dtype=bool)
        erosion_cnt = 0
        for t in range(T):
            x = returns[t]
            # Predictiva Student-t por hipótesis
            nu = 2.0 * a_arr
            sigma = np.sqrt((b_arr * (k_arr + 1.0)) / (a_arr * k_arr))
            # Calcular log-likelihoods (jit)
            ll = _student_t_logpdf_vec(x, m_arr, sigma, nu)
            # Growth: r -> r+1
            log_growth = np.log(1.0 - hazard) + ll + Rlog
            # Changepoint/reset: r -> 0
            log_cp_terms = np.log(hazard) + ll + Rlog
            log_cp = _logsumexp_vec(log_cp_terms)
            # Montar R_next
            R_next = np.empty(Rlog.size + 1, dtype=np.float64)
            R_next[0] = log_cp
            R_next[1:] = log_growth
            # Normalizar
            lse = _logsumexp_vec(R_next)
            R_next -= lse
            # Probabilidad de CP y ERL
            cp_prob_t = float(np.exp(R_next[0]))
            cp_probs[t] = cp_prob_t
            # Expected run length
            idxs = np.arange(R_next.size, dtype=np.float64)
            erl = float(np.sum(idxs * np.exp(R_next)))
            erl_series[t] = erl
            # Triggers
            if t >= burn_in and cp_prob_t > kill_threshold:
                kill_shock[t] = True
            # Grace period para erosión: esperar hasta burn_in + l_min
            grace_period = burn_in + l_min
            if t > grace_period and erl < l_min:
                erosion_cnt += 1
            else:
                erosion_cnt = 0
            if erosion_cnt >= m_consecutive:
                kill_erosion[t] = True
            # PRUNING: descartar hipótesis de baja masa
            max_log = float(np.max(R_next))
            keep = R_next >= (max_log - 10.0)
            if not np.all(keep):
                R_next = R_next[keep]
                # Actualizar parámetros acorde a hipótesis conservadas
                # Shift 0 es reset; otros son growth desde índices conservados-1
            # Actualizar parámetros para siguiente tick:
            # 1) Para r=0 (reset): prior actualizado con x
            k0 = prior_k0 + 1.0
            m0 = (prior_k0 * prior_m0 + x) / k0
            a0 = prior_a0 + 0.5
            b0 = prior_b0 + 0.5 * (prior_k0 * (x - prior_m0) * (x - prior_m0)) / k0
            # 2) Para growth: actualizar todos con x
            k_g = k_arr + 1.0
            m_g = (k_arr * m_arr + x) / k_g
            a_g = a_arr + 0.5
            b_g = b_arr + 0.5 * (k_arr * (x - m_arr) * (x - m_arr)) / k_g
            # Reconstruir arrays alineados con R_next
            # La primera hipótesis corresponde al reset (r=0)
            m_new = np.empty(R_next.size, dtype=np.float64)
            k_new = np.empty(R_next.size, dtype=np.float64)
            a_new = np.empty(R_next.size, dtype=np.float64)
            b_new = np.empty(R_next.size, dtype=np.float64)
            m_new[0] = m0
            k_new[0] = k0
            a_new[0] = a0
            b_new[0] = b0
            # El resto proviene de growth; si se aplicó pruning, recortar
            growth_keep = np.ones(Rlog.size, dtype=bool)
            # Mapear keep a growth (R_next[1:]) cuando hubo pruning
            if R_next.size != (Rlog.size + 1):
                # reconstruimos mask de next sin el primer elemento
                # aqui aproximamos: tomamos las top growth por probabilidad
                # Seleccion simple: usar los mayores log_growth que quepan
                log_growth_full = np.log(1.0 - hazard) + ll + Rlog
                order = np.argsort(-log_growth_full)  # descendente
                n_take = R_next.size - 1
                selected = order[:n_take]
                growth_keep = np.zeros(Rlog.size, dtype=bool)
                growth_keep[selected] = True
                m_g_s = m_g[growth_keep]
                k_g_s = k_g[growth_keep]
                a_g_s = a_g[growth_keep]
                b_g_s = b_g[growth_keep]
            else:
                m_g_s = m_g
                k_g_s = k_g
                a_g_s = a_g
                b_g_s = b_g
            m_new[1:] = m_g_s
            k_new[1:] = k_g_s
            a_new[1:] = a_g_s
            b_new[1:] = b_g_s
            # Siguiente iteración
            Rlog = R_next
            m_arr = m_new
            k_arr = k_new
            a_arr = a_new
            b_arr = b_new
        regime_stable = not (kill_shock.any() or kill_erosion.any())
        return {
            'kill_signals_shock': kill_shock,
            'kill_signals_erosion': kill_erosion,
            'changepoint_probs': cp_probs,
            'exp_run_lengths': erl_series,
            'regime_stable': bool(regime_stable)
        }
    except Exception as e:
        if debug:
            try:
                print(f"⚠️ DEBUG - bocpd_guard: excepción interna: {e}")
            except Exception:
                pass
        # Si algo falla, actuar conservador: no matar por BOCPD
        return {
            'kill_signals_shock': np.array([], dtype=bool),
            'kill_signals_erosion': np.array([], dtype=bool),
            'changepoint_probs': np.array([], dtype=np.float64),
            'exp_run_lengths': np.array([], dtype=np.float64),
            'regime_stable': True
        }