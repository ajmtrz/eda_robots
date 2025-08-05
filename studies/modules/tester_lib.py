import threading
from numba import njit
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import onnxruntime as ort
from typing import List, Tuple, Union

# Suprimir warnings temporalmente (nivel ERROR=3)
ort.set_default_logger_severity(3)

def tester(
        dataset: pd.DataFrame,
        model_main: object,
        model_meta: object,
        model_main_cols: list[str],
        model_meta_cols: list[str],
        direction: str = 'both',
        timeframe: str = 'H1',
        label_type: str = 'classification',
        model_main_threshold: float = 0.5,
        model_meta_threshold: float = 0.5,
        model_max_orders: int = 1,
        model_delay_bars: int = 1,
        debug: bool = False) -> tuple[float, pd.DataFrame]:
    """
    Evalúa una estrategia para una o ambas direcciones, usando ejecución realista:
    - Las operaciones se abren y cierran al precio 'open' de la barra actual (índice t),
      ya que las features y señales de t son válidas para operar en t.
    - Solo se pasan los arrays estrictamente necesarios a la función jiteada backtest.

    Parameters
    ----------
    label_type : str, default='classification'
        Tipo de modelo main: 'classification' o 'regression'.
        El modelo meta siempre es de clasificación.

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

        # Calcular predicciones usando ambos modelos según label_type
        if label_type == 'classification':
            # Modelo main: clasificación (probabilidades)
            main = predict_proba_onnx_models(model_main, ds_main)
        else:  # regression
            # Modelo main: regresión (valores continuos)
            main = predict_regression_onnx_models(model_main, ds_main)
        
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
            print(f"🔍 DEBUG tester - Información de entrada:")
            print(f"🔍   label_type: {label_type}")
            print(f"🔍   direction: {direction}")
            print(f"🔍   model_main_threshold: {model_main_threshold}")
            print(f"🔍   model_meta_threshold: {model_meta_threshold}")
            print(f"🔍   main.shape: {main.shape}, main.dtype: {main.dtype}")
            print(f"🔍   meta.shape: {meta.shape}, meta.dtype: {meta.dtype}")
            print(f"🔍   open_.shape: {open_.shape}, open_.dtype: {open_.dtype}")
            print(f"🔍   main.min(): {main.min():.6f}, main.max(): {main.max():.6f}")
            print(f"🔍   meta.min(): {meta.min():.6f}, meta.max(): {meta.max():.6f}")
            
            # 🔍 DEBUG: Análisis detallado de predicciones
            print(f"🔍 DEBUG tester - Análisis de predicciones:")
            print(f"🔍   main percentiles: {np.percentile(main, [10, 25, 50, 75, 90])}")
            print(f"🔍   main > threshold: {(main > model_main_threshold).sum()}/{len(main)} ({100*(main > model_main_threshold).mean():.1f}%)")
            print(f"🔍   main > threshold*1.5: {(main > model_main_threshold*1.5).sum()}/{len(main)} ({100*(main > model_main_threshold*1.5).mean():.1f}%)")
            print(f"🔍   main > threshold*2.0: {(main > model_main_threshold*2.0).sum()}/{len(main)} ({100*(main > model_main_threshold*2.0).mean():.1f}%)")
            print(f"🔍   Gap mínimo: {main.min() - model_main_threshold:.4f}")
            print(f"🔍   Gap promedio: {main.mean() - model_main_threshold:.4f}")

        # ── BACKTEST ────────────────────────────────────────────────
        # Mapeo para jit
        direction_int = {"buy": 0, "sell": 1, "both": 2}[direction]
        label_type_int = {"classification": 0, "regression": 1}[label_type]

        # DEBUG: Señales antes del backtest
        if debug:
            print(f"🔍 DEBUG tester - Señales generadas:")
            print(f"🔍   main.min(): {main.min():.6f}, main.max(): {main.max():.6f}")
            print(f"🔍   meta.min(): {meta.min():.6f}, meta.max(): {meta.max():.6f}")
            print(f"🔍   main > main_thr: {(main > model_main_threshold).sum()}")
            print(f"🔍   meta > meta_thr: {(meta > model_meta_threshold).sum()}")

        rpt, trade_stats, trade_profits = backtest(
            open_,
            main_predictions = main,
            meta_predictions = meta,
            main_thr   = model_main_threshold,
            meta_thr   = model_meta_threshold,
            direction_int  = direction_int,
            label_type_int = label_type_int,
            max_orders = model_max_orders,
            delay_bars = model_delay_bars
        )

        # DEBUG: Resultados del backtest
        if debug:
            print(f"🔍 DEBUG tester - Resultados del backtest:")
            print(f"🔍   trade_stats: {trade_stats}")
            print(f"🔍   len(trade_profits): {len(trade_profits)}")
            print(f"🔍   len(rpt): {len(rpt)}")
            if len(trade_profits) > 0:
                print(f"🔍   trade_profits.min(): {min(trade_profits):.6f}")
                print(f"🔍   trade_profits.max(): {max(trade_profits):.6f}")
                print(f"🔍   trade_profits.mean(): {np.mean(trade_profits):.6f}")
            if len(rpt) > 1:
                print(f"🔍   rpt[0]: {rpt[0]:.6f}")
                print(f"🔍   rpt[-1]: {rpt[-1]:.6f}")
                print(f"🔍   rpt.max(): {np.max(rpt):.6f}")
                print(f"🔍   rpt.min(): {np.min(rpt):.6f}")

        trade_nl, rdd_nl, r2, slope_nl, wf_nl = evaluate_report(rpt, trade_profits=trade_profits)
        
        # DEBUG: Métricas de evaluación
        if debug:
            print(f"🔍 DEBUG tester - Métricas de evaluación:")
            print(f"🔍   trade_nl: {trade_nl:.6f}")
            print(f"🔍   rdd_nl: {rdd_nl:.6f}")
            print(f"🔍   r2: {r2:.6f}")
            print(f"🔍   slope_nl: {slope_nl:.6f}")
            print(f"🔍   wf_nl: {wf_nl:.6f}")
        
        if (trade_nl <= -1.0 and rdd_nl <= -1.0 and r2 <= -1.0 and slope_nl <= -1.0 and wf_nl <= -1.0):
            if debug:
                print(f"🔍 DEBUG tester - TODAS las métricas son -1.0, retornando -1.0")
            return -1.0
        
        # Pesos optimizados para favorecer estabilidad temporal y linealidad constante
        # Prioriza: 1) Consistencia temporal (45%), 2) Linealidad+Pendiente (40%), 3) Otros (15%)
        score = (
                0.20 * r2 +        # Linealidad de la curva (R²)
                0.20 * slope_nl +  # Pendiente positiva consistente
                0.10 * rdd_nl +    # Ratio retorno/drawdown
                0.05 * trade_nl +  # Número de trades (menor importancia)
                0.45 * wf_nl       # Consistencia temporal (máxima prioridad)
        )
        if score < 0.0:
            if debug:
                print(f"🔍 DEBUG tester - Score < 0.0 ({score:.6f}), retornando -1.0")
            return -1.0
        if debug:
            print(f"🔍 DEBUG - Label type: {label_type}, Main threshold: {model_main_threshold}, Meta threshold: {model_meta_threshold}, Max orders: {model_max_orders}, Delay bars: {model_delay_bars}")
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

        return score

    except Exception as e:
        print(f"🔍 DEBUG: Error en tester: {e}")
        return -1.0

@njit(cache=True)
def evaluate_report(
    equity_curve: np.ndarray,
    trade_profits: np.ndarray,
    min_trades: int = 200,
    rdd_floor: float = 1.0
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

    if r2 < 0.7:  # Umbral más estricto para R²
        r2 = r2 * 0.5  # Penalización severa para R² < 0.7
    else:
        r2 = 0.35 + (r2 - 0.7) * 2.17  # Reescalar 0.7-1.0 a 0.35-1.0

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
            label_type_int = 0,  # 0=classification, 1=regression (mapeado)
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

    last_trade_bar = -delay_bars-1

    for bar in range(open_.size):
        main_val = main_predictions[bar]
        meta_val = meta_predictions[bar]
        price = open_[bar]
        
        # Los datos ya están limpios de NaN en tester(), no necesitamos verificar aquí

        # Generación de señales según tipo y dirección
        if label_type_int == 1:  # REGRESIÓN
            if direction_int == 0:  # Solo BUY
                buy_sig = main_val > main_thr
                sell_sig = False
            elif direction_int == 1:  # Solo SELL
                buy_sig = False
                sell_sig = abs(main_val) > main_thr
            else:  # direction_int == 2 (BOTH)
                # Distinguir por signo: positivo=buy, negativo=sell
                buy_sig = (main_val > main_thr) and (main_val > 0)
                sell_sig = (abs(main_val) > main_thr) and (main_val < 0)
        
        else:  # label_type_int == 0 (CLASIFICACIÓN)
            if direction_int == 0:  # Solo BUY
                buy_sig = main_val > main_thr
                sell_sig = False
            elif direction_int == 1:  # Solo SELL
                buy_sig = False
                sell_sig = main_val > main_thr
            else:  # direction_int == 2 (BOTH)
                # Usar probabilidades directamente
                buy_sig = main_val > main_thr
                sell_sig = main_val > main_thr

        # Meta siempre es clasificación
        meta_ok = meta_val > meta_thr

        # 1) CIERRE: cerrar posiciones cuyo tipo ya no tiene señal
        i = 0
        while i < n_open:
            pos_type = open_positions_type[i]
            
            # Determinar si cerrar según tipo y dirección
            should_close = False
            
            # Lógica de cierre según label_type y direction
            # Usar meta model para cierre también (doble filtrado)
            if label_type_int == 1:  # REGRESIÓN
                if direction_int == 0:  # Solo BUY
                    should_close = (pos_type == LONG and (not buy_sig or not meta_ok))
                elif direction_int == 1:  # Solo SELL
                    should_close = (pos_type == SHORT and (not sell_sig or not meta_ok))
                else:  # direction_int == 2 (BOTH)
                    if pos_type == LONG:
                        should_close = not buy_sig or main_val <= 0 or not meta_ok
                    else:  # SHORT
                        should_close = not sell_sig or main_val >= 0 or not meta_ok
            
            else:  # label_type_int == 0 (CLASIFICACIÓN)
                if direction_int == 0:  # Solo BUY
                    should_close = (pos_type == LONG and (not buy_sig or not meta_ok))
                elif direction_int == 1:  # Solo SELL
                    should_close = (pos_type == SHORT and (not sell_sig or not meta_ok))
                else:  # direction_int == 2 (BOTH)
                    should_close = (pos_type == LONG and (not buy_sig or not meta_ok)) or (pos_type == SHORT and (not sell_sig or not meta_ok))
            
            if should_close:
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
    
    # Ventanas más largas para mayor robustez
    base_window = max(10, n_trades // 20)  # Al menos 10, o 5% del total
    
    if n_trades < base_window:
        return 0.0

    # Evaluar múltiples escalas temporales con ventanas monótonamente crecientes
    # Asegurar que cada ventana sea mayor que la anterior y no exceda el límite
    window1 = base_window
    window2 = min(window1 * 2, n_trades // 8)  # Ajustado para evitar ventanas muy pequeñas
    window3 = min(window1 * 3, n_trades // 4)  # Ajustado para evitar ventanas muy pequeñas
    
    # Filtrar ventanas válidas (debe ser al menos 5 trades por ventana)
    windows = []
    weights = []
    
    if window1 >= 5:
        windows.append(window1)
        weights.append(0.5)
    
    if window2 > window1 and window2 >= 5:
        windows.append(window2)
        weights.append(0.3)
    
    if window3 > window2 and window3 >= 5:
        windows.append(window3)
        weights.append(0.2)
    
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
            
        step = max(1, window // 4)  # Solapamiento del 75%
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
                stability_penalty = 1.0 / (1.0 + cv * 2.0)  # Más penalización por alta volatilidad
        
        # Score para esta ventana
        window_score = prop_ventanas_rentables * avg_win_ratio * stability_penalty
        total_score += window_score * weight

    # Aplicar función sigmoide para mayor discriminación
    # Penalizar más fuertemente scores bajos
    final_score = total_score ** 1.5  # Exponente > 1 para penalizar más los valores bajos
    
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

def predict_regression_onnx_models(
    onnx_paths: Union[str, List[str]],
    X: np.ndarray,
) -> np.ndarray:
    """
    Devuelve las predicciones de regresión para uno o varios modelos ONNX.
    
    Parameters
    ----------
    onnx_paths : str o list[str]
        Ruta o rutas a los ficheros .onnx.
    X : np.ndarray  shape (n_samples, n_features)
        Matriz de características.

    Returns
    -------
    np.ndarray
        Si se pasa una sola ruta → shape (n_samples,)
        Si se pasa una lista → shape (n_models, n_samples)
    """
    X = X.astype(np.float32, copy=False)

    # Caso: un solo modelo (str)
    if isinstance(onnx_paths, str):
        sess, inp = _get_ort_session(onnx_paths)
        try:
            raw = sess.run(['predictions'], {inp: X})[0]
        except Exception:
            raw = sess.run(None, {inp: X})[0]
        
        # Aplanar la salida a 1D independientemente de su forma original
        return raw.ravel()

    # Caso: múltiples modelos (lista)
    n_models = len(onnx_paths)
    n_samples = X.shape[0]
    predictions = np.empty((n_models, n_samples), dtype=np.float32)

    for k, path in enumerate(onnx_paths):
        sess, inp = _get_ort_session(path)
        try:
            raw = sess.run(['predictions'], {inp: X})[0]
        except Exception:
            raw = sess.run(None, {inp: X})[0]
        
        # Aplanar y almacenar
        predictions[k] = raw.ravel()
        
    return predictions