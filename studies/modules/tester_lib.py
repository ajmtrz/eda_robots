import threading
from numba import njit, prange, float64, int64
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from functools import lru_cache
from scipy import stats

# Optional import for ONNX runtime
try:
    import onnxruntime as rt
    ONNX_AVAILABLE = True
except ImportError:
    rt = None
    ONNX_AVAILABLE = False

# Configuración thread-safe de ONNX Runtime
_onnx_configured = False
_onnx_lock = threading.RLock()

def _configure_onnx_runtime():
    """Configuración thread-safe de ONNX Runtime"""
    global _onnx_configured
    with _onnx_lock:
        if not _onnx_configured and ONNX_AVAILABLE:
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

# ────────── NUEVAS FUNCIONES PARA VALIDACIÓN ROBUSTA ──────────────────────────────────

@njit(cache=True, fastmath=True)
def _generate_random_signals(n_periods, signal_probs, seed=42):
    """Genera señales aleatorias preservando proporciones originales"""
    np.random.seed(seed)
    return np.random.choice(np.array([0, 1]), size=n_periods, p=signal_probs)

@njit(cache=True, fastmath=True)
def _simulate_monkey_strategy(close, n_simulations=1000):
    """
    Ejecuta el Monkey Test (Null Hypothesis Benchmark) de forma vectorizada.
    Simula estrategias aleatorias para establecer una línea base estadística.
    """
    n_periods = close.size
    monkey_returns = np.zeros(n_simulations, dtype=np.float64)
    
    # Proporciones típicas de señales (balanced)
    signal_probs = np.array([0.3, 0.7])  # 30% short, 70% long signals
    
    for sim in range(n_simulations):
        # Generar señales aleatorias
        np.random.seed(sim + 42)  # Seed diferente para cada simulación
        signals = np.random.choice(np.array([0, 1]), size=n_periods, p=signal_probs)
        
        # Simular estrategia simple: posición basada en señal
        position = 0.0
        last_price = close[0]
        total_return = 0.0
        
        for i in range(1, n_periods):
            if signals[i-1] == 1 and position <= 0:  # Long signal
                if position < 0:  # Close short first
                    total_return += (last_price - close[i-1]) / last_price
                position = 1.0
                last_price = close[i-1]
            elif signals[i-1] == 0 and position >= 0:  # Short signal
                if position > 0:  # Close long first
                    total_return += (close[i-1] - last_price) / last_price
                position = -1.0
                last_price = close[i-1]
        
        # Close final position
        if position > 0:
            total_return += (close[-1] - last_price) / last_price
        elif position < 0:
            total_return += (last_price - close[-1]) / last_price
            
        monkey_returns[sim] = total_return
    
    return monkey_returns

@njit(cache=True, fastmath=True)
def _calculate_deflated_sharpe(observed_sharpe, n_trials, n_periods, skewness, kurtosis):
    """
    Calcula el Deflated Sharpe Ratio según López de Prado.
    Ajusta el Sharpe por múltiples pruebas y características de la distribución.
    """
    if n_trials <= 1 or n_periods <= 1:
        return observed_sharpe
    
    # Varianza del estimador del Sharpe Ratio
    var_sr = (1.0 + 0.5 * observed_sharpe * observed_sharpe - 
              skewness * observed_sharpe + 
              (kurtosis - 3.0) / 4.0 * observed_sharpe * observed_sharpe) / n_periods
    
    # Ajuste por múltiples pruebas (aproximación Bonferroni)
    alpha_adjusted = 0.05 / n_trials
    
    # Z-score crítico ajustado
    z_critical = np.sqrt(2.0 * np.log(n_trials))  # Aproximación para colas extremas
    
    # Threshold de significancia ajustado
    sr_threshold = z_critical * np.sqrt(var_sr)
    
    # Deflated Sharpe Ratio
    if observed_sharpe <= sr_threshold:
        return 0.0
    else:
        # Ajuste conservador
        deflation_factor = 1.0 - (sr_threshold / max(observed_sharpe, 1e-8))
        return observed_sharpe * deflation_factor

@njit(cache=True, fastmath=True)
def _apply_transaction_costs(equity_curve, position_changes, cost_per_trade=0.001):
    """
    Aplica costos de transacción realistas de forma vectorizada.
    
    Args:
        equity_curve: Curva de equity original
        position_changes: Número de cambios de posición
        cost_per_trade: Costo proporcional por operación
    """
    if position_changes <= 0:
        return equity_curve
    
    # Estimar costos totales basados en el valor promedio del portfolio
    avg_value = np.mean(equity_curve)
    total_costs = position_changes * avg_value * cost_per_trade
    
    # Distribuir costos proporcionalmente a lo largo de la curva
    cost_per_period = total_costs / len(equity_curve)
    
    # Aplicar costos de forma acumulativa
    adjusted_curve = equity_curve.copy()
    for i in range(1, len(adjusted_curve)):
        adjusted_curve[i] -= cost_per_period * i
    
    return adjusted_curve

@njit(cache=True, fastmath=True)
def _walk_forward_validation(eq, window_size=252):
    """
    Implementa Walk-Forward Analysis simplificado.
    Evalúa la consistencia de performance en ventanas temporales.
    """
    n_periods = eq.size
    if n_periods < window_size * 2:
        return 1.0  # No hay suficientes datos para WF
    
    n_windows = (n_periods - window_size) // (window_size // 2)
    if n_windows < 2:
        return 1.0
    
    window_returns = np.zeros(n_windows, dtype=np.float64)
    
    for i in range(n_windows):
        start_idx = i * (window_size // 2)
        end_idx = start_idx + window_size
        
        if end_idx >= n_periods:
            break
            
        window = eq[start_idx:end_idx]
        if window.size > 1:
            window_return = (window[-1] - window[0]) / max(abs(window[0]), 1e-8)
            window_returns[i] = window_return
    
    # Calcular consistencia: % de ventanas positivas
    positive_windows = np.sum(window_returns > 0)
    consistency = positive_windows / len(window_returns) if len(window_returns) > 0 else 0.0
    
    return consistency

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
def _robust_sharpe_calculation(eq, periods_per_year=6240.0):
    """Calcula Sharpe ratio robusto con ajustes estadísticos"""
    if eq.size < 2:
        return 0.0, 0.0, 0.0  # sharpe, skewness, kurtosis
    
    # Calcular returns
    returns = np.diff(eq) / (eq[:-1] + 1e-8)
    
    if returns.size == 0:
        return 0.0, 0.0, 0.0
    
    # Estadísticas básicas
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    
    if std_return == 0:
        return 0.0, 0.0, 0.0
    
    # Sharpe anualizado
    annualized_return = mean_return * periods_per_year
    annualized_vol = std_return * np.sqrt(periods_per_year)
    sharpe = (annualized_return - RISK_FREE_RATE) / annualized_vol
    
    # Skewness y Kurtosis para Deflated Sharpe
    centered_returns = returns - mean_return
    m2 = np.mean(centered_returns ** 2)
    m3 = np.mean(centered_returns ** 3)
    m4 = np.mean(centered_returns ** 4)
    
    skewness = m3 / (m2 ** 1.5) if m2 > 0 else 0.0
    kurtosis = m4 / (m2 ** 2) if m2 > 0 else 3.0
    
    return sharpe, skewness, kurtosis

@njit(cache=True, fastmath=True)
def evaluate_report(eq: np.ndarray, ppy: float = 6240.0):
    """
    Sistema de scoring ROBUSTO que integra las técnicas del artículo:
    - Mantiene favorabilidad por curvas lineales ascendentes
    - Añade validación estadística robusta
    - Incluye costos de transacción
    - Implementa técnicas de validación del artículo
    - Optimizado para ejecutar en < 1 segundo
    
    Returns:
        tuple: (score, metrics_tuple) donde score integra todas las validaciones
    """
    # Validaciones básicas optimizadas
    if eq.size < 300 or not np.isfinite(eq).all():
        return (-1.0, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    
    # Protección para valores negativos
    eq_min = np.min(eq)
    if eq_min <= 0.0:
        eq = eq - eq_min + 1.0
    
    # === 1. MÉTRICAS TRADICIONALES OPTIMIZADAS ===
    
    # R² con bonus por pendiente positiva (mantenido)
    r2 = _signed_r2(eq)
    
    # Bonus específico por linealidad ascendente (mantenido)
    linearity_bonus = _linearity_bonus(eq)
    
    # Consistencia del crecimiento (mantenido)
    consistency = _consistency_score(eq)
    
    # Recompensa por pendiente fuerte (mantenido)
    slope_reward = _slope_reward(eq)
    
    # === 2. NUEVAS VALIDACIONES ROBUSTAS DEL ARTÍCULO ===
    
    # Sharpe ratio robusto con estadísticas para Deflated Sharpe
    sharpe, skewness, kurtosis = _robust_sharpe_calculation(eq, ppy)
    
    # Walk-Forward Analysis - consistencia temporal
    wf_consistency = _walk_forward_validation(eq, window_size=min(252, eq.size // 4))
    
    # Aplicar costos de transacción (estimación conservadora)
    # Estimamos cambios de posición basados en volatilidad de la curva
    returns = np.diff(eq)
    estimated_trades = max(1, int(np.sum(np.abs(np.diff(returns > np.median(returns)))) * 0.5))
    eq_with_costs = _apply_transaction_costs(eq, estimated_trades, 0.001)
    
    # Recalcular métricas post-costos
    cost_adjusted_return = (eq_with_costs[-1] - eq_with_costs[0]) / max(abs(eq_with_costs[0]), 1.0)
    
    # Deflated Sharpe Ratio (asumiendo múltiples pruebas moderadas)
    n_trials_estimated = 50  # Estimación conservadora de pruebas realizadas
    deflated_sharpe = _calculate_deflated_sharpe(sharpe, n_trials_estimated, eq.size, skewness, kurtosis)
    
    # === 3. PENALIZACIONES Y AJUSTES ===
    
    # Drawdown mejorado
    peak = eq[0]
    max_dd = 0.0
    for val in eq:
        if val > peak:
            peak = val
        else:
            dd = (peak - val) / peak if peak > 0 else 0.0
            max_dd = max(max_dd, dd)
    
    # Penalización por drawdown exponencial (mantenido pero mejorado)
    dd_penalty = np.exp(-max_dd * 12.0)  # Más estricto
    
    # Penalización por volatilidad excesiva
    if returns.size > 0:
        volatility_ratio = np.std(returns) / (abs(np.mean(returns)) + 1e-8)
        vol_penalty = 1.0 / (1.0 + volatility_ratio * 0.5)
    else:
        volatility_ratio = 0.0
        vol_penalty = 1.0
    
    # === 4. SCORE INTEGRADO ROBUSTO ===
    
    # Componentes principales (mantenidos con mejoras)
    linearity_component = (r2 + linearity_bonus) / 2.0  # [0,1]
    growth_component = (slope_reward + consistency) / 2.0  # [0,1]
    
    # Nuevos componentes de robustez
    robustness_component = (
        (deflated_sharpe / max(abs(deflated_sharpe), 1.0) + 1.0) / 2.0 * 0.3 +  # 30% Deflated Sharpe
        wf_consistency * 0.4 +  # 40% Walk-Forward consistency
        (cost_adjusted_return / max(abs(cost_adjusted_return), 1.0) + 1.0) / 2.0 * 0.3  # 30% Cost-adjusted return
    )
    robustness_component = max(0.0, min(1.0, robustness_component))
    
    # Score base integrado
    base_score = (
        linearity_component * 0.4 +      # 40% peso a linealidad (mantenido alto)
        growth_component * 0.25 +        # 25% peso a crecimiento
        robustness_component * 0.25 +    # 25% peso a robustez estadística
        max(0.0, min(1.0, (sharpe + 2.0) / 4.0)) * 0.1  # 10% Sharpe tradicional
    )
    
    # Aplicar penalizaciones
    final_score = base_score * dd_penalty * vol_penalty
    
    # Bonus especial para curvas lineales perfectas (mantenido)
    if r2 > 0.98 and slope_reward > 0.5 and max_dd < 0.01 and wf_consistency > 0.8:
        final_score = min(1.0, final_score * 1.15)  # Bonus del 15%
    
    # Asegurar rango [0,1]
    final_score = max(0.0, min(1.0, final_score))
    
    # === 5. MÉTRICAS EXPANDIDAS PARA DEBUGGING ===
    total_return = cost_adjusted_return
    
    metrics_tuple = (
        r2, linearity_bonus, consistency, slope_reward,
        total_return, max_dd, dd_penalty, linearity_component,
        growth_component, base_score, final_score, sharpe,
        deflated_sharpe, wf_consistency, robustness_component, skewness,
        kurtosis, vol_penalty, estimated_trades, cost_adjusted_return, 
        volatility_ratio, 0.0
    )
    
    return final_score, metrics_tuple

def metrics_tuple_to_dict(score: float, metrics_tuple: tuple, periods_per_year: float) -> dict:
    """Convierte la tupla de métricas expandida a diccionario"""
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
        'sharpe_ratio': metrics_tuple[11],
        'deflated_sharpe': metrics_tuple[12],
        'wf_consistency': metrics_tuple[13],
        'robustness_component': metrics_tuple[14],
        'skewness': metrics_tuple[15],
        'kurtosis': metrics_tuple[16],
        'vol_penalty': metrics_tuple[17],
        'estimated_trades': metrics_tuple[18],
        'cost_adjusted_return': metrics_tuple[19],
        'volatility_ratio': metrics_tuple[20],
        'periods_per_year': periods_per_year
    }

# === FUNCIONES ADICIONALES DE VALIDACIÓN ===

def run_monkey_test(equity_curve, close_prices=None, n_simulations=1000):
    """
    Ejecuta el Monkey Test (Null Hypothesis Benchmark) según el artículo.
    
    Args:
        equity_curve: Curva de equity de la estrategia
        close_prices: Precios de cierre (si están disponibles)
        n_simulations: Número de simulaciones Monte Carlo
    
    Returns:
        dict: Resultados del test estadístico
    """
    if close_prices is None:
        # Generar precios sintéticos basados en la equity curve
        close_prices = equity_curve + np.random.normal(0, np.std(np.diff(equity_curve)), len(equity_curve))
    
    # Obtener return de la estrategia real
    strategy_return = (equity_curve[-1] - equity_curve[0]) / abs(equity_curve[0])
    
    # Simular estrategias aleatorias
    monkey_returns = _simulate_monkey_strategy(close_prices, n_simulations)
    
    # Calcular p-value
    p_value = np.sum(monkey_returns >= strategy_return) / n_simulations
    
    return {
        'strategy_return': strategy_return,
        'monkey_returns_mean': np.mean(monkey_returns),
        'monkey_returns_std': np.std(monkey_returns),
        'p_value': p_value,
        'is_significant': p_value < 0.05,
        'percentile': (1.0 - p_value) * 100
    }

def comprehensive_strategy_validation(equity_curve, close_prices=None, periods_per_year=6240.0):
    """
    Validación comprensiva que implementa todas las técnicas del artículo.
    
    Returns:
        dict: Resultados completos de validación
    """
    # Evaluación principal
    score, metrics_tuple = evaluate_report(equity_curve, periods_per_year)
    metrics = metrics_tuple_to_dict(score, metrics_tuple, periods_per_year)
    
    # Monkey Test
    monkey_results = run_monkey_test(equity_curve, close_prices)
    
    # Walk-Forward detallado
    wf_consistency = _walk_forward_validation(equity_curve)
    
    # Análisis de costos
    estimated_trades = metrics['estimated_trades']
    cost_impact = abs(metrics['total_return'] - metrics['cost_adjusted_return'])
    
    return {
        'primary_score': score,
        'metrics': metrics,
        'monkey_test': monkey_results,
        'walk_forward_consistency': wf_consistency,
        'cost_analysis': {
            'estimated_trades': estimated_trades,
            'cost_impact_percent': cost_impact * 100,
            'cost_adjusted_return': metrics['cost_adjusted_return']
        },
        'validation_summary': {
            'passes_monkey_test': monkey_results['is_significant'],
            'robust_performance': score > 0.5,
            'consistent_performance': wf_consistency > 0.6,
            'cost_acceptable': cost_impact < 0.1,
            'overall_robust': all([
                monkey_results['is_significant'],
                score > 0.5,
                wf_consistency > 0.6,
                cost_impact < 0.1
            ])
        }
    }

# ────────── impresor de métricas ─────────────────────────────────────────────
def print_detailed_metrics(metrics: dict, title: str = "Strategy Metrics"):
    """
    Imprime las métricas detalladas expandidas en formato de depuración.
    
    Args:
        metrics: Diccionario devuelto por metrics_tuple_to_dict
        title:   Encabezado para el bloque de debug
    """
    print(f"🔍 DEBUG: {title} - Score: {metrics['score']:.4f}\n"
          f"  📈 LINEARITY METRICS:\n"
          f"     • R²={metrics['r2']:.4f} | Linearity Bonus={metrics['linearity_bonus']:.4f}\n"
          f"     • Slope Reward={metrics['slope_reward']:.4f} | Consistency={metrics['consistency']:.4f}\n"
          f"  📊 ROBUSTNESS METRICS:\n"
          f"     • Sharpe={metrics['sharpe_ratio']:.4f} | Deflated Sharpe={metrics['deflated_sharpe']:.4f}\n"
          f"     • WF Consistency={metrics['wf_consistency']:.4f} | Vol Penalty={metrics['vol_penalty']:.4f}\n"
          f"  💰 RISK & COSTS:\n"
          f"     • Max Drawdown={metrics['max_drawdown']:.4f} | DD Penalty={metrics['dd_penalty']:.4f}\n"
          f"     • Est. Trades={metrics['estimated_trades']:.0f} | Cost Adj. Return={metrics['cost_adjusted_return']:.4f}\n"
          f"  🏗️ COMPONENTS:\n"
          f"     • Linearity Comp={metrics['linearity_component']:.4f} | Growth Comp={metrics['growth_component']:.4f}\n"
          f"     • Robustness Comp={metrics['robustness_component']:.4f} | Final Score={metrics['final_score']:.4f}\n"
          f"  📏 PERIODS/YEAR: {metrics['periods_per_year']:.2f}")

def _ort_session(model_path: str):
    """Thread-safe ONNX session cache"""
    if not ONNX_AVAILABLE:
        raise RuntimeError("ONNX Runtime not available. Install onnxruntime to use ONNX models.")
    
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