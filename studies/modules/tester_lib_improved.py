import numpy as np
from numba import njit

@njit(cache=True, fastmath=True)
def _calculate_r2_improved(eq):
    """R² mejorado con mejor manejo de casos especiales"""
    n = eq.size
    if n < 2:
        return 0.0
        
    t = np.arange(n, dtype=np.float64)
    xm = t.mean()
    ym = eq.mean()
    
    # Calcular covarianza y varianzas
    cov = ((t-xm)*(eq-ym)).sum()
    var_t = ((t-xm)**2).sum()
    var_y = ((eq-ym)**2).sum()
    
    # Manejar casos especiales
    if var_t == 0.0 or var_y == 0.0:
        return 0.0
    
    # R² básico
    r2 = (cov*cov)/(var_t*var_y)
    
    # Ajustar por dirección
    slope = cov / var_t
    if slope > 0:
        return min(1.0, r2)
    else:
        # Penalización menos severa para pendientes negativas
        return max(-0.5, -r2 * 0.5)


@njit(cache=True, fastmath=True)
def _calculate_linearity_improved(eq):
    """Métrica de linealidad mejorada con tolerancia al ruido"""
    n = eq.size
    if n < 10:
        return 0.0
    
    # Ajuste lineal
    t = np.arange(n, dtype=np.float64)
    x_mean = t.mean()
    y_mean = eq.mean()
    
    num = np.sum((t - x_mean) * (eq - y_mean))
    den = np.sum((t - x_mean) ** 2)
    
    if den == 0:
        return 0.0
    
    slope = num / den
    intercept = y_mean - slope * x_mean
    
    # Solo evaluar pendientes positivas
    if slope <= 0:
        return 0.0
    
    # Línea ideal
    y_ideal = slope * t + intercept
    
    # Calcular error normalizado
    residuals = eq - y_ideal
    
    # Usar desviación estándar robusta (MAD - Median Absolute Deviation)
    median_residual = np.median(residuals)
    mad = np.median(np.abs(residuals - median_residual))
    
    # Normalizar por el rango
    data_range = np.max(eq) - np.min(eq)
    if data_range < 1e-8:
        return 0.0
    
    # Score basado en MAD normalizado (más robusto al ruido)
    normalized_mad = mad / data_range
    linearity_score = np.exp(-normalized_mad * 10.0)
    
    # Bonus por pendiente consistente
    slope_factor = min(1.0, slope / 0.5)  # Pendiente ideal alrededor de 0.5
    
    return linearity_score * (0.7 + 0.3 * slope_factor)


@njit(cache=True, fastmath=True)
def _calculate_consistency_improved(eq):
    """Consistencia mejorada con mejor manejo de volatilidad"""
    if eq.size < 3:
        return 0.0
    
    # Diferencias (returns)
    diffs = np.diff(eq)
    
    if len(diffs) == 0:
        return 0.0
    
    # Proporción de períodos positivos
    positive_ratio = np.sum(diffs > 0) / len(diffs)
    
    # Proporción de períodos no negativos
    non_negative_ratio = np.sum(diffs >= 0) / len(diffs)
    
    # Score direccional
    direction_score = positive_ratio * 0.7 + non_negative_ratio * 0.3
    
    # Volatilidad relativa (usando mediana para robustez)
    median_diff = np.median(np.abs(diffs))
    mean_diff = np.mean(diffs)
    
    if mean_diff > 0 and median_diff > 0:
        # Coeficiente de variación robusto
        volatility_ratio = median_diff / mean_diff
        volatility_penalty = 1.0 / (1.0 + volatility_ratio)
    else:
        volatility_penalty = direction_score * 0.5
    
    return direction_score * volatility_penalty


@njit(cache=True, fastmath=True)
def _calculate_drawdown_penalty_improved(eq):
    """Penalización por drawdown más equilibrada"""
    if eq.size < 2:
        return 1.0
    
    peak = eq[0]
    max_dd = 0.0
    dd_duration = 0
    max_dd_duration = 0
    
    for val in eq:
        if val >= peak:
            peak = val
            dd_duration = 0
        else:
            dd_duration += 1
            max_dd_duration = max(max_dd_duration, dd_duration)
            dd = (peak - val) / peak if peak > 0 else 0.0
            max_dd = max(max_dd, dd)
    
    # Penalización base menos severa
    if max_dd < 0.05:
        dd_penalty = 1.0
    elif max_dd < 0.1:
        dd_penalty = 0.9 + (0.1 - max_dd) / 0.05 * 0.1
    elif max_dd < 0.2:
        dd_penalty = 0.7 + (0.2 - max_dd) / 0.1 * 0.2
    elif max_dd < 0.3:
        dd_penalty = 0.5 + (0.3 - max_dd) / 0.1 * 0.2
    else:
        dd_penalty = np.exp(-max_dd * 3.0) * 0.5
    
    # Penalización por duración
    duration_penalty = 1.0 / (1.0 + max_dd_duration / (eq.size * 0.2))
    
    return dd_penalty * (0.8 + 0.2 * duration_penalty)


@njit(cache=True, fastmath=True)
def _calculate_slope_quality(eq):
    """Evalúa la calidad de la pendiente de crecimiento"""
    n = eq.size
    if n < 2:
        return 0.0
    
    # Crecimiento total normalizado
    total_growth = eq[-1] - eq[0]
    initial_value = max(abs(eq[0]), 1.0)
    normalized_growth = total_growth / initial_value
    
    # Tasa de crecimiento por período
    growth_rate = normalized_growth / (n - 1)
    
    # Score basado en tasa de crecimiento ideal
    if growth_rate <= 0:
        return 0.0
    elif growth_rate < 0.0001:  # Muy lento
        return growth_rate / 0.0001 * 0.3
    elif growth_rate < 0.001:   # Lento pero aceptable
        return 0.3 + (growth_rate - 0.0001) / 0.0009 * 0.4
    elif growth_rate < 0.01:    # Rango ideal
        return 0.7 + (growth_rate - 0.001) / 0.009 * 0.3
    else:  # Muy rápido
        return 1.0 * np.exp(-(growth_rate - 0.01) * 50.0)


@njit(cache=True, fastmath=True)
def _calculate_smoothness_improved(eq):
    """Métrica de suavidad mejorada"""
    if eq.size < 5:
        return 0.0
    
    # Primera diferencia (velocidad)
    first_diff = np.diff(eq)
    if len(first_diff) < 2:
        return 0.0
    
    # Segunda diferencia (aceleración)
    second_diff = np.diff(first_diff)
    
    # Normalizar por rango de datos
    data_range = np.max(eq) - np.min(eq)
    if data_range < 1e-8:
        return 0.0
    
    # Usar percentil 75 en lugar de mean para robustez
    acceleration_75 = np.percentile(np.abs(second_diff), 75)
    normalized_acceleration = acceleration_75 / data_range
    
    # Score exponencial
    smoothness = np.exp(-normalized_acceleration * 20.0)
    
    return smoothness


@njit(cache=True, fastmath=True)
def _calculate_trade_efficiency(trade_stats: np.ndarray, eq_length: int):
    """Evalúa la eficiencia del trading"""
    if eq_length < 10:
        return 0.0
    
    total_trades = trade_stats[0]
    positive_trades = trade_stats[1]
    win_rate = trade_stats[4]
    
    if total_trades <= 0:
        return 0.0
    
    # Frecuencia normalizada
    trade_frequency = total_trades / eq_length
    
    # Score de frecuencia (favorece actividad moderada)
    if trade_frequency < 0.01:
        freq_score = trade_frequency / 0.01 * 0.5
    elif trade_frequency < 0.1:
        freq_score = 0.5 + (trade_frequency - 0.01) / 0.09 * 0.4
    elif trade_frequency < 0.2:
        freq_score = 0.9 + (trade_frequency - 0.1) / 0.1 * 0.1
    else:
        freq_score = 1.0 * np.exp(-(trade_frequency - 0.2) * 5.0)
    
    # Score de calidad
    quality_score = win_rate ** 2  # Favorece win rates altos
    
    # Combinar scores
    efficiency = freq_score * 0.4 + quality_score * 0.6
    
    return efficiency


@njit(cache=True, fastmath=True)
def evaluate_report_improved(eq: np.ndarray, trade_stats: np.ndarray) -> tuple:
    """
    Versión mejorada de evaluate_report con mayor robustez y precisión.
    
    Cambios principales:
    - Validaciones menos estrictas
    - Mejor manejo de ruido y volatilidad
    - Penalizaciones más equilibradas
    - Métricas más robustas
    """
    # Validaciones básicas más flexibles
    if eq.size < 10:  # Reducido de 50
        return (-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    
    # Verificar que no sea todo NaN o infinito
    if not np.isfinite(eq).all():
        return (-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    
    # Ajustar valores negativos (shift)
    eq_min = np.min(eq)
    if eq_min <= 0.0:
        eq = eq - eq_min + 1.0
    
    # === CALCULAR MÉTRICAS ===
    
    # 1. R² mejorado
    r2 = _calculate_r2_improved(eq)
    
    # 2. Linealidad robusta
    linearity = _calculate_linearity_improved(eq)
    
    # 3. Consistencia mejorada
    consistency = _calculate_consistency_improved(eq)
    
    # 4. Calidad de pendiente
    slope_quality = _calculate_slope_quality(eq)
    
    # 5. Suavidad mejorada
    smoothness = _calculate_smoothness_improved(eq)
    
    # 6. Penalización por drawdown equilibrada
    dd_penalty = _calculate_drawdown_penalty_improved(eq)
    
    # 7. Eficiencia de trading
    trade_efficiency = _calculate_trade_efficiency(trade_stats, eq.size)
    
    # 8. Retorno total
    total_return = (eq[-1] - eq[0]) / max(abs(eq[0]), 1.0)
    return_score = min(1.0, max(0.0, total_return / 2.0))  # Normalizado a [0,1]
    
    # === COMBINAR MÉTRICAS ===
    
    # Componentes principales
    trend_component = (
        r2 * 0.3 +
        linearity * 0.4 +
        slope_quality * 0.3
    )
    
    quality_component = (
        consistency * 0.4 +
        smoothness * 0.3 +
        return_score * 0.3
    )
    
    trading_component = trade_efficiency
    
    # Score base
    base_score = (
        trend_component * 0.45 +
        quality_component * 0.35 +
        trading_component * 0.20
    )
    
    # Aplicar penalización por drawdown
    penalized_score = base_score * dd_penalty
    
    # === BONIFICACIONES ===
    
    final_score = penalized_score
    
    # Bonus por excelencia en múltiples métricas
    excellence_count = 0
    if r2 > 0.9: excellence_count += 1
    if linearity > 0.85: excellence_count += 1
    if consistency > 0.8: excellence_count += 1
    if smoothness > 0.8: excellence_count += 1
    if dd_penalty > 0.9: excellence_count += 1
    
    if excellence_count >= 3:
        excellence_bonus = 0.05 * excellence_count
        final_score = min(1.0, final_score * (1.0 + excellence_bonus))
    
    # Asegurar rango [0,1]
    final_score = max(0.0, min(1.0, final_score))
    
    # Dummy values para otras métricas (para compatibilidad)
    metrics_tuple = (
        final_score, r2, linearity, 0.0, consistency,
        slope_quality, 0.0, smoothness, return_score, dd_penalty,
        trend_component, quality_component, trading_component, base_score, penalized_score,
        trade_efficiency, trading_component
    )
    
    return metrics_tuple