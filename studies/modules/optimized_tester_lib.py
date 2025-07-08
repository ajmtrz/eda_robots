
import numpy as np
from numba import njit
from tester_lib import metrics_tuple_to_dict

# Constantes financieras
RISK_FREE_RATE = 0.02    # Tasa libre de riesgo anual (2%)


@njit(cache=True, fastmath=True)
def _signed_r2_optimized(eq):
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
def evaluate_report_optimized(eq: np.ndarray, ppy: float = 6240.0):
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
    r2_optimized = _signed_r2_optimized(eq)
    
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
    linearity_component = (r2_optimized + linearity_bonus) / 2.0  # [0,1]
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
    if r2_optimized > 0.98 and slope_reward > 0.5 and max_dd < 0.01:
        final_score = min(1.0, final_score * 1.2)  # Bonus del 20%
    
    # Asegurar rango [0,1]
    final_score = max(0.0, min(1.0, final_score))
    
    # === MÉTRICAS PARA DEBUGGING ===
    metrics_tuple = (
        r2_optimized, linearity_bonus, consistency, slope_reward,
        total_return, max_dd, dd_penalty, linearity_component,
        growth_component, base_score, final_score,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0  # Padding
    )
    
    return final_score, metrics_tuple


def metrics_tuple_to_dict_optimized(score: float, metrics_tuple: tuple, periods_per_year: float) -> dict:
    """Convierte la tupla de métricas optimizada a diccionario"""
    return {
        'score': score,
        'r2_optimized': metrics_tuple[0],
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


def compare_scoring_systems(eq: np.ndarray, ppy: float = 6240.0):
    """
    Compara el sistema original vs optimizado para una curva dada.
    
    Returns:
        dict: Comparación de ambos sistemas
    """
    # Importar función original
    from tester_lib import evaluate_report
    
    # Evaluar con sistema original
    score_orig, metrics_orig = evaluate_report(eq, ppy)
    
    # Evaluar con sistema optimizado
    score_opt, metrics_opt = evaluate_report_optimized(eq, ppy)
    
    return {
        'original_score': score_orig,
        'optimized_score': score_opt,
        'improvement': score_opt - score_orig,
        'original_metrics': metrics_tuple_to_dict(score_orig, metrics_orig, ppy),
        'optimized_metrics': metrics_tuple_to_dict_optimized(score_opt, metrics_opt, ppy)
    }