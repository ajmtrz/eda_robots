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

# Thread-safe session cache
_session_cache = {}
_session_lock = threading.RLock()

# Thread-safe plotting
_plot_lock = threading.RLock()

def _safe_plot(equity_curve, metrics_dict):
    """Thread-safe plotting function"""
    with _plot_lock:
        score = metrics_dict.get('final_score', -1.0)
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
            rpt, _ = process_data(close, main, meta)
        else:
            direction_map = {'buy': 0, 'sell': 1}
            direction_int = direction_map.get(direction, 0)
            rpt, _ = process_data_one_direction(close, main, meta, direction_int)

        if rpt.size < 2:
            return -1.0

        metrics_tuple = evaluate_report(rpt)
        if print_metrics:
            metrics_dict = metrics_tuple_to_dict(metrics_tuple)
            _safe_plot(rpt, metrics_dict)
            print_detailed_metrics(metrics_dict)

        return metrics_tuple[0]
    
    except Exception as e:
        print(f"🔍 DEBUG: Error en tester: {e}")
        return -1.0

# ────────── helpers ──────────────────────────────────────────────────────────

@njit(cache=True, fastmath=True)
def _signed_r2(eq):
    """R² con signo - versión ultra-optimizada que favorece pendientes positivas perfectas"""
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
    
    if slope > 0:
        # Sistema de potenciación exponencial para pendientes positivas perfectas
        slope_factor = min(3.0, max(1.0, slope * 2.0))  # Factor entre 1-3
        r2_enhanced = min(1.0, r2 * (1.0 + slope_factor/5.0))
        
        # Bonus adicional por linealidad casi perfecta
        if r2 > 0.95:
            perfection_bonus = (r2 - 0.95) / 0.05 * 0.3  # Hasta 30% bonus
            r2_enhanced = min(1.0, r2_enhanced + perfection_bonus)
        
        return r2_enhanced
    else:
        # Penalización exponencial para pendientes negativas
        return -r2 * 2.0


@njit(cache=True, fastmath=True)
def _perfect_linearity_score(eq):
    """Nueva métrica: detecta y recompensa curvas perfectamente lineales"""
    n = eq.size
    if n < 20:
        return 0.0
    
    # Ajuste lineal de alta precisión
    t = np.arange(n, dtype=np.float64)
    x_mean = t.mean()
    y_mean = eq.mean()
    
    # Calcular pendiente y intercepto
    num = np.sum((t - x_mean) * (eq - y_mean))
    den = np.sum((t - x_mean) ** 2)
    
    if den == 0:
        return 0.0
    
    slope = num / den
    intercept = y_mean - slope * x_mean
    
    # Solo procesar pendientes positivas
    if slope <= 0:
        return 0.0
    
    # Línea teórica perfecta
    y_perfect = slope * t + intercept
    
    # Calcular desviación promedio absoluta normalizada
    deviations = np.abs(eq - y_perfect)
    mean_deviation = np.mean(deviations)
    
    # Normalizar por el rango de la serie
    data_range = max(eq[-1] - eq[0], 1e-8)
    normalized_deviation = mean_deviation / data_range
    
    # Score de linealidad perfecta (0 = perfecto, 1 = muy desviado)
    linearity_score = np.exp(-normalized_deviation * 20.0)  # Penalización exponencial
    
    # Bonus por pendiente ideal (0.1 a 2.0)
    if 0.1 <= slope <= 2.0:
        slope_bonus = 1.0
    elif slope < 0.1:
        slope_bonus = slope / 0.1
    else:
        slope_bonus = np.exp(-(slope - 2.0) * 0.5)
    
    return linearity_score * slope_bonus


@njit(cache=True, fastmath=True)
def _monotonic_growth_score(eq):
    """Evalúa crecimiento monótono casi perfecto"""
    if eq.size < 10:
        return 0.0
    
    # Calcular diferencias consecutivas
    diffs = np.diff(eq)
    
    # Porcentaje de períodos con crecimiento no negativo
    non_negative_periods = np.sum(diffs >= 0) / len(diffs)
    
    # Porcentaje de períodos con crecimiento positivo
    positive_periods = np.sum(diffs > 0) / len(diffs)
    
    # Evaluar consistencia de crecimiento
    # Ideal: 95%+ períodos con crecimiento no negativo
    monotonic_score = min(1.0, non_negative_periods * 1.05)
    
    # Bonus por períodos con crecimiento activo
    growth_activity = min(1.0, positive_periods * 1.2)
    
    # Penalización por volatilidad excesiva
    if len(diffs) > 0 and np.mean(diffs) > 0:
        cv = np.std(diffs) / (np.mean(diffs) + 1e-8)  # Coeficiente de variación
        volatility_penalty = 1.0 / (1.0 + cv * 1.5)
    else:
        volatility_penalty = 0.0
    
    return (monotonic_score * 0.6 + growth_activity * 0.4) * volatility_penalty


@njit(cache=True, fastmath=True)
def _smoothness_score(eq):
    """Evalúa la suavidad de la curva (ausencia de ruido excesivo)"""
    if eq.size < 5:
        return 0.0
    
    # Calcular segundas diferencias (aceleración)
    first_diff = np.diff(eq)
    if len(first_diff) < 2:
        return 0.0
    
    second_diff = np.diff(first_diff)
    
    # La suavidad ideal tiene segundas diferencias pequeñas
    # (cambios graduales en la velocidad de crecimiento)
    
    if len(second_diff) == 0:
        return 1.0
    
    # Normalizar por el rango de primeras diferencias
    first_diff_range = max(np.max(first_diff) - np.min(first_diff), 1e-8)
    normalized_second_diff = np.abs(second_diff) / first_diff_range
    
    # Score de suavidad (menor variación = mayor score)
    mean_volatility = np.mean(normalized_second_diff)
    smoothness = np.exp(-mean_volatility * 10.0)
    
    return smoothness


@njit(cache=True, fastmath=True)
def _advanced_drawdown_penalty(eq):
    """Sistema avanzado de penalización por drawdown"""
    if eq.size < 2:
        return 1.0
    
    peak = eq[0]
    max_dd = 0.0
    consecutive_dd_periods = 0
    max_consecutive_dd = 0
    
    for val in eq:
        if val >= peak:
            peak = val
            consecutive_dd_periods = 0
        else:
            consecutive_dd_periods += 1
            max_consecutive_dd = max(max_consecutive_dd, consecutive_dd_periods)
            dd = (peak - val) / peak if peak > 0 else 0.0
            max_dd = max(max_dd, dd)
    
    # Penalización base por drawdown máximo
    dd_penalty_base = np.exp(-max_dd * 15.0)  # Muy estricto con drawdowns
    
    # Penalización adicional por períodos consecutivos de drawdown
    periods_penalty = np.exp(-max_consecutive_dd / 20.0)
    
    # Combinar penalizaciones
    final_penalty = dd_penalty_base * (0.7 + 0.3 * periods_penalty)
    
    return max(0.0, min(1.0, final_penalty))


@njit(cache=True, fastmath=True)
def _linearity_bonus(eq):
    """Calcula un bonus específico por linealidad ascendente perfecta - MEJORADO"""
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
    
    # Calcular linealidad (R²) con mayor precisión
    y_pred = slope * (t - x_mean) + y_mean
    ss_res = np.sum((eq - y_pred) ** 2)
    ss_tot = np.sum((eq - y_mean) ** 2)
    
    if ss_tot == 0:
        return 1.0 if slope > 0 else 0.0
    
    r2 = 1.0 - (ss_res / ss_tot)
    
    # Sistema de bonus más agresivo para linealidad alta
    if r2 > 0.98:
        # Linealidad casi perfecta: bonus exponencial
        perfection_factor = (r2 - 0.98) / 0.02  # 0-1
        bonus_multiplier = 1.0 + perfection_factor * 2.0  # 1.0-3.0
    elif r2 > 0.95:
        # Linealidad muy alta: bonus moderado
        bonus_multiplier = 1.0 + (r2 - 0.95) / 0.03 * 0.5  # 1.0-1.5
    else:
        bonus_multiplier = 1.0
    
    # Pendiente normalizada más agresiva
    if slope >= 0.5:
        slope_factor = min(1.0, slope / 1.5)  # Ideal: 0.5-1.5
    else:
        slope_factor = slope / 0.5 * 0.8  # Reducido para pendientes pequeñas
    
    linear_bonus = r2 * slope_factor * bonus_multiplier
    
    return max(0.0, min(2.0, linear_bonus))  # Permitir valores > 1.0


@njit(cache=True, fastmath=True)
def _consistency_score(eq):
    """Evalúa la consistencia del crecimiento - MEJORADO"""
    if eq.size < 3:
        return 0.0
    
    # Calcular diferencias (returns)
    diffs = np.diff(eq)
    
    # Porcentaje de períodos con crecimiento positivo
    positive_periods = np.sum(diffs > 0) / len(diffs)
    
    # Porcentaje de períodos sin pérdidas (incluyendo flat)
    non_negative_periods = np.sum(diffs >= 0) / len(diffs)
    
    # Consistencia direccional mejorada
    # Ideal: 85%+ períodos positivos, 95%+ períodos no negativos
    direction_score = (positive_periods * 0.85 + non_negative_periods * 0.15)
    direction_consistency = min(1.0, direction_score * 1.1)
    
    # Análisis de volatilidad más sofisticado
    if len(diffs) > 0:
        mean_growth = np.mean(diffs)
        if mean_growth > 0:
            # Coeficiente de variación
            cv = np.std(diffs) / (mean_growth + 1e-8)
            
            # Volatility penalty más estricto
            vol_penalty = 1.0 / (1.0 + cv * 3.0)
            
            # Bonus por crecimiento estable
            if cv < 0.5:  # Baja volatilidad
                stability_bonus = 1.0 + (0.5 - cv) * 0.4
            else:
                stability_bonus = 1.0
                
            vol_penalty *= stability_bonus
        else:
            vol_penalty = 0.0
    else:
        vol_penalty = 1.0
    
    return direction_consistency * vol_penalty


@njit(cache=True, fastmath=True)
def _slope_reward(eq):
    """Recompensa específica por pendiente ascendente fuerte - MEJORADO"""
    n = eq.size
    if n < 2:
        return 0.0
    
    # Pendiente global: (final - inicial) / tiempo
    total_growth = eq[-1] - eq[0]
    time_span = n - 1
    
    if time_span == 0:
        return 0.0
    
    slope = total_growth / time_span
    
    # Solo recompensar pendientes positivas
    if slope <= 0:
        return 0.0
    
    # Sistema de recompensa más agresivo y específico
    if slope < 0.05:
        # Pendientes muy pequeñas: reward mínimo
        return slope / 0.05 * 0.1
    elif slope < 0.2:
        # Pendientes pequeñas pero aceptables
        return 0.1 + (slope - 0.05) / 0.15 * 0.3
    elif slope <= 1.5:
        # Rango ideal: reward casi lineal con bonus
        base_reward = 0.4 + (slope - 0.2) / 1.3 * 0.5
        
        # Bonus por estar en el rango perfecto (0.5-1.0)
        if 0.5 <= slope <= 1.0:
            ideal_bonus = 1.0 + 0.3  # 30% bonus
        else:
            ideal_bonus = 1.0
            
        return base_reward * ideal_bonus
    else:
        # Pendientes muy altas: reward alto pero con decaimiento suave
        excess = slope - 1.5
        base_reward = 0.9
        decay = np.exp(-excess * 0.3)
        return base_reward * decay


@njit(cache=True, fastmath=True)
def evaluate_report(eq: np.ndarray) -> tuple:
    """
    Sistema de scoring ULTRA-OPTIMIZADO para curvas de equity perfectamente lineales.
    
    Cambios principales:
    - Nuevas métricas avanzadas para detectar linealidad perfecta
    - Pesos rebalanceados para maximizar detección de curvas lineales
    - Sistema de bonificación más agresivo
    - Penalizaciones más estrictas para desviaciones
    
    Returns:
        tuple: (metrics_tuple)
    """
    # Validaciones básicas
    if eq.size < 50 or not np.isfinite(eq).all():  # Reducido de 200 a 50
        return (-1.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0)
    
    # Protección para valores negativos
    eq_min = np.min(eq)
    if eq_min <= 0.0:
        eq = eq - eq_min + 1.0
    
    # === MÉTRICAS AVANZADAS OPTIMIZADAS ===
    
    # 1. R² con sistema de bonificación exponencial
    r2 = _signed_r2(eq)
    if r2 < 0.0:
        return (-1.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0)
    
    # 2. Score de linealidad perfecta (NUEVA MÉTRICA)
    perfect_linearity = _perfect_linearity_score(eq)
    
    # 3. Bonus de linealidad mejorado
    linearity_bonus = _linearity_bonus(eq)
    
    # 4. Consistencia mejorada
    consistency = _consistency_score(eq)
    
    # 5. Recompensa por pendiente optimizada
    slope_reward = _slope_reward(eq)
    
    # 6. Score de crecimiento monótono (NUEVA MÉTRICA)
    monotonic_growth = _monotonic_growth_score(eq)
    
    # 7. Score de suavidad (NUEVA MÉTRICA)
    smoothness = _smoothness_score(eq)
    
    # 8. Retorno total normalizado
    total_return = max(0.0, (eq[-1] - eq[0]) / max(abs(eq[0]), 1.0))
    
    # 9. Penalización avanzada por drawdown
    dd_penalty = _advanced_drawdown_penalty(eq)
    
    # === SISTEMA DE SCORING ULTRA-OPTIMIZADO ===
    
    # Componente de linealidad perfecta (peso principal)
    linearity_component = (
        r2 * 0.3 +                    # R² base
        perfect_linearity * 0.4 +     # Linealidad perfecta (NUEVA)
        linearity_bonus * 0.3         # Bonus linealidad
    )
    
    # Componente de crecimiento consistente
    growth_component = (
        slope_reward * 0.4 +          # Recompensa pendiente
        consistency * 0.3 +           # Consistencia
        monotonic_growth * 0.3        # Crecimiento monótono (NUEVA)
    )
    
    # Componente de calidad técnica
    quality_component = (
        smoothness * 0.6 +            # Suavidad (NUEVA)
        min(1.0, total_return) * 0.4  # Retorno total
    )
    
    # Score base con pesos rebalanceados para máxima linealidad
    base_score = (
        linearity_component * 0.55 +  # 55% peso a linealidad (aumentado)
        growth_component * 0.30 +     # 30% peso a crecimiento
        quality_component * 0.15      # 15% peso a calidad
    )
    
    # Aplicar penalización por drawdown
    penalized_score = base_score * dd_penalty
    
    # === SISTEMA DE BONIFICACIÓN ULTRA-AGRESIVO ===
    
    final_score = penalized_score
    
    # Bonus por linealidad casi perfecta (R² > 0.98)
    if r2 > 0.98:
        perfection_bonus = (r2 - 0.98) / 0.02 * 0.25  # Hasta 25% bonus
        final_score = min(1.0, final_score * (1.0 + perfection_bonus))
    
    # Bonus por combinación perfecta de métricas
    if (perfect_linearity > 0.9 and monotonic_growth > 0.9 and 
        smoothness > 0.8 and slope_reward > 0.7):
        elite_bonus = 0.15  # 15% bonus por excelencia total
        final_score = min(1.0, final_score * (1.0 + elite_bonus))
    
    # Bonus por crecimiento monótono perfecto
    if monotonic_growth > 0.95:
        monotonic_bonus = (monotonic_growth - 0.95) / 0.05 * 0.1  # Hasta 10% bonus
        final_score = min(1.0, final_score * (1.0 + monotonic_bonus))
    
    # Asegurar rango [0,1]
    final_score = max(0.0, min(1.0, final_score))
    
    # === MÉTRICAS EXPANDIDAS PARA DEBUGGING ===
    metrics_tuple = (
        final_score, r2, perfect_linearity, linearity_bonus, consistency, 
        slope_reward, monotonic_growth, smoothness, total_return, dd_penalty,
        linearity_component, growth_component, quality_component, base_score, penalized_score
    )
    
    return metrics_tuple


def metrics_tuple_to_dict(metrics_tuple: tuple) -> dict:
    """Convierte la tupla de métricas optimizada a diccionario - EXPANDIDO"""
    # Asegura que la clave 'final_score' esté presente y consistente con la tupla
    return {
        'final_score': metrics_tuple[0],
        'r2': metrics_tuple[1],
        'perfect_linearity': metrics_tuple[2],
        'linearity_bonus': metrics_tuple[3],
        'consistency': metrics_tuple[4],
        'slope_reward': metrics_tuple[5],
        'monotonic_growth': metrics_tuple[6],
        'smoothness': metrics_tuple[7],
        'total_return': metrics_tuple[8],
        'dd_penalty': metrics_tuple[9],
        'linearity_component': metrics_tuple[10],
        'growth_component': metrics_tuple[11],
        'quality_component': metrics_tuple[12],
        'base_score': metrics_tuple[13],
        'penalized_score': metrics_tuple[14]
    }

# ────────── impresor de métricas ─────────────────────────────────────────────
def print_detailed_metrics(metrics_dict: tuple):
    """
    Imprime las métricas detalladas en formato de depuración - EXPANDIDO.
    
    Args:
        metrics: Diccionario devuelto por metrics_tuple_to_dict
        title:   Encabezado para el bloque de debug
    """
    print(f"🔍 DEBUG: Strategy Metrics - Score: {metrics_dict['final_score']:.6f}\n"
          f"  📈 LINEALITY METRICS:\n"
          f"    • R²={metrics_dict['r2']:.6f} | Perfect Linearity={metrics_dict['perfect_linearity']:.6f}\n"
          f"    • Linearity Bonus={metrics_dict['linearity_bonus']:.6f}\n"
          f"  📊 GROWTH METRICS:\n"
          f"    • Consistency={metrics_dict['consistency']:.6f} | Slope Reward={metrics_dict['slope_reward']:.6f}\n"
          f"    • Monotonic Growth={metrics_dict['monotonic_growth']:.6f}\n"
          f"  🎯 QUALITY METRICS:\n"
          f"    • Smoothness={metrics_dict['smoothness']:.6f} | Total Return={metrics_dict['total_return']:.6f}\n"
          f"    • DD Penalty={metrics_dict['dd_penalty']:.6f}\n"
          f"  🔧 COMPONENT SCORES:\n"
          f"    • Linearity Comp={metrics_dict['linearity_component']:.6f} | Growth Comp={metrics_dict['growth_component']:.6f}\n"
          f"    • Quality Comp={metrics_dict['quality_component']:.6f}\n"
          f"  🏆 FINAL SCORES:\n"
          f"    • Base={metrics_dict['base_score']:.6f} | Penalized={metrics_dict['penalized_score']:.6f}\n")

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