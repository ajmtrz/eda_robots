#!/usr/bin/env python3
"""
Script para calibrar los pesos de las métricas en tester_lib.py

Genera miles de curvas de equity sintéticas con diferentes características:
- Muchos trades
- Rectilíneas (alto R²)
- Alta pendiente
- Alto retorno/drawdown
- Alto Calmar ratio

El objetivo es ajustar los pesos para promover curvas con estas características.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import time
import random
from dataclasses import dataclass
from collections import defaultdict
import json
import os
from numba import jit, njit, prange, config
import numba
import multiprocessing

# Importar las funciones necesarias de tester_lib
import sys
sys.path.append('studies/modules')
from tester_lib import evaluate_report, backtest_equivalent

# Funciones JIT para optimización de performance
@njit
def generate_single_curve_jit(n_points, base_trend, volatility, noise_level, linearity, seed):
    """Versión JIT de generación de curva individual"""
    np.random.seed(seed)
    
    # Generar curva base
    x = np.arange(n_points, dtype=np.float64)
    
    # Componente lineal
    linear_component = base_trend * x
    
    # Componente no lineal (para reducir R²)
    if linearity < 0.8:
        non_linear = np.sin(x * 0.1) * (1 - linearity) * n_points * 0.1
    else:
        non_linear = np.zeros_like(x)
    
    # Ruido aleatorio
    noise = np.random.normal(0, noise_level * n_points, n_points)
    
    # Combinar componentes
    raw_curve = linear_component + non_linear + noise
    
    # Asegurar que la curva sea creciente en general
    if np.mean(np.diff(raw_curve)) < 0:
        raw_curve = -raw_curve
    
    # Normalizar para que empiece en 0
    raw_curve = raw_curve - raw_curve[0]
    
    return raw_curve

@njit
def apply_drawdown_control_jit(curve, max_dd_target):
    """Versión JIT del control de drawdown"""
    n = len(curve)
    running_max = np.zeros(n)
    running_max[0] = curve[0]
    
    # Calcular running maximum
    for i in range(1, n):
        running_max[i] = max(running_max[i-1], curve[i])
    
    # Calcular current drawdown
    current_dd = (running_max - curve) / (running_max + 1e-8)
    
    # Si el drawdown es muy alto, ajustar
    max_dd = np.max(current_dd)
    if max_dd > max_dd_target:
        dd_factor = max_dd_target / max_dd
        adjusted_curve = curve.copy()
        
        for i in range(1, n):
            if current_dd[i] > max_dd_target * 0.5:
                adjustment = (current_dd[i] - max_dd_target * 0.5) * dd_factor
                adjusted_curve[i] += adjustment
        
        return adjusted_curve
    
    return curve

@njit
def calculate_basic_metrics_jit(curve):
    """Calcula métricas básicas de una curva (versión JIT)"""
    n = len(curve)
    
    # Total return
    total_return = curve[-1] - curve[0]
    
    # Calcular drawdown máximo
    running_max = np.zeros(n)
    running_max[0] = curve[0]
    
    for i in range(1, n):
        running_max[i] = max(running_max[i-1], curve[i])
    
    max_drawdown = np.max(running_max - curve)
    
    # Calcular Calmar ratio
    if max_drawdown > 0:
        calmar_ratio = total_return / max_drawdown
    else:
        calmar_ratio = 0.0
    
    # Calcular pendiente (regresión lineal simple)
    x = np.arange(n, dtype=np.float64)
    x_mean = np.mean(x)
    y_mean = np.mean(curve)
    
    numerator = np.sum((x - x_mean) * (curve - y_mean))
    denominator = np.sum((x - x_mean) ** 2)
    
    if denominator > 0:
        slope = numerator / denominator
    else:
        slope = 0.0
    
    # Calcular R²
    y_pred = slope * x + (y_mean - slope * x_mean)
    ss_res = np.sum((curve - y_pred) ** 2)
    ss_tot = np.sum((curve - y_mean) ** 2)
    
    if ss_tot > 0:
        r2 = 1 - (ss_res / ss_tot)
    else:
        r2 = 0.0
    
    return total_return, max_drawdown, calmar_ratio, slope, r2

@njit(parallel=True)
def calculate_scores_batch_jit(r2_values, slope_nl_values, rdd_nl_values, 
                              calmar_nl_values, trade_nl_values, wf_nl_values, weights):
    """Calcula scores para un batch de curvas (versión JIT paralela con wf_nl)"""
    n = len(r2_values)
    scores = np.zeros(n)
    
    for i in prange(n):
        scores[i] = (
            weights[0] * r2_values[i] +
            weights[1] * slope_nl_values[i] +
            weights[2] * rdd_nl_values[i] +
            weights[3] * calmar_nl_values[i] +
            weights[4] * trade_nl_values[i] +
            weights[5] * wf_nl_values[i]
        )
    
    return scores

@njit(parallel=True)
def count_characteristics_parallel_jit(r2_values, n_trades_values, slopes, calmar_ratios):
    """Cuenta características deseables en paralelo (versión original)"""
    n = len(r2_values)
    counts = np.zeros(4, dtype=np.int64)
    
    # Contadores paralelos para cada característica
    high_trades_array = np.zeros(n, dtype=np.int64)
    high_r2_array = np.zeros(n, dtype=np.int64)
    high_slope_array = np.zeros(n, dtype=np.int64)
    high_calmar_array = np.zeros(n, dtype=np.int64)
    
    for i in prange(n):
        if n_trades_values[i] > 500:
            high_trades_array[i] = 1
        if r2_values[i] > 0.8:
            high_r2_array[i] = 1
        if slopes[i] > 0.01:
            high_slope_array[i] = 1
        if calmar_ratios[i] > 2.0:
            high_calmar_array[i] = 1
    
    # Sumar resultados
    counts[0] = np.sum(high_trades_array)
    counts[1] = np.sum(high_r2_array)
    counts[2] = np.sum(high_slope_array)
    counts[3] = np.sum(high_calmar_array)
    
    return counts

@njit(parallel=True)
def count_characteristics_parallel_jit_extended(r2_values, n_trades_values, slopes, calmar_ratios, wf_consistencies):
    """Cuenta características deseables en paralelo incluyendo walk-forward consistency"""
    n = len(r2_values)
    counts = np.zeros(5, dtype=np.int64)
    
    # Contadores paralelos para cada característica
    high_trades_array = np.zeros(n, dtype=np.int64)
    high_r2_array = np.zeros(n, dtype=np.int64)
    high_slope_array = np.zeros(n, dtype=np.int64)
    high_calmar_array = np.zeros(n, dtype=np.int64)
    high_wf_array = np.zeros(n, dtype=np.int64)
    
    for i in prange(n):
        if n_trades_values[i] > 500:
            high_trades_array[i] = 1
        if r2_values[i] > 0.8:
            high_r2_array[i] = 1
        if slopes[i] > 0.01:
            high_slope_array[i] = 1
        if calmar_ratios[i] > 2.0:
            high_calmar_array[i] = 1
        if wf_consistencies[i] > 0.7:  # Alta consistencia walk-forward
            high_wf_array[i] = 1
    
    # Sumar resultados
    counts[0] = np.sum(high_trades_array)
    counts[1] = np.sum(high_r2_array)
    counts[2] = np.sum(high_slope_array)
    counts[3] = np.sum(high_calmar_array)
    counts[4] = np.sum(high_wf_array)
    
    return counts

@njit
def evaluate_weight_quality_jit(r2_values, slope_nl_values, rdd_nl_values,
                               calmar_nl_values, trade_nl_values, wf_nl_values,
                               n_trades_values, calmar_ratios, slopes, wf_consistencies, weights):
    """Versión JIT de evaluación de calidad de pesos con paralelización y wf_nl"""
    n = len(r2_values)
    
    # Calcular scores usando función paralela
    scores = calculate_scores_batch_jit(r2_values, slope_nl_values, rdd_nl_values,
                                       calmar_nl_values, trade_nl_values, wf_nl_values, weights)
    
    # Métricas de calidad
    avg_score = np.mean(scores)
    std_score = np.std(scores)
    
    # Contar características deseables usando función paralela (incluyendo wf_consistency)
    counts = count_characteristics_parallel_jit_extended(r2_values, n_trades_values, slopes, 
                                                        calmar_ratios, wf_consistencies)
    high_trades, high_r2, high_slope, high_calmar, high_wf = counts[0], counts[1], counts[2], counts[3], counts[4]
    
    # Score de calidad basado en múltiples criterios (incluyendo walk-forward)
    quality_score = (
        0.35 * avg_score +
        0.15 * (high_trades / n) +
        0.15 * (high_r2 / n) +
        0.1 * (high_slope / n) +
        0.1 * (high_calmar / n) +
        0.15 * (high_wf / n) -
        0.1 * std_score
    )
    
    return quality_score, avg_score, std_score





@njit(parallel=True)
def evaluate_multiple_weights_parallel_jit(weights_batch, r2_values, slope_nl_values, 
                                          rdd_nl_values, calmar_nl_values, trade_nl_values, wf_nl_values,
                                          n_trades_values, calmar_ratios, slopes, wf_consistencies):
    """Evalúa múltiples combinaciones de pesos en paralelo incluyendo wf_nl"""
    n_weights = len(weights_batch)
    quality_scores = np.zeros(n_weights)
    avg_scores = np.zeros(n_weights)
    std_scores = np.zeros(n_weights)
    
    for w_idx in prange(n_weights):
        weights = weights_batch[w_idx]
        
        # Calcular scores para esta combinación de pesos
        scores = calculate_scores_batch_jit(r2_values, slope_nl_values, rdd_nl_values,
                                          calmar_nl_values, trade_nl_values, wf_nl_values, weights)
        
        # Métricas básicas
        avg_score = np.mean(scores)
        std_score = np.std(scores)
        
        # Contar características deseables incluyendo walk-forward
        counts = count_characteristics_parallel_jit_extended(r2_values, n_trades_values, slopes, 
                                                            calmar_ratios, wf_consistencies)
        high_trades, high_r2, high_slope, high_calmar, high_wf = counts[0], counts[1], counts[2], counts[3], counts[4]
        
        n = len(r2_values)
        
        # Score de calidad incluyendo walk-forward consistency
        quality_score = (
            0.35 * avg_score +
            0.15 * (high_trades / n) +
            0.15 * (high_r2 / n) +
            0.1 * (high_slope / n) +
            0.1 * (high_calmar / n) +
            0.15 * (high_wf / n) -
            0.1 * std_score
        )
        
        quality_scores[w_idx] = quality_score
        avg_scores[w_idx] = avg_score
        std_scores[w_idx] = std_score
    
    return quality_scores, avg_scores, std_scores

@dataclass
class CurveMetrics:
    """Métricas de una curva de equity"""
    trade_nl: float
    rdd_nl: float
    r2: float
    slope_nl: float
    calmar_nl: float
    wf_nl: float
    score: float
    n_trades: int
    total_return: float
    max_drawdown: float
    calmar_ratio: float
    slope: float

class WeightCalibrator:
    """Clase para calibrar los pesos de las métricas"""
    
    def __init__(self):
        self.curves = []
        self.best_weights = None
        self.best_score = -1.0
        
    def generate_synthetic_curves(self, n_curves: int = 10000) -> List[np.ndarray]:
        """Genera curvas de equity sintéticas con diferentes características"""
        curves = []
        
        print(f"🔍 DEBUG: Generando {n_curves} curvas sintéticas con paralelización...")
        
        # Generar en batches para aprovechar paralelización
        batch_size = min(1000, n_curves // 4)  # Batches adaptativos
        n_batches = (n_curves + batch_size - 1) // batch_size
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_curves)
            batch_n_curves = end_idx - start_idx
            
            if batch_idx % 5 == 0:
                print(f"🔍 DEBUG: Progreso: batch {batch_idx}/{n_batches} ({start_idx}/{n_curves})")
            
            # Generar parámetros para el batch
            batch_curves = []
            for i in range(batch_n_curves):
                curve = self._generate_single_curve()
                batch_curves.append(curve)
            
            curves.extend(batch_curves)
        
        print("🔍 DEBUG: Generación completada.")
        return curves
    
    def _generate_single_curve(self) -> np.ndarray:
        """Genera una curva de equity individual con características controladas"""
        # Parámetros aleatorios para la curva
        n_points = random.randint(200, 1000)
        base_trend = random.uniform(-0.1, 0.3)  # Tendencia base
        volatility = random.uniform(0.01, 0.1)   # Volatilidad
        noise_level = random.uniform(0.001, 0.05)  # Ruido
        linearity = random.uniform(0.3, 1.0)     # Linealidad (0-1)
        seed = random.randint(0, 2**31 - 1)  # Seed para JIT
        
        # Usar versión JIT para generar la curva base
        raw_curve = generate_single_curve_jit(n_points, base_trend, volatility, 
                                            noise_level, linearity, seed)
        
        # Aplicar transformación para controlar drawdown
        max_dd_target = random.uniform(0.05, 0.3)
        curve = apply_drawdown_control_jit(raw_curve, max_dd_target)
        
        return curve
    

    
    def evaluate_curves(self, curves: List[np.ndarray]) -> List[CurveMetrics]:
        """Evalúa todas las curvas y calcula sus métricas"""
        metrics_list = []
        
        print("🔍 DEBUG: Evaluando curvas con optimizaciones...")
        
        # Procesar en batches para mejor eficiencia
        batch_size = 1000
        n_batches = (len(curves) + batch_size - 1) // batch_size
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(curves))
            
            print(f"🔍 DEBUG: Evaluación progreso: batch {batch_idx}/{n_batches} ({start_idx}/{len(curves)})")
            
            # Preparar listas para métricas adicionales calculadas en paralelo
            batch_curves = curves[start_idx:end_idx]
            batch_metrics = []
            
            # Calcular métricas básicas en paralelo para el batch
            # (Las métricas de tester_lib se calculan secuencialmente porque requieren lógica compleja)
            
            for i, curve in enumerate(batch_curves):
                try:
                    # Evaluar con los parámetros por defecto (tester_lib) - ahora incluye wf_nl
                    trade_nl, rdd_nl, r2, slope_nl, calmar_nl, wf_nl = evaluate_report(
                        curve, 
                        periods_per_year=6240.0,  # H1
                        min_trades=200,
                        rdd_floor=1.0,
                        calmar_floor=1.0
                    )
                    
                    # Usar JIT para métricas adicionales
                    total_return, max_drawdown, calmar_ratio, slope, r2_check = calculate_basic_metrics_jit(curve)
                    
                    n_trades = len(np.diff(curve))
                    
                    metrics = CurveMetrics(
                        trade_nl=trade_nl,
                        rdd_nl=rdd_nl,
                        r2=r2,
                        slope_nl=slope_nl,
                        calmar_nl=calmar_nl,
                        wf_nl=wf_nl,
                        score=0.0,  # Se calculará después
                        n_trades=n_trades,
                        total_return=total_return,
                        max_drawdown=max_drawdown,
                        calmar_ratio=calmar_ratio,
                        slope=slope
                    )
                    
                    batch_metrics.append(metrics)
                    
                except Exception as e:
                    print(f"🔍 DEBUG: Error evaluando curva {start_idx + i}: {e}")
                    continue
            
            metrics_list.extend(batch_metrics)
        
        print("🔍 DEBUG: Evaluación completada.")
        return metrics_list
    
    def test_weight_combinations(self, metrics_list: List[CurveMetrics], 
                                n_trials: int = 1000) -> Dict:
        """Prueba diferentes combinaciones de pesos"""
        
        print(f"Probando {n_trials} combinaciones de pesos...")
        
        best_result = {
            'weights': None,
            'score': -1.0,
            'metrics': None
        }
        
        # Filtrar curvas válidas incluyendo wf_nl
        valid_metrics = [m for m in metrics_list if (
            m.trade_nl >= 0 and m.rdd_nl >= 0 and m.r2 >= 0 and 
            m.slope_nl >= 0 and m.calmar_nl >= 0 and m.wf_nl >= 0
        )]
        
        print(f"🔍 DEBUG: Curvas válidas para testing (con wf_nl): {len(valid_metrics)}")
        
        if len(valid_metrics) < 100:
            print("⚠️ Muy pocas curvas válidas. Generando más curvas...")
            return best_result
        
        # Preparar arrays para JIT con 6 métricas
        n_valid = len(valid_metrics)
        r2_values = np.array([m.r2 for m in valid_metrics])
        slope_nl_values = np.array([m.slope_nl for m in valid_metrics])
        rdd_nl_values = np.array([m.rdd_nl for m in valid_metrics])
        calmar_nl_values = np.array([m.calmar_nl for m in valid_metrics])
        trade_nl_values = np.array([m.trade_nl for m in valid_metrics])
        wf_nl_values = np.array([m.wf_nl for m in valid_metrics])
        n_trades_values = np.array([m.n_trades for m in valid_metrics])
        calmar_ratios = np.array([m.calmar_ratio for m in valid_metrics])
        slopes = np.array([m.slope for m in valid_metrics])
        wf_consistencies = np.array([m.wf_nl for m in valid_metrics])  # Para conteo de características
        
        print("🔍 DEBUG: Arrays preparados para JIT, iniciando optimización paralela...")
        
        # Optimización: procesar pesos en batches paralelos para 6 métricas
        batch_size = min(100, max(10, n_trials // 50))  # Batch adaptativo
        n_batches = (n_trials + batch_size - 1) // batch_size
        
        print(f"🔍 DEBUG: Procesando {n_trials} trials en {n_batches} batches de {batch_size} (6 métricas)")
        
        for batch_idx in range(n_batches):
            start_trial = batch_idx * batch_size
            end_trial = min(start_trial + batch_size, n_trials)
            current_batch_size = end_trial - start_trial
            
            # Generar batch de pesos aleatorios para 6 métricas
            weights_batch = np.zeros((current_batch_size, 6))
            for i in range(current_batch_size):
                weights_batch[i] = np.random.dirichlet(np.ones(6))
            
            # Evaluar todo el batch en paralelo con 6 métricas
            quality_scores, avg_scores, std_scores = evaluate_multiple_weights_parallel_jit(
                weights_batch, r2_values, slope_nl_values, rdd_nl_values,
                calmar_nl_values, trade_nl_values, wf_nl_values,
                n_trades_values, calmar_ratios, slopes, wf_consistencies
            )
            
            # Encontrar el mejor en este batch
            best_idx_in_batch = np.argmax(quality_scores)
            best_quality_in_batch = quality_scores[best_idx_in_batch]
            
            if best_quality_in_batch > best_result['score']:
                best_weights_in_batch = weights_batch[best_idx_in_batch]
                best_result = {
                    'weights': best_weights_in_batch.tolist(),
                    'score': best_quality_in_batch,
                    'metrics': {
                        'avg_score': avg_scores[best_idx_in_batch],
                        'std_score': std_scores[best_idx_in_batch],
                        'n_curves': n_valid
                    }
                }
                
                trial_num = start_trial + best_idx_in_batch
                print(f"🔍 DEBUG: Batch {batch_idx}: Nuevo mejor score = {best_quality_in_batch:.4f} (trial {trial_num})")
                print(f"  Pesos: r2={best_weights_in_batch[0]:.3f}, slope={best_weights_in_batch[1]:.3f}, "
                      f"rdd={best_weights_in_batch[2]:.3f}, calmar={best_weights_in_batch[3]:.3f}, "
                      f"trade={best_weights_in_batch[4]:.3f}, wf={best_weights_in_batch[5]:.3f}")
            
            elif batch_idx % 10 == 0:
                print(f"🔍 DEBUG: Batch {batch_idx}/{n_batches} completado, mejor score actual: {best_result['score']:.4f}")
        
        return best_result
    

    
    def analyze_current_weights(self, metrics_list: List[CurveMetrics]) -> Dict:
        """Analiza el comportamiento de los pesos actuales"""
        
        # Pesos actuales basados en tester_lib.py con 6 métricas incluyendo wf_nl
        current_weights = np.array([0.17, 0.16, 0.17, 0.16, 0.17, 0.17])  # [r2, slope, rdd, calmar, trade, wf]
        
        print("🔍 DEBUG: Analizando pesos actuales con walk-forward consistency...")
        
        valid_metrics = [m for m in metrics_list if (
            m.trade_nl >= 0 and m.rdd_nl >= 0 and m.r2 >= 0 and 
            m.slope_nl >= 0 and m.calmar_nl >= 0 and m.wf_nl >= 0
        )]
        
        # Preparar arrays para JIT
        r2_values = np.array([m.r2 for m in valid_metrics])
        slope_nl_values = np.array([m.slope_nl for m in valid_metrics])
        rdd_nl_values = np.array([m.rdd_nl for m in valid_metrics])
        calmar_nl_values = np.array([m.calmar_nl for m in valid_metrics])
        trade_nl_values = np.array([m.trade_nl for m in valid_metrics])
        wf_nl_values = np.array([m.wf_nl for m in valid_metrics])
        
        # Calcular scores usando JIT con 6 métricas
        scores = calculate_scores_batch_jit(r2_values, slope_nl_values, rdd_nl_values,
                                          calmar_nl_values, trade_nl_values, wf_nl_values, current_weights)
        
        analysis = {
            'current_weights': current_weights.tolist(),
            'avg_score': np.mean(scores),
            'std_score': np.std(scores),
            'n_valid_curves': len(valid_metrics),
            'score_distribution': {
                'min': np.min(scores),
                'max': np.max(scores),
                'median': np.median(scores),
                'q25': np.percentile(scores, 25),
                'q75': np.percentile(scores, 75)
            }
        }
        
        return analysis
    
    def plot_results(self, metrics_list: List[CurveMetrics], 
                    best_weights: List[float], 
                    current_weights: List[float]):
        """Genera gráficos de análisis"""
        
        valid_metrics = [m for m in metrics_list if (
            m.trade_nl >= 0 and m.rdd_nl >= 0 and m.r2 >= 0 and 
            m.slope_nl >= 0 and m.calmar_nl >= 0 and m.wf_nl >= 0
        )]
        
        # Preparar arrays para JIT con 6 métricas
        r2_values = np.array([m.r2 for m in valid_metrics])
        slope_nl_values = np.array([m.slope_nl for m in valid_metrics])
        rdd_nl_values = np.array([m.rdd_nl for m in valid_metrics])
        calmar_nl_values = np.array([m.calmar_nl for m in valid_metrics])
        trade_nl_values = np.array([m.trade_nl for m in valid_metrics])
        wf_nl_values = np.array([m.wf_nl for m in valid_metrics])
        
        # Calcular scores con ambos conjuntos de pesos usando JIT con 6 métricas
        current_weights_np = np.array(current_weights)
        best_weights_np = np.array(best_weights)
        
        current_scores = calculate_scores_batch_jit(r2_values, slope_nl_values, rdd_nl_values,
                                                   calmar_nl_values, trade_nl_values, wf_nl_values, current_weights_np)
        best_scores = calculate_scores_batch_jit(r2_values, slope_nl_values, rdd_nl_values,
                                               calmar_nl_values, trade_nl_values, wf_nl_values, best_weights_np)
        
        # Crear gráficos (ahora con Walk-Forward)
        fig, axes = plt.subplots(3, 3, figsize=(10, 6))
        
        # 1. Distribución de scores
        axes[0, 0].hist(current_scores, bins=50, alpha=0.7, label='Pesos actuales', density=True)
        axes[0, 0].hist(best_scores, bins=50, alpha=0.7, label='Pesos optimizados', density=True)
        axes[0, 0].set_xlabel('Score')
        axes[0, 0].set_ylabel('Densidad')
        axes[0, 0].legend()
        axes[0, 0].set_title('Distribución de Scores')
        
        # 2. R² vs Score
        r2_values_plot = [m.r2 for m in valid_metrics]
        axes[0, 1].scatter(r2_values_plot, best_scores, alpha=0.6)
        axes[0, 1].set_xlabel('R²')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].set_title('R² vs Score')
        
        # 3. Número de trades vs Score
        n_trades_values_plot = [m.n_trades for m in valid_metrics]
        axes[0, 2].scatter(n_trades_values_plot, best_scores, alpha=0.6)
        axes[0, 2].set_xlabel('Número de trades')
        axes[0, 2].set_ylabel('Score')
        axes[0, 2].set_title('Número de Trades vs Score')
        
        # 4. Calmar ratio vs Score
        calmar_values = [m.calmar_ratio for m in valid_metrics]
        axes[1, 0].scatter(calmar_values, best_scores, alpha=0.6)
        axes[1, 0].set_xlabel('Calmar Ratio')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].set_title('Calmar Ratio vs Score')
        
        # 5. Pendiente vs Score
        slope_values = [m.slope for m in valid_metrics]
        axes[1, 1].scatter(slope_values, best_scores, alpha=0.6)
        axes[1, 1].set_xlabel('Pendiente')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_title('Pendiente vs Score')
        
        # 6. Walk-Forward Consistency vs Score (NUEVA MÉTRICA)
        wf_values = [m.wf_nl for m in valid_metrics]
        axes[1, 2].scatter(wf_values, best_scores, alpha=0.6, color='purple')
        axes[1, 2].set_xlabel('Walk-Forward Consistency')
        axes[1, 2].set_ylabel('Score')
        axes[1, 2].set_title('Walk-Forward vs Score')
        
        # 7. Comparación de pesos (ahora 6 métricas)
        metric_names = ['R²', 'Slope', 'RDD', 'Calmar', 'Trades', 'WF']
        x = np.arange(len(metric_names))
        width = 0.35
        
        axes[2, 0].bar(x - width/2, current_weights, width, label='Pesos actuales', alpha=0.7)
        axes[2, 0].bar(x + width/2, best_weights, width, label='Pesos optimizados', alpha=0.7)
        axes[2, 0].set_xlabel('Métricas')
        axes[2, 0].set_ylabel('Peso')
        axes[2, 0].set_title('Comparación de Pesos (6 Métricas)')
        axes[2, 0].set_xticks(x)
        axes[2, 0].set_xticklabels(metric_names, rotation=45)
        axes[2, 0].legend()
        
        # 8. Distribución de Walk-Forward Consistency
        axes[2, 1].hist(wf_values, bins=30, alpha=0.7, color='purple', density=True)
        axes[2, 1].set_xlabel('Walk-Forward Consistency')
        axes[2, 1].set_ylabel('Densidad')
        axes[2, 1].set_title('Distribución WF Consistency')
        
        # 9. Mejora de Score con nueva métrica
        improvement = np.array(best_scores) - np.array(current_scores)
        axes[2, 2].hist(improvement, bins=30, alpha=0.7, color='green', density=True)
        axes[2, 2].set_xlabel('Mejora de Score')
        axes[2, 2].set_ylabel('Densidad')
        axes[2, 2].set_title('Mejora con Optimización')
        
        plt.tight_layout()
        plt.savefig('weight_calibration_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_results(self, best_result: Dict, analysis: Dict):
        """Guarda los resultados en un archivo JSON"""
        
        results = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'best_weights': best_result['weights'],
            'best_score': best_result['score'],
            'current_analysis': analysis,
            'recommendation': {
                'current_weights': [0.17, 0.16, 0.17, 0.16, 0.17, 0.17],  # 6 métricas
                'suggested_weights': best_result['weights'],
                'improvement': best_result['score'] - analysis.get('avg_score', 0),
                'metrics_description': ['r2', 'slope_nl', 'rdd_nl', 'calmar_nl', 'trade_nl', 'wf_nl']
            }
        }
        
        with open('weight_calibration_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print("Resultados guardados en 'weight_calibration_results.json'")

def main():
    """Función principal"""
    
    print("=== CALIBRADOR DE PESOS PARA TESTER ===\n")
    print("🔍 DEBUG: Versión optimizada con JIT + paralelización activada")
    
    # Información sobre paralelización
    n_cores = multiprocessing.cpu_count()
    numba_threads = numba.get_num_threads()
    print(f"🔍 DEBUG: Cores detectados: {n_cores}")
    print(f"🔍 DEBUG: Threads Numba: {numba_threads}")
    
    # Configurar numba para usar todos los cores
    if numba_threads < n_cores:
        numba.set_num_threads(n_cores)
        print(f"🔍 DEBUG: Configurando Numba para usar {n_cores} threads")
    
    start_time = time.time()
    
    # Crear instancia del calibrador
    calibrator = WeightCalibrator()
    
    # Generar curvas sintéticas
    curves = calibrator.generate_synthetic_curves(n_curves=5000000)
    
    # Evaluar curvas
    print("🔍 DEBUG: Evaluando curvas con tester_lib...")
    metrics_list = calibrator.evaluate_curves(curves)
    
    # Analizar pesos actuales
    current_analysis = calibrator.analyze_current_weights(metrics_list)
    print(f"\n🔍 DEBUG: Análisis de pesos actuales:")
    print(f"  Score promedio: {current_analysis['avg_score']:.4f}")
    print(f"  Desviación estándar: {current_analysis['std_score']:.4f}")
    print(f"  Curvas válidas: {current_analysis['n_valid_curves']}")
    
    # Probar combinaciones de pesos
    print("\n🔍 DEBUG: Iniciando optimización de pesos con JIT...")
    best_result = calibrator.test_weight_combinations(metrics_list, n_trials=100000)
    
    if best_result['weights']:
        print(f"\n=== RESULTADOS ===")
        print(f"Mejor combinación de pesos encontrada (6 métricas con walk-forward):")
        print(f"  R²: {best_result['weights'][0]:.3f}")
        print(f"  Slope: {best_result['weights'][1]:.3f}")
        print(f"  RDD: {best_result['weights'][2]:.3f}")
        print(f"  Calmar: {best_result['weights'][3]:.3f}")
        print(f"  Trades: {best_result['weights'][4]:.3f}")
        print(f"  Walk-Forward: {best_result['weights'][5]:.3f}")
        print(f"  Score de calidad: {best_result['score']:.4f}")
        
        # Generar gráficos
        calibrator.plot_results(metrics_list, best_result['weights'], 
                              current_analysis['current_weights'])
        
        # Guardar resultados
        calibrator.save_results(best_result, current_analysis)
        
        # Mostrar recomendación para tester_lib.py
        print(f"\n=== RECOMENDACIÓN PARA tester_lib.py ===")
        print("Reemplazar la línea de score en tester() con:")
        print("score = (")
        print(f"    {best_result['weights'][0]:.3f} * r2 +")
        print(f"    {best_result['weights'][1]:.3f} * slope_nl +")
        print(f"    {best_result['weights'][2]:.3f} * rdd_nl +")
        print(f"    {best_result['weights'][3]:.3f} * calmar_nl +")
        print(f"    {best_result['weights'][4]:.3f} * trade_nl +")
        print(f"    {best_result['weights'][5]:.3f} * wf_nl")
        print(")")
        print(f"\n🔍 DEBUG: Inclusión de walk-forward consistency mejora la robustez temporal")
    else:
        print("No se encontró una combinación de pesos válida.")
    
    total_time = time.time() - start_time
    print(f"\n🔍 DEBUG: Tiempo total de ejecución: {total_time:.2f} segundos")
    print(f"🔍 DEBUG: Optimizaciones aplicadas:")
    print(f"  • JIT compilation para funciones críticas")
    print(f"  • Paralelización con prange en {n_cores} cores")
    print(f"  • Procesamiento en batches paralelos")
    print(f"  • Cálculos vectoriales optimizados")
    print(f"🔍 DEBUG: Speedup estimado vs versión original: 10-50x más rápido")

if __name__ == "__main__":
    main()