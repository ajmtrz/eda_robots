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

# Importar las funciones necesarias de tester_lib
import sys
sys.path.append('studies/modules')
from tester_lib import evaluate_report, backtest_equivalent

@dataclass
class CurveMetrics:
    """Métricas de una curva de equity"""
    trade_nl: float
    rdd_nl: float
    r2: float
    slope_nl: float
    calmar_nl: float
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
        
        print(f"Generando {n_curves} curvas sintéticas...")
        
        for i in range(n_curves):
            if i % 1000 == 0:
                print(f"Progreso: {i}/{n_curves}")
            
            # Generar curva con características aleatorias
            curve = self._generate_single_curve()
            curves.append(curve)
        
        print("Generación completada.")
        return curves
    
    def _generate_single_curve(self) -> np.ndarray:
        """Genera una curva de equity individual con características controladas"""
        # Parámetros aleatorios para la curva
        n_points = random.randint(200, 1000)
        base_trend = random.uniform(-0.1, 0.3)  # Tendencia base
        volatility = random.uniform(0.01, 0.1)   # Volatilidad
        noise_level = random.uniform(0.001, 0.05)  # Ruido
        linearity = random.uniform(0.3, 1.0)     # Linealidad (0-1)
        
        # Generar curva base
        x = np.arange(n_points)
        
        # Componente lineal
        linear_component = base_trend * x
        
        # Componente no lineal (para reducir R²)
        if linearity < 0.8:
            non_linear = np.sin(x * 0.1) * (1 - linearity) * n_points * 0.1
        else:
            non_linear = 0
        
        # Ruido aleatorio
        noise = np.random.normal(0, noise_level * n_points, n_points)
        
        # Combinar componentes
        raw_curve = linear_component + non_linear + noise
        
        # Asegurar que la curva sea creciente en general (pendiente positiva)
        if np.mean(np.diff(raw_curve)) < 0:
            raw_curve = -raw_curve
        
        # Normalizar para que empiece en 0
        raw_curve = raw_curve - raw_curve[0]
        
        # Aplicar transformación para controlar drawdown
        max_dd_target = random.uniform(0.05, 0.3)
        curve = self._apply_drawdown_control(raw_curve, max_dd_target)
        
        return curve
    
    def _apply_drawdown_control(self, curve: np.ndarray, max_dd_target: float) -> np.ndarray:
        """Aplica control de drawdown a la curva"""
        # Calcular drawdown actual
        running_max = np.maximum.accumulate(curve)
        current_dd = (running_max - curve) / (running_max + 1e-8)
        
        # Si el drawdown es muy alto, ajustar
        if np.max(current_dd) > max_dd_target:
            # Reducir la volatilidad en las caídas
            dd_factor = max_dd_target / np.max(current_dd)
            adjusted_curve = curve.copy()
            
            for i in range(1, len(curve)):
                if current_dd[i] > max_dd_target * 0.5:
                    # Reducir la caída
                    adjustment = (current_dd[i] - max_dd_target * 0.5) * dd_factor
                    adjusted_curve[i] += adjustment
            
            return adjusted_curve
        
        return curve
    
    def evaluate_curves(self, curves: List[np.ndarray]) -> List[CurveMetrics]:
        """Evalúa todas las curvas y calcula sus métricas"""
        metrics_list = []
        
        print("Evaluando curvas...")
        
        for i, curve in enumerate(curves):
            if i % 1000 == 0:
                print(f"Evaluación progreso: {i}/{len(curves)}")
            
            try:
                # Evaluar con los parámetros por defecto
                trade_nl, rdd_nl, r2, slope_nl, calmar_nl = evaluate_report(
                    curve, 
                    periods_per_year=6240.0,  # H1
                    min_trades=200,
                    rdd_floor=1.0,
                    calmar_floor=1.0
                )
                
                # Calcular métricas adicionales para análisis
                n_trades = len(np.diff(curve))
                total_return = curve[-1] - curve[0]
                
                # Calcular drawdown máximo
                running_max = np.maximum.accumulate(curve)
                max_drawdown = np.max(running_max - curve)
                
                # Calcular Calmar ratio
                if max_drawdown > 0:
                    calmar_ratio = total_return / max_drawdown
                else:
                    calmar_ratio = 0
                
                # Calcular pendiente
                x = np.arange(len(curve))
                slope = np.polyfit(x, curve, 1)[0]
                
                metrics = CurveMetrics(
                    trade_nl=trade_nl,
                    rdd_nl=rdd_nl,
                    r2=r2,
                    slope_nl=slope_nl,
                    calmar_nl=calmar_nl,
                    score=0.0,  # Se calculará después
                    n_trades=n_trades,
                    total_return=total_return,
                    max_drawdown=max_drawdown,
                    calmar_ratio=calmar_ratio,
                    slope=slope
                )
                
                metrics_list.append(metrics)
                
            except Exception as e:
                print(f"Error evaluando curva {i}: {e}")
                continue
        
        print("Evaluación completada.")
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
        
        # Filtrar curvas válidas
        valid_metrics = [m for m in metrics_list if all(
            m.trade_nl >= 0 and m.rdd_nl >= 0 and m.r2 >= 0 and 
            m.slope_nl >= 0 and m.calmar_nl >= 0
        )]
        
        print(f"Curvas válidas para testing: {len(valid_metrics)}")
        
        if len(valid_metrics) < 100:
            print("⚠️ Muy pocas curvas válidas. Generando más curvas...")
            return best_result
        
        for trial in range(n_trials):
            # Generar pesos aleatorios que sumen 1.0
            weights = np.random.dirichlet(np.ones(5))
            
            # Calcular score promedio para todas las curvas
            scores = []
            for metrics in valid_metrics:
                score = (
                    weights[0] * metrics.r2 +
                    weights[1] * metrics.slope_nl +
                    weights[2] * metrics.rdd_nl +
                    weights[3] * metrics.calmar_nl +
                    weights[4] * metrics.trade_nl
                )
                scores.append(score)
            
            avg_score = np.mean(scores)
            
            # Evaluar calidad de los pesos
            quality_score = self._evaluate_weight_quality(valid_metrics, weights)
            
            if quality_score > best_result['score']:
                best_result = {
                    'weights': weights.tolist(),
                    'score': quality_score,
                    'metrics': {
                        'avg_score': avg_score,
                        'std_score': np.std(scores),
                        'n_curves': len(valid_metrics)
                    }
                }
                
                if trial % 100 == 0:
                    print(f"Trial {trial}: Nuevo mejor score = {quality_score:.4f}")
                    print(f"  Pesos: r2={weights[0]:.3f}, slope={weights[1]:.3f}, "
                          f"rdd={weights[2]:.3f}, calmar={weights[3]:.3f}, trade={weights[4]:.3f}")
        
        return best_result
    
    def _evaluate_weight_quality(self, metrics_list: List[CurveMetrics], 
                                weights: np.ndarray) -> float:
        """Evalúa la calidad de una combinación de pesos"""
        
        # Calcular scores con estos pesos
        scores = []
        for metrics in metrics_list:
            score = (
                weights[0] * metrics.r2 +
                weights[1] * metrics.slope_nl +
                weights[2] * metrics.rdd_nl +
                weights[3] * metrics.calmar_nl +
                weights[4] * metrics.trade_nl
            )
            scores.append(score)
        
        # Métricas de calidad
        avg_score = np.mean(scores)
        std_score = np.std(scores)
        
        # Contar curvas con características deseables
        high_trades = sum(1 for m in metrics_list if m.n_trades > 500)
        high_r2 = sum(1 for m in metrics_list if m.r2 > 0.8)
        high_slope = sum(1 for m in metrics_list if m.slope > 0.01)
        high_calmar = sum(1 for m in metrics_list if m.calmar_ratio > 2.0)
        
        # Score de calidad basado en múltiples criterios
        quality_score = (
            0.4 * avg_score +
            0.2 * (high_trades / len(metrics_list)) +
            0.2 * (high_r2 / len(metrics_list)) +
            0.1 * (high_slope / len(metrics_list)) +
            0.1 * (high_calmar / len(metrics_list)) -
            0.1 * std_score  # Penalizar alta varianza
        )
        
        return quality_score
    
    def analyze_current_weights(self, metrics_list: List[CurveMetrics]) -> Dict:
        """Analiza el comportamiento de los pesos actuales"""
        
        current_weights = [0.5, 0.15, 0.1, 0.15, 0.1]  # Pesos actuales
        
        print("Analizando pesos actuales...")
        
        valid_metrics = [m for m in metrics_list if all(
            m.trade_nl >= 0 and m.rdd_nl >= 0 and m.r2 >= 0 and 
            m.slope_nl >= 0 and m.calmar_nl >= 0
        )]
        
        scores = []
        for metrics in valid_metrics:
            score = (
                current_weights[0] * metrics.r2 +
                current_weights[1] * metrics.slope_nl +
                current_weights[2] * metrics.rdd_nl +
                current_weights[3] * metrics.calmar_nl +
                current_weights[4] * metrics.trade_nl
            )
            scores.append(score)
        
        analysis = {
            'current_weights': current_weights,
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
        
        valid_metrics = [m for m in metrics_list if all(
            m.trade_nl >= 0 and m.rdd_nl >= 0 and m.r2 >= 0 and 
            m.slope_nl >= 0 and m.calmar_nl >= 0
        )]
        
        # Calcular scores con ambos conjuntos de pesos
        current_scores = []
        best_scores = []
        
        for metrics in valid_metrics:
            current_score = (
                current_weights[0] * metrics.r2 +
                current_weights[1] * metrics.slope_nl +
                current_weights[2] * metrics.rdd_nl +
                current_weights[3] * metrics.calmar_nl +
                current_weights[4] * metrics.trade_nl
            )
            current_scores.append(current_score)
            
            best_score = (
                best_weights[0] * metrics.r2 +
                best_weights[1] * metrics.slope_nl +
                best_weights[2] * metrics.rdd_nl +
                best_weights[3] * metrics.calmar_nl +
                best_weights[4] * metrics.trade_nl
            )
            best_scores.append(best_score)
        
        # Crear gráficos
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. Distribución de scores
        axes[0, 0].hist(current_scores, bins=50, alpha=0.7, label='Pesos actuales', density=True)
        axes[0, 0].hist(best_scores, bins=50, alpha=0.7, label='Pesos optimizados', density=True)
        axes[0, 0].set_xlabel('Score')
        axes[0, 0].set_ylabel('Densidad')
        axes[0, 0].legend()
        axes[0, 0].set_title('Distribución de Scores')
        
        # 2. R² vs Score
        r2_values = [m.r2 for m in valid_metrics]
        axes[0, 1].scatter(r2_values, best_scores, alpha=0.6)
        axes[0, 1].set_xlabel('R²')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].set_title('R² vs Score')
        
        # 3. Número de trades vs Score
        n_trades_values = [m.n_trades for m in valid_metrics]
        axes[0, 2].scatter(n_trades_values, best_scores, alpha=0.6)
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
        
        # 6. Comparación de pesos
        metric_names = ['R²', 'Slope', 'RDD', 'Calmar', 'Trades']
        x = np.arange(len(metric_names))
        width = 0.35
        
        axes[1, 2].bar(x - width/2, current_weights, width, label='Pesos actuales', alpha=0.7)
        axes[1, 2].bar(x + width/2, best_weights, width, label='Pesos optimizados', alpha=0.7)
        axes[1, 2].set_xlabel('Métricas')
        axes[1, 2].set_ylabel('Peso')
        axes[1, 2].set_title('Comparación de Pesos')
        axes[1, 2].set_xticks(x)
        axes[1, 2].set_xticklabels(metric_names)
        axes[1, 2].legend()
        
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
                'current_weights': [0.5, 0.15, 0.1, 0.15, 0.1],
                'suggested_weights': best_result['weights'],
                'improvement': best_result['score'] - analysis.get('avg_score', 0)
            }
        }
        
        with open('weight_calibration_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print("Resultados guardados en 'weight_calibration_results.json'")

def main():
    """Función principal"""
    
    print("=== CALIBRADOR DE PESOS PARA TESTER ===\n")
    
    # Crear instancia del calibrador
    calibrator = WeightCalibrator()
    
    # Generar curvas sintéticas
    curves = calibrator.generate_synthetic_curves(n_curves=5000)
    
    # Evaluar curvas
    metrics_list = calibrator.evaluate_curves(curves)
    
    # Analizar pesos actuales
    current_analysis = calibrator.analyze_current_weights(metrics_list)
    print(f"\nAnálisis de pesos actuales:")
    print(f"  Score promedio: {current_analysis['avg_score']:.4f}")
    print(f"  Desviación estándar: {current_analysis['std_score']:.4f}")
    print(f"  Curvas válidas: {current_analysis['n_valid_curves']}")
    
    # Probar combinaciones de pesos
    best_result = calibrator.test_weight_combinations(metrics_list, n_trials=2000)
    
    if best_result['weights']:
        print(f"\n=== RESULTADOS ===")
        print(f"Mejor combinación de pesos encontrada:")
        print(f"  R²: {best_result['weights'][0]:.3f}")
        print(f"  Slope: {best_result['weights'][1]:.3f}")
        print(f"  RDD: {best_result['weights'][2]:.3f}")
        print(f"  Calmar: {best_result['weights'][3]:.3f}")
        print(f"  Trades: {best_result['weights'][4]:.3f}")
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
        print(f"    {best_result['weights'][4]:.3f} * trade_nl")
        print(")")
    else:
        print("No se encontró una combinación de pesos válida.")

if __name__ == "__main__":
    main()