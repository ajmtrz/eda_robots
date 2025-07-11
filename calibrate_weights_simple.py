#!/usr/bin/env python3
"""
Script simplificado para calibrar los pesos de las métricas en tester_lib.py

Genera curvas de equity sintéticas y prueba diferentes combinaciones de pesos
para promover curvas con muchas trades, rectilíneas, alta pendiente y alto retorno/drawdown.
"""

import math
import random
import time
import json
from typing import List, Tuple, Dict

# Importar las funciones necesarias de tester_lib
import sys
sys.path.append('studies/modules')

try:
    from tester_lib import evaluate_report
    print("✅ Módulo tester_lib importado correctamente")
except ImportError as e:
    print(f"❌ Error importando tester_lib: {e}")
    print("Asegúrate de que el archivo tester_lib.py esté en studies/modules/")
    sys.exit(1)

class SimpleWeightCalibrator:
    """Clase simplificada para calibrar los pesos de las métricas"""
    
    def __init__(self):
        self.curves = []
        self.best_weights = None
        self.best_score = -1.0
        
    def generate_synthetic_curves(self, n_curves: int = 1000) -> List[List[float]]:
        """Genera curvas de equity sintéticas con diferentes características"""
        curves = []
        
        print(f"Generando {n_curves} curvas sintéticas...")
        
        for i in range(n_curves):
            if i % 100 == 0:
                print(f"Progreso: {i}/{n_curves}")
            
            # Generar curva con características aleatorias
            curve = self._generate_single_curve()
            curves.append(curve)
        
        print("Generación completada.")
        return curves
    
    def _generate_single_curve(self) -> List[float]:
        """Genera una curva de equity individual con características controladas"""
        # Parámetros aleatorios para la curva
        n_points = random.randint(200, 800)
        base_trend = random.uniform(-0.05, 0.2)  # Tendencia base
        noise_level = random.uniform(0.001, 0.03)  # Ruido
        linearity = random.uniform(0.4, 1.0)     # Linealidad (0-1)
        
        # Generar curva base
        curve = [0.0]  # Empezar en 0
        
        for i in range(1, n_points):
            # Componente lineal
            linear_component = base_trend * i
            
            # Componente no lineal (para reducir R²)
            if linearity < 0.8:
                non_linear = math.sin(i * 0.1) * (1 - linearity) * n_points * 0.05
            else:
                non_linear = 0
            
            # Ruido aleatorio
            noise = random.gauss(0, noise_level * n_points)
            
            # Combinar componentes
            value = linear_component + non_linear + noise
            curve.append(value)
        
        # Asegurar que la curva sea creciente en general (pendiente positiva)
        if curve[-1] < curve[0]:
            curve = [-v for v in curve]
        
        # Normalizar para que empiece en 0
        start_value = curve[0]
        curve = [v - start_value for v in curve]
        
        # Aplicar control de drawdown
        curve = self._apply_drawdown_control(curve)
        
        return curve
    
    def _apply_drawdown_control(self, curve: List[float]) -> List[float]:
        """Aplica control de drawdown a la curva"""
        max_dd_target = random.uniform(0.05, 0.25)
        
        # Calcular drawdown actual
        running_max = [curve[0]]
        for i in range(1, len(curve)):
            running_max.append(max(running_max[-1], curve[i]))
        
        current_dd = []
        for i in range(len(curve)):
            if running_max[i] > 0:
                dd = (running_max[i] - curve[i]) / running_max[i]
            else:
                dd = 0
            current_dd.append(dd)
        
        # Si el drawdown es muy alto, ajustar
        max_current_dd = max(current_dd)
        if max_current_dd > max_dd_target:
            # Reducir la volatilidad en las caídas
            dd_factor = max_dd_target / max_current_dd
            adjusted_curve = curve.copy()
            
            for i in range(1, len(curve)):
                if current_dd[i] > max_dd_target * 0.5:
                    # Reducir la caída
                    adjustment = (current_dd[i] - max_dd_target * 0.5) * dd_factor
                    adjusted_curve[i] += adjustment
            
            return adjusted_curve
        
        return curve
    
    def evaluate_curves(self, curves: List[List[float]]) -> List[Dict]:
        """Evalúa todas las curvas y calcula sus métricas"""
        metrics_list = []
        
        print("Evaluando curvas...")
        
        for i, curve in enumerate(curves):
            if i % 100 == 0:
                print(f"Evaluación progreso: {i}/{len(curves)}")
            
            try:
                # Convertir a numpy array para evaluate_report
                import numpy as np
                curve_array = np.array(curve, dtype=np.float64)
                
                # Evaluar con los parámetros por defecto
                trade_nl, rdd_nl, r2, slope_nl, calmar_nl = evaluate_report(
                    curve_array, 
                    periods_per_year=6240.0,  # H1
                    min_trades=200,
                    rdd_floor=1.0,
                    calmar_floor=1.0
                )
                
                # Calcular métricas adicionales para análisis
                n_trades = len(curve) - 1
                total_return = curve[-1] - curve[0]
                
                # Calcular drawdown máximo
                running_max = [curve[0]]
                for j in range(1, len(curve)):
                    running_max.append(max(running_max[-1], curve[j]))
                
                max_drawdown = 0
                for j in range(len(curve)):
                    dd = running_max[j] - curve[j]
                    max_drawdown = max(max_drawdown, dd)
                
                # Calcular Calmar ratio
                if max_drawdown > 0:
                    calmar_ratio = total_return / max_drawdown
                else:
                    calmar_ratio = 0
                
                # Calcular pendiente
                x = list(range(len(curve)))
                slope = self._calculate_slope(x, curve)
                
                metrics = {
                    'trade_nl': trade_nl,
                    'rdd_nl': rdd_nl,
                    'r2': r2,
                    'slope_nl': slope_nl,
                    'calmar_nl': calmar_nl,
                    'n_trades': n_trades,
                    'total_return': total_return,
                    'max_drawdown': max_drawdown,
                    'calmar_ratio': calmar_ratio,
                    'slope': slope
                }
                
                metrics_list.append(metrics)
                
            except Exception as e:
                print(f"Error evaluando curva {i}: {e}")
                continue
        
        print("Evaluación completada.")
        return metrics_list
    
    def _calculate_slope(self, x: List[float], y: List[float]) -> float:
        """Calcula la pendiente usando regresión lineal simple"""
        n = len(x)
        if n < 2:
            return 0.0
        
        # Medias
        x_mean = sum(x) / n
        y_mean = sum(y) / n
        
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
            return 0.0
        
        # Pendiente
        slope = numerator / denominator
        return slope
    
    def test_weight_combinations(self, metrics_list: List[Dict], 
                                n_trials: int = 500) -> Dict:
        """Prueba diferentes combinaciones de pesos"""
        
        print(f"Probando {n_trials} combinaciones de pesos...")
        
        best_result = {
            'weights': None,
            'score': -1.0,
            'metrics': None
        }
        
        # Filtrar curvas válidas
        valid_metrics = [m for m in metrics_list if all(
            m['trade_nl'] >= 0 and m['rdd_nl'] >= 0 and m['r2'] >= 0 and 
            m['slope_nl'] >= 0 and m['calmar_nl'] >= 0
        )]
        
        print(f"Curvas válidas para testing: {len(valid_metrics)}")
        
        if len(valid_metrics) < 50:
            print("⚠️ Muy pocas curvas válidas.")
            return best_result
        
        for trial in range(n_trials):
            # Generar pesos aleatorios que sumen 1.0
            weights = self._generate_random_weights()
            
            # Calcular score promedio para todas las curvas
            scores = []
            for metrics in valid_metrics:
                score = (
                    weights[0] * metrics['r2'] +
                    weights[1] * metrics['slope_nl'] +
                    weights[2] * metrics['rdd_nl'] +
                    weights[3] * metrics['calmar_nl'] +
                    weights[4] * metrics['trade_nl']
                )
                scores.append(score)
            
            avg_score = sum(scores) / len(scores)
            
            # Evaluar calidad de los pesos
            quality_score = self._evaluate_weight_quality(valid_metrics, weights)
            
            if quality_score > best_result['score']:
                best_result = {
                    'weights': weights,
                    'score': quality_score,
                    'metrics': {
                        'avg_score': avg_score,
                        'std_score': self._calculate_std(scores),
                        'n_curves': len(valid_metrics)
                    }
                }
                
                if trial % 50 == 0:
                    print(f"Trial {trial}: Nuevo mejor score = {quality_score:.4f}")
                    print(f"  Pesos: r2={weights[0]:.3f}, slope={weights[1]:.3f}, "
                          f"rdd={weights[2]:.3f}, calmar={weights[3]:.3f}, trade={weights[4]:.3f}")
        
        return best_result
    
    def _generate_random_weights(self) -> List[float]:
        """Genera pesos aleatorios que sumen 1.0"""
        weights = [random.random() for _ in range(5)]
        total = sum(weights)
        return [w / total for w in weights]
    
    def _calculate_std(self, values: List[float]) -> float:
        """Calcula la desviación estándar"""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return math.sqrt(variance)
    
    def _evaluate_weight_quality(self, metrics_list: List[Dict], 
                                weights: List[float]) -> float:
        """Evalúa la calidad de una combinación de pesos"""
        
        # Calcular scores con estos pesos
        scores = []
        for metrics in metrics_list:
            score = (
                weights[0] * metrics['r2'] +
                weights[1] * metrics['slope_nl'] +
                weights[2] * metrics['rdd_nl'] +
                weights[3] * metrics['calmar_nl'] +
                weights[4] * metrics['trade_nl']
            )
            scores.append(score)
        
        # Métricas de calidad
        avg_score = sum(scores) / len(scores)
        std_score = self._calculate_std(scores)
        
        # Contar curvas con características deseables
        high_trades = sum(1 for m in metrics_list if m['n_trades'] > 400)
        high_r2 = sum(1 for m in metrics_list if m['r2'] > 0.7)
        high_slope = sum(1 for m in metrics_list if m['slope'] > 0.005)
        high_calmar = sum(1 for m in metrics_list if m['calmar_ratio'] > 1.5)
        
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
    
    def analyze_current_weights(self, metrics_list: List[Dict]) -> Dict:
        """Analiza el comportamiento de los pesos actuales"""
        
        current_weights = [0.5, 0.15, 0.1, 0.15, 0.1]  # Pesos actuales
        
        print("Analizando pesos actuales...")
        
        valid_metrics = [m for m in metrics_list if all(
            m['trade_nl'] >= 0 and m['rdd_nl'] >= 0 and m['r2'] >= 0 and 
            m['slope_nl'] >= 0 and m['calmar_nl'] >= 0
        )]
        
        scores = []
        for metrics in valid_metrics:
            score = (
                current_weights[0] * metrics['r2'] +
                current_weights[1] * metrics['slope_nl'] +
                current_weights[2] * metrics['rdd_nl'] +
                current_weights[3] * metrics['calmar_nl'] +
                current_weights[4] * metrics['trade_nl']
            )
            scores.append(score)
        
        analysis = {
            'current_weights': current_weights,
            'avg_score': sum(scores) / len(scores) if scores else 0,
            'std_score': self._calculate_std(scores),
            'n_valid_curves': len(valid_metrics),
            'score_distribution': {
                'min': min(scores) if scores else 0,
                'max': max(scores) if scores else 0,
                'median': sorted(scores)[len(scores)//2] if scores else 0
            }
        }
        
        return analysis
    
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
    
    print("=== CALIBRADOR DE PESOS PARA TESTER (VERSIÓN SIMPLIFICADA) ===\n")
    
    # Crear instancia del calibrador
    calibrator = SimpleWeightCalibrator()
    
    # Generar curvas sintéticas
    curves = calibrator.generate_synthetic_curves(n_curves=1000)
    
    # Evaluar curvas
    metrics_list = calibrator.evaluate_curves(curves)
    
    # Analizar pesos actuales
    current_analysis = calibrator.analyze_current_weights(metrics_list)
    print(f"\nAnálisis de pesos actuales:")
    print(f"  Score promedio: {current_analysis['avg_score']:.4f}")
    print(f"  Desviación estándar: {current_analysis['std_score']:.4f}")
    print(f"  Curvas válidas: {current_analysis['n_valid_curves']}")
    
    # Probar combinaciones de pesos
    best_result = calibrator.test_weight_combinations(metrics_list, n_trials=500)
    
    if best_result['weights']:
        print(f"\n=== RESULTADOS ===")
        print(f"Mejor combinación de pesos encontrada:")
        print(f"  R²: {best_result['weights'][0]:.3f}")
        print(f"  Slope: {best_result['weights'][1]:.3f}")
        print(f"  RDD: {best_result['weights'][2]:.3f}")
        print(f"  Calmar: {best_result['weights'][3]:.3f}")
        print(f"  Trades: {best_result['weights'][4]:.3f}")
        print(f"  Score de calidad: {best_result['score']:.4f}")
        
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
        
        # Mostrar estadísticas adicionales
        print(f"\n=== ESTADÍSTICAS ADICIONALES ===")
        valid_metrics = [m for m in metrics_list if all(
            m['trade_nl'] >= 0 and m['rdd_nl'] >= 0 and m['r2'] >= 0 and 
            m['slope_nl'] >= 0 and m['calmar_nl'] >= 0
        )]
        
        if valid_metrics:
            high_trades = sum(1 for m in valid_metrics if m['n_trades'] > 400)
            high_r2 = sum(1 for m in valid_metrics if m['r2'] > 0.7)
            high_slope = sum(1 for m in valid_metrics if m['slope'] > 0.005)
            high_calmar = sum(1 for m in valid_metrics if m['calmar_ratio'] > 1.5)
            
            print(f"Curvas con >400 trades: {high_trades}/{len(valid_metrics)} ({high_trades/len(valid_metrics)*100:.1f}%)")
            print(f"Curvas con R² > 0.7: {high_r2}/{len(valid_metrics)} ({high_r2/len(valid_metrics)*100:.1f}%)")
            print(f"Curvas con pendiente > 0.005: {high_slope}/{len(valid_metrics)} ({high_slope/len(valid_metrics)*100:.1f}%)")
            print(f"Curvas con Calmar > 1.5: {high_calmar}/{len(valid_metrics)} ({high_calmar/len(valid_metrics)*100:.1f}%)")
    else:
        print("No se encontró una combinación de pesos válida.")

if __name__ == "__main__":
    main()