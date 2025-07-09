#!/usr/bin/env python3
"""
SISTEMA DE TESTING EXHAUSTIVO PARA evaluate_report()

Este módulo implementa los tests rigurosos diseñados en el análisis completo
de la función evaluate_report del módulo tester_lib.py.

TESTS IMPLEMENTADOS:
- Suite A: Validación de curvas lineales perfectas
- Suite B: Casos patológicos y edge cases  
- Suite C: Robustez de métricas de trades
- Suite D: Consistencia matemática individual
- Suite E: Benchmarking masivo con miles de curvas controladas
- Corrección y validación de bugs identificados
"""

import sys
import os
import numpy as np
import pandas as pd
import random
import time
from typing import List, Dict, Tuple, Any
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Agregar el path del módulo bajo test
sys.path.append('/workspace/studies/modules')

try:
    from tester_lib import (
        evaluate_report, 
        _signed_r2, 
        _perfect_linearity_score,
        _linearity_bonus,
        _consistency_score,
        _slope_reward,
        _monotonic_growth_score,
        _smoothness_score,
        _advanced_drawdown_penalty,
        _trade_activity_score,
        _trade_consistency_score,
        metrics_tuple_to_dict
    )
    print("✅ Módulo tester_lib importado exitosamente")
except ImportError as e:
    print(f"❌ Error importando tester_lib: {e}")
    sys.exit(1)

# ============================================================================
# UTILITIES PARA GENERACIÓN DE DATOS CONTROLADOS
# ============================================================================

def generate_ideal_trade_stats(eq_length: int, n_trades: int = None, win_rate: float = 0.8) -> np.ndarray:
    """Genera trade_stats ideales para testing"""
    if n_trades is None:
        n_trades = max(10, eq_length // 20)  # 5% de actividad por defecto
    
    positive_trades = int(n_trades * win_rate)
    negative_trades = n_trades - positive_trades
    zero_trades = 0
    avg_positive = 0.05  # 5% ganancia promedio por trade positivo
    avg_negative = -0.02  # 2% pérdida promedio por trade negativo
    
    return np.array([
        float(n_trades),       # total_trades
        float(positive_trades), # positive_trades  
        float(negative_trades), # negative_trades
        float(zero_trades),    # zero_trades
        float(win_rate),       # win_rate
        avg_positive,          # avg_positive
        avg_negative           # avg_negative
    ], dtype=np.float64)

def generate_realistic_trade_stats(n_trades: int, win_rate: float, eq_length: int) -> np.ndarray:
    """Genera trade_stats realistas con variación controlada"""
    if n_trades == 0:
        return np.zeros(7, dtype=np.float64)
    
    positive_trades = int(n_trades * win_rate)
    negative_trades = n_trades - positive_trades
    zero_trades = 0
    
    # Valores realistas con algo de ruido
    avg_positive = random.uniform(0.02, 0.08)
    avg_negative = random.uniform(-0.05, -0.01)
    
    return np.array([
        float(n_trades), float(positive_trades), float(negative_trades),
        float(zero_trades), float(win_rate), avg_positive, avg_negative
    ], dtype=np.float64)

def generate_perfect_linear_curve(length: int, slope: float, start_value: float = 1.0) -> np.ndarray:
    """Genera curva perfectamente lineal"""
    return np.linspace(start_value, start_value + slope * length, length)

def generate_noisy_linear_curve(length: int, slope: float, noise_level: float, 
                               start_value: float = 1.0, seed: int = None) -> np.ndarray:
    """Genera curva lineal con ruido controlado"""
    if seed is not None:
        np.random.seed(seed)
    
    base_curve = generate_perfect_linear_curve(length, slope, start_value)
    noise = np.random.normal(0, noise_level, length)
    return base_curve + noise

# ============================================================================
# TEST SUITE A: VALIDACIÓN DE CURVAS LINEALES PERFECTAS
# ============================================================================

class TestPerfectLinearCurves:
    """Suite A: Valida que curvas perfectamente lineales obtengan scores máximos"""
    
    def __init__(self):
        self.results = []
        self.failed_tests = []
    
    def test_perfect_linear_curves(self):
        """Test principal: curvas lineales perfectas deben obtener scores altos"""
        print("\n🧪 TEST SUITE A: Curvas Lineales Perfectas")
        print("=" * 60)
        
        test_cases = [
            # (slope, length, expected_score_range, description)
            (0.1, 100, (0.70, 1.0), "Pendiente muy suave"),
            (0.5, 200, (0.80, 1.0), "Pendiente moderada ideal"),
            (1.0, 500, (0.85, 1.0), "Pendiente fuerte ideal"),
            (1.5, 1000, (0.80, 1.0), "Pendiente alta"),
            (2.0, 1000, (0.75, 0.95), "Pendiente muy fuerte"),
            (0.05, 2000, (0.65, 0.90), "Pendiente mínima, serie larga"),
        ]
        
        for i, (slope, length, expected_range, description) in enumerate(test_cases):
            eq = generate_perfect_linear_curve(length, slope)
            trade_stats = generate_ideal_trade_stats(length)
            
            result = evaluate_report(eq, trade_stats)
            score = result[0]
            
            # Validar rango esperado
            passed = expected_range[0] <= score <= expected_range[1]
            status = "✅ PASS" if passed else "❌ FAIL"
            
            print(f"  Test {i+1}: {description}")
            print(f"    Slope: {slope:.3f}, Length: {length}, Score: {score:.6f}")
            print(f"    Expected: [{expected_range[0]:.3f}, {expected_range[1]:.3f}] - {status}")
            
            self.results.append({
                'test_name': f"perfect_linear_{i+1}",
                'slope': slope,
                'length': length,
                'score': score,
                'expected_min': expected_range[0],
                'expected_max': expected_range[1],
                'passed': passed,
                'description': description
            })
            
            if not passed:
                self.failed_tests.append({
                    'test': f"Perfect Linear {i+1}",
                    'details': f"Score {score:.6f} outside [{expected_range[0]}, {expected_range[1]}]",
                    'slope': slope,
                    'length': length
                })
        
        # Estadísticas
        passed_count = sum(1 for r in self.results if r['passed'])
        total_count = len(self.results)
        pass_rate = passed_count / total_count * 100
        
        print(f"\n📊 Resultados Suite A: {passed_count}/{total_count} tests pasaron ({pass_rate:.1f}%)")
        
        return pass_rate > 80.0  # 80% de tests deben pasar

# ============================================================================
# TEST SUITE B: CASOS PATOLÓGICOS Y EDGE CASES
# ============================================================================

class TestPathologicalCases:
    """Suite B: Valida manejo correcto de casos extremos"""
    
    def __init__(self):
        self.results = []
        self.failed_tests = []
    
    def test_pathological_cases(self):
        """Test de casos extremos y edge cases"""
        print("\n🧪 TEST SUITE B: Casos Patológicos")
        print("=" * 60)
        
        # Test 1: Serie muy corta
        print("  Test 1: Serie muy corta (< 50 elementos)")
        eq_short = np.array([1.0, 1.1, 1.2])
        trade_stats_empty = np.zeros(7)
        result = evaluate_report(eq_short, trade_stats_empty)
        passed_1 = result[0] == -1.0
        print(f"    Score: {result[0]:.6f} - {'✅ PASS' if passed_1 else '❌ FAIL'}")
        
        # Test 2: Valores infinitos/NaN
        print("  Test 2: Valores infinitos en equity curve")
        eq_invalid = np.array([1.0, np.inf, 1.2] + [1.3] * 50)
        result = evaluate_report(eq_invalid, trade_stats_empty)
        passed_2 = result[0] == -1.0
        print(f"    Score: {result[0]:.6f} - {'✅ PASS' if passed_2 else '❌ FAIL'}")
        
        # Test 3: Equity curve negativa
        print("  Test 3: Equity curve con valores negativos")
        eq_negative = np.linspace(-10.0, -5.0, 100)
        result = evaluate_report(eq_negative, trade_stats_empty)
        passed_3 = result[0] >= 0.0
        print(f"    Score: {result[0]:.6f} - {'✅ PASS' if passed_3 else '❌ FAIL'}")
        
        # Test 4: Trade stats malformadas
        print("  Test 4: Trade stats con estructura incorrecta")
        eq_normal = generate_perfect_linear_curve(100, 0.5)
        trade_stats_wrong = np.array([1, 2, 3])  # Solo 3 elementos en lugar de 7
        try:
            result = evaluate_report(eq_normal, trade_stats_wrong)
            passed_4 = result[0] == -1.0
        except:
            passed_4 = True  # Error esperado
        print(f"    Manejo de error - {'✅ PASS' if passed_4 else '❌ FAIL'}")
        
        # Test 5: Equity curve plana (sin crecimiento)
        print("  Test 5: Equity curve completamente plana")
        eq_flat = np.ones(100)
        trade_stats_normal = generate_ideal_trade_stats(100)
        result = evaluate_report(eq_flat, trade_stats_normal)
        passed_5 = result[0] < 0.3  # Score bajo esperado
        print(f"    Score: {result[0]:.6f} - {'✅ PASS' if passed_5 else '❌ FAIL'}")
        
        # Test 6: Equity curve con pendiente negativa fuerte
        print("  Test 6: Equity curve con fuerte pendiente negativa")
        eq_declining = np.linspace(10.0, 1.0, 100)
        result = evaluate_report(eq_declining, trade_stats_normal)
        passed_6 = result[0] <= 0.2  # Score muy bajo esperado
        print(f"    Score: {result[0]:.6f} - {'✅ PASS' if passed_6 else '❌ FAIL'}")
        
        # Compilar resultados
        all_passed = [passed_1, passed_2, passed_3, passed_4, passed_5, passed_6]
        total_passed = sum(all_passed)
        pass_rate = total_passed / len(all_passed) * 100
        
        print(f"\n📊 Resultados Suite B: {total_passed}/{len(all_passed)} tests pasaron ({pass_rate:.1f}%)")
        
        return pass_rate > 80.0

# ============================================================================
# TEST SUITE C: VALIDACIÓN DE TRADE METRICS
# ============================================================================

class TestTradeMetrics:
    """Suite C: Valida comportamiento específico de métricas de trades"""
    
    def __init__(self):
        self.results = []
    
    def test_trade_metrics_behavior(self):
        """Valida que las métricas de trades se comporten correctamente"""
        print("\n🧪 TEST SUITE C: Métricas de Trades")
        print("=" * 60)
        
        base_eq = generate_perfect_linear_curve(1000, 0.5)  # Curva lineal ideal
        
        # Escenario 1: Sin trades
        print("  Test 1: Sin actividad de trades")
        no_trades = np.zeros(7)
        result_no_trades = evaluate_report(base_eq, no_trades)
        score_no_trades = result_no_trades[0]
        print(f"    Score sin trades: {score_no_trades:.6f}")
        
        # Escenario 2: Muchos trades exitosos
        print("  Test 2: Alta actividad de trades exitosos")
        many_good_trades = np.array([100, 85, 15, 0, 0.85, 0.06, -0.02])
        result_good_trades = evaluate_report(base_eq, many_good_trades)
        score_good_trades = result_good_trades[0]
        print(f"    Score con muchos trades buenos: {score_good_trades:.6f}")
        
        # Escenario 3: Pocos trades exitosos
        print("  Test 3: Baja actividad de trades exitosos")
        few_good_trades = np.array([10, 8, 2, 0, 0.8, 0.05, -0.02])
        result_few_trades = evaluate_report(base_eq, few_good_trades)
        score_few_trades = result_few_trades[0]
        print(f"    Score con pocos trades buenos: {score_few_trades:.6f}")
        
        # Escenario 4: Muchos trades con baja win rate
        print("  Test 4: Alta actividad con baja win rate")
        many_bad_trades = np.array([100, 30, 70, 0, 0.3, 0.06, -0.03])
        result_bad_trades = evaluate_report(base_eq, many_bad_trades)
        score_bad_trades = result_bad_trades[0]
        print(f"    Score con muchos trades malos: {score_bad_trades:.6f}")
        
        # Validaciones de comportamiento esperado
        print("\n  📊 Validaciones de comportamiento:")
        
        # 1. Trades exitosos deben mejorar score
        improvement_1 = score_good_trades > score_no_trades
        print(f"    1. Trades exitosos mejoran score: {'✅' if improvement_1 else '❌'}")
        print(f"       {score_good_trades:.6f} > {score_no_trades:.6f}")
        
        # 2. Más trades exitosos deben mejorar score
        improvement_2 = score_good_trades > score_few_trades
        print(f"    2. Más trades exitosos mejoran score: {'✅' if improvement_2 else '❌'}")
        print(f"       {score_good_trades:.6f} > {score_few_trades:.6f}")
        
        # 3. Baja win rate debe penalizar score
        improvement_3 = score_good_trades > score_bad_trades
        print(f"    3. Baja win rate penaliza score: {'✅' if improvement_3 else '❌'}")
        print(f"       {score_good_trades:.6f} > {score_bad_trades:.6f}")
        
        # 4. Sin trades no debe ser el peor caso (trades malos sí)
        improvement_4 = score_no_trades > score_bad_trades
        print(f"    4. Sin trades mejor que trades malos: {'✅' if improvement_4 else '❌'}")
        print(f"       {score_no_trades:.6f} > {score_bad_trades:.6f}")
        
        all_improvements = [improvement_1, improvement_2, improvement_3, improvement_4]
        pass_rate = sum(all_improvements) / len(all_improvements) * 100
        
        print(f"\n📊 Resultados Suite C: {sum(all_improvements)}/{len(all_improvements)} tests pasaron ({pass_rate:.1f}%)")
        
        return pass_rate > 75.0

# ============================================================================
# TEST SUITE D: CONSISTENCIA MATEMÁTICA INDIVIDUAL
# ============================================================================

class TestIndividualMetrics:
    """Suite D: Valida que cada métrica individual se comporte correctamente"""
    
    def __init__(self):
        self.results = []
    
    def test_individual_metrics_consistency(self):
        """Valida comportamiento de métricas individuales"""
        print("\n🧪 TEST SUITE D: Métricas Individuales")
        print("=" * 60)
        
        # Preparar datos de test
        perfect_line = generate_perfect_linear_curve(100, 0.5, 1.0)
        negative_slope = np.linspace(2.0, 1.0, 100)
        noisy_line = generate_noisy_linear_curve(100, 0.5, 0.1, 1.0, seed=42)
        
        tests_passed = []
        
        # Test 1: _signed_r2()
        print("  Test 1: _signed_r2() behavior")
        try:
            r2_perfect = _signed_r2(perfect_line)
            r2_negative = _signed_r2(negative_slope)
            r2_noisy = _signed_r2(noisy_line)
            
            test_1a = 0.95 <= r2_perfect <= 1.5
            test_1b = r2_negative <= 0
            test_1c = 0.8 <= r2_noisy <= 1.2
            
            print(f"    Perfect line R²: {r2_perfect:.6f} ∈ [0.95, 1.5] - {'✅' if test_1a else '❌'}")
            print(f"    Negative slope R²: {r2_negative:.6f} ≤ 0 - {'✅' if test_1b else '❌'}")
            print(f"    Noisy line R²: {r2_noisy:.6f} ∈ [0.8, 1.2] - {'✅' if test_1c else '❌'}")
            
            tests_passed.extend([test_1a, test_1b, test_1c])
        except Exception as e:
            print(f"    ❌ Error en _signed_r2: {e}")
            tests_passed.extend([False, False, False])
        
        # Test 2: _perfect_linearity_score()
        print("  Test 2: _perfect_linearity_score() behavior")
        try:
            linearity_perfect = _perfect_linearity_score(perfect_line)
            linearity_negative = _perfect_linearity_score(negative_slope)
            linearity_noisy = _perfect_linearity_score(noisy_line)
            
            test_2a = 0.9 <= linearity_perfect <= 1.0
            test_2b = linearity_negative == 0.0  # Pendiente negativa
            test_2c = 0.3 <= linearity_noisy <= 0.9
            
            print(f"    Perfect linearity: {linearity_perfect:.6f} ∈ [0.9, 1.0] - {'✅' if test_2a else '❌'}")
            print(f"    Negative slope linearity: {linearity_negative:.6f} = 0.0 - {'✅' if test_2b else '❌'}")
            print(f"    Noisy linearity: {linearity_noisy:.6f} ∈ [0.3, 0.9] - {'✅' if test_2c else '❌'}")
            
            tests_passed.extend([test_2a, test_2b, test_2c])
        except Exception as e:
            print(f"    ❌ Error en _perfect_linearity_score: {e}")
            tests_passed.extend([False, False, False])
        
        # Test 3: Trade metrics
        print("  Test 3: Trade metrics behavior")
        try:
            eq_length = 1000
            trade_stats_high = np.array([250, 200, 50, 0, 0.8, 0.1, -0.05])
            trade_stats_low = np.array([10, 8, 2, 0, 0.8, 0.1, -0.05])
            trade_stats_zero = np.zeros(7)
            
            activity_high = _trade_activity_score(trade_stats_high, eq_length)
            activity_low = _trade_activity_score(trade_stats_low, eq_length)
            activity_zero = _trade_activity_score(trade_stats_zero, eq_length)
            
            consistency_high = _trade_consistency_score(trade_stats_high, perfect_line)
            consistency_low = _trade_consistency_score(trade_stats_low, perfect_line)
            consistency_zero = _trade_consistency_score(trade_stats_zero, perfect_line)
            
            test_3a = 0.0 <= activity_high <= 0.3
            test_3b = activity_high > activity_low > activity_zero
            test_3c = 0.0 <= consistency_high <= 0.2
            test_3d = consistency_high >= consistency_low >= consistency_zero
            
            print(f"    High activity score: {activity_high:.6f} ∈ [0.0, 0.3] - {'✅' if test_3a else '❌'}")
            print(f"    Activity ordering: {activity_high:.4f} > {activity_low:.4f} > {activity_zero:.4f} - {'✅' if test_3b else '❌'}")
            print(f"    High consistency score: {consistency_high:.6f} ∈ [0.0, 0.2] - {'✅' if test_3c else '❌'}")
            print(f"    Consistency ordering: {consistency_high:.4f} ≥ {consistency_low:.4f} ≥ {consistency_zero:.4f} - {'✅' if test_3d else '❌'}")
            
            tests_passed.extend([test_3a, test_3b, test_3c, test_3d])
        except Exception as e:
            print(f"    ❌ Error en trade metrics: {e}")
            tests_passed.extend([False, False, False, False])
        
        # Estadísticas finales
        pass_rate = sum(tests_passed) / len(tests_passed) * 100
        print(f"\n📊 Resultados Suite D: {sum(tests_passed)}/{len(tests_passed)} tests pasaron ({pass_rate:.1f}%)")
        
        return pass_rate > 70.0

# ============================================================================
# TEST SUITE E: BENCHMARKING MASIVO
# ============================================================================

class TestMassiveBenchmark:
    """Suite E: Validación con miles de curvas controladas"""
    
    def __init__(self):
        self.results = []
    
    def test_massive_controlled_curves(self, n_tests: int = 1000):
        """Genera y testea miles de curvas controladas"""
        print(f"\n🧪 TEST SUITE E: Benchmarking Masivo ({n_tests:,} curvas)")
        print("=" * 60)
        
        results = []
        start_time = time.time()
        
        print("  Generando y evaluando curvas controladas...")
        
        for i in range(n_tests):
            if (i + 1) % (n_tests // 10) == 0:
                print(f"    Progreso: {i+1:,}/{n_tests:,} ({(i+1)/n_tests*100:.1f}%)")
            
            # Generar parámetros aleatorios controlados
            length = random.randint(100, 1000)
            slope = random.uniform(-1.0, 2.0)
            noise_level = random.uniform(0.0, 0.3)
            
            # Generar curva con características controladas
            if noise_level > 0:
                eq = generate_noisy_linear_curve(length, slope, noise_level, 1.0, seed=i)
            else:
                eq = generate_perfect_linear_curve(length, slope, 1.0)
            
            # Generar trade stats realistas
            n_trades = random.randint(0, length // 10)
            win_rate = random.uniform(0.4, 0.9)
            trade_stats = generate_realistic_trade_stats(n_trades, win_rate, length)
            
            # Evaluar
            result = evaluate_report(eq, trade_stats)
            score = result[0]
            
            # Registrar para análisis estadístico
            results.append({
                'slope': slope,
                'noise': noise_level,
                'trades': n_trades,
                'win_rate': win_rate,
                'score': score,
                'length': length,
                'is_positive_slope': slope > 0,
                'is_low_noise': noise_level < 0.1,
                'has_trades': n_trades > 0
            })
        
        elapsed_time = time.time() - start_time
        print(f"  ✅ Completado en {elapsed_time:.2f} segundos")
        print(f"    Velocidad promedio: {n_tests/elapsed_time:.1f} evaluaciones/segundo")
        
        # Análisis estadístico
        self._analyze_results(results)
        
        return True
    
    def _analyze_results(self, results: List[Dict]):
        """Analiza estadísticamente los resultados del benchmark masivo"""
        print("\n  📊 ANÁLISIS ESTADÍSTICO:")
        
        df = pd.DataFrame(results)
        
        # Estadísticas generales
        print(f"    Score promedio: {df['score'].mean():.6f}")
        print(f"    Score mediano: {df['score'].median():.6f}")
        print(f"    Score std: {df['score'].std():.6f}")
        print(f"    Scores válidos (>-1): {(df['score'] > -1).sum():,}/{len(df):,} ({(df['score'] > -1).mean()*100:.1f}%)")
        
        # Análisis por características
        positive_slopes = df[df['is_positive_slope']]
        negative_slopes = df[~df['is_positive_slope']]
        
        print(f"    Score promedio pendientes positivas: {positive_slopes['score'].mean():.6f}")
        print(f"    Score promedio pendientes negativas: {negative_slopes['score'].mean():.6f}")
        
        low_noise = df[df['is_low_noise']]
        high_noise = df[~df['is_low_noise']]
        
        print(f"    Score promedio bajo ruido: {low_noise['score'].mean():.6f}")
        print(f"    Score promedio alto ruido: {high_noise['score'].mean():.6f}")
        
        with_trades = df[df['has_trades']]
        without_trades = df[~df['has_trades']]
        
        if len(with_trades) > 0 and len(without_trades) > 0:
            print(f"    Score promedio con trades: {with_trades['score'].mean():.6f}")
            print(f"    Score promedio sin trades: {without_trades['score'].mean():.6f}")
        
        # Correlaciones importantes
        print("\n  🔗 CORRELACIONES:")
        
        valid_scores = df[df['score'] > -1]
        if len(valid_scores) > 10:
            corr_slope = valid_scores['score'].corr(valid_scores['slope'])
            corr_noise = valid_scores['score'].corr(valid_scores['noise'])
            corr_trades = valid_scores['score'].corr(valid_scores['trades'])
            corr_win_rate = valid_scores['score'].corr(valid_scores['win_rate'])
            
            print(f"    Score vs Slope: {corr_slope:.3f}")
            print(f"    Score vs Noise: {corr_noise:.3f}")
            print(f"    Score vs # Trades: {corr_trades:.3f}")
            print(f"    Score vs Win Rate: {corr_win_rate:.3f}")
            
            # Evaluación de correlaciones esperadas
            print("\n  ✅ VALIDACIÓN DE CORRELACIONES:")
            print(f"    Score-Slope positiva (>0.3): {'✅' if corr_slope > 0.3 else '❌'} ({corr_slope:.3f})")
            print(f"    Score-Noise negativa (<-0.2): {'✅' if corr_noise < -0.2 else '❌'} ({corr_noise:.3f})")
            print(f"    Score-Trades positiva (>0.1): {'✅' if corr_trades > 0.1 else '❌'} ({corr_trades:.3f})")
        
        self.results = results

# ============================================================================
# TEST DE CORRECCIÓN DE BUGS IDENTIFICADOS
# ============================================================================

class TestBugFixes:
    """Tests específicos para verificar corrección de bugs identificados"""
    
    def test_trade_activity_bug(self):
        """Verifica el bug en _trade_activity_score multiplicación por 0.15"""
        print("\n🐛 TEST CORRECCIÓN BUG: _trade_activity_score()")
        print("=" * 60)
        
        eq_length = 1000
        high_activity_trades = np.array([200, 160, 40, 0, 0.8, 0.05, -0.02])
        
        # El score actual debería estar incorrectamente limitado por la multiplicación por 0.15
        activity_score = _trade_activity_score(high_activity_trades, eq_length)
        
        print(f"  Score de actividad actual: {activity_score:.6f}")
        print(f"  Score máximo teórico: 0.3")
        print(f"  Score máximo real con bug: {0.3 * 0.15:.6f}")
        
        # El bug hace que el score sea muchísimo menor de lo esperado
        bug_present = activity_score < 0.1  # Con el bug, debería ser ~0.045 máximo
        
        if bug_present:
            print("  ❌ BUG CONFIRMADO: Score artificialmente bajo por multiplicación por 0.15")
        else:
            print("  ✅ BUG CORREGIDO: Score en rango esperado")
        
        return not bug_present
    
    def test_trade_consistency_bug(self):
        """Verifica el bug en _trade_consistency_score doble penalización"""
        print("\n🐛 TEST CORRECCIÓN BUG: _trade_consistency_score()")
        print("=" * 60)
        
        eq = generate_perfect_linear_curve(1000, 0.5)
        high_consistency_trades = np.array([100, 80, 20, 0, 0.8, 0.05, -0.02])
        
        consistency_score = _trade_consistency_score(high_consistency_trades, eq)
        
        print(f"  Score de consistencia actual: {consistency_score:.6f}")
        print(f"  Score máximo teórico: 0.2")
        print(f"  Score máximo real con bug: {0.2 * 0.2:.6f}")
        
        # El bug hace que el score sea 0.04 máximo en lugar de 0.2
        bug_present = consistency_score < 0.05
        
        if bug_present:
            print("  ❌ BUG CONFIRMADO: Score artificialmente bajo por doble penalización")
        else:
            print("  ✅ BUG CORREGIDO: Score en rango esperado")
        
        return not bug_present

# ============================================================================
# RUNNER PRINCIPAL DE TESTS
# ============================================================================

class TestRunner:
    """Ejecutor principal de todos los tests"""
    
    def __init__(self):
        self.test_results = {}
    
    def run_all_tests(self, quick_mode: bool = False):
        """Ejecuta todos los tests de manera secuencial"""
        print("🚀 INICIANDO TESTS EXHAUSTIVOS DE evaluate_report()")
        print("=" * 80)
        
        start_time = time.time()
        
        # Suite A: Curvas lineales perfectas
        test_a = TestPerfectLinearCurves()
        self.test_results['perfect_linear'] = test_a.test_perfect_linear_curves()
        
        # Suite B: Casos patológicos
        test_b = TestPathologicalCases()
        self.test_results['pathological'] = test_b.test_pathological_cases()
        
        # Suite C: Métricas de trades
        test_c = TestTradeMetrics()
        self.test_results['trade_metrics'] = test_c.test_trade_metrics_behavior()
        
        # Suite D: Métricas individuales
        test_d = TestIndividualMetrics()
        self.test_results['individual_metrics'] = test_d.test_individual_metrics_consistency()
        
        # Suite E: Benchmarking masivo
        test_e = TestMassiveBenchmark()
        n_benchmark = 500 if quick_mode else 2000
        self.test_results['massive_benchmark'] = test_e.test_massive_controlled_curves(n_benchmark)
        
        # Tests de bugs
        test_bugs = TestBugFixes()
        self.test_results['bug_trade_activity'] = test_bugs.test_trade_activity_bug()
        self.test_results['bug_trade_consistency'] = test_bugs.test_trade_consistency_bug()
        
        total_time = time.time() - start_time
        
        # Resumen final
        self._print_final_summary(total_time)
        
        return self.test_results
    
    def _print_final_summary(self, total_time: float):
        """Imprime resumen final de todos los tests"""
        print("\n" + "=" * 80)
        print("📋 RESUMEN FINAL DE TESTS")
        print("=" * 80)
        
        passed_tests = sum(1 for result in self.test_results.values() if result)
        total_tests = len(self.test_results)
        overall_pass_rate = passed_tests / total_tests * 100
        
        print(f"⏱️  Tiempo total de ejecución: {total_time:.2f} segundos")
        print(f"📊 Tests pasados: {passed_tests}/{total_tests} ({overall_pass_rate:.1f}%)")
        print()
        
        for test_name, passed in self.test_results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"  {test_name}: {status}")
        
        print("\n" + "=" * 80)
        
        if overall_pass_rate >= 80:
            print("🎉 RESULTADO GENERAL: ✅ ÉXITO - La función funciona correctamente")
        elif overall_pass_rate >= 60:
            print("⚠️  RESULTADO GENERAL: 🟡 PARCIAL - Función necesita mejoras")
        else:
            print("🚨 RESULTADO GENERAL: ❌ CRÍTICO - Función tiene problemas serios")
        
        print("=" * 80)

# ============================================================================
# PUNTO DE ENTRADA PRINCIPAL
# ============================================================================

if __name__ == "__main__":
    """Ejecuta el sistema completo de tests"""
    
    # Configuración
    quick_mode = "--quick" in sys.argv
    
    print("🔬 SISTEMA DE TESTING EXHAUSTIVO PARA evaluate_report()")
    print("📁 Archivo: /workspace/studies/modules/tester_lib.py")
    print("🎯 Objetivo: Validar función de scoring de curvas de equity")
    
    if quick_mode:
        print("⚡ Modo rápido activado (menos tests de benchmark)")
    
    print()
    
    # Ejecutar tests
    runner = TestRunner()
    results = runner.run_all_tests(quick_mode=quick_mode)
    
    # Salir con código apropiado
    overall_success = sum(results.values()) / len(results) >= 0.8
    sys.exit(0 if overall_success else 1)