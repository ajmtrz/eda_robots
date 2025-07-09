#!/usr/bin/env python3
"""
TEST SIMPLIFICADO PARA evaluate_report()

Version sin dependencias externas para validar bugs críticos identificados.
"""

import sys
import os
import math
import time
import random

# Agregar el path del módulo bajo test
sys.path.append('/workspace/studies/modules')

try:
    import numpy as np
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
    IMPORTS_OK = True
except ImportError as e:
    print(f"❌ Error importando: {e}")
    IMPORTS_OK = False

def create_simple_array(start, end, length):
    """Crea array equivalente a np.linspace sin numpy"""
    step = (end - start) / (length - 1)
    return [start + i * step for i in range(length)]

def to_numpy_array(data):
    """Convierte lista a numpy array"""
    return np.array(data, dtype=np.float64)

def test_basic_functionality():
    """Test básico de funcionalidad sin dependencias externas"""
    print("🧪 TEST BÁSICO: Funcionalidad de evaluate_report")
    print("=" * 60)
    
    if not IMPORTS_OK:
        print("❌ No se pudo importar tester_lib - Saltando tests")
        return False
    
    try:
        # Test 1: Curva lineal perfecta ascendente
        print("  Test 1: Curva lineal perfecta ascendente")
        eq_perfect = to_numpy_array(create_simple_array(1.0, 2.0, 100))
        trade_stats_ideal = to_numpy_array([20, 16, 4, 0, 0.8, 0.05, -0.02])
        
        result = evaluate_report(eq_perfect, trade_stats_ideal)
        score_perfect = result[0]
        
        print(f"    Score curva perfecta: {score_perfect:.6f}")
        test_1_pass = score_perfect > 0.7  # Debe ser alto
        print(f"    ✅ PASS" if test_1_pass else f"    ❌ FAIL - Score demasiado bajo")
        
        # Test 2: Curva plana (sin crecimiento)
        print("  Test 2: Curva completamente plana")
        eq_flat = to_numpy_array([1.0] * 100)
        result = evaluate_report(eq_flat, trade_stats_ideal)
        score_flat = result[0]
        
        print(f"    Score curva plana: {score_flat:.6f}")
        test_2_pass = score_flat < 0.3  # Debe ser bajo
        print(f"    ✅ PASS" if test_2_pass else f"    ❌ FAIL - Score demasiado alto")
        
        # Test 3: Curva con pendiente negativa
        print("  Test 3: Curva con pendiente negativa")
        eq_negative = to_numpy_array(create_simple_array(2.0, 1.0, 100))
        result = evaluate_report(eq_negative, trade_stats_ideal)
        score_negative = result[0]
        
        print(f"    Score curva negativa: {score_negative:.6f}")
        test_3_pass = score_negative < score_perfect  # Debe ser menor que perfecta
        print(f"    ✅ PASS" if test_3_pass else f"    ❌ FAIL - Score no penaliza pendiente negativa")
        
        # Test 4: Serie muy corta (debe fallar)
        print("  Test 4: Serie muy corta (edge case)")
        eq_short = to_numpy_array([1.0, 1.1, 1.2])
        result = evaluate_report(eq_short, trade_stats_ideal)
        score_short = result[0]
        
        print(f"    Score serie corta: {score_short:.6f}")
        test_4_pass = score_short == -1.0  # Debe retornar -1
        print(f"    ✅ PASS" if test_4_pass else f"    ❌ FAIL - No maneja series cortas correctamente")
        
        # Resumen
        all_tests = [test_1_pass, test_2_pass, test_3_pass, test_4_pass]
        passed = sum(all_tests)
        total = len(all_tests)
        
        print(f"\n📊 Resultados: {passed}/{total} tests básicos pasaron ({passed/total*100:.1f}%)")
        
        return passed >= 3  # Al menos 3 de 4 tests deben pasar
        
    except Exception as e:
        print(f"❌ Error en tests básicos: {e}")
        return False

def test_trade_metrics_bugs():
    """Test específico para validar bugs en trade metrics"""
    print("\n🐛 TEST BUGS: Métricas de Trades")
    print("=" * 60)
    
    if not IMPORTS_OK:
        print("❌ No se pudo importar tester_lib - Saltando tests")
        return False
    
    try:
        # Bug 1: _trade_activity_score multiplicación incorrecta por 0.15
        print("  Bug 1: _trade_activity_score multiplicación por 0.15")
        eq_length = 1000
        high_activity_trades = to_numpy_array([200, 160, 40, 0, 0.8, 0.05, -0.02])
        
        activity_score = _trade_activity_score(high_activity_trades, eq_length)
        print(f"    Score actividad actual: {activity_score:.6f}")
        print(f"    Score máximo teórico sin bug: 0.30")
        print(f"    Score máximo real con bug: {0.30 * 0.15:.6f}")
        
        # Con el bug, el score debería estar limitado a ~0.045
        bug_1_present = activity_score < 0.10
        if bug_1_present:
            print("    ❌ BUG CONFIRMADO: Score artificialmente reducido")
        else:
            print("    ✅ BUG CORREGIDO: Score en rango esperado")
        
        # Bug 2: _trade_consistency_score doble penalización
        print("  Bug 2: _trade_consistency_score doble penalización")
        eq_perfect = to_numpy_array(create_simple_array(1.0, 2.0, 1000))
        consistency_score = _trade_consistency_score(high_activity_trades, eq_perfect)
        
        print(f"    Score consistencia actual: {consistency_score:.6f}")
        print(f"    Score máximo teórico sin bug: 0.20")
        print(f"    Score máximo real con bug: {0.20 * 0.20:.6f}")
        
        # Con el bug, el score debería estar limitado a ~0.04
        bug_2_present = consistency_score < 0.08
        if bug_2_present:
            print("    ❌ BUG CONFIRMADO: Score con doble penalización")
        else:
            print("    ✅ BUG CORREGIDO: Score en rango esperado")
        
        print(f"\n📊 Bugs detectados: {int(bug_1_present) + int(bug_2_present)}/2")
        
        # Retornar True si NO hay bugs (bugs corregidos)
        return not (bug_1_present or bug_2_present)
        
    except Exception as e:
        print(f"❌ Error en test de bugs: {e}")
        return False

def test_individual_metrics():
    """Test de métricas individuales"""
    print("\n🔬 TEST INDIVIDUAL: Métricas Específicas")
    print("=" * 60)
    
    if not IMPORTS_OK:
        print("❌ No se pudo importar tester_lib - Saltando tests")
        return False
    
    try:
        # Preparar datos
        perfect_line = to_numpy_array(create_simple_array(1.0, 2.0, 100))
        negative_line = to_numpy_array(create_simple_array(2.0, 1.0, 100))
        flat_line = to_numpy_array([1.0] * 100)
        
        tests_passed = []
        
        # Test _signed_r2
        print("  Test _signed_r2():")
        r2_perfect = _signed_r2(perfect_line)
        r2_negative = _signed_r2(negative_line)
        r2_flat = _signed_r2(flat_line)
        
        print(f"    Línea perfecta R²: {r2_perfect:.6f}")
        print(f"    Línea negativa R²: {r2_negative:.6f}")
        print(f"    Línea plana R²: {r2_flat:.6f}")
        
        test_r2_perfect = r2_perfect > 0.9
        test_r2_negative = r2_negative <= 0
        test_r2_behavior = r2_perfect > abs(r2_negative)
        
        tests_passed.extend([test_r2_perfect, test_r2_negative, test_r2_behavior])
        print(f"    R² tests: {'✅' if all([test_r2_perfect, test_r2_negative, test_r2_behavior]) else '❌'}")
        
        # Test _perfect_linearity_score
        print("  Test _perfect_linearity_score():")
        linearity_perfect = _perfect_linearity_score(perfect_line)
        linearity_negative = _perfect_linearity_score(negative_line)
        
        print(f"    Línea perfecta linealidad: {linearity_perfect:.6f}")
        print(f"    Línea negativa linealidad: {linearity_negative:.6f}")
        
        test_linearity_perfect = linearity_perfect > 0.8
        test_linearity_negative = linearity_negative == 0.0  # Debe ser 0 para pendiente negativa
        
        tests_passed.extend([test_linearity_perfect, test_linearity_negative])
        print(f"    Linealidad tests: {'✅' if all([test_linearity_perfect, test_linearity_negative]) else '❌'}")
        
        # Test _slope_reward
        print("  Test _slope_reward():")
        slope_perfect = _slope_reward(perfect_line)
        slope_negative = _slope_reward(negative_line)
        slope_flat = _slope_reward(flat_line)
        
        print(f"    Línea perfecta slope reward: {slope_perfect:.6f}")
        print(f"    Línea negativa slope reward: {slope_negative:.6f}")
        print(f"    Línea plana slope reward: {slope_flat:.6f}")
        
        test_slope_perfect = slope_perfect > 0.5
        test_slope_negative = slope_negative == 0.0
        test_slope_flat = slope_flat == 0.0
        
        tests_passed.extend([test_slope_perfect, test_slope_negative, test_slope_flat])
        print(f"    Slope tests: {'✅' if all([test_slope_perfect, test_slope_negative, test_slope_flat]) else '❌'}")
        
        # Estadísticas
        passed_count = sum(tests_passed)
        total_count = len(tests_passed)
        pass_rate = passed_count / total_count * 100
        
        print(f"\n📊 Métricas individuales: {passed_count}/{total_count} tests pasaron ({pass_rate:.1f}%)")
        
        return pass_rate > 70.0
        
    except Exception as e:
        print(f"❌ Error en test de métricas individuales: {e}")
        return False

def test_scoring_behavior():
    """Test del comportamiento del scoring general"""
    print("\n⚖️  TEST SCORING: Comportamiento General")
    print("=" * 60)
    
    if not IMPORTS_OK:
        print("❌ No se pudo importar tester_lib - Saltando tests")
        return False
    
    try:
        # Diferentes tipos de curvas para comparar
        eq_perfect = to_numpy_array(create_simple_array(1.0, 2.0, 500))  # Perfecta lineal ascendente
        eq_steep = to_numpy_array(create_simple_array(1.0, 4.0, 500))    # Muy empinada
        eq_gentle = to_numpy_array(create_simple_array(1.0, 1.2, 500))   # Muy suave
        eq_negative = to_numpy_array(create_simple_array(2.0, 1.0, 500)) # Descendente
        
        # Trade stats diferentes
        good_trades = to_numpy_array([50, 40, 10, 0, 0.8, 0.05, -0.02])
        no_trades = to_numpy_array([0, 0, 0, 0, 0, 0, 0])
        bad_trades = to_numpy_array([50, 15, 35, 0, 0.3, 0.03, -0.05])
        
        # Evaluaciones
        score_perfect_good = evaluate_report(eq_perfect, good_trades)[0]
        score_perfect_no = evaluate_report(eq_perfect, no_trades)[0]
        score_steep_good = evaluate_report(eq_steep, good_trades)[0]
        score_gentle_good = evaluate_report(eq_gentle, good_trades)[0]
        score_negative_good = evaluate_report(eq_negative, good_trades)[0]
        score_perfect_bad = evaluate_report(eq_perfect, bad_trades)[0]
        
        print(f"  Scores obtenidos:")
        print(f"    Perfecta + Buenos trades: {score_perfect_good:.6f}")
        print(f"    Perfecta + Sin trades: {score_perfect_no:.6f}")
        print(f"    Perfecta + Malos trades: {score_perfect_bad:.6f}")
        print(f"    Empinada + Buenos trades: {score_steep_good:.6f}")
        print(f"    Suave + Buenos trades: {score_gentle_good:.6f}")
        print(f"    Negativa + Buenos trades: {score_negative_good:.6f}")
        
        # Validaciones de comportamiento esperado
        tests = []
        
        # 1. Curva perfecta debe ser la mejor
        test_1 = score_perfect_good >= max(score_steep_good, score_gentle_good, score_negative_good)
        tests.append(test_1)
        print(f"    1. Curva perfecta es la mejor: {'✅' if test_1 else '❌'}")
        
        # 2. Buenos trades deben mejorar score vs sin trades
        test_2 = score_perfect_good >= score_perfect_no
        tests.append(test_2)
        print(f"    2. Buenos trades mejoran score: {'✅' if test_2 else '❌'}")
        
        # 3. Malos trades deben empeorar score vs buenos trades  
        test_3 = score_perfect_good > score_perfect_bad
        tests.append(test_3)
        print(f"    3. Malos trades empeoran score: {'✅' if test_3 else '❌'}")
        
        # 4. Pendiente negativa debe tener score muy bajo
        test_4 = score_negative_good < 0.5
        tests.append(test_4)
        print(f"    4. Pendiente negativa penalizada: {'✅' if test_4 else '❌'}")
        
        # 5. Todos los scores válidos deben estar en [0,1]
        all_scores = [score_perfect_good, score_perfect_no, score_steep_good, 
                     score_gentle_good, score_negative_good, score_perfect_bad]
        test_5 = all(0 <= s <= 1 or s == -1 for s in all_scores)
        tests.append(test_5)
        print(f"    5. Scores en rango válido [0,1]: {'✅' if test_5 else '❌'}")
        
        pass_rate = sum(tests) / len(tests) * 100
        print(f"\n📊 Comportamiento scoring: {sum(tests)}/{len(tests)} tests pasaron ({pass_rate:.1f}%)")
        
        return pass_rate > 80.0
        
    except Exception as e:
        print(f"❌ Error en test de comportamiento: {e}")
        return False

def main():
    """Función principal que ejecuta todos los tests"""
    print("🔬 SISTEMA DE TESTING SIMPLIFICADO PARA evaluate_report()")
    print("📁 Archivo: /workspace/studies/modules/tester_lib.py")
    print("🎯 Objetivo: Detectar bugs críticos y validar comportamiento")
    print("=" * 80)
    
    start_time = time.time()
    
    # Ejecutar todos los tests
    results = {}
    results['basic_functionality'] = test_basic_functionality()
    results['trade_metrics_bugs'] = test_trade_metrics_bugs()
    results['individual_metrics'] = test_individual_metrics()
    results['scoring_behavior'] = test_scoring_behavior()
    
    total_time = time.time() - start_time
    
    # Resumen final
    print("\n" + "=" * 80)
    print("📋 RESUMEN FINAL")
    print("=" * 80)
    
    passed_tests = sum(results.values())
    total_tests = len(results)
    overall_pass_rate = passed_tests / total_tests * 100
    
    print(f"⏱️  Tiempo total: {total_time:.2f} segundos")
    print(f"📊 Tests pasados: {passed_tests}/{total_tests} ({overall_pass_rate:.1f}%)")
    print()
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {test_name}: {status}")
    
    print("\n" + "=" * 80)
    
    if overall_pass_rate >= 80:
        print("🎉 RESULTADO: ✅ FUNCIÓN FUNCIONA CORRECTAMENTE")
        conclusion = "ÉXITO"
    elif overall_pass_rate >= 60:
        print("⚠️  RESULTADO: 🟡 FUNCIÓN NECESITA MEJORAS")
        conclusion = "PARCIAL"
    else:
        print("🚨 RESULTADO: ❌ FUNCIÓN TIENE PROBLEMAS CRÍTICOS")
        conclusion = "FALLO"
    
    print("=" * 80)
    
    # Recomendaciones basadas en los resultados
    if not results['trade_metrics_bugs']:
        print("\n🚨 RECOMENDACIÓN CRÍTICA:")
        print("   Los bugs en trade metrics deben corregirse INMEDIATAMENTE")
        print("   - _trade_activity_score: eliminar multiplicación por 0.15")
        print("   - _trade_consistency_score: eliminar doble penalización")
    
    if not results['basic_functionality']:
        print("\n🚨 PROBLEMA FUNDAMENTAL:")
        print("   La funcionalidad básica no funciona correctamente")
        print("   Revisar la lógica principal de evaluate_report()")
    
    return overall_pass_rate >= 80

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)