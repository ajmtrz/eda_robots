"""
EJEMPLO DE USO: VALIDACIÓN AVANZADA DE ESTRATEGIAS
=================================================

Este script demuestra cómo utilizar las nuevas funciones de validación 
avanzada añadidas al módulo tester_lib.py basadas en las recomendaciones
del artículo de backtesting.

Autor: Sistema de Validación Avanzada
Fecha: 2024
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Añadir el módulo al path
sys.path.append('/workspace/studies/modules')
from tester_lib import (
    monkey_test,
    enhanced_deflated_sharpe_ratio, 
    walk_forward_analysis,
    probability_backtest_overfitting,
    comprehensive_strategy_validation,
    tester
)

def create_mock_models():
    """
    Crea modelos mock para demostración.
    En uso real, estos serían sus modelos entrenados.
    """
    class MockModel:
        def __init__(self, bias=0.5):
            self.bias = bias
            
        def predict_proba(self, X):
            # Simulamos predicciones con sesgo hacia una clase
            n_samples = X.shape[0]
            probs_neg = np.random.random(n_samples) * (1 - self.bias)
            probs_pos = 1 - probs_neg
            return np.column_stack([probs_neg, probs_pos])
    
    return MockModel(bias=0.6), MockModel(bias=0.55)

def create_sample_dataset(n_periods=5000):
    """
    Crea un dataset de ejemplo para demostración.
    En uso real, este sería su dataset histórico real.
    """
    np.random.seed(42)
    
    # Simular precios con tendencia alcista y volatilidad
    returns = np.random.normal(0.0002, 0.01, n_periods)  # 0.02% promedio, 1% vol diaria
    prices = 1000 * np.cumprod(1 + returns)
    
    # Crear features sintéticas
    feature_1 = np.random.randn(n_periods)
    feature_2 = np.random.randn(n_periods) 
    feature_3 = np.random.randn(n_periods)
    feature_4 = np.random.randn(n_periods)
    
    # Crear DataFrame
    df = pd.DataFrame({
        'close': prices,
        'feature_main_1': feature_1,
        'feature_main_2': feature_2,
        'feature_meta_1': feature_3, 
        'feature_meta_2': feature_4
    })
    
    return df

def example_individual_tests():
    """
    Ejemplo de uso de tests individuales de validación.
    """
    print("🔬 EJEMPLO: TESTS INDIVIDUALES DE VALIDACIÓN")
    print("=" * 60)
    
    # Preparar datos y modelos
    dataset = create_sample_dataset(n_periods=3000)
    model_main, model_meta = create_mock_models()
    
    main_cols = ['feature_main_1', 'feature_main_2']
    meta_cols = ['feature_meta_1', 'feature_meta_2']
    
    print(f"Dataset creado: {len(dataset)} períodos")
    print(f"Rango de precios: ${dataset['close'].min():.2f} - ${dataset['close'].max():.2f}")
    
    # 1. MONKEY TEST
    print("\n1️⃣ EJECUTANDO MONKEY TEST")
    print("-" * 40)
    
    monkey_results = monkey_test(
        dataset=dataset,
        model_main=model_main,
        model_meta=model_meta,
        model_main_cols=main_cols,
        model_meta_cols=meta_cols,
        direction='both',
        timeframe='H1',
        n_monkeys=500  # Reducido para demo rápida
    )
    
    print(f"\n📊 Resultados Monkey Test:")
    print(f"   Status: {monkey_results['status']}")
    print(f"   P-value: {monkey_results['p_value']:.4f}")
    print(f"   Significativo: {monkey_results['is_significant']}")
    
    # 2. WALK-FORWARD ANALYSIS
    print("\n2️⃣ EJECUTANDO WALK-FORWARD ANALYSIS")
    print("-" * 40)
    
    wf_results = walk_forward_analysis(
        dataset=dataset,
        model_main=model_main,
        model_meta=model_meta,
        model_main_cols=main_cols,
        model_meta_cols=meta_cols,
        direction='both',
        timeframe='H1',
        train_size=1000,
        test_size=300,
        step_size=150
    )
    
    print(f"\n📊 Resultados Walk-Forward:")
    print(f"   Status: {wf_results['status']}")
    if 'summary' in wf_results:
        summary = wf_results['summary']
        print(f"   Ventanas analizadas: {summary['num_windows']}")
        print(f"   Score promedio: {summary['score_mean']:.4f}")
        print(f"   Estabilidad: {summary['is_stable']}")
    
    # 3. PROBABILITY OF BACKTEST OVERFITTING
    print("\n3️⃣ EJECUTANDO PBO ANALYSIS")
    print("-" * 40)
    
    pbo_results = probability_backtest_overfitting(
        dataset=dataset,
        model_main=model_main,
        model_meta=model_meta,
        model_main_cols=main_cols,
        model_meta_cols=meta_cols,
        direction='both',
        timeframe='H1',
        n_splits=8  # Reducido para demo
    )
    
    print(f"\n📊 Resultados PBO:")
    print(f"   Status: {pbo_results['status']}")
    print(f"   PBO: {pbo_results.get('pbo', 'N/A')}")
    print(f"   Overfitting detectado: {pbo_results.get('is_overfitted', 'N/A')}")

def example_comprehensive_validation():
    """
    Ejemplo de validación integral usando el framework completo.
    """
    print("\n\n🎯 EJEMPLO: VALIDACIÓN INTEGRAL COMPLETA")
    print("=" * 60)
    
    # Preparar datos y modelos
    dataset = create_sample_dataset(n_periods=4000)
    model_main, model_meta = create_mock_models()
    
    main_cols = ['feature_main_1', 'feature_main_2']
    meta_cols = ['feature_meta_1', 'feature_meta_2']
    
    # Configuración personalizada para demo rápida
    validation_config = {
        'monkey_test': {'n_monkeys': 500},
        'walk_forward': {'train_size': 1000, 'test_size': 300, 'step_size': 200},
        'pbo': {'n_splits': 8},
        'dsr': {'confidence_level': 0.95}
    }
    
    # Ejecutar validación integral
    results = comprehensive_strategy_validation(
        dataset=dataset,
        model_main=model_main,
        model_meta=model_meta,
        model_main_cols=main_cols,
        model_meta_cols=meta_cols,
        direction='both',
        timeframe='H1',
        num_trials=10,  # Simular que probamos 10 configuraciones
        validation_config=validation_config
    )
    
    # Mostrar resultados finales
    print(f"\n📋 RESUMEN DE VALIDACIÓN INTEGRAL:")
    print("-" * 40)
    print(f"Status final: {results['overall_status']}")
    
    if 'final_verdict' in results:
        verdict = results['final_verdict']
        print(f"Tests pasados: {verdict['passed_tests']}/{verdict['total_tests']}")
        print(f"Fallas críticas: {verdict['critical_failures']}")
        print(f"Recomendación: {verdict['recommendation']}")
    
    # Detalles por test
    print(f"\n📊 DETALLES POR TEST:")
    for test_name, test_result in results['tests_results'].items():
        status = test_result.get('status', 'UNKNOWN')
        print(f"   {test_name.upper()}: {status}")

def example_enhanced_dsr():
    """
    Ejemplo específico del Enhanced Deflated Sharpe Ratio.
    """
    print("\n\n📈 EJEMPLO: ENHANCED DEFLATED SHARPE RATIO")
    print("=" * 60)
    
    # Crear una equity curve sintética
    np.random.seed(123)
    n_periods = 1000
    daily_returns = np.random.normal(0.001, 0.02, n_periods)  # 0.1% diario, 2% vol
    equity_curve = 100000 * np.cumprod(1 + daily_returns)
    
    print(f"Equity curve creada: {n_periods} períodos")
    print(f"Retorno total: {(equity_curve[-1]/equity_curve[0] - 1)*100:.1f}%")
    
    # Calcular DSR para diferentes números de trials
    for num_trials in [1, 10, 50, 100]:
        dsr_results = enhanced_deflated_sharpe_ratio(
            equity_curve=equity_curve,
            num_trials=num_trials,
            periods_per_year=6240.0,  # H1 timeframe
            confidence_level=0.95
        )
        
        print(f"\n🔍 DSR con {num_trials} trials probados:")
        if dsr_results['status'] == 'PASSED':
            print(f"   Sharpe observado: {dsr_results['sharpe_observed']:.3f}")
            print(f"   Sharpe deflated: {dsr_results['deflated_sharpe']:.3f}")
            print(f"   Threshold: {dsr_results['threshold_sharpe']:.3f}")
            print(f"   Significativo: {dsr_results['is_significant']}")
        else:
            print(f"   Status: {dsr_results['status']}")

def main():
    """
    Función principal que ejecuta todos los ejemplos.
    """
    print("🚀 INICIANDO EJEMPLOS DE VALIDACIÓN AVANZADA")
    print("=" * 60)
    print("Este script demuestra el uso de las nuevas funciones de")
    print("validación de estrategias basadas en las recomendaciones")
    print("del artículo de backtesting de Quant Beckman.")
    print()
    
    try:
        # Ejecutar ejemplos
        example_individual_tests()
        example_comprehensive_validation() 
        example_enhanced_dsr()
        
        print("\n\n✅ TODOS LOS EJEMPLOS COMPLETADOS EXITOSAMENTE")
        print("=" * 60)
        print("\n📖 CÓMO USAR EN SUS PROYECTOS:")
        print("1. Importe las funciones desde tester_lib:")
        print("   from tester_lib import comprehensive_strategy_validation")
        print()
        print("2. Prepare sus datos y modelos entrenados")
        print()
        print("3. Ejecute la validación integral:")
        print("   results = comprehensive_strategy_validation(...)")
        print()
        print("4. Evalúe el veredicto final:")
        print("   if results['overall_status'] == 'VALIDATED':")
        print("       # Estrategia apta para trading en vivo")
        print()
        print("⚠️  IMPORTANTE: Estas funciones complementan, no reemplazan,")
        print("   la función evaluate_report existente para estrategias lineales.")
        
    except Exception as e:
        print(f"\n❌ ERROR EN LA EJECUCIÓN: {str(e)}")
        print("Verifique que el módulo tester_lib.py esté correctamente instalado")
        print("y que todas las dependencias estén disponibles.")

if __name__ == "__main__":
    main()