import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append('/workspace/studies/modules')

from tester_lib import evaluate_report, metrics_tuple_to_dict
from tester_lib_improved import evaluate_report_improved

# Importar el generador de curvas del test anterior
from test_evaluate_report import EquityCurveGenerator

def compare_functions():
    """Compara las dos versiones de evaluate_report"""
    generator = EquityCurveGenerator()
    
    # Casos de prueba idénticos a los anteriores
    test_cases = [
        # Curvas perfectamente lineales (deberían tener score muy alto)
        ('perfect_linear', {'slope': 0.1, 'noise': 0.0}, (0.85, 1.0)),
        ('perfect_linear', {'slope': 0.5, 'noise': 0.0}, (0.90, 1.0)),
        ('perfect_linear', {'slope': 1.0, 'noise': 0.0}, (0.90, 1.0)),
        ('perfect_linear', {'slope': 2.0, 'noise': 0.0}, (0.85, 1.0)),
        
        # Curvas lineales con ruido (score alto pero menor)
        ('perfect_linear', {'slope': 0.5, 'noise': 0.5}, (0.70, 0.90)),
        ('perfect_linear', {'slope': 0.5, 'noise': 1.0}, (0.60, 0.85)),
        ('perfect_linear', {'slope': 0.5, 'noise': 2.0}, (0.50, 0.75)),
        
        # Curvas exponenciales (score medio-alto)
        ('exponential', {'rate': 0.0005}, (0.60, 0.85)),
        ('exponential', {'rate': 0.001}, (0.55, 0.80)),
        ('exponential', {'rate': 0.002}, (0.50, 0.75)),
        
        # Curvas escalonadas (score medio)
        ('stepped', {'step_size': 5.0, 'step_freq': 50}, (0.50, 0.75)),
        ('stepped', {'step_size': 10.0, 'step_freq': 100}, (0.45, 0.70)),
        ('stepped', {'step_size': -5.0, 'step_freq': 50}, (0.0, 0.20)),
        
        # Curvas volátiles (score bajo)
        ('volatile', {'trend': 0.1, 'volatility': 5.0}, (0.20, 0.50)),
        ('volatile', {'trend': 0.0, 'volatility': 10.0}, (0.10, 0.40)),
        ('volatile', {'trend': -0.1, 'volatility': 5.0}, (0.0, 0.20)),
        
        # Curvas con drawdown (score medio-bajo)
        ('drawdown_recovery', {'dd_depth': 0.1, 'dd_location': 0.5}, (0.60, 0.85)),
        ('drawdown_recovery', {'dd_depth': 0.3, 'dd_location': 0.5}, (0.40, 0.65)),
        ('drawdown_recovery', {'dd_depth': 0.5, 'dd_location': 0.3}, (0.20, 0.45)),
        
        # Curvas en diente de sierra (score bajo-medio)
        ('sawtooth', {'period': 100, 'amplitude': 20.0}, (0.30, 0.60)),
        ('sawtooth', {'period': 50, 'amplitude': 10.0}, (0.35, 0.65)),
        
        # Random walks (score muy bajo)
        ('random_walk', {'drift': 0.0}, (0.0, 0.30)),
        ('random_walk', {'drift': 0.01}, (0.10, 0.40)),
        ('random_walk', {'drift': -0.01}, (0.0, 0.20)),
        
        # Curvas planas (score muy bajo)
        ('flat', {'value': 100.0}, (0.0, 0.10)),
        
        # Curvas sinusoidales con tendencia (score medio)
        ('sine_trend', {'trend': 0.1, 'amplitude': 10.0, 'frequency': 0.05}, (0.40, 0.70)),
        ('sine_trend', {'trend': 0.2, 'amplitude': 5.0, 'frequency': 0.1}, (0.50, 0.80)),
        
        # Curvas compuestas
        ('compound', {'components': [
            {'type': 'linear', 'slope': 0.3},
            {'type': 'sine', 'amplitude': 5.0, 'frequency': 0.02},
            {'type': 'noise', 'std': 0.5}
        ]}, (0.65, 0.85)),
    ]
    
    results = []
    original_passed = 0
    improved_passed = 0
    
    print("=" * 100)
    print("COMPARACIÓN DE FUNCIONES evaluate_report")
    print("=" * 100)
    print(f"{'Tipo de Curva':<25} {'Params':<35} {'Original':<12} {'Mejorada':<12} {'Esperado':<15}")
    print("-" * 100)
    
    for curve_type, params, expected_range in test_cases:
        # Generar curva
        if curve_type == 'perfect_linear':
            equity, trade_stats = generator.generate_perfect_linear(**params)
        elif curve_type == 'exponential':
            equity, trade_stats = generator.generate_exponential(**params)
        elif curve_type == 'stepped':
            equity, trade_stats = generator.generate_stepped(**params)
        elif curve_type == 'volatile':
            equity, trade_stats = generator.generate_volatile(**params)
        elif curve_type == 'drawdown_recovery':
            equity, trade_stats = generator.generate_drawdown_recovery(**params)
        elif curve_type == 'sawtooth':
            equity, trade_stats = generator.generate_sawtooth(**params)
        elif curve_type == 'random_walk':
            equity, trade_stats = generator.generate_random_walk(**params)
        elif curve_type == 'flat':
            equity, trade_stats = generator.generate_flat(**params)
        elif curve_type == 'sine_trend':
            equity, trade_stats = generator.generate_sine_trend(**params)
        elif curve_type == 'compound':
            equity, trade_stats = generator.generate_compound_curve(**params)
        
        # Evaluar con ambas funciones
        score_original = evaluate_report(equity, trade_stats)[0]
        score_improved = evaluate_report_improved(equity, trade_stats)[0]
        
        # Verificar si están en rango
        original_in_range = expected_range[0] <= score_original <= expected_range[1]
        improved_in_range = expected_range[0] <= score_improved <= expected_range[1]
        
        if original_in_range:
            original_passed += 1
        if improved_in_range:
            improved_passed += 1
        
        # Formatear resultados
        original_str = f"{score_original:6.4f} {'✓' if original_in_range else '✗'}"
        improved_str = f"{score_improved:6.4f} {'✓' if improved_in_range else '✗'}"
        
        print(f"{curve_type:<25} {str(params):<35} {original_str:<12} {improved_str:<12} {expected_range}")
        
        results.append({
            'curve_type': curve_type,
            'params': params,
            'score_original': score_original,
            'score_improved': score_improved,
            'expected_min': expected_range[0],
            'expected_max': expected_range[1],
            'original_passed': original_in_range,
            'improved_passed': improved_in_range,
            'improvement': score_improved - score_original
        })
    
    print("-" * 100)
    print(f"\nRESUMEN:")
    print(f"Función Original: {original_passed}/{len(test_cases)} tests pasados ({original_passed/len(test_cases)*100:.1f}%)")
    print(f"Función Mejorada: {improved_passed}/{len(test_cases)} tests pasados ({improved_passed/len(test_cases)*100:.1f}%)")
    print(f"Mejora: {(improved_passed - original_passed)/len(test_cases)*100:.1f}%")
    
    # Análisis de mejoras por tipo
    df_results = pd.DataFrame(results)
    
    print("\nMEJORA PROMEDIO POR TIPO DE CURVA:")
    print("-" * 50)
    
    improvement_by_type = df_results.groupby('curve_type').agg({
        'improvement': 'mean',
        'original_passed': 'mean',
        'improved_passed': 'mean'
    }).sort_values('improvement', ascending=False)
    
    for curve_type, row in improvement_by_type.iterrows():
        print(f"{curve_type:<20} Mejora: {row['improvement']:+.4f} | "
              f"Original: {row['original_passed']*100:.0f}% | "
              f"Mejorada: {row['improved_passed']*100:.0f}%")
    
    # Crear visualización comparativa
    plt.figure(figsize=(14, 8))
    
    # Subplot 1: Comparación de scores
    plt.subplot(2, 2, 1)
    x = range(len(df_results))
    plt.plot(x, df_results['score_original'], 'b-', label='Original', alpha=0.7)
    plt.plot(x, df_results['score_improved'], 'g-', label='Mejorada', alpha=0.7)
    plt.fill_between(x, df_results['expected_min'], df_results['expected_max'], 
                     alpha=0.2, color='gray', label='Rango esperado')
    plt.xlabel('Test Case')
    plt.ylabel('Score')
    plt.title('Comparación de Scores')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Scatter plot original vs mejorada
    plt.subplot(2, 2, 2)
    plt.scatter(df_results['score_original'], df_results['score_improved'], 
                c=df_results['improved_passed'], cmap='RdYlGn', s=100, alpha=0.7)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('Score Original')
    plt.ylabel('Score Mejorada')
    plt.title('Original vs Mejorada')
    plt.colorbar(label='Pasó test')
    plt.grid(True, alpha=0.3)
    
    # Subplot 3: Mejora por tipo
    plt.subplot(2, 2, 3)
    improvement_by_type['improvement'].plot(kind='barh')
    plt.xlabel('Mejora promedio en score')
    plt.title('Mejora por Tipo de Curva')
    plt.grid(True, alpha=0.3)
    
    # Subplot 4: Tasa de éxito por tipo
    plt.subplot(2, 2, 4)
    success_data = pd.DataFrame({
        'Original': improvement_by_type['original_passed'] * 100,
        'Mejorada': improvement_by_type['improved_passed'] * 100
    })
    success_data.plot(kind='bar', rot=45)
    plt.ylabel('Tasa de éxito (%)')
    plt.title('Tasa de Éxito por Tipo')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('/workspace/studies/function_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return df_results

def test_statistical_robustness():
    """Prueba la robustez estadística de ambas funciones"""
    generator = EquityCurveGenerator()
    
    print("\n" + "=" * 80)
    print("PRUEBA DE ROBUSTEZ ESTADÍSTICA (100 iteraciones)")
    print("=" * 80)
    
    test_configs = [
        ('Linear perfecta', lambda: generator.generate_perfect_linear(slope=0.5, noise=0.0)),
        ('Linear con ruido', lambda: generator.generate_perfect_linear(slope=0.5, noise=1.0)),
        ('Volátil positiva', lambda: generator.generate_volatile(trend=0.1, volatility=5.0)),
        ('Drawdown 30%', lambda: generator.generate_drawdown_recovery(dd_depth=0.3, dd_location=0.5)),
    ]
    
    for test_name, curve_generator in test_configs:
        scores_original = []
        scores_improved = []
        
        for _ in range(100):
            equity, trade_stats = curve_generator()
            scores_original.append(evaluate_report(equity, trade_stats)[0])
            scores_improved.append(evaluate_report_improved(equity, trade_stats)[0])
        
        scores_original = np.array(scores_original)
        scores_improved = np.array(scores_improved)
        
        # Calcular estadísticas
        print(f"\n{test_name}:")
        print(f"  Original: {np.mean(scores_original):.4f} ± {np.std(scores_original):.4f} "
              f"[{np.min(scores_original):.4f}, {np.max(scores_original):.4f}]")
        print(f"  Mejorada: {np.mean(scores_improved):.4f} ± {np.std(scores_improved):.4f} "
              f"[{np.min(scores_improved):.4f}, {np.max(scores_improved):.4f}]")
        
        # Contar scores válidos (no -1.0)
        valid_original = np.sum(scores_original > -0.5)
        valid_improved = np.sum(scores_improved > -0.5)
        print(f"  Válidos: Original {valid_original}/100, Mejorada {valid_improved}/100")

if __name__ == "__main__":
    # Ejecutar comparación principal
    df_results = compare_functions()
    
    # Ejecutar prueba de robustez
    test_statistical_robustness()
    
    # Guardar resultados
    df_results.to_csv('/workspace/studies/function_comparison_results.csv', index=False)
    print("\nResultados guardados en: function_comparison_results.csv")
    print("Gráficos guardados en: function_comparison.png")