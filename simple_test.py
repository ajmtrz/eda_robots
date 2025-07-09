#!/usr/bin/env python3
"""
Script de prueba simplificado para testear la función evaluate_report optimizada
sin dependencias externas complejas.
"""

import sys
import os
sys.path.append('/workspace/studies')

# Importar solo las funciones necesarias
try:
    from modules.tester_lib import evaluate_report, metrics_tuple_to_dict, print_detailed_metrics
except ImportError as e:
    print(f"Error importing tester_lib: {e}")
    print("Trying alternative import...")
    try:
        sys.path.append('/workspace')
        from studies.modules.tester_lib import evaluate_report, metrics_tuple_to_dict, print_detailed_metrics
    except ImportError as e2:
        print(f"Alternative import also failed: {e2}")
        sys.exit(1)

def generate_simple_test_data():
    """Genera datos de prueba simples sin numpy"""
    
    # Crear arrays simples para testing
    def create_array(values):
        return [float(x) for x in values]
    
    test_cases = {
        'linear_perfect': {
            'equity': create_array([1.0 + 0.02 * i for i in range(200)]),
            'trade_stats': [40, 35, 5, 0, 0.875, 0.05, -0.02, 0.875]
        },
        'high_trades': {
            'equity': create_array([1.0 + 0.01 * i + 0.005 * (i % 10) for i in range(200)]),
            'trade_stats': [80, 65, 15, 0, 0.8125, 0.03, -0.01, 0.8125]
        },
        'high_profit': {
            'equity': create_array([1.0 + 0.05 * i for i in range(200)]),
            'trade_stats': [25, 20, 5, 0, 0.8, 0.08, -0.02, 0.8]
        },
        'perfect_combination': {
            'equity': create_array([1.0 + 0.03 * i + 0.002 * (i % 20) for i in range(200)]),
            'trade_stats': [60, 50, 10, 0, 0.833, 0.05, -0.015, 0.833]
        }
    }
    
    return test_cases

def test_case(name, equity, trade_stats):
    """Testea un caso específico"""
    print(f"\n{'='*60}")
    print(f"TESTING: {name.upper()}")
    print(f"{'='*60}")
    
    # Convertir a arrays numpy para la función
    import numpy as np
    equity_array = np.array(equity, dtype=np.float64)
    trade_stats_array = np.array(trade_stats, dtype=np.float64)
    
    # Evaluar la curva
    metrics_tuple = evaluate_report(equity_array, trade_stats_array)
    metrics_dict = metrics_tuple_to_dict(metrics_tuple)
    
    # Mostrar métricas detalladas
    print_detailed_metrics(metrics_dict)
    
    # Mostrar estadísticas de trades
    print(f"\n📊 TRADE STATISTICS:")
    print(f"  • Total Trades: {trade_stats[0]}")
    print(f"  • Positive Trades: {trade_stats[1]}")
    print(f"  • Negative Trades: {trade_stats[2]}")
    print(f"  • Win Rate: {trade_stats[4]:.3f}")
    print(f"  • Avg Positive: {trade_stats[5]:.4f}")
    print(f"  • Avg Negative: {trade_stats[6]:.4f}")
    
    # Mostrar características de la curva
    total_return = (equity[-1] - equity[0]) / equity[0]
    slope = (equity[-1] - equity[0]) / len(equity)
    print(f"\n📈 CURVE CHARACTERISTICS:")
    print(f"  • Total Return: {total_return:.4f}")
    print(f"  • Average Slope: {slope:.6f}")
    print(f"  • Final Value: {equity[-1]:.4f}")
    
    return metrics_dict['final_score']

def run_simple_test():
    """Ejecuta una prueba simple"""
    print("🚀 SIMPLE TEST OF OPTIMIZED EVALUATE_REPORT FUNCTION")
    print("="*80)
    
    # Generar datos de prueba
    test_cases = generate_simple_test_data()
    
    # Testear cada caso
    results = {}
    for name, data in test_cases.items():
        score = test_case(name, data['equity'], data['trade_stats'])
        results[name] = {
            'score': score,
            'trades': data['trade_stats'][0],
            'positive_trades': data['trade_stats'][1],
            'win_rate': data['trade_stats'][4],
            'total_return': (data['equity'][-1] - data['equity'][0]) / data['equity'][0]
        }
    
    # Mostrar resumen comparativo
    print(f"\n{'='*80}")
    print("📊 COMPARATIVE SUMMARY")
    print(f"{'='*80}")
    
    # Ordenar por score
    sorted_results = sorted(results.items(), key=lambda x: x[1]['score'], reverse=True)
    
    print(f"{'Name':<20} {'Score':<10} {'Trades':<8} {'WinRate':<8} {'Return':<8}")
    print("-" * 60)
    
    for name, data in sorted_results:
        print(f"{name:<20} {data['score']:<10.4f} {data['trades']:<8} {data['win_rate']:<8.3f} {data['total_return']:<8.4f}")
    
    # Análisis de correlaciones
    print(f"\n🔍 ANALYSIS:")
    scores = [r[1]['score'] for r in sorted_results]
    trades = [r[1]['trades'] for r in sorted_results]
    win_rates = [r[1]['win_rate'] for r in sorted_results]
    returns = [r[1]['total_return'] for r in sorted_results]
    
    # Correlación simple
    def simple_correlation(x, y):
        if len(x) != len(y):
            return 0.0
        n = len(x)
        if n == 0:
            return 0.0
        mean_x = sum(x) / n
        mean_y = sum(y) / n
        numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
        denominator_x = sum((x[i] - mean_x) ** 2 for i in range(n))
        denominator_y = sum((y[i] - mean_y) ** 2 for i in range(n))
        if denominator_x == 0 or denominator_y == 0:
            return 0.0
        return numerator / (denominator_x * denominator_y) ** 0.5
    
    print(f"Score vs Trades: {simple_correlation(scores, trades):.4f}")
    print(f"Score vs Win Rate: {simple_correlation(scores, win_rates):.4f}")
    print(f"Score vs Total Return: {simple_correlation(scores, returns):.4f}")
    
    # Identificar mejores estrategias
    print(f"\n🏆 TOP PERFORMERS:")
    for i, (name, data) in enumerate(sorted_results[:3], 1):
        print(f"{i}. {name}: Score={data['score']:.4f}, Trades={data['trades']}, WinRate={data['win_rate']:.3f}, Return={data['total_return']:.4f}")

if __name__ == "__main__":
    run_simple_test()