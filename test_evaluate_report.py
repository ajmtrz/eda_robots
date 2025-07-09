#!/usr/bin/env python3
"""
Script de prueba para testear la función evaluate_report optimizada
con diferentes tipos de curvas de equity.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from studies.modules.tester_lib import evaluate_report, metrics_tuple_to_dict, print_detailed_metrics

def generate_test_curves():
    """Genera diferentes tipos de curvas de equity para testing"""
    
    curves = {}
    
    # 1. Curva lineal perfecta ascendente (IDEAL)
    n = 200
    t = np.arange(n)
    curves['linear_perfect'] = {
        'equity': 1.0 + 0.02 * t,  # Pendiente constante de 0.02
        'trade_stats': np.array([40, 35, 5, 0, 0.875, 0.05, -0.02, 0.875])  # 40 trades, 87.5% win rate
    }
    
    # 2. Curva lineal con ruido moderado
    noise = np.random.normal(0, 0.01, n)
    curves['linear_noisy'] = {
        'equity': 1.0 + 0.015 * t + np.cumsum(noise),
        'trade_stats': np.array([35, 28, 7, 0, 0.8, 0.04, -0.015, 0.8])
    }
    
    # 3. Curva con muchos trades positivos (PROMOCIÓN DE TRADES)
    curves['high_trades'] = {
        'equity': 1.0 + 0.01 * t + 0.005 * np.sin(t * 0.1),
        'trade_stats': np.array([80, 65, 15, 0, 0.8125, 0.03, -0.01, 0.8125])  # 80 trades, 81.25% win rate
    }
    
    # 4. Curva con alta pendiente y ganancias (PROMOCIÓN DE GANANCIAS)
    curves['high_profit'] = {
        'equity': 1.0 + 0.05 * t,  # Pendiente alta de 0.05
        'trade_stats': np.array([25, 20, 5, 0, 0.8, 0.08, -0.02, 0.8])
    }
    
    # 5. Curva con combinación perfecta de trades y ganancias
    curves['perfect_combination'] = {
        'equity': 1.0 + 0.03 * t + 0.002 * np.sin(t * 0.05),
        'trade_stats': np.array([60, 50, 10, 0, 0.833, 0.05, -0.015, 0.833])
    }
    
    # 6. Curva con drawdown moderado
    drawdown = np.maximum(0, 0.1 * np.sin(t * 0.02))
    curves['moderate_drawdown'] = {
        'equity': 1.0 + 0.02 * t - drawdown,
        'trade_stats': np.array([30, 24, 6, 0, 0.8, 0.04, -0.02, 0.8])
    }
    
    # 7. Curva con pocos trades pero muy rentable
    curves['few_trades_high_profit'] = {
        'equity': 1.0 + 0.04 * t,
        'trade_stats': np.array([15, 12, 3, 0, 0.8, 0.1, -0.025, 0.8])
    }
    
    # 8. Curva con muchos trades pequeños
    curves['many_small_trades'] = {
        'equity': 1.0 + 0.008 * t + 0.001 * np.sin(t * 0.2),
        'trade_stats': np.array([100, 75, 25, 0, 0.75, 0.02, -0.008, 0.75])
    }
    
    return curves

def test_curve(name, equity, trade_stats):
    """Testea una curva específica y muestra los resultados"""
    print(f"\n{'='*60}")
    print(f"TESTING: {name.upper()}")
    print(f"{'='*60}")
    
    # Evaluar la curva
    metrics_tuple = evaluate_report(equity, trade_stats)
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
    
    # Plotear la curva
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(equity, 'b-', linewidth=2, label=f'Score: {metrics_dict["final_score"]:.4f}')
    plt.title(f'Equity Curve: {name}')
    plt.xlabel('Time')
    plt.ylabel('Equity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plotear trades
    plt.subplot(1, 2, 2)
    trade_frequency = trade_stats[0] / len(equity)
    positive_frequency = trade_stats[1] / len(equity)
    
    bars = plt.bar(['Total Trades', 'Positive Trades'], 
                   [trade_frequency, positive_frequency], 
                   color=['blue', 'green'], alpha=0.7)
    plt.title('Trade Activity (Normalized)')
    plt.ylabel('Frequency')
    plt.ylim(0, max(trade_frequency, positive_frequency) * 1.2)
    
    # Añadir valores en las barras
    for bar, value in zip(bars, [trade_frequency, positive_frequency]):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

def run_comprehensive_test():
    """Ejecuta una prueba comprehensiva de todas las curvas"""
    print("🚀 COMPREHENSIVE TEST OF OPTIMIZED EVALUATE_REPORT FUNCTION")
    print("="*80)
    
    # Generar curvas de prueba
    curves = generate_test_curves()
    
    # Testear cada curva
    results = {}
    for name, data in curves.items():
        test_curve(name, data['equity'], data['trade_stats'])
        results[name] = {
            'score': evaluate_report(data['equity'], data['trade_stats'])[0],
            'trades': data['trade_stats'][0],
            'positive_trades': data['trade_stats'][1],
            'win_rate': data['trade_stats'][4],
            'total_return': (data['equity'][-1] - data['equity'][0]) / data['equity'][0]
        }
    
    # Mostrar resumen comparativo
    print(f"\n{'='*80}")
    print("📊 COMPARATIVE SUMMARY")
    print(f"{'='*80}")
    
    # Crear DataFrame para comparación
    df = pd.DataFrame(results).T
    df = df.sort_values('score', ascending=False)
    
    print(df.round(4))
    
    # Análisis de correlaciones
    print(f"\n🔍 CORRELATION ANALYSIS:")
    print(f"Score vs Trades: {df['score'].corr(df['trades']):.4f}")
    print(f"Score vs Win Rate: {df['score'].corr(df['win_rate']):.4f}")
    print(f"Score vs Total Return: {df['score'].corr(df['total_return']):.4f}")
    
    # Identificar mejores estrategias
    print(f"\n🏆 TOP PERFORMERS:")
    top_3 = df.head(3)
    for i, (name, row) in enumerate(top_3.iterrows(), 1):
        print(f"{i}. {name}: Score={row['score']:.4f}, Trades={row['trades']}, WinRate={row['win_rate']:.3f}, Return={row['total_return']:.4f}")

if __name__ == "__main__":
    run_comprehensive_test()