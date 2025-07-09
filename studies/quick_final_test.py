import numpy as np
import pandas as pd
import sys
sys.path.append('/workspace/studies/modules')

from tester_lib import evaluate_report, metrics_tuple_to_dict
from test_evaluate_report import EquityCurveGenerator

def quick_validation():
    """Test rápido de validación de la función mejorada"""
    generator = EquityCurveGenerator()
    
    print("VALIDACIÓN RÁPIDA DE LA FUNCIÓN evaluate_report MEJORADA")
    print("=" * 80)
    
    # Casos de prueba específicos
    test_cases = [
        ('Lineal perfecta (pendiente baja)', 'perfect_linear', {'slope': 0.1, 'noise': 0.0}),
        ('Lineal perfecta (pendiente media)', 'perfect_linear', {'slope': 0.5, 'noise': 0.0}),
        ('Lineal perfecta (pendiente alta)', 'perfect_linear', {'slope': 2.0, 'noise': 0.0}),
        ('Lineal con ruido bajo', 'perfect_linear', {'slope': 0.5, 'noise': 0.5}),
        ('Lineal con ruido medio', 'perfect_linear', {'slope': 0.5, 'noise': 1.0}),
        ('Lineal con ruido alto', 'perfect_linear', {'slope': 0.5, 'noise': 2.0}),
        ('Drawdown 10%', 'drawdown_recovery', {'dd_depth': 0.1, 'dd_location': 0.5}),
        ('Drawdown 30%', 'drawdown_recovery', {'dd_depth': 0.3, 'dd_location': 0.5}),
        ('Drawdown 50%', 'drawdown_recovery', {'dd_depth': 0.5, 'dd_location': 0.3}),
        ('Volátil positiva', 'volatile', {'trend': 0.1, 'volatility': 5.0}),
        ('Volátil negativa', 'volatile', {'trend': -0.1, 'volatility': 5.0}),
        ('Exponencial', 'exponential', {'rate': 0.001}),
        ('Escalonada positiva', 'stepped', {'step_size': 10.0, 'step_freq': 50}),
        ('Escalonada negativa', 'stepped', {'step_size': -5.0, 'step_freq': 50}),
        ('Sinusoidal con tendencia', 'sine_trend', {'trend': 0.2, 'amplitude': 5.0, 'frequency': 0.1}),
        ('Random walk', 'random_walk', {'drift': 0.0}),
    ]
    
    results = []
    
    print(f"\n{'Descripción':<30} {'Score':<10} {'R²':<10} {'Lin':<10} {'Trades':<10} {'Estado'}")
    print("-" * 80)
    
    for desc, curve_type, params in test_cases:
        # Generar curva
        if curve_type == 'perfect_linear':
            equity, trade_stats = generator.generate_perfect_linear(**params)
        elif curve_type == 'drawdown_recovery':
            equity, trade_stats = generator.generate_drawdown_recovery(**params)
        elif curve_type == 'volatile':
            equity, trade_stats = generator.generate_volatile(**params)
        elif curve_type == 'exponential':
            equity, trade_stats = generator.generate_exponential(**params)
        elif curve_type == 'stepped':
            equity, trade_stats = generator.generate_stepped(**params)
        elif curve_type == 'sine_trend':
            equity, trade_stats = generator.generate_sine_trend(**params)
        elif curve_type == 'random_walk':
            equity, trade_stats = generator.generate_random_walk(**params)
        
        # Evaluar
        metrics_tuple = evaluate_report(equity, trade_stats)
        metrics = metrics_tuple_to_dict(metrics_tuple)
        
        score = metrics['final_score']
        r2 = metrics['r2']
        linearity = metrics['perfect_linearity']
        n_trades = int(trade_stats[0])
        
        # Determinar estado
        if score < 0:
            status = "❌ INVÁLIDO"
        elif score < 0.3:
            status = "⚠️  BAJO"
        elif score < 0.7:
            status = "✓ MEDIO"
        else:
            status = "✅ ALTO"
        
        print(f"{desc:<30} {score:<10.4f} {r2:<10.4f} {linearity:<10.4f} {n_trades:<10} {status}")
        
        results.append({
            'description': desc,
            'score': score,
            'r2': r2,
            'linearity': linearity,
            'consistency': metrics['consistency'],
            'trade_activity': metrics['trade_activity'],
            'n_trades': n_trades,
            'valid': score > -0.5
        })
    
    # Resumen
    df_results = pd.DataFrame(results)
    valid_results = df_results[df_results['valid']]
    
    print("\n" + "=" * 80)
    print("RESUMEN DE VALIDACIÓN:")
    print(f"- Tests válidos: {df_results['valid'].sum()}/{len(df_results)} ({df_results['valid'].mean()*100:.1f}%)")
    print(f"- Score promedio (válidos): {valid_results['score'].mean():.4f}")
    print(f"- Rango de scores: [{valid_results['score'].min():.4f}, {valid_results['score'].max():.4f}]")
    
    # Verificar objetivos clave
    print("\nVERIFICACIÓN DE OBJETIVOS:")
    
    # 1. Curvas lineales perfectas
    linear_perfect = df_results[df_results['description'].str.contains('perfecta')]
    if linear_perfect['score'].min() > 0.8:
        print("✅ Curvas lineales perfectas: TODAS con score > 0.8")
    else:
        print(f"⚠️  Curvas lineales perfectas: Score mínimo = {linear_perfect['score'].min():.4f}")
    
    # 2. Tolerancia al ruido
    linear_noisy = df_results[df_results['description'].str.contains('ruido')]
    if linear_noisy['valid'].all():
        print("✅ Tolerancia al ruido: TODAS las curvas con ruido son válidas")
    else:
        print(f"❌ Tolerancia al ruido: {linear_noisy['valid'].sum()}/{len(linear_noisy)} válidas")
    
    # 3. Manejo de drawdowns
    drawdown = df_results[df_results['description'].str.contains('Drawdown')]
    if drawdown['valid'].all():
        print("✅ Manejo de drawdowns: TODAS las curvas con drawdown son válidas")
        print(f"   Score promedio con drawdown: {drawdown['score'].mean():.4f}")
    else:
        print(f"❌ Manejo de drawdowns: {drawdown['valid'].sum()}/{len(drawdown)} válidas")
    
    # 4. Curvas volátiles
    volatile = df_results[df_results['description'].str.contains('Volátil')]
    if volatile['valid'].all():
        print("✅ Curvas volátiles: TODAS evaluadas correctamente (no -1.0)")
    else:
        print(f"❌ Curvas volátiles: {volatile['valid'].sum()}/{len(volatile)} válidas")
    
    return df_results

if __name__ == "__main__":
    results = quick_validation()
    
    # Guardar resultados
    results.to_csv('/workspace/studies/quick_validation_results.csv', index=False)
    print("\nResultados guardados en: quick_validation_results.csv")
