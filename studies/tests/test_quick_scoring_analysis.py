
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Backend sin GUI
import matplotlib.pyplot as plt

# Añadir el path del módulo
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'modules')))
from tester_lib import evaluate_report, metrics_tuple_to_dict

PERIODS_PER_YEAR = 6240.0  # H1
N_PERIODS = 1000  # Longitud estándar de las curvas


def generate_test_curves():
    """Genera un conjunto pequeño de curvas de prueba."""
    curves = {}
    
    # 1. Curva lineal perfecta
    t = np.arange(N_PERIODS)
    perfect_linear = 0.5 * t  # Pendiente constante de 0.5
    curves['perfect_linear'] = perfect_linear
    
    # 2. Curva lineal con ruido mínimo
    np.random.seed(42)
    noise = np.random.normal(0, 0.02, N_PERIODS)
    linear_with_noise = 0.5 * t + np.cumsum(noise)
    curves['linear_with_noise'] = linear_with_noise
    
    # 3. Curva exponencial suave
    exponential = np.exp(0.005 * t) - 1.0
    curves['exponential'] = exponential
    
    # 4. Curva volátil pero ascendente
    trend = 0.3 * t
    volatile_noise = np.random.normal(0.01, 0.1, N_PERIODS)
    volatile_up = trend + np.cumsum(volatile_noise)
    curves['volatile_uptrend'] = volatile_up
    
    # 5. Curva con drawdown moderado
    base_trend = 0.4 * t
    # Crear un drawdown en el medio
    dd_curve = base_trend.copy()
    dd_curve[400:600] -= np.linspace(0, 50, 200)  # Drawdown
    dd_curve[600:] -= 50  # Mantener el drawdown
    curves['moderate_drawdown'] = dd_curve
    
    # 6. Curva lateral (sin tendencia)
    lateral_noise = np.random.normal(0, 0.05, N_PERIODS)
    lateral_noise -= lateral_noise.mean()  # Centrar en cero
    lateral = np.cumsum(lateral_noise)
    curves['sideways'] = lateral
    
    # 7. Curva descendente
    declining = -0.2 * t + np.cumsum(np.random.normal(0, 0.02, N_PERIODS))
    curves['declining'] = declining
    
    return curves


def quick_analysis():
    """Realiza un análisis rápido del sistema de scoring."""
    print("🧪 ANÁLISIS RÁPIDO DEL SISTEMA DE SCORING")
    print("="*50)
    
    # Generar curvas de prueba
    curves = generate_test_curves()
    
    # Evaluar cada curva
    results = []
    for name, curve in curves.items():
        try:
            score, metrics_tuple = evaluate_report(curve, PERIODS_PER_YEAR)
            metrics = metrics_tuple_to_dict(score, metrics_tuple, PERIODS_PER_YEAR)
            
            result = {
                'curve_name': name,
                'score': score,
                **metrics
            }
            results.append(result)
            
        except Exception as e:
            print(f"❌ Error evaluando {name}: {e}")
            continue
    
    # Crear DataFrame y mostrar resultados
    df = pd.DataFrame(results)
    df = df.sort_values('score', ascending=False)
    
    print("\n📊 RESULTADOS DEL SCORING:")
    print("-" * 80)
    key_cols = ['curve_name', 'score', 'r2', 'sharpe_ratio', 'max_drawdown_relative', 'total_return']
    print(df[key_cols].round(4).to_string(index=False))
    
    # Análisis de correlaciones
    print("\n🔗 CORRELACIONES CON EL SCORE:")
    metrics_cols = ['r2', 'sharpe_ratio', 'sortino_ratio', 'max_drawdown_relative', 'total_return']
    correlations = df[metrics_cols + ['score']].corr()['score'].drop('score')
    correlations = correlations.sort_values(ascending=False)
    for metric, corr in correlations.items():
        print(f"  {metric}: {corr:.4f}")
    
    # Crear visualización
    create_quick_visualization(curves, df)
    
    return df


def create_quick_visualization(curves, results_df):
    """Crea una visualización rápida de las curvas y sus puntuaciones."""
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    
    # Ordenar por score
    ordered_names = results_df.sort_values('score', ascending=False)['curve_name'].tolist()
    
    for i, name in enumerate(ordered_names):
        if i >= len(axes):
            break
            
        ax = axes[i]
        curve = curves[name]
        score = results_df[results_df['curve_name'] == name]['score'].iloc[0]
        r2 = results_df[results_df['curve_name'] == name]['r2'].iloc[0]
        
        ax.plot(curve, linewidth=1.5)
        ax.set_title(f"{name}\nScore: {score:.4f}, R²: {r2:.4f}")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Períodos")
        ax.set_ylabel("P&L")
    
    # Ocultar ejes no usados
    for i in range(len(ordered_names), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('quick_scoring_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n📊 Gráfico guardado como: quick_scoring_analysis.png")


def test_linear_progression():
    """Prueba específica para curvas lineales con diferentes pendientes."""
    print("\n🔍 PRUEBA ESPECÍFICA: CURVAS LINEALES")
    print("-" * 50)
    
    slopes = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
    results = []
    
    for slope in slopes:
        # Curva lineal perfecta
        t = np.arange(N_PERIODS)
        perfect_curve = slope * t
        
        # Curva lineal con ruido mínimo
        np.random.seed(42)
        noise = np.random.normal(0, 0.01 * slope, N_PERIODS)
        noisy_curve = perfect_curve + np.cumsum(noise)
        
        for curve_type, curve in [('perfect', perfect_curve), ('with_noise', noisy_curve)]:
            try:
                score, metrics_tuple = evaluate_report(curve, PERIODS_PER_YEAR)
                metrics = metrics_tuple_to_dict(score, metrics_tuple, PERIODS_PER_YEAR)
                
                results.append({
                    'slope': slope,
                    'curve_type': curve_type,
                    'score': score,
                    'r2': metrics['r2'],
                    'sharpe_ratio': metrics['sharpe_ratio'],
                    'total_return': metrics['total_return']
                })
                
            except Exception as e:
                print(f"❌ Error con pendiente {slope}, tipo {curve_type}: {e}")
    
    df = pd.DataFrame(results)
    
    print("\nResultados por pendiente:")
    for slope in slopes:
        slope_data = df[df['slope'] == slope]
        if not slope_data.empty:
            perfect_score = slope_data[slope_data['curve_type'] == 'perfect']['score'].iloc[0] if len(slope_data[slope_data['curve_type'] == 'perfect']) > 0 else 0
            noise_score = slope_data[slope_data['curve_type'] == 'with_noise']['score'].iloc[0] if len(slope_data[slope_data['curve_type'] == 'with_noise']) > 0 else 0
            print(f"  Pendiente {slope:5.2f}: Perfect={perfect_score:.4f}, With_noise={noise_score:.4f}")
    
    return df


def main():
    """Función principal para la prueba rápida."""
    # Análisis general
    general_results = quick_analysis()
    
    # Prueba específica de lineales
    linear_results = test_linear_progression()
    
    print("\n✅ ANÁLISIS RÁPIDO COMPLETADO")
    print("\nObservaciones clave:")
    print("1. ¿Las curvas lineales perfectas obtienen las mejores puntuaciones?")
    print("2. ¿El R² tiene alta correlación con el score?")
    print("3. ¿Las pendientes mayores obtienen mejores scores?")
    print("4. ¿El ruido penaliza significativamente el score?")
    
    return general_results, linear_results


if __name__ == "__main__":
    main()