
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
from optimized_tester_lib import evaluate_report_optimized, metrics_tuple_to_dict_optimized, compare_scoring_systems

PERIODS_PER_YEAR = 6240.0  # H1
N_PERIODS = 1000  # Longitud estándar de las curvas


def generate_test_curves():
    """Genera curvas de prueba específicas para validar la optimización."""
    curves = {}
    
    # 1. Curva lineal perfecta - DEBE SER LA MEJOR
    t = np.arange(N_PERIODS)
    perfect_linear = 0.5 * t  # Pendiente moderada
    curves['perfect_linear'] = perfect_linear
    
    # 2. Curva lineal con pendiente alta - DEBE SER MUY BUENA
    steep_linear = 1.5 * t  # Pendiente alta
    curves['steep_linear'] = steep_linear
    
    # 3. Curva lineal con pendiente baja - BUENA PERO MENOR
    shallow_linear = 0.1 * t  # Pendiente baja
    curves['shallow_linear'] = shallow_linear
    
    # 4. Curva lineal con ruido mínimo - BUENA
    np.random.seed(42)
    noise = np.random.normal(0, 0.02, N_PERIODS)
    linear_with_noise = 0.5 * t + np.cumsum(noise)
    curves['linear_with_noise'] = linear_with_noise
    
    # 5. Curva exponencial suave - DEBE SER INFERIOR A LINEALES
    exponential = np.exp(0.005 * t) - 1.0
    curves['exponential'] = exponential
    
    # 6. Curva volátil pero ascendente - INFERIOR POR VOLATILIDAD
    trend = 0.3 * t
    volatile_noise = np.random.normal(0.01, 0.1, N_PERIODS)
    volatile_up = trend + np.cumsum(volatile_noise)
    curves['volatile_uptrend'] = volatile_up
    
    # 7. Curva con drawdown - PENALIZADA
    base_trend = 0.4 * t
    dd_curve = base_trend.copy()
    dd_curve[400:600] -= np.linspace(0, 50, 200)  # Drawdown
    dd_curve[600:] -= 50  # Mantener el drawdown
    curves['moderate_drawdown'] = dd_curve
    
    # 8. Curva lateral - MUY BAJA
    lateral_noise = np.random.normal(0, 0.05, N_PERIODS)
    lateral_noise -= lateral_noise.mean()  # Centrar en cero
    lateral = np.cumsum(lateral_noise)
    curves['sideways'] = lateral
    
    # 9. Curva descendente - SCORE MÍNIMO
    declining = -0.2 * t + np.cumsum(np.random.normal(0, 0.02, N_PERIODS))
    curves['declining'] = declining
    
    return curves


def test_slope_preference():
    """Prueba específica para verificar que pendientes mayores obtienen mejores scores."""
    print("\n🔺 PRUEBA DE PREFERENCIA POR PENDIENTES ALTAS")
    print("-" * 60)
    
    slopes = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
    results = []
    
    t = np.arange(N_PERIODS)
    
    for slope in slopes:
        # Curva lineal perfecta con pendiente específica
        perfect_curve = slope * t
        
        # Evaluar con ambos sistemas
        orig_score, _ = evaluate_report(perfect_curve, PERIODS_PER_YEAR)
        opt_score, _ = evaluate_report_optimized(perfect_curve, PERIODS_PER_YEAR)
        
        results.append({
            'slope': slope,
            'original_score': orig_score,
            'optimized_score': opt_score,
            'improvement': opt_score - orig_score
        })
        
        print(f"Pendiente {slope:5.2f}: Original={orig_score:.4f}, Optimizado={opt_score:.4f}, Mejora={opt_score-orig_score:+.4f}")
    
    df = pd.DataFrame(results)
    
    # Verificar que las pendientes altas tengan mejores scores en el sistema optimizado
    correlation_orig = df['slope'].corr(df['original_score'])
    correlation_opt = df['slope'].corr(df['optimized_score'])
    
    print(f"\n📊 Correlación pendiente-score:")
    print(f"  Sistema Original: {correlation_orig:.4f}")
    print(f"  Sistema Optimizado: {correlation_opt:.4f}")
    
    if correlation_opt > correlation_orig:
        print("✅ ¡El sistema optimizado favorece mejor las pendientes altas!")
    else:
        print("❌ El sistema optimizado necesita más ajustes.")
    
    return df


def comprehensive_comparison():
    """Comparación comprensiva de ambos sistemas de scoring."""
    print("\n🔍 COMPARACIÓN COMPRENSIVA DE SISTEMAS DE SCORING")
    print("=" * 70)
    
    curves = generate_test_curves()
    results = []
    
    for name, curve in curves.items():
        try:
            comparison = compare_scoring_systems(curve, PERIODS_PER_YEAR)
            
            result = {
                'curve_name': name,
                'original_score': comparison['original_score'],
                'optimized_score': comparison['optimized_score'],
                'improvement': comparison['improvement'],
                'original_r2': comparison['original_metrics']['r2'],
                'optimized_r2': comparison['optimized_metrics']['r2_optimized'],
                'linearity_bonus': comparison['optimized_metrics']['linearity_bonus'],
                'slope_reward': comparison['optimized_metrics']['slope_reward']
            }
            results.append(result)
            
        except Exception as e:
            print(f"❌ Error evaluando {name}: {e}")
            continue
    
    df = pd.DataFrame(results)
    
    # Mostrar resultados ordenados por score optimizado
    df_sorted = df.sort_values('optimized_score', ascending=False)
    
    print("\n📊 RANKING DE CURVAS (Sistema Optimizado):")
    print("-" * 80)
    display_cols = ['curve_name', 'optimized_score', 'original_score', 'improvement', 'linearity_bonus', 'slope_reward']
    print(df_sorted[display_cols].round(4).to_string(index=False))
    
    # Verificar expectativas clave
    print("\n🎯 VERIFICACIÓN DE EXPECTATIVAS:")
    print("-" * 40)
    
    # 1. Las curvas lineales perfectas deben estar en el top
    linear_curves = ['perfect_linear', 'steep_linear', 'shallow_linear']
    linear_scores = df[df['curve_name'].isin(linear_curves)]['optimized_score']
    non_linear_scores = df[~df['curve_name'].isin(linear_curves)]['optimized_score']
    
    avg_linear = linear_scores.mean()
    avg_non_linear = non_linear_scores.mean()
    
    print(f"1. Curvas lineales vs no-lineales:")
    print(f"   Promedio lineales: {avg_linear:.4f}")
    print(f"   Promedio no-lineales: {avg_non_linear:.4f}")
    if avg_linear > avg_non_linear:
        print("   ✅ Las curvas lineales obtienen mejores scores")
    else:
        print("   ❌ Las curvas lineales NO obtienen mejores scores")
    
    # 2. La pendiente empinada debe superar a la exponencial
    steep_score = df[df['curve_name'] == 'steep_linear']['optimized_score'].iloc[0]
    exp_score = df[df['curve_name'] == 'exponential']['optimized_score'].iloc[0]
    
    print(f"\n2. Pendiente empinada vs exponencial:")
    print(f"   Steep linear: {steep_score:.4f}")
    print(f"   Exponential: {exp_score:.4f}")
    if steep_score > exp_score:
        print("   ✅ La curva lineal empinada supera a la exponencial")
    else:
        print("   ❌ La exponencial sigue superando a la lineal empinada")
    
    # 3. El sistema debe penalizar curvas con drawdown
    dd_score = df[df['curve_name'] == 'moderate_drawdown']['optimized_score'].iloc[0]
    perfect_score = df[df['curve_name'] == 'perfect_linear']['optimized_score'].iloc[0]
    
    print(f"\n3. Penalización por drawdown:")
    print(f"   Perfect linear: {perfect_score:.4f}")
    print(f"   With drawdown: {dd_score:.4f}")
    if perfect_score > dd_score:
        print("   ✅ El drawdown es penalizado correctamente")
    else:
        print("   ❌ El drawdown NO es suficientemente penalizado")
    
    return df


def create_comparison_visualization(df_curves, df_slopes):
    """Crea visualizaciones comparando ambos sistemas."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Comparación de scores por tipo de curva
    ax1 = axes[0, 0]
    x_pos = np.arange(len(df_curves))
    width = 0.35
    
    ax1.bar(x_pos - width/2, df_curves['original_score'], width, label='Original', alpha=0.7)
    ax1.bar(x_pos + width/2, df_curves['optimized_score'], width, label='Optimizado', alpha=0.7)
    
    ax1.set_xlabel('Tipo de Curva')
    ax1.set_ylabel('Score')
    ax1.set_title('Comparación de Scores por Tipo de Curva')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(df_curves['curve_name'], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Mejora por tipo de curva
    ax2 = axes[0, 1]
    colors = ['green' if x > 0 else 'red' for x in df_curves['improvement']]
    ax2.bar(x_pos, df_curves['improvement'], color=colors, alpha=0.7)
    ax2.set_xlabel('Tipo de Curva')
    ax2.set_ylabel('Mejora en Score')
    ax2.set_title('Mejora del Sistema Optimizado')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(df_curves['curve_name'], rotation=45, ha='right')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.grid(True, alpha=0.3)
    
    # 3. Relación pendiente-score
    ax3 = axes[1, 0]
    ax3.plot(df_slopes['slope'], df_slopes['original_score'], 'o-', label='Original', alpha=0.7)
    ax3.plot(df_slopes['slope'], df_slopes['optimized_score'], 's-', label='Optimizado', alpha=0.7)
    ax3.set_xlabel('Pendiente')
    ax3.set_ylabel('Score')
    ax3.set_title('Relación Pendiente-Score')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xscale('log')
    
    # 4. Distribución de mejoras
    ax4 = axes[1, 1]
    ax4.hist(df_curves['improvement'], bins=10, alpha=0.7, edgecolor='black')
    ax4.set_xlabel('Mejora en Score')
    ax4.set_ylabel('Frecuencia')
    ax4.set_title('Distribución de Mejoras')
    ax4.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Sin mejora')
    ax4.axvline(x=df_curves['improvement'].mean(), color='green', linestyle='--', alpha=0.7, label='Promedio')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('scoring_system_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n📊 Visualización guardada como: scoring_system_comparison.png")


def main():
    """Función principal del test."""
    print("🎯 EVALUACIÓN DEL SISTEMA DE SCORING OPTIMIZADO")
    print("="*70)
    
    # Test 1: Comparación comprensiva
    df_curves = comprehensive_comparison()
    
    # Test 2: Preferencia por pendientes
    df_slopes = test_slope_preference()
    
    # Crear visualizaciones
    create_comparison_visualization(df_curves, df_slopes)
    
    # Resumen final
    print("\n🏆 RESUMEN FINAL:")
    print("-" * 30)
    mejoras_positivas = (df_curves['improvement'] > 0).sum()
    total_curvas = len(df_curves)
    
    print(f"Curvas con mejora: {mejoras_positivas}/{total_curvas}")
    print(f"Mejora promedio: {df_curves['improvement'].mean():.4f}")
    print(f"Mejora máxima: {df_curves['improvement'].max():.4f}")
    print(f"Mejora mínima: {df_curves['improvement'].min():.4f}")
    
    # Verificar si el sistema cumple los objetivos
    linear_dominance = df_curves[df_curves['curve_name'].isin(['perfect_linear', 'steep_linear'])]['optimized_score'].mean()
    exp_score = df_curves[df_curves['curve_name'] == 'exponential']['optimized_score'].iloc[0]
    
    if linear_dominance > exp_score:
        print("\n✅ ¡ÉXITO! El sistema optimizado favorece las curvas lineales sobre las exponenciales")
    else:
        print("\n❌ El sistema necesita más ajustes para favorecer curvas lineales")
    
    print("\n✅ EVALUACIÓN COMPLETADA")


if __name__ == "__main__":
    main()