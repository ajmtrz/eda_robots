
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from tqdm import tqdm
import json
from datetime import datetime

# Añadir el path del módulo
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'modules')))
from tester_lib import evaluate_report, metrics_tuple_to_dict

PERIODS_PER_YEAR = 6240.0  # H1
N_PERIODS = 1000  # Longitud estándar de las curvas


def generate_perfect_linear_curves(n_curves=10000, seed=42):
    """Genera curvas lineales perfectas ascendentes con diferentes pendientes y ruido."""
    rng = np.random.default_rng(seed)
    curves = []
    
    for i in range(n_curves):
        # Pendiente aleatoria entre 0.01 y 1.0
        slope = rng.uniform(0.01, 1.0)
        
        # Ruido gaussiano con diferentes niveles
        noise_level = rng.uniform(0.0, 0.1 * slope)  # Ruido proporcional a la pendiente
        noise = rng.normal(0, noise_level, N_PERIODS)
        
        # Curva base lineal
        t = np.arange(N_PERIODS)
        base_curve = slope * t
        
        # Añadir ruido acumulativo suave
        cumulative_noise = np.cumsum(noise)
        curve = base_curve + cumulative_noise
        
        curves.append({
            'curve': curve,
            'slope': slope,
            'noise_level': noise_level,
            'type': 'perfect_linear'
        })
    
    return curves


def generate_exponential_curves(n_curves=10000, seed=43):
    """Genera curvas exponenciales suaves ascendentes."""
    rng = np.random.default_rng(seed)
    curves = []
    
    for i in range(n_curves):
        # Parámetro de crecimiento exponencial
        growth_rate = rng.uniform(0.001, 0.01)
        
        # Ruido
        noise_level = rng.uniform(0.0, 0.05)
        noise = rng.normal(0, noise_level, N_PERIODS)
        
        # Curva exponencial
        t = np.arange(N_PERIODS)
        base_curve = np.exp(growth_rate * t) - 1.0
        
        # Añadir ruido
        curve = base_curve + np.cumsum(noise)
        
        curves.append({
            'curve': curve,
            'growth_rate': growth_rate,
            'noise_level': noise_level,
            'type': 'exponential'
        })
    
    return curves


def generate_volatile_uptrend_curves(n_curves=10000, seed=44):
    """Genera curvas con tendencia alcista pero alta volatilidad."""
    rng = np.random.default_rng(seed)
    curves = []
    
    for i in range(n_curves):
        # Tendencia base
        base_slope = rng.uniform(0.1, 0.8)
        
        # Alta volatilidad
        volatility = rng.uniform(0.1, 0.5)
        
        # Genera la curva
        t = np.arange(N_PERIODS)
        trend = base_slope * t
        
        # Ruido volátil con sesgo positivo
        vol_noise = rng.normal(0.02, volatility, N_PERIODS)  # Sesgo positivo pequeño
        cumulative_vol = np.cumsum(vol_noise)
        
        curve = trend + cumulative_vol
        
        curves.append({
            'curve': curve,
            'base_slope': base_slope,
            'volatility': volatility,
            'type': 'volatile_uptrend'
        })
    
    return curves


def generate_drawdown_curves(n_curves=10000, seed=45):
    """Genera curvas con drawdowns en diferentes puntos."""
    rng = np.random.default_rng(seed)
    curves = []
    
    for i in range(n_curves):
        # Parámetros del drawdown
        dd_start = rng.integers(200, 600)
        dd_duration = rng.integers(50, 200)
        dd_severity = rng.uniform(0.1, 0.6)  # % del valor pico
        
        # Tendencia base
        base_slope = rng.uniform(0.05, 0.5)
        
        # Construir la curva
        t = np.arange(N_PERIODS)
        base_curve = base_slope * t
        
        # Añadir drawdown
        curve = base_curve.copy()
        if dd_start + dd_duration < N_PERIODS:
            peak_value = curve[dd_start]
            dd_depth = peak_value * dd_severity
            
            # Drawdown gradual
            for j in range(dd_duration):
                if dd_start + j < N_PERIODS:
                    progress = j / dd_duration
                    # Drawdown parabólico (primero cae rápido, luego se recupera)
                    dd_factor = 4 * progress * (1 - progress)
                    curve[dd_start + j] -= dd_depth * dd_factor
        
        # Ruido pequeño
        noise = rng.normal(0, 0.02, N_PERIODS)
        curve += np.cumsum(noise)
        
        curves.append({
            'curve': curve,
            'base_slope': base_slope,
            'dd_severity': dd_severity,
            'dd_start': dd_start,
            'dd_duration': dd_duration,
            'type': 'drawdown'
        })
    
    return curves


def generate_sideways_curves(n_curves=5000, seed=46):
    """Genera curvas laterales (sin tendencia clara)."""
    rng = np.random.default_rng(seed)
    curves = []
    
    for i in range(n_curves):
        # Volatilidad lateral
        volatility = rng.uniform(0.02, 0.2)
        
        # Drift muy pequeño o nulo
        drift = rng.uniform(-0.01, 0.01)
        
        # Construir curva lateral
        noise = rng.normal(drift, volatility, N_PERIODS)
        # Centrar alrededor de cero
        noise -= noise.mean()
        curve = np.cumsum(noise)
        
        curves.append({
            'curve': curve,
            'volatility': volatility,
            'drift': drift,
            'type': 'sideways'
        })
    
    return curves


def evaluate_curve_batch(curves_batch):
    """Evalúa un lote de curvas."""
    results = []
    
    for curve_data in curves_batch:
        try:
            curve = curve_data['curve']
            score, metrics_tuple = evaluate_report(curve, PERIODS_PER_YEAR)
            
            # Convertir métricas a dict
            metrics_dict = metrics_tuple_to_dict(score, metrics_tuple, PERIODS_PER_YEAR)
            
            # Añadir información de la curva
            result = {
                'score': score,
                'curve_type': curve_data['type'],
                **{k: v for k, v in curve_data.items() if k != 'curve'},
                **metrics_dict
            }
            
            results.append(result)
            
        except Exception as e:
            print(f"Error evaluating curve: {e}")
            continue
    
    return results


def generate_and_evaluate_massive_curves(n_total=1000000, batch_size=1000, n_workers=None):
    """Genera y evalúa millones de curvas usando procesamiento paralelo."""
    
    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)
    
    print(f"🚀 Generando y evaluando {n_total:,} curvas usando {n_workers} procesos...")
    
    # Distribuir las curvas por tipo
    n_perfect = int(n_total * 0.3)    # 30% curvas lineales perfectas
    n_exp = int(n_total * 0.2)        # 20% exponenciales
    n_volatile = int(n_total * 0.2)   # 20% volátiles alcistas
    n_dd = int(n_total * 0.2)         # 20% con drawdowns
    n_sideways = int(n_total * 0.1)   # 10% laterales
    
    all_curves = []
    
    print("📈 Generando curvas lineales perfectas...")
    all_curves.extend(generate_perfect_linear_curves(n_perfect, seed=42))
    
    print("📈 Generando curvas exponenciales...")
    all_curves.extend(generate_exponential_curves(n_exp, seed=43))
    
    print("📈 Generando curvas volátiles alcistas...")
    all_curves.extend(generate_volatile_uptrend_curves(n_volatile, seed=44))
    
    print("📈 Generando curvas con drawdowns...")
    all_curves.extend(generate_drawdown_curves(n_dd, seed=45))
    
    print("📈 Generando curvas laterales...")
    all_curves.extend(generate_sideways_curves(n_sideways, seed=46))
    
    print(f"✅ Total de curvas generadas: {len(all_curves):,}")
    
    # Mezclar aleatoriamente
    rng = np.random.default_rng(42)
    rng.shuffle(all_curves)
    
    # Dividir en lotes
    batches = [all_curves[i:i+batch_size] for i in range(0, len(all_curves), batch_size)]
    
    print(f"🔄 Procesando {len(batches)} lotes de {batch_size} curvas cada uno...")
    
    all_results = []
    
    # Procesamiento paralelo
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Enviar todos los lotes
        future_to_batch = {executor.submit(evaluate_curve_batch, batch): i 
                          for i, batch in enumerate(batches)}
        
        # Recoger resultados con barra de progreso
        for future in tqdm(as_completed(future_to_batch), total=len(batches), 
                          desc="Evaluando lotes"):
            try:
                batch_results = future.result()
                all_results.extend(batch_results)
            except Exception as e:
                print(f"Error en lote: {e}")
    
    print(f"✅ Evaluación completada. {len(all_results):,} curvas procesadas.")
    
    return all_results


def analyze_scoring_results(results):
    """Analiza los resultados del scoring masivo."""
    df = pd.DataFrame(results)
    
    print("\n" + "="*60)
    print("📊 ANÁLISIS DE RESULTADOS DEL SCORING MASIVO")
    print("="*60)
    
    # Estadísticas por tipo de curva
    print("\n🏆 RANKING PROMEDIO POR TIPO DE CURVA:")
    ranking = df.groupby('curve_type')['score'].agg(['mean', 'std', 'count']).round(4)
    ranking = ranking.sort_values('mean', ascending=False)
    print(ranking)
    
    # Top curvas
    print("\n🥇 TOP 10 CURVAS CON MAYOR PUNTUACIÓN:")
    top_curves = df.nlargest(10, 'score')[['score', 'curve_type', 'r2', 'sharpe_ratio', 
                                          'max_drawdown_relative', 'total_return']]
    print(top_curves.round(4))
    
    # Análisis de correlaciones
    print("\n🔗 CORRELACIÓN ENTRE SCORE Y MÉTRICAS CLAVE:")
    metrics = ['r2', 'sharpe_ratio', 'sortino_ratio', 'max_drawdown_relative', 
              'total_return', 'activity']
    correlations = df[metrics + ['score']].corr()['score'].drop('score').sort_values(ascending=False)
    print(correlations.round(4))
    
    # Análisis específico de curvas lineales perfectas
    linear_curves = df[df['curve_type'] == 'perfect_linear']
    if not linear_curves.empty:
        print("\n📏 ANÁLISIS DE CURVAS LINEALES PERFECTAS:")
        print(f"Puntuación promedio: {linear_curves['score'].mean():.4f}")
        print(f"R² promedio: {linear_curves['r2'].mean():.4f}")
        print(f"Sharpe promedio: {linear_curves['sharpe_ratio'].mean():.4f}")
        
        # Correlación entre pendiente y score en curvas lineales
        if 'slope' in linear_curves.columns:
            slope_score_corr = linear_curves['slope'].corr(linear_curves['score'])
            print(f"Correlación pendiente-score: {slope_score_corr:.4f}")
    
    return df, ranking


def create_visualizations(df, output_dir="curve_analysis_results"):
    """Crea visualizaciones de los resultados."""
    os.makedirs(output_dir, exist_ok=True)
    
    plt.style.use('seaborn-v0_8')
    
    # 1. Distribución de scores por tipo
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=df, x='curve_type', y='score')
    plt.xticks(rotation=45)
    plt.title('Distribución de Puntuaciones por Tipo de Curva')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/score_distribution_by_type.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Scatter plot: R² vs Score
    plt.figure(figsize=(10, 8))
    for curve_type in df['curve_type'].unique():
        subset = df[df['curve_type'] == curve_type].sample(min(1000, len(df)))
        plt.scatter(subset['r2'], subset['score'], alpha=0.6, label=curve_type, s=20)
    plt.xlabel('R² (Linealidad)')
    plt.ylabel('Score')
    plt.title('Relación entre R² y Puntuación Final')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/r2_vs_score.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Heatmap de correlaciones
    metrics = ['score', 'r2', 'sharpe_ratio', 'sortino_ratio', 'calmar_ratio',
              'max_drawdown_relative', 'total_return', 'activity']
    corr_matrix = df[metrics].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, 
                square=True, fmt='.3f')
    plt.title('Matriz de Correlación entre Métricas')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Análisis específico de curvas lineales
    linear_df = df[df['curve_type'] == 'perfect_linear']
    if not linear_df.empty and 'slope' in linear_df.columns:
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.scatter(linear_df['slope'], linear_df['score'], alpha=0.6, s=20)
        plt.xlabel('Pendiente de la Curva Lineal')
        plt.ylabel('Score')
        plt.title('Pendiente vs Puntuación (Curvas Lineales)')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.scatter(linear_df['noise_level'], linear_df['score'], alpha=0.6, s=20)
        plt.xlabel('Nivel de Ruido')
        plt.ylabel('Score')
        plt.title('Ruido vs Puntuación (Curvas Lineales)')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/linear_curves_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"📊 Visualizaciones guardadas en: {output_dir}/")


def save_results(df, ranking, output_dir="curve_analysis_results"):
    """Guarda los resultados en archivos."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Guardar DataFrame completo (muestra)
    sample_df = df.sample(min(10000, len(df)))  # Muestra para evitar archivos gigantes
    sample_df.to_csv(f'{output_dir}/curve_analysis_sample.csv', index=False)
    
    # Guardar ranking
    ranking.to_csv(f'{output_dir}/type_ranking.csv')
    
    # Guardar resumen estadístico
    summary = {
        'total_curves_analyzed': len(df),
        'analysis_date': datetime.now().isoformat(),
        'periods_per_year': PERIODS_PER_YEAR,
        'curve_length': N_PERIODS,
        'type_distribution': df['curve_type'].value_counts().to_dict(),
        'score_statistics': {
            'mean': float(df['score'].mean()),
            'std': float(df['score'].std()),
            'min': float(df['score'].min()),
            'max': float(df['score'].max()),
            'median': float(df['score'].median())
        }
    }
    
    with open(f'{output_dir}/analysis_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"💾 Resultados guardados en: {output_dir}/")


def main():
    """Función principal para ejecutar el análisis masivo."""
    print("🎯 ANÁLISIS MASIVO DE SISTEMA DE SCORING DE CURVAS")
    print("="*60)
    
    # Configuración
    n_total = 1000000  # 1 millón de curvas
    batch_size = 1000
    
    # Generar y evaluar
    results = generate_and_evaluate_massive_curves(n_total, batch_size)
    
    # Analizar
    df, ranking = analyze_scoring_results(results)
    
    # Crear visualizaciones
    create_visualizations(df)
    
    # Guardar resultados
    save_results(df, ranking)
    
    print("\n✅ ANÁLISIS COMPLETADO")
    print("Revisa los archivos generados para optimizar el sistema de scoring.")


if __name__ == "__main__":
    main()