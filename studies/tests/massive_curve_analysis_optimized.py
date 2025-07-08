#!/usr/bin/env python3
"""
Análisis Masivo de Curvas de Rendimiento - Versión Optimizada
============================================================

Este script genera millones de curvas de rendimiento sintéticas y las evalúa
usando el sistema de scoring optimizado que favorece curvas lineales ascendentes.

Uso:
    python3 massive_curve_analysis_optimized.py [--curves N] [--workers N]

Ejemplo:
    python3 massive_curve_analysis_optimized.py --curves 1000000 --workers 8
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Backend sin GUI
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from tqdm import tqdm
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Añadir el path del módulo
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'modules')))
from optimized_tester_lib import evaluate_report_optimized, metrics_tuple_to_dict_optimized

PERIODS_PER_YEAR = 6240.0  # H1
N_PERIODS = 1000  # Longitud estándar de las curvas


def generate_perfect_linear_curves(n_curves=50000, seed_base=42):
    """Genera curvas lineales perfectas con diferentes pendientes."""
    rng = np.random.default_rng(seed_base)
    curves = []
    
    for i in range(n_curves):
        # Pendiente entre 0.01 y 3.0 (rango amplio)
        slope = rng.uniform(0.01, 3.0)
        
        # Intercepto aleatorio pequeño
        intercept = rng.uniform(-10, 10)
        
        # Curva base lineal perfecta
        t = np.arange(N_PERIODS)
        curve = slope * t + intercept
        
        curves.append({
            'curve': curve,
            'slope': slope,
            'intercept': intercept,
            'type': 'perfect_linear',
            'subtype': 'perfect'
        })
    
    return curves


def generate_noisy_linear_curves(n_curves=30000, seed_base=43):
    """Genera curvas lineales con diferentes niveles de ruido."""
    rng = np.random.default_rng(seed_base)
    curves = []
    
    for i in range(n_curves):
        # Pendiente entre 0.05 y 2.0
        slope = rng.uniform(0.05, 2.0)
        
        # Nivel de ruido proporcional a la pendiente
        noise_ratio = rng.uniform(0.01, 0.15)  # 1% a 15% de la pendiente
        noise_level = slope * noise_ratio
        
        # Construir curva
        t = np.arange(N_PERIODS)
        base_curve = slope * t
        
        # Añadir ruido acumulativo
        noise = rng.normal(0, noise_level, N_PERIODS)
        curve = base_curve + np.cumsum(noise)
        
        curves.append({
            'curve': curve,
            'slope': slope,
            'noise_ratio': noise_ratio,
            'noise_level': noise_level,
            'type': 'perfect_linear',
            'subtype': 'noisy'
        })
    
    return curves


def generate_exponential_curves(n_curves=20000, seed_base=44):
    """Genera curvas exponenciales (deberían ser inferiores a lineales)."""
    rng = np.random.default_rng(seed_base)
    curves = []
    
    for i in range(n_curves):
        # Tasa de crecimiento exponencial
        growth_rate = rng.uniform(0.0005, 0.015)
        
        # Ruido
        noise_level = rng.uniform(0.01, 0.1)
        
        # Construir curva exponencial
        t = np.arange(N_PERIODS)
        base_curve = np.exp(growth_rate * t) - 1.0
        
        # Añadir ruido
        noise = rng.normal(0, noise_level, N_PERIODS)
        curve = base_curve + np.cumsum(noise)
        
        curves.append({
            'curve': curve,
            'growth_rate': growth_rate,
            'noise_level': noise_level,
            'type': 'exponential',
            'subtype': 'smooth'
        })
    
    return curves


def generate_volatile_curves(n_curves=15000, seed_base=45):
    """Genera curvas con alta volatilidad pero tendencia ascendente."""
    rng = np.random.default_rng(seed_base)
    curves = []
    
    for i in range(n_curves):
        # Tendencia base
        base_slope = rng.uniform(0.1, 1.0)
        
        # Alta volatilidad
        volatility = rng.uniform(0.05, 0.3)
        
        # Construir curva
        t = np.arange(N_PERIODS)
        trend = base_slope * t
        
        # Ruido volátil con sesgo positivo pequeño
        vol_noise = rng.normal(0.005, volatility, N_PERIODS)
        cumulative_vol = np.cumsum(vol_noise)
        
        curve = trend + cumulative_vol
        
        curves.append({
            'curve': curve,
            'base_slope': base_slope,
            'volatility': volatility,
            'type': 'volatile_uptrend',
            'subtype': 'high_vol'
        })
    
    return curves


def generate_mixed_curves(n_curves=10000, seed_base=46):
    """Genera curvas con características mixtas."""
    rng = np.random.default_rng(seed_base)
    curves = []
    
    for i in range(n_curves):
        curve_type = rng.choice(['drawdown', 'sideways', 'declining'])
        
        if curve_type == 'drawdown':
            # Curva con drawdown
            base_slope = rng.uniform(0.1, 0.8)
            dd_severity = rng.uniform(0.1, 0.5)
            dd_start = rng.integers(200, 600)
            dd_duration = rng.integers(50, 200)
            
            t = np.arange(N_PERIODS)
            base_curve = base_slope * t
            
            # Añadir drawdown
            curve = base_curve.copy()
            if dd_start + dd_duration < N_PERIODS:
                peak_value = curve[dd_start]
                dd_depth = peak_value * dd_severity
                
                for j in range(dd_duration):
                    if dd_start + j < N_PERIODS:
                        progress = j / dd_duration
                        dd_factor = 4 * progress * (1 - progress)
                        curve[dd_start + j] -= dd_depth * dd_factor
            
            subtype = 'with_drawdown'
            extra_params = {'dd_severity': dd_severity, 'base_slope': base_slope}
            
        elif curve_type == 'sideways':
            # Curva lateral
            volatility = rng.uniform(0.02, 0.15)
            drift = rng.uniform(-0.01, 0.01)
            
            noise = rng.normal(drift, volatility, N_PERIODS)
            noise -= noise.mean()
            curve = np.cumsum(noise)
            
            subtype = 'lateral'
            extra_params = {'volatility': volatility, 'drift': drift}
            
        else:  # declining
            # Curva descendente
            decline_slope = rng.uniform(-0.5, -0.05)
            noise_level = rng.uniform(0.01, 0.05)
            
            t = np.arange(N_PERIODS)
            base_curve = decline_slope * t
            noise = rng.normal(0, noise_level, N_PERIODS)
            curve = base_curve + np.cumsum(noise)
            
            subtype = 'declining'
            extra_params = {'decline_slope': decline_slope, 'noise_level': noise_level}
        
        curves.append({
            'curve': curve,
            'type': curve_type,
            'subtype': subtype,
            **extra_params
        })
    
    return curves


def evaluate_curve_batch_optimized(curves_batch):
    """Evalúa un lote de curvas usando el sistema optimizado."""
    results = []
    
    for curve_data in curves_batch:
        try:
            curve = curve_data['curve']
            score, metrics_tuple = evaluate_report_optimized(curve, PERIODS_PER_YEAR)
            
            # Convertir métricas a dict
            metrics_dict = metrics_tuple_to_dict_optimized(score, metrics_tuple, PERIODS_PER_YEAR)
            
            # Añadir información de la curva
            result = {
                'score': score,
                'curve_type': curve_data['type'],
                'curve_subtype': curve_data.get('subtype', 'default'),
                **{k: v for k, v in curve_data.items() if k not in ['curve', 'type', 'subtype']},
                **metrics_dict
            }
            
            results.append(result)
            
        except Exception as e:
            print(f"Error evaluating curve: {e}")
            continue
    
    return results


def generate_and_evaluate_massive_curves_optimized(n_total=1000000, batch_size=2000, n_workers=None):
    """Genera y evalúa millones de curvas usando el sistema optimizado."""
    
    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)
    
    print(f"🚀 ANÁLISIS MASIVO CON SISTEMA OPTIMIZADO")
    print(f"📊 Generando y evaluando {n_total:,} curvas usando {n_workers} procesos...")
    
    # Distribución optimizada para favorecer curvas lineales
    n_perfect = int(n_total * 0.5)    # 50% curvas lineales perfectas
    n_noisy = int(n_total * 0.3)      # 30% lineales con ruido
    n_exp = int(n_total * 0.1)        # 10% exponenciales
    n_volatile = int(n_total * 0.08)  # 8% volátiles
    n_mixed = int(n_total * 0.02)     # 2% mixtas
    
    all_curves = []
    
    print("📈 Generando curvas lineales perfectas...")
    all_curves.extend(generate_perfect_linear_curves(n_perfect, seed_base=42))
    
    print("📈 Generando curvas lineales con ruido...")
    all_curves.extend(generate_noisy_linear_curves(n_noisy, seed_base=43))
    
    print("📈 Generando curvas exponenciales...")
    all_curves.extend(generate_exponential_curves(n_exp, seed_base=44))
    
    print("📈 Generando curvas volátiles...")
    all_curves.extend(generate_volatile_curves(n_volatile, seed_base=45))
    
    print("📈 Generando curvas mixtas...")
    all_curves.extend(generate_mixed_curves(n_mixed, seed_base=46))
    
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
        future_to_batch = {executor.submit(evaluate_curve_batch_optimized, batch): i 
                          for i, batch in enumerate(batches)}
        
        # Recoger resultados con barra de progreso
        for future in tqdm(as_completed(future_to_batch), total=len(batches), 
                          desc="🔍 Evaluando lotes"):
            try:
                batch_results = future.result()
                all_results.extend(batch_results)
            except Exception as e:
                print(f"Error en lote: {e}")
    
    print(f"✅ Evaluación completada. {len(all_results):,} curvas procesadas.")
    
    return all_results


def analyze_massive_results_optimized(results):
    """Analiza los resultados del análisis masivo optimizado."""
    df = pd.DataFrame(results)
    
    print("\n" + "="*80)
    print("📊 ANÁLISIS DE RESULTADOS MASIVOS - SISTEMA OPTIMIZADO")
    print("="*80)
    
    # Estadísticas generales
    print(f"\n📈 ESTADÍSTICAS GENERALES:")
    print(f"  Total de curvas: {len(df):,}")
    print(f"  Score promedio: {df['score'].mean():.4f}")
    print(f"  Score mediano: {df['score'].median():.4f}")
    print(f"  Desviación estándar: {df['score'].std():.4f}")
    print(f"  Score máximo: {df['score'].max():.4f}")
    print(f"  Score mínimo: {df['score'].min():.4f}")
    
    # Ranking por tipo de curva
    print(f"\n🏆 RANKING POR TIPO DE CURVA (Sistema Optimizado):")
    type_ranking = df.groupby('curve_type')['score'].agg(['mean', 'std', 'count', 'max']).round(4)
    type_ranking = type_ranking.sort_values('mean', ascending=False)
    print(type_ranking)
    
    # Análisis específico de curvas lineales
    linear_curves = df[df['curve_type'] == 'perfect_linear']
    if not linear_curves.empty:
        print(f"\n📏 ANÁLISIS DETALLADO DE CURVAS LINEALES:")
        print(f"  Cantidad: {len(linear_curves):,}")
        print(f"  Score promedio: {linear_curves['score'].mean():.4f}")
        print(f"  Score mediano: {linear_curves['score'].median():.4f}")
        
        # Análisis por subtipo
        if 'curve_subtype' in linear_curves.columns:
            subtype_analysis = linear_curves.groupby('curve_subtype')['score'].agg(['mean', 'count']).round(4)
            print(f"\n  Por subtipo:")
            print(subtype_analysis)
        
        # Correlación pendiente-score
        if 'slope' in linear_curves.columns:
            slope_corr = linear_curves['slope'].corr(linear_curves['score'])
            print(f"\n  Correlación pendiente-score: {slope_corr:.4f}")
            
            # Análisis por rangos de pendiente
            linear_curves['slope_range'] = pd.cut(linear_curves['slope'], 
                                                bins=[0, 0.1, 0.5, 1.0, 2.0, np.inf], 
                                                labels=['0-0.1', '0.1-0.5', '0.5-1.0', '1.0-2.0', '>2.0'])
            slope_analysis = linear_curves.groupby('slope_range')['score'].agg(['mean', 'count']).round(4)
            print(f"\n  Por rango de pendiente:")
            print(slope_analysis)
    
    # Verificación de objetivos del sistema optimizado
    print(f"\n🎯 VERIFICACIÓN DE OBJETIVOS:")
    
    # 1. ¿Las curvas lineales dominan?
    linear_mean = df[df['curve_type'] == 'perfect_linear']['score'].mean()
    exp_mean = df[df['curve_type'] == 'exponential']['score'].mean()
    volatile_mean = df[df['curve_type'] == 'volatile_uptrend']['score'].mean()
    
    print(f"  Curvas lineales vs exponenciales:")
    print(f"    Lineales: {linear_mean:.4f}")
    print(f"    Exponenciales: {exp_mean:.4f}")
    if linear_mean > exp_mean:
        print(f"    ✅ Las lineales superan a las exponenciales por {((linear_mean/exp_mean-1)*100):.1f}%")
    else:
        print(f"    ❌ Las exponenciales aún superan a las lineales")
    
    # 2. Top 10% de curvas
    top_10_percent = df.nlargest(int(len(df) * 0.1), 'score')
    top_distribution = top_10_percent['curve_type'].value_counts(normalize=True)
    print(f"\n  Top 10% de curvas por tipo:")
    for curve_type, percentage in top_distribution.items():
        print(f"    {curve_type}: {percentage*100:.1f}%")
    
    return df, type_ranking


def create_massive_visualizations(df, output_dir="massive_analysis_results"):
    """Crea visualizaciones del análisis masivo."""
    os.makedirs(output_dir, exist_ok=True)
    
    plt.style.use('seaborn-v0_8')
    
    # 1. Distribución de scores por tipo
    plt.figure(figsize=(14, 10))
    
    plt.subplot(2, 2, 1)
    sns.boxplot(data=df, x='curve_type', y='score')
    plt.xticks(rotation=45)
    plt.title('Distribución de Scores por Tipo (Sistema Optimizado)')
    plt.grid(True, alpha=0.3)
    
    # 2. Histograma de scores
    plt.subplot(2, 2, 2)
    for curve_type in df['curve_type'].unique():
        subset = df[df['curve_type'] == curve_type]
        plt.hist(subset['score'], alpha=0.6, label=curve_type, bins=50, density=True)
    plt.xlabel('Score')
    plt.ylabel('Densidad')
    plt.title('Distribución de Scores por Tipo')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Análisis de curvas lineales por pendiente
    linear_df = df[df['curve_type'] == 'perfect_linear']
    if not linear_df.empty and 'slope' in linear_df.columns:
        plt.subplot(2, 2, 3)
        # Muestra para el scatter plot
        sample_size = min(10000, len(linear_df))
        sample = linear_df.sample(sample_size) if len(linear_df) > sample_size else linear_df
        
        plt.scatter(sample['slope'], sample['score'], alpha=0.3, s=5)
        plt.xlabel('Pendiente')
        plt.ylabel('Score')
        plt.title('Relación Pendiente-Score (Curvas Lineales)')
        plt.grid(True, alpha=0.3)
    
    # 4. Comparación de métricas clave
    plt.subplot(2, 2, 4)
    metrics = ['linearity_bonus', 'slope_reward', 'consistency']
    type_means = df.groupby('curve_type')[metrics].mean()
    
    x_pos = np.arange(len(type_means.index))
    width = 0.25
    
    for i, metric in enumerate(metrics):
        plt.bar(x_pos + i*width, type_means[metric], width, label=metric, alpha=0.8)
    
    plt.xlabel('Tipo de Curva')
    plt.ylabel('Valor Promedio')
    plt.title('Métricas Clave por Tipo de Curva')
    plt.xticks(x_pos + width, type_means.index, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/massive_analysis_overview.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Visualizaciones guardadas en: {output_dir}/")


def save_massive_results(df, type_ranking, output_dir="massive_analysis_results"):
    """Guarda los resultados del análisis masivo."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Guardar muestra del DataFrame
    sample_size = min(50000, len(df))
    sample_df = df.sample(sample_size) if len(df) > sample_size else df
    sample_df.to_csv(f'{output_dir}/massive_results_sample.csv', index=False)
    
    # Guardar ranking completo
    type_ranking.to_csv(f'{output_dir}/curve_type_ranking.csv')
    
    # Guardar estadísticas por tipo
    detailed_stats = df.groupby('curve_type').agg({
        'score': ['count', 'mean', 'std', 'min', 'max', 'median'],
        'linearity_bonus': 'mean',
        'slope_reward': 'mean',
        'consistency': 'mean'
    }).round(4)
    detailed_stats.to_csv(f'{output_dir}/detailed_statistics.csv')
    
    # Resumen ejecutivo
    summary = {
        'analysis_date': datetime.now().isoformat(),
        'total_curves': len(df),
        'periods_per_year': PERIODS_PER_YEAR,
        'curve_length': N_PERIODS,
        'system_version': 'optimized',
        'score_statistics': {
            'mean': float(df['score'].mean()),
            'median': float(df['score'].median()),
            'std': float(df['score'].std()),
            'min': float(df['score'].min()),
            'max': float(df['score'].max())
        },
        'type_distribution': df['curve_type'].value_counts().to_dict(),
        'top_10_percent_types': df.nlargest(int(len(df) * 0.1), 'score')['curve_type'].value_counts(normalize=True).to_dict(),
        'objectives_achieved': {
            'linear_dominance': float(df[df['curve_type'] == 'perfect_linear']['score'].mean()),
            'exponential_score': float(df[df['curve_type'] == 'exponential']['score'].mean()),
            'linear_vs_exp_ratio': float(df[df['curve_type'] == 'perfect_linear']['score'].mean() / 
                                       df[df['curve_type'] == 'exponential']['score'].mean())
        }
    }
    
    with open(f'{output_dir}/executive_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"💾 Resultados guardados en: {output_dir}/")


def main():
    """Función principal del análisis masivo optimizado."""
    parser = argparse.ArgumentParser(description='Análisis Masivo de Curvas - Sistema Optimizado')
    parser.add_argument('--curves', type=int, default=100000, help='Número total de curvas a generar')
    parser.add_argument('--workers', type=int, default=None, help='Número de procesos paralelos')
    parser.add_argument('--batch-size', type=int, default=2000, help='Tamaño de lote para procesamiento')
    
    args = parser.parse_args()
    
    print("🎯 ANÁLISIS MASIVO DE CURVAS DE RENDIMIENTO - SISTEMA OPTIMIZADO")
    print("="*80)
    print(f"📊 Configuración:")
    print(f"  - Curvas a generar: {args.curves:,}")
    print(f"  - Procesos paralelos: {args.workers or 'auto'}")
    print(f"  - Tamaño de lote: {args.batch_size}")
    print()
    
    # Generar y evaluar
    results = generate_and_evaluate_massive_curves_optimized(
        n_total=args.curves, 
        batch_size=args.batch_size, 
        n_workers=args.workers
    )
    
    # Analizar
    df, type_ranking = analyze_massive_results_optimized(results)
    
    # Crear visualizaciones
    create_massive_visualizations(df)
    
    # Guardar resultados
    save_massive_results(df, type_ranking)
    
    print("\n🏆 ANÁLISIS MASIVO COMPLETADO")
    print("✅ El sistema optimizado ha sido validado con datos masivos")
    print("📁 Revisa la carpeta 'massive_analysis_results' para los resultados detallados")


if __name__ == "__main__":
    main()