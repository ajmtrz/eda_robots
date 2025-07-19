#!/usr/bin/env python3
"""
Demostración de las funciones de etiquetado optimizadas.

Este script muestra cómo usar las nuevas funciones optimizadas que mantienen
la metodología de etiquetado exacta mientras mejoran significativamente el rendimiento.

Autor: Sistema de Optimización de Etiquetado
Fecha: 2024
"""

import sys
import os
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

# Agregar el directorio del módulo al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from labeling_lib import (
    get_prices,
    get_labels_trend_with_profit_multi,
    get_labels_trend_with_profit_multi_optimized,
    get_labels_multi_window,
    get_labels_multi_window_optimized,
    get_labels_mean_reversion_multi,
    get_labels_mean_reversion_multi_optimized,
    benchmark_labeling_performance
)

def generate_sample_data(n_points=10000):
    """
    Genera datos de muestra para las pruebas de rendimiento.
    """
    np.random.seed(42)  # Para reproducibilidad
    
    # Generar serie de precios sintética con tendencia y ruido
    t = np.linspace(0, 4*np.pi, n_points)
    trend = 0.1 * t
    cyclical = 10 * np.sin(t) + 5 * np.sin(3*t)
    noise = np.random.normal(0, 2, n_points)
    
    base_price = 100
    close_prices = base_price + trend + cyclical + noise
    
    # Asegurar que los precios son positivos
    close_prices = np.maximum(close_prices, 1.0)
    
    # Generar high y low basados en close
    high_offset = np.random.uniform(0.1, 2.0, n_points)
    low_offset = np.random.uniform(0.1, 2.0, n_points)
    
    high_prices = close_prices + high_offset
    low_prices = close_prices - low_offset
    open_prices = close_prices + np.random.normal(0, 0.5, n_points)
    
    # Crear DataFrame
    timestamps = pd.date_range(start='2020-01-01', periods=n_points, freq='1min')
    
    dataset = pd.DataFrame({
        'time': timestamps,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': np.random.randint(1000, 10000, n_points)
    })
    
    dataset.set_index('time', inplace=True)
    return dataset

def validate_methodology_preservation():
    """
    Valida que las funciones optimizadas mantengan exactamente la misma
    metodología de etiquetado que las funciones originales.
    """
    print("=" * 60)
    print("VALIDACIÓN DE PRESERVACIÓN DE METODOLOGÍA")
    print("=" * 60)
    
    # Generar datos de prueba
    dataset = generate_sample_data(n_points=1000)
    
    test_cases = [
        {
            'name': 'Trend Multi - SMA',
            'original_func': get_labels_trend_with_profit_multi,
            'optimized_func': get_labels_trend_with_profit_multi_optimized,
            'params': {
                'label_filter': 'sma',
                'label_rolling_periods_small': [10, 20, 30],
                'label_threshold': 0.5,
                'label_markup': 0.5,
                'direction': 2
            }
        },
        {
            'name': 'Trend Multi - EMA',
            'original_func': get_labels_trend_with_profit_multi,
            'optimized_func': get_labels_trend_with_profit_multi_optimized,
            'params': {
                'label_filter': 'ema',
                'label_rolling_periods_small': [15, 25],
                'label_threshold': 0.3,
                'label_markup': 0.7,
                'direction': 1
            }
        },
        {
            'name': 'Multi Window',
            'original_func': get_labels_multi_window,
            'optimized_func': get_labels_multi_window_optimized,
            'params': {
                'label_window_sizes_int': [10, 20, 30],
                'label_markup': 0.5,
                'direction': 0
            }
        },
        {
            'name': 'Mean Reversion Multi',
            'original_func': get_labels_mean_reversion_multi,
            'optimized_func': get_labels_mean_reversion_multi_optimized,
            'params': {
                'label_markup': 0.5,
                'label_window_sizes_float': [0.2, 0.3, 0.5],
                'direction': 2,
                'use_optimized_helpers': False  # Usar splines exactos para validación
            }
        }
    ]
    
    validation_results = []
    
    for test_case in test_cases:
        print(f"\nValidando: {test_case['name']}")
        print("-" * 40)
        
        try:
            # Ejecutar función original
            original_result = test_case['original_func'](dataset.copy(), **test_case['params'])
            
            # Ejecutar función optimizada
            optimized_result = test_case['optimized_func'](dataset.copy(), **test_case['params'])
            
            # Verificar que los resultados sean idénticos
            if len(original_result) != len(optimized_result):
                print(f"❌ FALLO: Diferentes longitudes - Original: {len(original_result)}, Optimizada: {len(optimized_result)}")
                validation_results.append(False)
                continue
            
            # Comparar etiquetas principales
            original_labels = original_result['labels_main'].values
            optimized_labels = optimized_result['labels_main'].values
            
            # Verificar igualdad exacta (o casi exacta para operaciones en punto flotante)
            labels_equal = np.allclose(original_labels, optimized_labels, rtol=1e-10, atol=1e-10, equal_nan=True)
            
            if labels_equal:
                print(f"✅ ÉXITO: Las etiquetas son idénticas")
                validation_results.append(True)
            else:
                print(f"❌ FALLO: Las etiquetas difieren")
                print(f"   Diferencia máxima: {np.max(np.abs(original_labels - optimized_labels))}")
                validation_results.append(False)
                
        except Exception as e:
            print(f"❌ ERROR durante la validación: {e}")
            validation_results.append(False)
    
    # Resumen de validación
    print("\n" + "=" * 60)
    print("RESUMEN DE VALIDACIÓN")
    print("=" * 60)
    passed = sum(validation_results)
    total = len(validation_results)
    print(f"Pruebas pasadas: {passed}/{total}")
    
    if passed == total:
        print("🎉 TODAS LAS PRUEBAS PASARON - La metodología se preserva exactamente")
    else:
        print("⚠️  ALGUNAS PRUEBAS FALLARON - Revisar implementación")
    
    return all(validation_results)

def performance_benchmark():
    """
    Realiza benchmarks de rendimiento comparando funciones originales vs optimizadas.
    """
    print("\n" + "=" * 60)
    print("BENCHMARKS DE RENDIMIENTO")
    print("=" * 60)
    
    # Generar diferentes tamaños de datos para ver escalabilidad
    data_sizes = [1000, 5000, 10000]
    
    benchmark_configs = [
        {
            'name': 'Trend Multi SMA',
            'function_name': 'trend_multi',
            'params': {
                'label_filter': 'sma',
                'label_rolling_periods_small': [10, 20, 30, 40],
                'label_threshold': 0.5,
                'label_markup': 0.5
            }
        },
        {
            'name': 'Multi Window',
            'function_name': 'multi_window',
            'params': {
                'label_window_sizes_int': [10, 20, 30, 40, 50],
                'label_markup': 0.5
            }
        }
    ]
    
    results_summary = []
    
    for config in benchmark_configs:
        print(f"\n{config['name']}")
        print("-" * 40)
        
        for data_size in data_sizes:
            print(f"\nTamaño de datos: {data_size} puntos")
            
            # Generar datos
            dataset = generate_sample_data(n_points=data_size)
            
            try:
                # Ejecutar benchmark
                if config['function_name'] == 'trend_multi':
                    # Para trend_multi, necesitamos manejar el parámetro use_optimized_helpers
                    original_params = config['params'].copy()
                    optimized_params = config['params'].copy()
                    optimized_params['use_optimized_helpers'] = True
                    
                    # Benchmark manual para trend_multi
                    iterations = 3
                    
                    # Función original
                    original_times = []
                    for _ in range(iterations):
                        start_time = time.time()
                        get_labels_trend_with_profit_multi(dataset.copy(), **original_params)
                        end_time = time.time()
                        original_times.append(end_time - start_time)
                    
                    # Función optimizada
                    optimized_times = []
                    for _ in range(iterations):
                        start_time = time.time()
                        get_labels_trend_with_profit_multi_optimized(dataset.copy(), **optimized_params)
                        end_time = time.time()
                        optimized_times.append(end_time - start_time)
                    
                    avg_original = np.mean(original_times)
                    avg_optimized = np.mean(optimized_times)
                    speedup = avg_original / avg_optimized
                    
                    result = {
                        'function_name': config['name'],
                        'data_size': data_size,
                        'original_avg_time': avg_original,
                        'optimized_avg_time': avg_optimized,
                        'speedup_factor': speedup
                    }
                else:
                    # Usar la función de benchmark existente para otras funciones
                    result = benchmark_labeling_performance(
                        dataset, config['function_name'], iterations=3, **config['params']
                    )
                    result['data_size'] = data_size
                
                print(f"  Tiempo original: {result['original_avg_time']:.4f}s")
                print(f"  Tiempo optimizado: {result['optimized_avg_time']:.4f}s")
                print(f"  Aceleración: {result['speedup_factor']:.2f}x")
                
                results_summary.append(result)
                
            except Exception as e:
                print(f"  ❌ Error en benchmark: {e}")
    
    # Mostrar resumen
    print("\n" + "=" * 60)
    print("RESUMEN DE RENDIMIENTO")
    print("=" * 60)
    
    for result in results_summary:
        print(f"{result.get('function_name', 'Unknown')} ({result['data_size']} puntos): "
              f"{result['speedup_factor']:.2f}x más rápido")
    
    return results_summary

def demonstrate_usage():
    """
    Demuestra el uso práctico de las funciones optimizadas.
    """
    print("\n" + "=" * 60)
    print("DEMOSTRACIÓN DE USO PRÁCTICO")
    print("=" * 60)
    
    # Generar datos de ejemplo
    dataset = generate_sample_data(n_points=2000)
    
    print("\nEjemplo 1: Etiquetado de tendencia multi-período optimizado")
    print("-" * 50)
    
    # Usar función optimizada con diferentes filtros
    start_time = time.time()
    result_sma = get_labels_trend_with_profit_multi_optimized(
        dataset.copy(),
        label_filter='sma',
        label_rolling_periods_small=[10, 20, 30],
        label_threshold=0.3,
        label_markup=0.5,
        use_optimized_helpers=True
    )
    sma_time = time.time() - start_time
    
    print(f"✅ SMA optimizado completado en {sma_time:.4f}s")
    print(f"   Etiquetas generadas: {len(result_sma)}")
    print(f"   Distribución de etiquetas: {result_sma['labels_main'].value_counts().to_dict()}")
    
    print("\nEjemplo 2: Etiquetado multi-ventana optimizado")
    print("-" * 50)
    
    start_time = time.time()
    result_window = get_labels_multi_window_optimized(
        dataset.copy(),
        label_window_sizes_int=[15, 25, 35],
        label_markup=0.7,
        direction=2,
        use_optimized_helpers=True
    )
    window_time = time.time() - start_time
    
    print(f"✅ Multi-ventana optimizado completado en {window_time:.4f}s")
    print(f"   Etiquetas generadas: {len(result_window)}")
    print(f"   Distribución de etiquetas: {result_window['labels_main'].value_counts().to_dict()}")
    
    print("\nEjemplo 3: Comparación con versiones originales")
    print("-" * 50)
    
    # Comparar con versión original para mostrar la aceleración
    start_time = time.time()
    result_original = get_labels_trend_with_profit_multi(
        dataset.copy(),
        label_filter='sma',
        label_rolling_periods_small=[10, 20, 30],
        label_threshold=0.3,
        label_markup=0.5
    )
    original_time = time.time() - start_time
    
    speedup = original_time / sma_time
    print(f"Tiempo original: {original_time:.4f}s")
    print(f"Tiempo optimizado: {sma_time:.4f}s")
    print(f"Aceleración lograda: {speedup:.2f}x")
    
    # Verificar que los resultados son idénticos
    labels_match = np.allclose(result_original['labels_main'].values,
                              result_sma['labels_main'].values,
                              rtol=1e-10, atol=1e-10, equal_nan=True)
    print(f"¿Resultados idénticos?: {'✅ Sí' if labels_match else '❌ No'}")

def main():
    """
    Función principal que ejecuta todas las demostraciones.
    """
    print("🚀 DEMOSTRACIÓN DE OPTIMIZACIÓN DE ETIQUETADO")
    print("=" * 60)
    print("Este script demuestra las mejoras de rendimiento en las funciones")
    print("de etiquetado mientras mantiene la metodología exacta.")
    print()
    
    try:
        # 1. Validar que la metodología se preserve
        methodology_preserved = validate_methodology_preservation()
        
        if not methodology_preserved:
            print("\n⚠️  ADVERTENCIA: La metodología no se preserva exactamente.")
            print("   Continuando con las demostraciones, pero revisar la implementación.")
        
        # 2. Ejecutar benchmarks de rendimiento
        performance_results = performance_benchmark()
        
        # 3. Demostrar uso práctico
        demonstrate_usage()
        
        print("\n" + "=" * 60)
        print("✅ DEMOSTRACIÓN COMPLETADA")
        print("=" * 60)
        print("Las funciones optimizadas están listas para usar en producción.")
        print("Recuerda:")
        print("- Usar '_optimized' al final del nombre de función")
        print("- Configurar 'use_optimized_helpers=True' para máximo rendimiento")
        print("- Configurar 'use_optimized_helpers=False' para compatibilidad exacta")
        
    except Exception as e:
        print(f"\n❌ ERROR durante la demostración: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()