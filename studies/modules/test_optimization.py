#!/usr/bin/env python3
"""
Test rápido de las funciones optimizadas de etiquetado.
"""

import sys
import os
import numpy as np
import pandas as pd

# Agregar el directorio del módulo al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from labeling_lib import (
    get_labels_trend_with_profit_multi_optimized,
    get_labels_multi_window_optimized,
    get_labels_mean_reversion_multi_optimized
)

def create_test_data(n_points=1000):
    """Crear datos de prueba sintéticos."""
    np.random.seed(42)
    
    # Serie de precios sintética
    t = np.linspace(0, 4*np.pi, n_points)
    trend = 0.05 * t
    cyclical = 5 * np.sin(t) + 2 * np.sin(3*t)
    noise = np.random.normal(0, 1, n_points)
    
    close_prices = 100 + trend + cyclical + noise
    close_prices = np.maximum(close_prices, 1.0)  # Asegurar precios positivos
    
    high_prices = close_prices + np.random.uniform(0.1, 1.0, n_points)
    low_prices = close_prices - np.random.uniform(0.1, 1.0, n_points)
    open_prices = close_prices + np.random.normal(0, 0.3, n_points)
    
    timestamps = pd.date_range(start='2020-01-01', periods=n_points, freq='1min')
    
    dataset = pd.DataFrame({
        'time': timestamps,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': np.random.randint(1000, 5000, n_points)
    })
    
    dataset.set_index('time', inplace=True)
    return dataset

def test_trend_multi():
    """Test de la función trend multi optimizada."""
    print("Probando get_labels_trend_with_profit_multi_optimized...")
    
    dataset = create_test_data(1000)
    
    try:
        # Test con SMA (optimizado)
        result_sma = get_labels_trend_with_profit_multi_optimized(
            dataset.copy(),
            label_filter='sma',
            label_rolling_periods_small=[10, 20],
            label_threshold=0.3,
            label_markup=0.5,
            use_optimized_helpers=True
        )
        print(f"✅ SMA optimizado: {len(result_sma)} etiquetas generadas")
        
        # Test con EMA (optimizado)
        result_ema = get_labels_trend_with_profit_multi_optimized(
            dataset.copy(),
            label_filter='ema',
            label_rolling_periods_small=[15, 25],
            label_threshold=0.3,
            label_markup=0.5,
            use_optimized_helpers=True
        )
        print(f"✅ EMA optimizado: {len(result_ema)} etiquetas generadas")
        
        # Test con Savgol (fallback)
        result_savgol = get_labels_trend_with_profit_multi_optimized(
            dataset.copy(),
            label_filter='savgol',
            label_rolling_periods_small=[10, 20],
            label_polyorder=3,
            label_threshold=0.3,
            label_markup=0.5,
            use_optimized_helpers=False
        )
        print(f"✅ Savgol fallback: {len(result_savgol)} etiquetas generadas")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en trend_multi: {e}")
        return False

def test_multi_window():
    """Test de la función multi window optimizada."""
    print("\nProbando get_labels_multi_window_optimized...")
    
    dataset = create_test_data(1000)
    
    try:
        result = get_labels_multi_window_optimized(
            dataset.copy(),
            label_window_sizes_int=[10, 20, 30],
            label_markup=0.5,
            direction=2,
            use_optimized_helpers=True
        )
        print(f"✅ Multi window optimizado: {len(result)} etiquetas generadas")
        print(f"   Distribución: {result['labels_main'].value_counts().to_dict()}")
        return True
        
    except Exception as e:
        print(f"❌ Error en multi_window: {e}")
        return False

def test_mean_reversion():
    """Test de la función mean reversion optimizada."""
    print("\nProbando get_labels_mean_reversion_multi_optimized...")
    
    dataset = create_test_data(1000)
    
    try:
        # Test con aproximación optimizada
        result_fast = get_labels_mean_reversion_multi_optimized(
            dataset.copy(),
            label_markup=0.5,
            label_window_sizes_float=[0.2, 0.3],
            direction=2,
            use_optimized_helpers=True
        )
        print(f"✅ Mean reversion rápido: {len(result_fast)} etiquetas generadas")
        
        # Test con splines exactos
        result_exact = get_labels_mean_reversion_multi_optimized(
            dataset.copy(),
            label_markup=0.5,
            label_window_sizes_float=[0.2, 0.3],
            direction=2,
            use_optimized_helpers=False
        )
        print(f"✅ Mean reversion exacto: {len(result_exact)} etiquetas generadas")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en mean_reversion: {e}")
        return False

def main():
    """Ejecutar todos los tests."""
    print("🧪 TESTS DE FUNCIONES OPTIMIZADAS")
    print("=" * 50)
    
    results = []
    
    # Test individual functions
    results.append(test_trend_multi())
    results.append(test_multi_window())
    results.append(test_mean_reversion())
    
    # Summary
    print("\n" + "=" * 50)
    print("RESUMEN DE TESTS")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests pasados: {passed}/{total}")
    
    if passed == total:
        print("🎉 TODOS LOS TESTS PASARON")
        print("Las funciones optimizadas están funcionando correctamente.")
    else:
        print("⚠️  ALGUNOS TESTS FALLARON")
        print("Revisar la implementación.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)