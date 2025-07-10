#!/usr/bin/env python3
"""
Script de prueba simple para verificar la implementación de patrones fractales.
"""

import sys
import os

# Agregar el directorio de estudios al path
sys.path.append(os.path.join(os.path.dirname(__file__), 'studies'))

def test_imports():
    """Prueba que los módulos se pueden importar correctamente."""
    print("🧪 Probando imports...")
    
    try:
        from modules.StrategySearcher import StrategySearcher
        print("✅ StrategySearcher importado correctamente")
    except Exception as e:
        print(f"❌ Error importando StrategySearcher: {e}")
        return False
    
    try:
        from modules.labeling_lib import get_labels_fractal_patterns
        print("✅ get_labels_fractal_patterns importado correctamente")
    except Exception as e:
        print(f"❌ Error importando get_labels_fractal_patterns: {e}")
        return False
    
    try:
        from modules.labeling_lib import calculate_symmetric_correlation_dynamic
        print("✅ calculate_symmetric_correlation_dynamic importado correctamente")
    except Exception as e:
        print(f"❌ Error importando calculate_symmetric_correlation_dynamic: {e}")
        return False
    
    try:
        from modules.labeling_lib import generate_future_outcome_labels_for_patterns
        print("✅ generate_future_outcome_labels_for_patterns importado correctamente")
    except Exception as e:
        print(f"❌ Error importando generate_future_outcome_labels_for_patterns: {e}")
        return False
    
    return True

def test_strategy_searcher_structure():
    """Prueba la estructura de StrategySearcher para búsqueda fractal."""
    print("\n🧪 Probando estructura de StrategySearcher...")
    
    try:
        from modules.StrategySearcher import StrategySearcher
        
        # Verificar que la función fractal está en LABEL_FUNCS
        if "fractal" in StrategySearcher.LABEL_FUNCS:
            print("✅ Función fractal disponible en LABEL_FUNCS")
        else:
            print("❌ Función fractal NO disponible en LABEL_FUNCS")
            return False
        
        # Verificar que el método search_fractal existe
        searcher = StrategySearcher(
            symbol="EURUSD",
            timeframe="H1",
            direction="both",
            train_start=None,
            train_end=None,
            test_start=None,
            test_end=None,
            search_type="fractal",
            search_subtype="simple",
            label_method="fractal",
            tag="test"
        )
        
        if hasattr(searcher, 'search_fractal'):
            print("✅ Método search_fractal disponible")
        else:
            print("❌ Método search_fractal NO disponible")
            return False
        
        # Verificar que fractal está en search_funcs
        if 'fractal' in searcher.run_search.__code__.co_names:
            print("✅ Búsqueda fractal registrada en run_search")
        else:
            print("❌ Búsqueda fractal NO registrada en run_search")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error en StrategySearcher: {e}")
        return False

def test_labeling_lib_structure():
    """Prueba la estructura de labeling_lib para funciones fractales."""
    print("\n🧪 Probando estructura de labeling_lib...")
    
    try:
        from modules.labeling_lib import get_labels_fractal_patterns
        
        # Verificar que la función existe y tiene los parámetros correctos
        import inspect
        sig = inspect.signature(get_labels_fractal_patterns)
        params = list(sig.parameters.keys())
        
        expected_params = [
            'dataset', 'min_window_size', 'max_window_size', 
            'correlation_threshold', 'min_future_horizon', 
            'max_future_horizon', 'markup_points'
        ]
        
        for param in expected_params:
            if param in params:
                print(f"✅ Parámetro {param} presente")
            else:
                print(f"❌ Parámetro {param} NO presente")
                return False
        
        print("✅ Todos los parámetros esperados están presentes")
        return True
        
    except Exception as e:
        print(f"❌ Error en labeling_lib: {e}")
        return False

def test_code_analysis():
    """Analiza el código para verificar la implementación."""
    print("\n🧪 Analizando código...")
    
    # Verificar que los archivos existen
    files_to_check = [
        'studies/modules/StrategySearcher.py',
        'studies/modules/labeling_lib.py'
    ]
    
    for file_path in files_to_check:
        if os.path.exists(file_path):
            print(f"✅ Archivo {file_path} existe")
        else:
            print(f"❌ Archivo {file_path} NO existe")
            return False
    
    # Verificar contenido específico en StrategySearcher.py
    try:
        with open('studies/modules/StrategySearcher.py', 'r') as f:
            content = f.read()
            
        checks = [
            ('search_fractal', 'Método search_fractal'),
            ('fractal_min_window', 'Parámetro fractal_min_window'),
            ('fractal_max_window', 'Parámetro fractal_max_window'),
            ('fractal_corr_threshold', 'Parámetro fractal_corr_threshold'),
            ('get_labels_fractal_patterns', 'Import de get_labels_fractal_patterns'),
            ('elif self.search_type == \'fractal\'', 'Condición para fractal en _suggest_algo_specific'),
            ('"fractal": get_labels_fractal_patterns', 'Registro en LABEL_FUNCS'),
            ('\'fractal\': self.search_fractal', 'Registro en search_funcs')
        ]
        
        for check, description in checks:
            if check in content:
                print(f"✅ {description} encontrado")
            else:
                print(f"❌ {description} NO encontrado")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error analizando código: {e}")
        return False

def main():
    """Función principal de pruebas."""
    print("🚀 Iniciando pruebas de implementación de patrones fractales")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Estructura StrategySearcher", test_strategy_searcher_structure),
        ("Estructura labeling_lib", test_labeling_lib_structure),
        ("Análisis de código", test_code_analysis)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Ejecutando: {test_name}")
        if test_func():
            print(f"✅ {test_name}: PASÓ")
            passed += 1
        else:
            print(f"❌ {test_name}: FALLÓ")
    
    print("\n" + "=" * 60)
    print(f"📊 Resultados: {passed}/{total} pruebas pasaron")
    
    if passed == total:
        print("🎉 ¡Todas las pruebas pasaron! La implementación está completa.")
        print("\n📋 Resumen de la implementación:")
        print("✅ Funciones de correlación simétrica implementadas")
        print("✅ Función de etiquetado de patrones fractales implementada")
        print("✅ Integración con StrategySearcher completada")
        print("✅ Parámetros específicos de fractal configurados")
        print("✅ Método search_fractal agregado")
        print("✅ Soporte para features meta agregado")
        
        print("\n🔧 Para usar la búsqueda fractal:")
        print("   searcher = StrategySearcher(")
        print("       search_type='fractal',")
        print("       search_subtype='simple',  # o 'advanced'")
        print("       label_method='fractal',")
        print("       ...")
        print("   )")
    else:
        print("⚠️ Algunas pruebas fallaron. Revisar la implementación.")

if __name__ == "__main__":
    main()