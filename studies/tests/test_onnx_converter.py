#!/usr/bin/env python3
"""
Script de prueba para el convertidor ONNX personalizado para MQL5
"""

import numpy as np
import onnx
import onnxruntime as ort
from catboost import CatBoostRegressor
import tempfile
import os

# Importar nuestro convertidor
from modules.export_lib import convert_catboost_regression_to_mql5_compatible

def test_onnx_converter():
    """
    Prueba el convertidor ONNX personalizado
    """
    print("🧪 Iniciando prueba del convertidor ONNX personalizado...")
    
    # 1. Crear datos de prueba
    print("📊 Generando datos de prueba...")
    np.random.seed(42)
    X = np.random.randn(100, 5).astype(np.float32)
    y = np.random.randn(100).astype(np.float32)
    
    # 2. Entrenar modelo CatBoost
    print("🏋️ Entrenando modelo CatBoost...")
    model = CatBoostRegressor(iterations=50, depth=3, verbose=False)
    model.fit(X, y)
    
    # 3. Probar exportación normal (debería fallar en MQL5)
    print("📤 Exportando modelo normal...")
    normal_path = "test_normal_model.onnx"
    model.save_model(normal_path, format="onnx")
    
    # Verificar forma de salida normal
    normal_model = onnx.load(normal_path)
    normal_output_shape = normal_model.graph.output[0].type.tensor_type.shape
    print(f"📏 Forma de salida normal: {[dim.dim_value for dim in normal_output_shape.dim]}")
    
    # 4. Probar convertidor personalizado
    print("🔧 Aplicando convertidor personalizado...")
    converted_path = "test_converted_model.onnx"
    convert_catboost_regression_to_mql5_compatible(model, converted_path)
    
    # Verificar forma de salida convertida
    converted_model = onnx.load(converted_path)
    converted_output_shape = converted_model.graph.output[0].type.tensor_type.shape
    print(f"📏 Forma de salida convertida: {[dim.dim_value for dim in converted_output_shape.dim]}")
    
    # Debug adicional: verificar que el modelo sea válido
    try:
        onnx.checker.check_model(converted_model)
        print("✅ Modelo convertido es válido")
    except Exception as e:
        print(f"❌ Modelo convertido NO es válido: {e}")
    
    # Debug: mostrar información del modelo
    print(f"📊 Información del modelo convertido:")
    print(f"   - IR Version: {converted_model.ir_version}")
    print(f"   - Producer: {converted_model.producer_name}")
    print(f"   - Opset: {converted_model.opset_import[0].version}")
    print(f"   - Inputs: {len(converted_model.graph.input)}")
    print(f"   - Outputs: {len(converted_model.graph.output)}")
    print(f"   - Nodes: {len(converted_model.graph.node)}")
    
    # 5. Probar predicciones con ONNX Runtime
    print("🧮 Probando predicciones...")
    
    # Predicción con modelo normal
    session_normal = ort.InferenceSession(normal_path)
    input_name = session_normal.get_inputs()[0].name
    output_name = session_normal.get_outputs()[0].name
    
    prediction_normal = session_normal.run([output_name], {input_name: X[:5]})[0]
    print(f"📊 Predicción normal - forma: {prediction_normal.shape}, valores: {prediction_normal.flatten()[:3]}")
    
    # Predicción con modelo convertido
    session_converted = ort.InferenceSession(converted_path)
    input_name = session_converted.get_inputs()[0].name
    output_name = session_converted.get_outputs()[0].name
    
    prediction_converted = session_converted.run([output_name], {input_name: X[:5]})[0]
    print(f"📊 Predicción convertida - forma: {prediction_converted.shape}, valores: {prediction_converted.flatten()[:3]}")
    
    # 6. Verificar que las predicciones son equivalentes
    print("✅ Verificando equivalencia de predicciones...")
    normal_flat = prediction_normal.flatten()
    converted_flat = prediction_converted.flatten()
    
    if np.allclose(normal_flat, converted_flat, rtol=1e-5):
        print("✅ Las predicciones son equivalentes")
    else:
        print("❌ Las predicciones NO son equivalentes")
        print(f"   Diferencia máxima: {np.max(np.abs(normal_flat - converted_flat))}")
    
    # 7. Verificar compatibilidad con MQL5
    print("🔍 Verificando compatibilidad con MQL5...")
    if len(converted_output_shape.dim) == 2 and converted_output_shape.dim[1].dim_value == 1:
        print("✅ Forma compatible con MQL5: (N, 1)")
    else:
        print("❌ Forma NO compatible con MQL5")
    
    # 8. Limpiar archivos temporales
    print("🧹 Limpiando archivos temporales...")
    for path in [normal_path, converted_path]:
        if os.path.exists(path):
            os.remove(path)
    
    print("🎉 Prueba completada exitosamente!")

if __name__ == "__main__":
    test_onnx_converter() 