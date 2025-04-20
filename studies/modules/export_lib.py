import os
import re
from catboost import CatBoostClassifier
from skl2onnx import convert_sklearn, update_registered_converter
from skl2onnx.common.shape_calculator import calculate_linear_classifier_output_shapes
from skl2onnx.common.data_types import FloatTensorType, Int64TensorType
from skl2onnx._parse import _apply_zipmap, _get_sklearn_operator_name
from onnx.helper import get_attribute_value
from catboost.utils import convert_to_onnx_object

# ONNX para Pipeline con Catboost
def skl2onnx_parser_castboost_classifier(scope, model, inputs, custom_parsers=None):
    options = scope.get_options(model, dict(zipmap=True))
    no_zipmap = isinstance(options["zipmap"], bool) and not options["zipmap"]
    
    alias = _get_sklearn_operator_name(type(model))
    this_operator = scope.declare_local_operator(alias, model)
    this_operator.inputs = inputs
    
    label_variable = scope.declare_local_variable("label", Int64TensorType())
    probability_tensor_variable = scope.declare_local_variable("probabilities", FloatTensorType())
    
    this_operator.outputs.append(label_variable)
    this_operator.outputs.append(probability_tensor_variable)
    
    return _apply_zipmap(options["zipmap"], scope, model, inputs[0].type, this_operator.outputs)

def skl2onnx_convert_catboost(scope, operator, container):
    onx = convert_to_onnx_object(operator.raw_operator)
    node = onx.graph.node[0]
    
    container.add_node(
        node.op_type,
        [operator.inputs[0].full_name],
        [operator.outputs[0].full_name, operator.outputs[1].full_name],
        op_domain=node.domain,
        **{att.name: get_attribute_value(att) for att in node.attribute}
    )

def export_model_to_ONNX(best_models, **kwargs):
    models = best_models
    symbol = kwargs.get('symbol')
    timeframe = kwargs.get('timeframe')
    periods = kwargs.get('periods')
    periods_meta = kwargs.get('periods_meta')
    model_number = kwargs.get('model_number')
    models_export_path = kwargs.get('models_export_path')
    include_export_path = kwargs.get('include_export_path')

    # Register the custom converter
    update_registered_converter(
        CatBoostClassifier,
        "CatBoostClassifier",
        calculate_linear_classifier_output_shapes,
        skl2onnx_convert_catboost,
        parser=skl2onnx_parser_castboost_classifier,
        options={"nocl": [True, False], "zipmap": [True, False]}
    )
    
    # Modelo señal
    model_onnx = convert_sklearn(
        models[0],
        initial_types=[('input', FloatTensorType([None, len(models[0].feature_names_)]))],
        target_opset={"": 18, "ai.onnx.ml": 2},
        options={id(models[0]): {'zipmap': True}}
    )
    filename_model = f"dmitrievsky_model_{symbol}_{timeframe}_{model_number}.onnx"
    filepath_model = os.path.join(models_export_path, filename_model)
    with open(filepath_model, "wb") as f:
        f.write(model_onnx.SerializeToString())
    print(f"Modelo {filepath_model} ONNX exportado correctamente")
    
    # Modelo meta
    model_onnx = convert_sklearn(
        models[1],
        initial_types=[('input', FloatTensorType([None, len(models[1].feature_names_)]))],
        target_opset={"": 18, "ai.onnx.ml": 2},
        options={id(models[1]): {'zipmap': True}}
    )
    filename_model_m = f"dmitrievsky_model_m_{symbol}_{timeframe}_{model_number}.onnx"
    filepath_model_m = os.path.join(models_export_path, filename_model_m)
    with open(filepath_model_m, "wb") as f:
        f.write(model_onnx.SerializeToString())
    print(f"Modelo {filepath_model_m} ONNX exportado correctamente")

    code = r"#include <Math\Stat\Math.mqh>"
    code += '\n'
    code += rf'#resource "\\Files\\{filename_model}" as uchar ExtModel_[]'
    code += '\n'
    code += rf'#resource "\\Files\\{filename_model_m}" as uchar ExtModel_m_[]'
    code += '\n\n'
    code += 'int Periods_' + '[' + str(len(periods)) + \
        '] = {' + ','.join(map(str, periods)) + '};'
    code += '\n'
    code += 'int Periods_m_' + '[' + str(len(periods_meta)) + \
        '] = {' + ','.join(map(str, periods_meta)) + '};\n\n'
    code += '#define NUM_STATS           (ArraySize(stat_ptr))\n'
    code += '#define NUM_FEATURES        (ArraySize(Periods_))\n'
    code += '#define NUM_META_FEATURES   (ArraySize(Periods_m_))\n\n'
    code += 'double stat_std(const double &a[])\n'
    code += '  {\n'
    code += '   int n = ArraySize(a);\n'
    code += '   if(n <= 1) return 0.0;\n'
    code += '   double sum = 0.0, sum_sq = 0.0;\n'
    code += '   for(int i = 0; i < n; i++)\n'
    code += '     {\n'
    code += '      sum += a[i];\n'
    code += '      sum_sq += a[i] * a[i];\n'
    code += '     }\n'
    code += '   double mean = sum / n;\n'
    code += '   return MathSqrt((sum_sq - n * mean * mean) / (n - 1));\n'
    code += '  }\n\n'

    code += 'double stat_sk(const double &a[])\n'
    code += '  {\n'
    code += '   int n = ArraySize(a);\n'
    code += '   if(n <= 1) return 0.0;\n'
    code += '   double sum = 0.0, m3 = 0.0;\n'
    code += '   for(int i = 0; i < n; i++) sum += a[i];\n'
    code += '   double mean = sum / n;\n'
    code += '   double std = stat_std(a);\n'
    code += '   if(std == 0.0) return 0.0;\n'
    code += '   for(int i = 0; i < n; i++)\n'
    code += '      m3 += MathPow((a[i] - mean) / std, 3);\n'
    code += '   return m3 / n;\n'
    code += '  }\n\n'

    code += 'double stat_kur(const double &a[])\n'
    code += '  {\n'
    code += '   int n = ArraySize(a);\n'
    code += '   if(n <= 1) return 0.0;\n'
    code += '   double sum = 0.0, m4 = 0.0;\n'
    code += '   for(int i = 0; i < n; i++) sum += a[i];\n'
    code += '   double mean = sum / n;\n'
    code += '   double std = stat_std(a);\n'
    code += '   if(std == 0.0) return 0.0;\n'
    code += '   for(int i = 0; i < n; i++)\n'
    code += '      m4 += MathPow((a[i] - mean) / std, 4);\n'
    code += '   return m4 / n - 3.0;\n'
    code += '  }\n\n'

    code += 'typedef double (*StatFunc)(const double &[]);\n'
    code += 'StatFunc stat_ptr[] = { stat_std, stat_sk, stat_kur };\n\n'

    code += 'void fill_arays(double &features[])\n'
    code += '  {\n'
    code += '   double pr[];\n'
    code += '   double stat_value;\n'
    code += '   int index = 0;\n'
    code += '   for(int i=0; i<ArraySize(Periods_); i++)\n'
    code += '     {\n'
    code += '      CopyClose(NULL, PERIOD_H1, 1, Periods_[i], pr);\n'
    code += '      ArraySetAsSeries(pr, true);\n'
    code += '      for(int j = 0; j < ArraySize(stat_ptr); j++)\n'
    code += '        {\n'
    code += '         stat_value = stat_ptr[j](pr);\n'
    code += '         features[index++] = stat_value;\n'
    code += '        }\n'
    code += '     }\n'
    code += '  }\n\n'

    code += 'void fill_arays_m(double &features[])\n'
    code += '  {\n'
    code += '   double pr[];\n'
    code += '   double stat_value;\n'
    code += '   for(int i=0; i<ArraySize(Periods_m_); i++)\n'
    code += '     {\n'
    code += '      CopyClose(NULL, PERIOD_H1, 1, Periods_m_[i], pr);\n'
    code += '      ArraySetAsSeries(pr, true);\n'
    code += '      stat_value = stat_ptr[0](pr);\n'
    code += '      features[i] = stat_value;\n'
    code += '     }\n'
    code += '  }\n\n'

    file_name = os.path.join(include_export_path, f"{symbol}_{timeframe}_ONNX_include_{model_number}.mqh")
    with open(file_name, "w") as file:
        file.write(code)
    print('The file ' + file_name + ' has been written to disk')

def remove_inner_braces_and_second_bracket(text):
    # Регулярное выражение для поиска структуры double LeafValues[N][1] = { ... };
    pattern = re.compile(r'(double LeafValues\[\d+\]\[1\] = \{)(.*?)(\};)', re.DOTALL)

    # Функция для замены внутренних фигурных скобок и удаления второй квадратной скобки
    def replace_inner_braces_and_second_bracket(match):
        inner_content = match.group(2)
        # Удаление внутренних фигурных скобок
        inner_content = re.sub(r'\{([^{}]*)\}', r'\1', inner_content)
        # Удаление второй квадратной скобки
        return match.group(1).replace('[1]', '') + inner_content + match.group(3)

    # Замена внутренних фигурных скобок и удаление второй квадратной скобки
    result = pattern.sub(replace_inner_braces_and_second_bracket, text)

    return result