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
    model_seed = kwargs.get('best_model_seed')
    best_score = kwargs.get('best_score')
    periods_main = kwargs.get('best_periods_main')
    periods_meta = kwargs.get('best_periods_meta')
    stats_main = kwargs.get('best_stats_main')
    stats_meta = kwargs.get('best_stats_meta')
    symbol = kwargs.get('symbol')
    timeframe = kwargs.get('timeframe')
    direction = kwargs.get('direction')
    models_export_path = kwargs.get('models_export_path')
    include_export_path = kwargs.get('include_export_path')
    search_type = kwargs.get('search_type')
    search_subtype = kwargs.get('search_subtype')
    main_intervals = kwargs.get('best_main_intervals')
    meta_intervals = kwargs.get('best_meta_intervals')

    # Register the custom converter
    update_registered_converter(
        CatBoostClassifier,
        "CatBoostClassifier",
        calculate_linear_classifier_output_shapes,
        skl2onnx_convert_catboost,
        parser=skl2onnx_parser_castboost_classifier,
        options={"nocl": [True, False], "zipmap": [True, False]}
    )

    try:
        # Convierte los pipelines completos
        model_main_onnx = convert_sklearn(
            models[0],
            initial_types=[('input', FloatTensorType([None, len(models[0].feature_names_)]))],
            target_opset={"": 18, "ai.onnx.ml": 2},
            options={id(models[0]): {'zipmap': True}}
        )
        model_meta_onnx = convert_sklearn(
            models[1],
            initial_types=[('input', FloatTensorType([None, len(models[1].feature_names_)]))],
            target_opset={"": 18, "ai.onnx.ml": 2},
            options={id(models[1]): {'zipmap': True}}
        )

        # Eliminar inicializadores no utilizados
        for model_onnx in [model_main_onnx, model_meta_onnx]:
            initializers_to_remove = []
            for initializer in model_onnx.graph.initializer:
                if initializer.name == 'classes_ind':
                    initializers_to_remove.append(initializer)
            
            for initializer in initializers_to_remove:
                model_onnx.graph.initializer.remove(initializer)

        # Guarda los modelos ONNX
        filename_model = f"{symbol}_{timeframe}_{direction}_{search_type}_{search_subtype}_{model_seed}.onnx"
        filepath_model = os.path.join(models_export_path, filename_model)
        with open(filepath_model, "wb") as f:
            f.write(model_main_onnx.SerializeToString())
        
        filename_model_m = f"{symbol}_{timeframe}_{direction}_{search_type}_{search_subtype}_{model_seed}_m.onnx"
        filepath_model_m = os.path.join(models_export_path, filename_model_m)
        with open(filepath_model_m, "wb") as f:
            f.write(model_meta_onnx.SerializeToString())

        # Generar código MQL5
        code = r"#include <Math\Stat\Math.mqh>"
        code += '\n'
        code += rf'#resource "\\Files\\{filename_model}" as uchar ExtModel_[]'
        code += '\n'
        code += rf'#resource "\\Files\\{filename_model_m}" as uchar ExtModel_m_[]'
        code += '\n\n'
        code += '//+------------------------------------------------------------------+\n'
        code += f'//| BEST_SCORE           {best_score}{" " * 25} |\n'
        code += '//+------------------------------------------------------------------+\n'
        code += '\n\n'
        code += 'int periods_main' + '[' + str(len(periods_main)) + \
            '] = {' + ','.join(map(str, periods_main)) + '};'
        code += '\n'
        code += 'int periods_meta' + '[' + str(len(periods_meta)) + \
            '] = {' + ','.join(map(str, periods_meta)) + '};\n\n'
        code += '#define NUM_STATS_MAIN       (ArraySize(stat_main_ptr))\n'
        code += '#define NUM_STATS_META       (ArraySize(stat_meta_ptr))\n'
        code += '#define NUM_MAIN_FEATURES    (ArraySize(periods_main))\n'
        code += '#define NUM_META_FEATURES    (ArraySize(periods_meta))\n'
        code += f'#define SYMBOL               "{str(symbol)}"\n'
        code += f'#define TIMEFRAME            "{str(timeframe)}"\n'
        code += f'#define DIRECTION            "{str(direction)}"\n'
        code += f'#define MAGIC_NUMBER         {model_seed}\n\n'

        # Añadir código para intervalos de incertidumbre si están disponibles
        if main_intervals is not None and meta_intervals is not None:
            code += '// Intervalos de incertidumbre para el modelo principal\n'
            code += 'double main_intervals[] = {' + ','.join(map(str, main_intervals[1])) + '};\n'
            code += '// Intervalos de incertidumbre para el modelo meta\n'
            code += 'double meta_intervals[] = {' + ','.join(map(str, meta_intervals[1])) + '};\n\n'
            code += '// Función para filtrar señales con alta incertidumbre\n'
            code += 'bool filter_uncertain_signals(double main_prob, double meta_prob, int main_idx, int meta_idx)\n'
            code += '{\n'
            code += '    // Si el intervalo de incertidumbre es mayor que 0.5, descartar la señal\n'
            code += '    if(main_idx >= 0 && main_idx < ArraySize(main_intervals) && main_intervals[main_idx] > 0.5)\n'
            code += '        return false;\n'
            code += '    if(meta_idx >= 0 && meta_idx < ArraySize(meta_intervals) && meta_intervals[meta_idx] > 0.5)\n'
            code += '        return false;\n'
            code += '    return true;\n'
            code += '}\n\n'

        # Añadir el resto del código MQL5...
        stats_total = set(stats_main + stats_meta)
        if "mean" not in stats_total:
            code += stat_function_templates["mean"] + "\n"
        if "std" not in stats_total:
            code += stat_function_templates["std"] + "\n"
        if "median" not in stats_total:
            code += stat_function_templates["median"] + "\n"
        if "fisher" in stats_total:
            code += stat_function_templates["momentum"] + "\n"
        if "corr_skew" in stats_total:
            code += stat_function_templates["corr"] + "\n"
        for stat in stats_total:
            code += stat_function_templates[stat] + "\n"
        code += "\ntypedef double (*StatFunc)(const double &[]);\n"
        code += "StatFunc stat_main_ptr[] = { " + ", ".join(f"stat_{s}" for s in stats_main) + " };\n\n"
        code += "StatFunc stat_meta_ptr[] = { " + ", ".join(f"stat_{s}" for s in stats_meta) + " };\n\n"
        code += 'void fill_arays_main(double &features[])\n'
        code += '  {\n'
        code += '   double pr[];\n'
        code += '   double stat_value;\n'
        code += '   int index = 0;\n'
        code += '   for(int i=0; i<ArraySize(periods_main); i++)\n'
        code += '     {\n'
        code += '      CopyClose(NULL, PERIOD_'+timeframe+ ', 1, periods_main[i], pr);\n'
        code += '      ArraySetAsSeries(pr, true);\n'
        code += '      for(int j = 0; j < ArraySize(stat_main_ptr); j++)\n'
        code += '        {\n'
        code += '         stat_value = stat_main_ptr[j](pr);\n'
        code += '         features[index++] = stat_value;\n'
        code += '        }\n'
        code += '     }\n'
        code += '  }\n\n'

        code += 'void fill_arays_meta(double &features[])\n'
        code += '  {\n'
        code += '   double pr[];\n'
        code += '   double stat_value;\n'
        code += '   int index = 0;\n'
        code += '   for(int i=0; i<ArraySize(periods_meta); i++)\n'
        code += '     {\n'
        code += '      CopyClose(NULL, PERIOD_'+timeframe+ ', 1, periods_meta[i], pr);\n'
        code += '      ArraySetAsSeries(pr, true);\n'
        code += '      for(int j = 0; j < ArraySize(stat_meta_ptr); j++)\n'
        code += '        {\n'
        code += '         stat_value = stat_meta_ptr[j](pr);\n'
        code += '         features[index++] = stat_value;\n'
        code += '        }\n'
        code += '     }\n'
        code += '  }\n\n'

        file_name = os.path.join(include_export_path, f"{symbol}_{timeframe}_{direction}_ONNX_include_{model_seed}.mqh")
        with open(file_name, "w") as file:
            file.write(code)
        print('The file ' + file_name + ' has been written to disk')

    except Exception as e:
        print(f"Error exporting model: {str(e)}")
        raise

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