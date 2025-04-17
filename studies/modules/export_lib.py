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

def export_model_to_ONNX(**kwargs):
    models = kwargs.get('best_models')
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
        models[1],
        initial_types=[('input', FloatTensorType([None, len(models[1].feature_names_)]))],
        target_opset={"": 18, "ai.onnx.ml": 2},
        options={id(models[1]): {'zipmap': True}}
    )
    filename_model = f"dmitrievsky_model_{symbol}_{timeframe}_{model_number}.onnx"
    filepath_model = os.path.join(models_export_path, filename_model)
    with open(filepath_model, "wb") as f:
        f.write(model_onnx.SerializeToString())
    print(f"Modelo {filepath_model} ONNX exportado correctamente")
    
    # Modelo meta
    model_onnx = convert_sklearn(
        models[2],
        initial_types=[('input', FloatTensorType([None, len(models[2].feature_names_)]))],
        target_opset={"": 18, "ai.onnx.ml": 2},
        options={id(models[2]): {'zipmap': True}}
    )
    filename_model_m = f"dmitrievsky_model_m_{symbol}_{timeframe}_{model_number}.onnx"
    filepath_model_m = os.path.join(models_export_path, filename_model_m)
    with open(filepath_model_m, "wb") as f:
        f.write(model_onnx.SerializeToString())
    print(f"Modelo {filepath_model_m} ONNX exportado correctamente")

    code = r"#include <Math\Stat\Math.mqh>"
    code += '\n'
    code += rf'#resource "\\Files\\{filename_model}" as uchar ExtModel_' + str(model_number) + '[]'
    code += '\n'
    code += rf'#resource "\\Files\\{filename_model_m}" as uchar ExtModel_m_' + str(model_number) + '[]'
    code += '\n\n'
    code += 'int Periods_' + str(model_number) + '[' + str(len(periods)) + \
        '] = {' + ','.join(map(str, periods)) + '};'
    code += '\n'
    code += 'int Periods_m_' + str(model_number) + '[' + str(len(periods_meta)) + \
        '] = {' + ','.join(map(str, periods_meta)) + '};'
    code += '\n\n'

    # get features
    code += 'void fill_arays_' + str(model_number) + '( double &features[]) {\n'
    code += '   double pr[], ret[];\n'
    code += '   ArrayResize(ret, 1);\n'
    code += '   for(int i=ArraySize(Periods_' + str(model_number) + ')-1; i>=0; i--) {\n'
    code += '       CopyClose(NULL,PERIOD_' + timeframe + ',1,Periods_' + str(model_number) + '[i],pr);\n'
    code += '       ret[0] = MathStandardDeviation(pr);\n'
    code += '       ArrayInsert(features, ret, ArraySize(features), 0, WHOLE_ARRAY); }\n'
    code += '   ArraySetAsSeries(features, true);\n'
    code += '}\n\n'

    # get features
    code += 'void fill_arays_m_' + str(model_number) + '( double &features[]) {\n'
    code += '   double pr[], ret[];\n'
    code += '   ArrayResize(ret, 1);\n'
    code += '   for(int i=ArraySize(Periods_m_' + str(model_number) + ')-1; i>=0; i--) {\n'
    code += '       CopyClose(NULL,PERIOD_' + timeframe + ',1,Periods_m_' + str(model_number) + '[i],pr);\n'
    code += '       ret[0] = MathStandardDeviation(pr);\n'
    code += '       ArrayInsert(features, ret, ArraySize(features), 0, WHOLE_ARRAY); }\n'
    code += '   ArraySetAsSeries(features, true);\n'
    code += '}\n\n'

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

def export_model_to_MQL4_code(**kwargs):
    model = kwargs.get('model')
    symbol = kwargs.get('symbol')
    periods = kwargs.get('periods')
    periods_meta = kwargs.get('periods_meta')
    model_number = kwargs.get('model_number')
    export_path = kwargs.get('export_path')

    model[1].save_model('catmodel.h',
                     format="cpp",
                     export_parameters=None,
                     pool=None)
    model[2].save_model('meta_catmodel.h',
                     format="cpp",
                     export_parameters=None,
                     pool=None)
    
    # add variables
    code = 'int Periods' + '[' + str(len(periods)) + \
        '] = {' + ','.join(map(str, periods)) + '};'
    code += '\n'
    code += 'int Periods_m' + '[' + str(len(periods_meta)) + \
        '] = {' + ','.join(map(str, periods_meta)) + '};'
    code += '\n\n'

    # get features
    code += 'void fill_arays' + '( double &features[]) {\n'
    code += '   double pr[];\n'
    code += '   ArrayResize(features, ArraySize(Periods));\n'
    code += '   for(int i=ArraySize(Periods)-1; i>=0; i--) {\n'
    code += '       int copyed = CopyClose(NULL,PERIOD_H1,1,Periods[i],pr);\n'
    code += '       if (copyed != Periods[i]) break;\n'
    code += '       features[i] = MathMean(pr);\n'
    code += '}\n'
    code += '}\n\n'

    # get features
    code += 'void fill_arays_m' + '( double &features[]) {\n'
    code += '   double pr[];\n'
    code += '   ArrayResize(features, ArraySize(Periods_m));\n'
    code += '   for(int i=ArraySize(Periods_m)-1; i>=0; i--) {\n'
    code += '       int copyed = CopyClose(NULL,PERIOD_H1,1,Periods_m[i],pr);\n'
    code += '       if (copyed != Periods_m[i]) break;\n'
    code += '       features[i] = MathSkewness(pr);\n'
    code += '}\n'
    code += '}\n\n'

    # add CatBosst base model
    code += 'double catboost_model' + str(model_number) + '(const double &features[]) { \n'
    code += '    '
    with open('catmodel.h', 'r') as file:
        data = file.read()
        parsed_model_tree = data[data.find("unsigned int TreeDepth")
                               :data.find("double Scale = 1;")]
        code += remove_inner_braces_and_second_bracket(parsed_model_tree)
    code += '\n\n'
    code += 'return ' + \
        'ApplyCatboostModel' + str(model_number) + '(features, TreeDepth, TreeSplits , BorderCounts, Borders, LeafValues); } \n\n'

    # add CatBosst meta model
    code += 'double catboost_meta_model' + str(model_number) + '(const double &features[]) { \n'
    code += '    '
    with open('meta_catmodel.h', 'r') as file:
        data = file.read()
        parsed_model_tree = data[data.find("unsigned int TreeDepth")
                               :data.find("double Scale = 1;")]
        code += remove_inner_braces_and_second_bracket(parsed_model_tree)
    code += '\n\n'
    code += 'return ' + \
        'ApplyCatboostModel' + str(model_number) + '(features, TreeDepth, TreeSplits , BorderCounts, Borders, LeafValues); } \n\n'

    code += 'double ApplyCatboostModel' + str(model_number) + '(const double &features[],uint &TreeDepth_[],uint &TreeSplits_[],uint &BorderCounts_[],float &Borders_[],double &LeafValues_[]) {\n\
    uint FloatFeatureCount=ArrayRange(BorderCounts_,0);\n\
    uint BinaryFeatureCount=ArrayRange(Borders_,0);\n\
    uint TreeCount=ArrayRange(TreeDepth_,0);\n\
    bool     binaryFeatures[];\n\
    ArrayResize(binaryFeatures,BinaryFeatureCount);\n\
    uint binFeatureIndex=0;\n\
    for(uint i=0; i<FloatFeatureCount; i++) {\n\
       for(uint j=0; j<BorderCounts_[i]; j++) {\n\
          binaryFeatures[binFeatureIndex]=features[i]>Borders_[binFeatureIndex];\n\
          binFeatureIndex++;\n\
       }\n\
    }\n\
    double result=0.0;\n\
    uint treeSplitsPtr=0;\n\
    uint leafValuesForCurrentTreePtr=0;\n\
    for(uint treeId=0; treeId<TreeCount; treeId++) {\n\
       uint currentTreeDepth=TreeDepth_[treeId];\n\
       uint index=0;\n\
       for(uint depth=0; depth<currentTreeDepth; depth++) {\n\
          index|=(binaryFeatures[TreeSplits_[treeSplitsPtr+depth]]<<depth);\n\
       }\n\
       result+=LeafValues_[leafValuesForCurrentTreePtr+index];\n\
       treeSplitsPtr+=currentTreeDepth;\n\
       leafValuesForCurrentTreePtr+=(1<<currentTreeDepth);\n\
    }\n\
    return 1.0/(1.0+MathPow(M_E,-result));\n\
    }\n\n'

    file = open(export_path + str(symbol) + '_model_MQL_code_' + str(model_number) + '.mqh', "w")
    file.write(code)

    file.close()
    print('The file ' + 'cat_model' + '.mqh ' + 'has been written to disc')
