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
    model_number = kwargs.get('model_number')
    symbol = kwargs.get('symbol')
    timeframe = kwargs.get('timeframe')
    stats = kwargs.get('stats')
    periods = kwargs.get('periods')
    periods_meta = kwargs.get('periods_meta')
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

    stat_function_templates = {
        "std": """
            double stat_std(const double &a[])
            {
            int n = ArraySize(a);
            if(n <= 1) return 0.0;
            double sum = 0.0, sum_sq = 0.0;
            for(int i = 0; i < n; i++)
                {
                sum += a[i];
                sum_sq += a[i] * a[i];
                }
            double mean = sum / n;
            return MathSqrt((sum_sq - n * mean * mean) / (n - 1));
            }
            """,
        "skew": """
            double stat_skew(const double &a[])
            {
            int n = ArraySize(a);
            if(n <= 1) return 0.0;
            double sum = 0.0, m3 = 0.0;
            for(int i = 0; i < n; i++) sum += a[i];
            double mean = sum / n;
            double std = stat_std(a);
            if(std == 0.0) return 0.0;
            for(int i = 0; i < n; i++)
                m3 += MathPow((a[i] - mean) / std, 3);
            return m3 / n;
            }
            """,
        "kurt": """
            double stat_kurt(const double &a[])
            {
            int n = ArraySize(a);
            if(n <= 1) return 0.0;
            double sum = 0.0, m4 = 0.0;
            for(int i = 0; i < n; i++) sum += a[i];
            double mean = sum / n;
            double std = stat_std(a);
            if(std == 0.0) return 0.0;
            for(int i = 0; i < n; i++)
                m4 += MathPow((a[i] - mean) / std, 4);
            return m4 / n - 3.0;
            }
            """,
        "zscore": """
            double stat_zscore(const double &a[])
            {
            int n = ArraySize(a);
            if(n <= 1) return 0.0;
            double sum = 0.0;
            for(int i = 0; i < n; i++) sum += a[i];
            double mean = sum / n;
            double std = stat_std(a);
            return std == 0.0 ? 0.0 : (a[0] - mean) / std;
            }
            """,
        "mean": """
            double stat_mean(const double &a[])
            {
            double sum = 0.0;
            for(int i = 0; i < ArraySize(a); i++)
                sum += a[i];
            return sum / ArraySize(a);
            }
            """,
                "range": """
            double stat_range(const double &a[])
            {
            double minv = a[0], maxv = a[0];
            for(int i = 1; i < ArraySize(a); i++)
                {
                if(a[i] < minv) minv = a[i];
                if(a[i] > maxv) maxv = a[i];
                }
            return maxv - minv;
            }
            """,
        "median": """
            double stat_median(const double &a[])
            {
            double tmp[];
            ArrayCopy(tmp, a);
            ArraySort(tmp);
            int n = ArraySize(tmp);
            if(n % 2 == 0)
                return (tmp[n/2 - 1] + tmp[n/2]) / 2.0;
            else
                return tmp[n/2];
            }
            """,
        "mad": """
            double stat_mad(const double &a[])
            {
            double mean = stat_mean(a);
            double sum = 0.0;
            for(int i = 0; i < ArraySize(a); i++)
                sum += MathAbs(a[i] - mean);
            return sum / ArraySize(a);
            }
            """,
        "var": """
            double stat_var(const double &a[])
            {
            double mean = stat_mean(a);
            double sum = 0.0;
            for(int i = 0; i < ArraySize(a); i++)
                sum += (a[i] - mean) * (a[i] - mean);
            return sum / (ArraySize(a));
            }
            """,
        "entropy": """
            double stat_entropy(const double &a[])
            {
            int bins = 10;
            double minv = a[0], maxv = a[0];
            for(int i = 1; i < ArraySize(a); i++)
                {
                if(a[i] < minv) minv = a[i];
                if(a[i] > maxv) maxv = a[i];
                }
            double width = (maxv - minv) / bins;
            if(width == 0) return 0.0;
            double hist[10] = {0};
            for(int i = 0; i < ArraySize(a); i++)
                {
                int idx = int((a[i] - minv) / width);
                if(idx == bins) idx--; // borde superior
                hist[idx]++;
                }
            double total = ArraySize(a);
            double entropy = 0.0;
            for(int i = 0; i < bins; i++)
                {
                if(hist[i] > 0)
                    entropy -= (hist[i] / total) * MathLog(hist[i] / total);
                }
            return entropy;
            }
            """,
        "slope": """
            double stat_slope(const double &a[])
                {
                    int n = ArraySize(a);
                    if(n <= 1) return 0.0;
                    double x_mean = (n - 1) / 2.0;
                    double y_mean = 0.0;
                    for(int i = 0; i < n; i++)
                        y_mean += a[i];
                    y_mean /= n;

                    double num = 0.0, den = 0.0;
                    for(int i = 0; i < n; i++)
                    {
                        num += (i - x_mean) * (a[i] - y_mean);
                        den += (i - x_mean) * (i - x_mean);
                    }

                    return den == 0.0 ? 0.0 : num / den;
                }
            """,
        "momentum": """
            double stat_momentum(const double &x[])
            {
                int size = ArraySize(x);
                if(size == 0 || x[size-1] == 0) return 0.0;
                return (x[0] / x[size-1]) - 1.0;
            }
            """,
        "roc": """
            double stat_roc(const double &x[])
            {
                int size = ArraySize(x);
                if(size < 2 || x[size-1] == 0) return 0.0;
                return ((x[0] - x[size-1]) / x[size-1]) * 100;
            }
            """,
        "fractal": """
            double stat_fractal(const double &x[])
            {
                int size = ArraySize(x);
                if(size < 2) return 1.0;
                
                double mean = 0.0;
                for(int i = 0; i < size; i++) mean += x[i];
                mean /= size;
                
                double std_dev = 0.0;
                for(int i = 0; i < size; i++) std_dev += MathPow(x[i] - mean, 2);
                std_dev = MathSqrt(std_dev / (size-1));
                
                double eps = std_dev / 4.0;
                int count = 0;
                
                for(int i = 0; i < size-1; i++) {
                    if(MathAbs(x[i] - x[i+1]) > eps) count++;
                }
                
                if(count == 0) return 1.0;
                return 1.0 + MathLog(count) / MathLog(size);
            }
            """,
        "hurst": """
            double stat_hurst(const double &x[])
            {
                int n = ArraySize(x);
                if(n < 4) return 0.5;
                
                int lags = MathMin(n-1, 20);
                double rs[];
                ArrayResize(rs, lags);
                ArrayInitialize(rs, 0.0);
                
                for(int lag = 1; lag <= lags; lag++) {
                    int window_size = n - lag;
                    if(window_size < 1) continue;
                    
                    double mean = 0.0;
                    for(int i = 0; i < window_size; i++) mean += x[i];
                    mean /= window_size;
                    
                    double std_dev = 0.0;
                    for(int i = 0; i < window_size; i++) 
                        std_dev += MathPow(x[i] - mean, 2);
                    std_dev = MathSqrt(std_dev / (window_size-1));
                    
                    if(std_dev == 0) continue;
                    
                    double max_val = x[0];
                    for(int i = 1; i < window_size; i++)
                        if(x[i] > max_val) max_val = x[i];
                    
                    rs[lag-1] = max_val / std_dev;
                }
                
                int valid_count = 0;
                for(int i = 0; i < lags; i++)
                    if(rs[i] != 0) valid_count++;
                
                if(valid_count < 2) return 0.5;
                return 0.5;
            }
            """
    }
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
    code += '#define NUM_META_FEATURES   (ArraySize(Periods_m_))\n'
    code += f'#define SYMBOL              "{str(symbol)}"\n'
    code += f'#define TIMEFRAME           "{str(timeframe)}"\n'
    code += f'#define MODEL_NUMBER        "{str(model_number)}"\n\n'
    if "mean" not in stats:
        code += stat_function_templates["mean"] + "\n"
    if "std" not in stats:
        code += stat_function_templates["std"] + "\n"
    for stat in stats:
        code += stat_function_templates[stat] + "\n"
    code += "\ntypedef double (*StatFunc)(const double &[]);\n"
    code += "StatFunc stat_ptr[] = { " + ", ".join(f"stat_{s}" for s in stats) + " };\n\n"
    code += 'void fill_arays(double &features[])\n'
    code += '  {\n'
    code += '   double pr[];\n'
    code += '   double stat_value;\n'
    code += '   int index = 0;\n'
    code += '   for(int i=0; i<ArraySize(Periods_); i++)\n'
    code += '     {\n'
    code += '      CopyClose(NULL, PERIOD_'+timeframe+ ', 1, Periods_[i], pr);\n'
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
    code += '      CopyClose(NULL, PERIOD_'+timeframe+ ', 1, Periods_m_[i], pr);\n'
    code += '      ArraySetAsSeries(pr, true);\n'
    code += '      stat_value = stat_std(pr);\n'
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