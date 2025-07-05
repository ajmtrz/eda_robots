import os
import re
import tempfile
from catboost import CatBoostClassifier
from skl2onnx import convert_sklearn, update_registered_converter
from skl2onnx.common.shape_calculator import calculate_linear_classifier_output_shapes
from skl2onnx.common.data_types import FloatTensorType, Int64TensorType
from skl2onnx._parse import _apply_zipmap, _get_sklearn_operator_name
from onnx.helper import get_attribute_value
from catboost.utils import convert_to_onnx_object
    
# ONNX para Pipeline con Catboost
def skl2onnx_parser_catboost_classifier(scope, model, inputs, custom_parsers=None):
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
def export_models_to_ONNX(models):
    """
    Convierte una lista de modelos CatBoost a ONNX.
    
    :param models: Lista de modelos CatBoost a convertir.
    :param feature_names: Nombres de las características del modelo.
    :param target_opset: Versión del opset de ONNX a utilizar.
    :return: Lista de modelos convertidos a ONNX.
    """
    # Registrar el convertidor personalizado para CatBoostClassifier
    update_registered_converter(
        CatBoostClassifier,
        "CatBoostClassifier",
        calculate_linear_classifier_output_shapes,
        skl2onnx_convert_catboost,
        parser=skl2onnx_parser_catboost_classifier,
        options={"nocl": [True, False], "zipmap": [True, False]}
    )
    # Convertir cada modelo a ONNX
    onnx_models = []
    for model in models:
        onnx_model = convert_sklearn(
            model,
            initial_types=[('input', FloatTensorType([None, len(model.feature_names_)]))],
            target_opset={"": 18, "ai.onnx.ml": 2}
        )
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".onnx")
        tmp.write(onnx_model.SerializeToString())
        tmp.close()
        onnx_models.append(tmp.name)
    return onnx_models

def export_to_mql5(**kwargs):
    best_scores = kwargs.get('best_scores')
    model_paths = kwargs.get('best_model_paths')
    model_cols = kwargs.get('best_model_cols')
    stats_main = kwargs.get('best_stats_main')
    stats_meta = kwargs.get('best_stats_meta')
    symbol = kwargs.get('symbol')
    timeframe = kwargs.get('timeframe')
    direction = kwargs.get('direction')
    model_seed = kwargs.get('best_model_seed')
    label_method = kwargs.get('label_method')
    models_export_path = kwargs.get('models_export_path')
    include_export_path = kwargs.get('include_export_path')
    search_type = kwargs.get('search_type')
    search_subtype = kwargs.get('search_subtype')

    try:
        main_cols, meta_cols = model_cols
        # Copia los modelos ONNX desde los archivos temporales a la ruta de destino
        filename_model_main = f"{symbol}_{timeframe}_{direction}_{label_method}_{search_type}_{search_subtype}_main.onnx"
        filepath_model_main = os.path.join(models_export_path, filename_model_main)
        filename_model_meta = f"{symbol}_{timeframe}_{direction}_{label_method}_{search_type}_{search_subtype}_meta.onnx"
        filepath_model_meta = os.path.join(models_export_path, filename_model_meta)

        # model_paths[0] es el modelo main, model_paths[1] es el modelo meta
        if model_paths and len(model_paths) >= 2:
            with open(model_paths[0], "rb") as src, open(filepath_model_main, "wb") as dst:
                dst.write(src.read())
            with open(model_paths[1], "rb") as src, open(filepath_model_meta, "wb") as dst:
                dst.write(src.read())
        else:
            raise ValueError("No se encontraron suficientes rutas en model_paths para copiar los modelos ONNX.")

        # Remove temporary CatBoost model files if provided
        if model_paths:
            for p in model_paths:
                try:
                    os.remove(p)
                except Exception:
                    pass

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
                    return std == 0.0 ? 0.0 : (a[n-1] - mean) / std;
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
            "iqr": """
                double stat_iqr(const double &a[])
                {
                    double tmp[];
                    ArrayCopy(tmp, a);
                    ArraySort(tmp);
                    int n = ArraySize(tmp);
                    int q1 = int((n-1)*0.25);
                    int q3 = int((n-1)*0.75);
                    return tmp[q3] - tmp[q1];
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
            "cv": """
                double stat_cv(const double &a[])
                {
                    double mean = stat_mean(a);
                    if(mean == 0.0) return 0.0;
                    double sd = stat_std(a);
                    return sd/mean;
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
                    
                    // Crear el vector de índices x
                    double x[];
                    ArrayResize(x, n);
                    for(int i = 0; i < n; i++) x[i] = i;
                    
                    // Calcular medias usando las funciones existentes
                    double x_mean = stat_mean(x);
                    double y_mean = stat_mean(a);
                    
                    // Calcular covarianza
                    double cov = 0.0;
                    for(int i = 0; i < n; i++)
                        cov += (x[i] - x_mean) * (a[i] - y_mean);
                    cov /= n;
                    
                    // Calcular varianza de x usando stat_std
                    double x_std = stat_std(x);
                    double var_x = x_std * x_std * (n - 1) / n;  // Convertir de varianza muestral a poblacional
                    
                    return var_x == 0.0 ? 0.0 : cov / var_x;
                }
                """,
            "momentum": """
                double stat_momentum(const double &a[])
                {
                    int size = ArraySize(a);
                    if(size == 0 || a[size-1] == 0) return 0.0;
                    return (a[0] / a[size-1]) - 1.0;
                }
                """,
            "fractal": """
                double stat_fractal(const double &x[])
                {
                    int size = ArraySize(x);
                    if(size < 2) return 1.0;
                    
                    double mean = stat_mean(x);
                    double std_dev = stat_std(x);
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
                    if(n < 2) return 0.5;
                    
                    // Calcular rangos reescalados
                    double valid_rs[];
                    ArrayResize(valid_rs, n-1);
                    ArrayInitialize(valid_rs, 0.0);
                    
                    for(int i = 1; i < n; i++) {
                        // Calcular media y desviación estándar para cada subserie
                        double subseries[];
                        ArrayResize(subseries, i+1);
                        ArrayCopy(subseries, x, 0, 0, i+1);
                        
                        double m = stat_mean(subseries);
                        double s = stat_std(subseries);
                        if(s == 0) continue;
                        
                        // Calcular rango reescalado
                        double max_val = subseries[0];
                        double min_val = subseries[0];
                        for(int j = 1; j < ArraySize(subseries); j++) {
                            if(subseries[j] > max_val) max_val = subseries[j];
                            if(subseries[j] < min_val) min_val = subseries[j];
                        }
                        double r = max_val - min_val;
                        valid_rs[i-1] = r / s;
                    }
                    
                    // Calcular media de los logaritmos
                    double sum_log = 0.0;
                    int count = 0;
                    for(int i = 0; i < n-1; i++) {
                        if(valid_rs[i] > 0) {
                            sum_log += MathLog(valid_rs[i]);
                            count++;
                        }
                    }
                    
                    if(count == 0 || MathAbs(MathLog(n)) < 1e-10)
                        return 0.5;
                    return sum_log / count / MathLog(n);
                }
                """,
            "autocorr": """
                double stat_autocorr(const double &a[])
                {
                    int n = ArraySize(a);
                    if(n < 2) return 0.0;
                    double mean = stat_mean(a);
                    double num=0.0, den=0.0;
                    for(int i=0;i<n-1;i++)
                    {
                        double d0 = a[i]   - mean;
                        double d1 = a[i+1] - mean;
                        num += d0 * d1;
                        den += d0 * d0;
                    }
                    return den==0.0 ? 0.0 : num/den;
                }
                """,
            "maxdd":"""
                double stat_maxdd(const double &a[])
                {
                    int n = ArraySize(a);
                    if(n==0) return 0.0;
                    double peak = a[0];
                    double maxdd = 0.0;
                    for(int i=0;i<n;i++)
                    {
                        if(a[i] > peak) peak = a[i];
                        double dd = (peak - a[i]) / peak;
                        if(dd > maxdd) maxdd = dd;
                    }
                    return maxdd;
                }
            """,
            "sharpe": """
                double stat_sharpe(const double &a[])
                {
                    int n = ArraySize(a);
                    if(n < 2) return 0.0;
                    double sum=0.0, sum2=0.0;
                    for(int i=1;i<n;i++)
                    {
                        double r = a[i]/a[i-1] - 1.0;
                        sum  += r;
                        sum2 += r*r;
                    }
                    int m = n-1;
                    double mean = sum / m;
                    double std = stat_std(a);
                    return std==0.0 ? 0.0 : mean/std;
                }
            """,
            "fisher": """
                double stat_fisher(const double &a[])
                {
                    int n = ArraySize(a);
                    if(n <= 1) return 0.0;
                    
                    // Usar la función stat_momentum ya definida
                    double momentum = stat_momentum(a);
                    
                    // Aplicar transformación de Fisher
                    double x = MathMax(-0.9999, MathMin(0.9999, momentum));
                    return 0.5 * MathLog((1.0 + x)/(1.0 - x));
                }
                """,
            "chande": """
                double stat_chande(const double &a[])
                {
                    int n = ArraySize(a);
                    if(n < 2) return 0.0;
                    double up=0.0, down=0.0;
                    for(int i=1;i<n;i++)
                    {
                        double diff = a[i] - a[i-1];
                        if(diff > 0) up += diff;
                        else         down -= diff;  // diff es negativo
                    }
                    double sum = up + down;
                    return sum==0.0 ? 0.0 : (up - down)/sum;
                }
            """,
            "approxentropy": """
                double stat_approxentropy(const double &a[])
                {
                    int n = ArraySize(a);
                    int m = 2;
                    if(n <= m + 1) return 0.0;
                    
                    double sd = stat_std(a);
                    double r = 0.2 * sd;
                    
                    // Para m = 2
                    int count_m = 0;
                    for(int i = 0; i < n - 1; i++) {
                        for(int j = 0; j < n - 1; j++) {
                            if(i != j) {  // Excluir i == j como en Python
                                bool match_m = true;
                                for(int k = 0; k < m; k++) {
                                    if(MathAbs(a[i+k] - a[j+k]) > r) {
                                        match_m = false;
                                        break;
                                    }
                                }
                                if(match_m) count_m++;
                            }
                        }
                    }
                    double phi1 = count_m > 0 ? MathLog((double)count_m / (n - 1)) : 0.0;
                    
                    // Para m = 3
                    int count_m1 = 0;
                    for(int i = 0; i < n - 2; i++) {
                        for(int j = 0; j < n - 2; j++) {
                            if(i != j) {  // Excluir i == j como en Python
                                bool match_m1 = true;
                                for(int k = 0; k < m + 1; k++) {
                                    if(MathAbs(a[i+k] - a[j+k]) > r) {
                                        match_m1 = false;
                                        break;
                                    }
                                }
                                if(match_m1) count_m1++;
                            }
                        }
                    }
                    double phi2 = count_m1 > 0 ? MathLog((double)count_m1 / (n - 2)) : 0.0;
                    
                    return phi1 - phi2;
                }
                """,
            "effratio": """
                double stat_effratio(const double &a[])
                {
                    int n = ArraySize(a);
                    if(n < 2) return 0.0;
                    double directio = a[n-1] - a[0];
                    double volatility = 0.0;
                    for(int i=1;i<n;i++) volatility += MathAbs(a[i]-a[i-1]);
                    return volatility==0.0 ? 0.0 : directio/volatility;
                }
            """,
            "corr": """
            double stat_corr(const double &a[], const double &b[])
            {
                int n = ArraySize(a);
                if(n != ArraySize(b) || n < 2) return 0.0;
                
                // Calcular medias usando stat_mean
                double mean_a = stat_mean(a);
                double mean_b = stat_mean(b);
                
                // Calcular covarianza
                double cov = 0.0;
                for(int i=0; i<n; i++) {
                    cov += (a[i] - mean_a) * (b[i] - mean_b);
                }
                
                // Calcular desviaciones estándar usando stat_std
                double std_a = stat_std(a);
                double std_b = stat_std(b);
                
                if(std_a == 0.0 || std_b == 0.0) return 0.0;
                
                return cov / (n * std_a * std_b);
            }
            """,
            "corrskew": """
            double stat_corrskew(const double &a[])
            {
                int n = ArraySize(a);
                int lag = MathMin(5, n/2);
                if(n < lag+1) return 0.0;
                
                // Preparar arrays para correlación positiva
                double x1[], y1[];
                ArrayResize(x1, n-lag);
                ArrayResize(y1, n-lag);
                for(int i=0; i<n-lag; i++) {
                    x1[i] = a[i];
                    y1[i] = a[i+lag];
                }
                
                // Preparar arrays para correlación negativa
                double x2[], y2[];
                ArrayResize(x2, n-lag);
                ArrayResize(y2, n-lag);
                for(int i=0; i<n-lag; i++) {
                    x2[i] = -a[i];
                    y2[i] = a[i+lag];
                }
                
                double corr_pos = stat_corr(x1, y1);
                double corr_neg = stat_corr(x2, y2);
                
                return corr_pos - corr_neg;
            }
            """,
            "jumpvol": """
                double stat_jumpvol(const double &a[])
                {
                    int n = ArraySize(a);
                    if(n < 2) return 0.0;
                    
                    // Calcular log returns
                    double logret[];
                    ArrayResize(logret, n-1);
                    for(int i = 1; i < n; i++) 
                        logret[i-1] = MathLog(a[i-1]/a[i]);
                    
                    // Calcular mediana usando stat_median
                    double med = stat_median(logret);
                    
                    // Calcular MAD usando stat_median
                    double dev[];
                    ArrayResize(dev, n-1);
                    for(int i = 0; i < n-1; i++) 
                        dev[i] = MathAbs(logret[i] - med);
                    double mad = stat_median(dev);
                    
                    if(mad == 0.0) return 0.0;
                    
                    // Contar saltos
                    double thresh = 3.0 * mad;
                    int jumps = 0;
                    for(int i = 0; i < n-1; i++)
                        if(dev[i] > thresh) jumps++;
                    
                    return (double)jumps/(n-1);
                }
                """,
            "volskew": """
                double stat_volskew(const double &a[])
                {
                    int n = ArraySize(a);
                    if(n<2) return 0.0;
                    
                    // Calcular movimientos positivos y negativos
                    double up_moves[], down_moves[];
                    ArrayResize(up_moves, n-1);
                    ArrayResize(down_moves, n-1);
                    
                    for(int i=1; i<n; i++)
                    {
                        double diff = a[i] - a[i-1];
                        up_moves[i-1] = MathMax(diff, 0.0);
                        down_moves[i-1] = MathMax(-diff, 0.0);
                    }
                    
                    // Calcular desviación estándar usando stat_std
                    double up_vol = stat_std(up_moves);
                    double down_vol = stat_std(down_moves);
                    
                    // Calcular skew
                    double sum = up_vol + down_vol;
                    return sum==0.0 ? 0.0 : (up_vol - down_vol)/sum;
                }
            """,
        }
        code = r"#include <Math\Stat\Math.mqh>"
        code += '\n'
        code += rf'#resource "\\Files\\{filename_model_main}" as uchar ExtModel_[]'
        code += '\n'
        code += rf'#resource "\\Files\\{filename_model_meta}" as uchar ExtModel_m_[]'
        code += '\n\n'
        code += '//+------------------------------------------------------------------+\n'
        code += f'//| INS SCORE: {best_scores[0]} | OOS SCORE: {best_scores[1]}    |\n'
        code += '//+------------------------------------------------------------------+\n'
        code += '\n\n'
        code += f'#define SYMBOL               "{str(symbol)}"\n'
        code += f'#define TIMEFRAME            "{str(timeframe)}"\n'
        code += f'#define DIRECTION            "{str(direction)}"\n'
        code += f'#define MAGIC_NUMBER         {str(model_seed)}\n'
        code += "string MAIN_COLS[] = { " + ",".join('"' + c + '"' for c in main_cols) + " };\n"
        code += "string META_COLS[] = { " + ",".join('"' + c + '"' for c in meta_cols) + " };\n\n"
        stats_total = set(stats_main + stats_meta)
        if "mean" not in stats_total:
            code += stat_function_templates["mean"] + "\n"
        if "std" not in stats_total:
            code += stat_function_templates["std"] + "\n"
        if "median" not in stats_total:
            code += stat_function_templates["median"] + "\n"
        if "fisher" in stats_total and "momentum" not in stats_total:
            code += stat_function_templates["momentum"] + "\n"
        if "corrskew" in stats_total and "corr" not in stats_total:
            code += stat_function_templates["corr"] + "\n"
        for stat in stats_total:
            code += stat_function_templates[stat] + "\n\n"
        # ─────────────────────────  Dispatcher de estadísticos  ────────────
        code += 'double switch_stat(string stat,const double &data[])\n'
        code += '  {\n'
        for st in sorted(stats_total):
            code += f'   if(stat=="{st}") return stat_{st}(data);\n'
        code += '   return 0.0;              // estadístico desconocido\n'
        code += '  }\n\n'

        # ─────────────────────────  MAIN feature builder  ──────────────────
        code += 'void fill_arays_main(double &features[])\n'
        code += '  {\n'
        code += '   double pr[];\n'
        code += '   for(int k=0; k<ArraySize(MAIN_COLS); k++)\n'
        code += '     {\n'
        code += '      // \"24_std_feature\"  →  periodo 24  +  \"std\"\n'
        code += '      string parts[];\n'
        code += '      StringSplit(MAIN_COLS[k], \'_\', parts);\n'
        code += '      if(ArraySize(parts)<2) continue;\n'
        code += '      int    period = (int)StringToInteger(parts[0]);\n'
        code += '      string stat   = parts[1];\n\n'
        code += f'      CopyClose(NULL, PERIOD_{timeframe}, 1, period, pr);\n'
        code += '      ArraySetAsSeries(pr,true);\n\n'
        code += '      features[k] = switch_stat(stat, pr);\n'
        code += '     }\n'
        code += '  }\n\n'

        # ─────────────────────────  META feature builder  ──────────────────
        code += 'void fill_arays_meta(double &features[])\n'
        code += '  {\n'
        code += '   double pr[];\n'
        code += '   for(int k=0; k<ArraySize(META_COLS); k++)\n'
        code += '     {\n'
        code += '      string parts[];\n'
        code += '      StringSplit(META_COLS[k], \'_\', parts);\n'
        code += '      if(ArraySize(parts)<2) continue;\n'
        code += '      int    period = (int)StringToInteger(parts[0]);\n'
        code += '      string stat   = parts[1];\n\n'
        code += f'      CopyClose(NULL, PERIOD_{timeframe}, 1, period, pr);\n'
        code += '      ArraySetAsSeries(pr,true);\n\n'
        code += '      features[k] = switch_stat(stat, pr);\n'
        code += '     }\n'
        code += '  }\n\n'

        file_name = os.path.join(include_export_path, f"{symbol}_{timeframe}_{direction}_{label_method}_{search_type}_{search_subtype}.mqh")
        with open(file_name, "w") as file:
            file.write(code)

    except Exception as e:
        print(f"ERROR EN EXPORTACIÓN: {e}")
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