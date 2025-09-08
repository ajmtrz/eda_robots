import os
import tempfile
import hashlib
from catboost import CatBoostClassifier

def export_models_to_ONNX(models):
    """
    Convierte una lista de modelos CatBoost a ONNX usando el método nativo de CatBoost o el convertidor personalizado según el tipo de modelo.

    :param models: Lista o tupla de modelos CatBoost a convertir.
    :return: Lista de rutas de archivos ONNX temporales.
    """

    onnx_models = []
    for model in models:
        # Crear archivo temporal para el modelo ONNX
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".onnx")
        tmp.close()

        # Detectar tipo de modelo automáticamente
        if isinstance(model, CatBoostClassifier):
            # Para clasificación, usar exportación normal
            model.save_model(tmp.name, format="onnx")
        else:
            raise ValueError(f"Tipo de modelo no soportado para exportación ONNX: {type(model)}")

        onnx_models.append(tmp.name)

    return onnx_models

def export_dataset_to_csv(dataset, decimal_precision=6):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    dataset.to_csv(tmp.name, index=True, float_format=f'%.{decimal_precision}f', date_format='%Y.%m.%d %H:%M:%S')
    return tmp.name

def export_to_mql5(**kwargs):
    tag = kwargs.get('tag')
    best_score = kwargs.get('best_score')
    model_paths = kwargs.get('best_model_paths')
    model_cols = kwargs.get('best_model_cols')
    stats_main = kwargs.get('best_stats_main')
    stats_meta = kwargs.get('best_stats_meta')
    direction = kwargs.get('direction')
    models_export_path = kwargs.get('models_export_path')
    include_export_path = kwargs.get('include_export_path')
    decimal_precision = kwargs.get('decimal_precision')
    full_ds_with_labels_path = kwargs.get('best_full_ds_with_labels_path')
    main_threshold = kwargs.get('best_main_threshold')
    meta_threshold = kwargs.get('best_meta_threshold')

    def _should_use_returns(stat_name):
        """Determina si un estadístico debe usar retornos en lugar de precios."""
        return stat_name in ["mean", "median", "std", "iqr", "mad", "sharpe", "autocorr"]

    def _build_periods_funcs(cols: list[str]) -> tuple[list[str], list[str]]:
        """Devuelve dos listas: [periodos] y [punteros a stat_X]."""
        periods, funcs = [], []
        for c in cols:
            if c.endswith("_main_feature"):
                base = c[:-13]
            elif c.endswith("_meta_feature"):
                base = c[:-13]
            p_str, stat = base.split('_', 1)
            period = int(p_str)
            
            # ───── AJUSTAR PERÍODO PARA ESTADÍSTICOS CON RETORNOS ─────
            # Si el estadístico usa retornos, incrementamos el período en 1
            # para obtener el número correcto de retornos
            if _should_use_returns(stat):
                period += 1
            
            periods.append(period)
            funcs.append(f"stat_{stat}")
        return periods, funcs
    
    try:
        main_cols, meta_cols = model_cols
        hash_str = main_cols + meta_cols
        model_seed = int.from_bytes(hashlib.sha3_224(str(hash_str).encode('utf-8')).digest()[:5], 'big')
        main_periods, main_funcs = _build_periods_funcs(main_cols)
        meta_periods, meta_funcs = _build_periods_funcs(meta_cols)
        
        # Crear subcarpeta por tag y construir nombres basados en model_seed
        tag_models_dir = os.path.join(models_export_path, str(tag))
        os.makedirs(tag_models_dir, exist_ok=True)

        filename_model_main = f"{model_seed}_main.onnx"
        filepath_model_main = os.path.join(tag_models_dir, filename_model_main)
        filename_model_meta = f"{model_seed}_meta.onnx"
        filepath_model_meta = os.path.join(tag_models_dir, filename_model_meta)

        # model_paths[0] es el modelo main, model_paths[1] es el modelo meta
        if model_paths and len(model_paths) >= 2:
            with open(model_paths[0], "rb") as src, open(filepath_model_main, "wb") as dst:
                dst.write(src.read())
            with open(model_paths[1], "rb") as src, open(filepath_model_meta, "wb") as dst:
                dst.write(src.read())
        else:
            raise ValueError("No se encontraron suficientes rutas en model_paths para copiar los modelos ONNX.")
        if model_paths:
            for p in model_paths:
                try:
                    os.remove(p)
                except Exception:
                    pass

        # Copia el dataset con labels desde el archivo temporal a la ruta de destino
        data_dir = f"./data/{tag}"
        os.makedirs(data_dir, exist_ok=True)
        dataset_filename = f"{model_seed}.csv"
        dataset_path = os.path.join(data_dir, dataset_filename)
        if full_ds_with_labels_path:
            with open(full_ds_with_labels_path, "rb") as src, open(dataset_path, "wb") as dst:
                dst.write(src.read())
        else:
            raise ValueError("No se encontraron suficientes rutas en model_paths para copiar el dataset.")
        if full_ds_with_labels_path:
            os.remove(full_ds_with_labels_path)

        stat_function_templates = {
            "std": """
                double stat_std(const double &a[])
                {
                    int n = ArraySize(a);
                    if(n <= 1) return 0.0;
                    
                    // USAR LA MISMA FÓRMULA DIRECTA QUE PYTHON
                    // Python: np.sqrt(np.sum((x - m) ** 2) / (x.size - 1))
                    double mean = stat_mean(a);
                    double sum_sq_diff = 0.0;
                    for(int i = 0; i < n; i++)
                    {
                        double diff = a[i] - mean;
                        sum_sq_diff += diff * diff;
                    }
                    return MathSqrt(sum_sq_diff / (n - 1));
                }
                """,
            "skew": """
                double stat_skew(const double &a[])
                {
                    int n = ArraySize(a);
                    if(n <= 1) return 0.0;
                    
                    // REPLICAR PYTHON EXACTO: std_manual() y mean_manual() consistentes
                    double std = stat_std(a);
                    if(std == 0.0) return 0.0;
                    
                    double mean = stat_mean(a);
                    
                    // REPLICAR PYTHON EXACTO: mean_manual(((x - m) / s) ** 3)
                    // Crear array temporal para valores estandarizados al cubo
                    double standardized_cubed[];
                    ArrayResize(standardized_cubed, n);
                    for(int i = 0; i < n; i++) {
                        double standardized = (a[i] - mean) / std;
                        // USAR MULTIPLICACIÓN DIRECTA PARA MAYOR PRECISIÓN QUE MathPow
                        standardized_cubed[i] = standardized * standardized * standardized;
                    }
                    
                    // USAR stat_mean para consistencia exacta con Python mean_manual()
                    return stat_mean(standardized_cubed);
                }
                """,
            "kurt": """
                double stat_kurt(const double &a[])
                {
                    int n = ArraySize(a);
                    if(n <= 1) return 0.0;
                    
                    // REPLICAR PYTHON EXACTO: std_manual() y mean_manual() consistentes
                    double std = stat_std(a);
                    if(std == 0.0) return 0.0;
                    
                    double mean = stat_mean(a);
                    
                    // REPLICAR PYTHON EXACTO: mean_manual(((x - m) / s) ** 4)
                    // Crear array temporal para valores estandarizados a la 4ta potencia
                    double standardized_fourth[];
                    ArrayResize(standardized_fourth, n);
                    for(int i = 0; i < n; i++) {
                        double standardized = (a[i] - mean) / std;
                        // USAR MULTIPLICACIÓN DIRECTA PARA MAYOR PRECISIÓN QUE MathPow
                        double standardized_sq = standardized * standardized;
                        standardized_fourth[i] = standardized_sq * standardized_sq;
                    }
                    
                    // USAR stat_mean para consistencia exacta con Python mean_manual()
                    return stat_mean(standardized_fourth) - 3.0;
                }
                """,
            "zscore": """
                double stat_zscore(const double &a[])
                {
                    int n = ArraySize(a);
                    if(n <= 1) return 0.0;
                    
                    // REPLICAR PYTHON EXACTO: usar las funciones std y mean
                    double std = stat_std(a);
                    if(std == 0.0) return 0.0;
                    
                    double mean = stat_mean(a);
                    return (a[n-1] - mean) / std;
                }
                """,
            "vwapdevz": """
                double stat_vwapdevz(const double &a[])
                {
                    int n = ArraySize(a);
                    if(n == 0) return 0.0;

                    // Obtener ventanas High/Low/Volume del mismo tamaño y desplazamiento (shift=1)
                    double h[], l[];
                    long   vol[];
                    CopyHigh(_Symbol, _Period, 1, n, h);
                    CopyLow(_Symbol, _Period, 1, n, l);
                    CopyTickVolume(_Symbol, _Period, 1, n, vol);
                    ArraySetAsSeries(h, false);
                    ArraySetAsSeries(l, false);
                    ArraySetAsSeries(vol, false);

                    // Calcular VWAP rolling en la ventana [0..n-1]
                    double num = 0.0, den = 0.0;
                    for(int i = 0; i < n; i++)
                    {
                        double tp = (h[i] + l[i] + a[i]) / 3.0;
                        double v = (double)vol[i];
                        num += tp * v;
                        den += v;
                    }

                    double vwap;
                    if(den <= 0.0)
                        vwap = (h[n-1] + l[n-1] + a[n-1]) / 3.0;
                    else
                        vwap = num / den;

                    // Z-score respecto a la desviación estándar de la serie de cierres de la ventana
                    double sd = stat_std(a);
                    if(sd <= 1e-8) sd = 1e-8;
                    return (a[n-1] - vwap) / sd;
                }
                """,
            "mean": """
                double stat_mean(const double &a[])
                {
                    if(ArraySize(a) == 0) return 0.0;
                    double sum = 0.0;
                    for(int i = 0; i < ArraySize(a); i++)
                        sum += a[i];
                    return sum / ArraySize(a);
                }
                """,
            "range": """
                double stat_range(const double &a[])
                {
                    int n = ArraySize(a);
                    if(n == 0) return 0.0;
                    
                    // REPLICAR PYTHON EXACTO: np.max(window_data) - np.min(window_data)
                    double minv = a[0], maxv = a[0];
                    for(int i = 1; i < n; i++)
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
                    int n = ArraySize(a);
                    if(n == 0) return 0.0;  // Mantener 0.0 para consistencia con otros stats
                    
                    // REPLICAR PYTHON EXACTO: b = a.copy(); b.sort()
                    double tmp[];
                    ArrayResize(tmp, n);
                    
                    // Copia manual exacta con máxima precisión
                    for(int i = 0; i < n; i++) {
                        tmp[i] = a[i];
                    }
                    
                    // Ordenamiento con verificación de estabilidad
                    ArraySort(tmp);
                    
                    // Cálculo de mediana exacto como Python
                    int mid = n / 2;
                    if(n % 2 == 1) {
                        // n impar: elemento del medio
                        return tmp[mid];
                    } else {
                        // n par: promedio de los dos elementos del medio
                        // REPLICAR PYTHON EXACTO: 0.5 * (b[mid-1] + b[mid])
                        return 0.5 * (tmp[mid-1] + tmp[mid]);
                    }
                }
                """,
            "iqr": """
                double stat_iqr(const double &a[])
                {
                    int n = ArraySize(a);
                    if(n == 0) return 0.0;
                    
                    // REPLICAR PYTHON EXACTO: b = a.copy(); b.sort()
                    double tmp[];
                    ArrayResize(tmp, n);
                    for(int i = 0; i < n; i++) tmp[i] = a[i];  // Copia manual exacta
                    ArraySort(tmp);
                    
                    // REPLICAR PYTHON EXACTO: int(0.25 * (n - 1)) y int(0.75 * (n - 1))
                    // Usar casting explícito para asegurar el mismo comportamiento que Python
                    int q1_idx = (int)(0.25 * (double)(n - 1));
                    int q3_idx = (int)(0.75 * (double)(n - 1));
                    return tmp[q3_idx] - tmp[q1_idx];
                }
                """,
            "mad": """
                double stat_mad(const double &a[])
                {
                    if(ArraySize(a) == 0) return 0.0;
                    
                    // REPLICAR PYTHON EXACTO: m = mean_manual(window_data)
                    double mean = stat_mean(a);
                    
                    // REPLICAR PYTHON EXACTO: mean_manual(np.abs(window_data - m))
                    // Crear array temporal para las desviaciones absolutas
                    double abs_deviations[];
                    ArrayResize(abs_deviations, ArraySize(a));
                    
                    for(int i = 0; i < ArraySize(a); i++) {
                        abs_deviations[i] = MathAbs(a[i] - mean);
                    }
                    
                    // Usar stat_mean para calcular la media de las desviaciones absolutas
                    return stat_mean(abs_deviations);
                }
                """,
            "var": """
                double stat_var(const double &a[])
                {
                    int n = ArraySize(a);
                    if(n == 0) return 0.0;
                    
                    // REPLICAR PYTHON EXACTO: std * std * (window_data.size - 1) / window_data.size
                    double std = stat_std(a);
                    double variance_sample = std * std;  // Varianza muestral
                    
                    // Convertir de varianza muestral a poblacional como en Python
                    return variance_sample * ((double)(n - 1) / (double)n);
                }
                """,
            "cv": """
                double stat_cv(const double &a[])
                {
                    double mean = stat_mean(a);
                    if(mean == 0.0) return 0.0;
                    double sd = stat_std(a);
                    return sd / mean;
                }
                """,
            "entropy": """
                double stat_entropy(const double &a[])
                {
                    int n = ArraySize(a);
                    if(n == 0) return 0.0;
                    
                    int bins = 10;
                    double minv = a[0], maxv = a[0];
                    for(int i = 1; i < n; i++)
                    {
                        if(a[i] < minv) minv = a[i];
                        if(a[i] > maxv) maxv = a[i];
                    }
                    double width = (maxv - minv) / bins;
                    if(width == 0) return 0.0;
                    
                    double hist[10] = {0};
                    for(int i = 0; i < n; i++)
                    {
                        int idx = int((a[i] - minv) / width);
                        if(idx == bins) idx--; // borde superior
                        hist[idx]++;
                    }
                    
                    double total = n;
                    double entropy = 0.0;
                    for(int i = 0; i < bins; i++)
                    {
                        if(hist[i] > 0)
                        {
                            double p = hist[i] / total;
                            entropy -= p * MathLog(p);
                        }
                    }
                    return entropy;
                }
                """,
            "slope": """
                double stat_slope(const double &a[])
                {
                    int n = ArraySize(a);
                    if(n <= 1) return 0.0;
                    
                    // REPLICAR PYTHON EXACTO: x_idx = np.arange(n)
                    double x_idx[];
                    ArrayResize(x_idx, n);
                    for(int i = 0; i < n; i++) x_idx[i] = (double)i;
                    
                    // USAR FUNCIONES CONSISTENTES: mean_manual() y std_manual()
                    double x_mean = stat_mean(x_idx);
                    double y_mean = stat_mean(a);
                    
                    // Calcular covarianza exactamente como Python
                    double cov = 0.0;
                    for(int i = 0; i < n; i++) {
                        cov += (x_idx[i] - x_mean) * (a[i] - y_mean);
                    }
                    cov /= (double)n;
                    
                    // Calcular varianza usando stat_std para máxima precisión
                    double x_std = stat_std(x_idx);
                    double var_x = x_std * x_std * ((double)(n - 1) / (double)n);
                    
                    return (var_x != 0.0) ? (cov / var_x) : 0.0;
                }
                """,
            "momentum": """
                double stat_momentum(const double &a[])
                {
                    int size = ArraySize(a);
                    if(size < 2) return 0.0;
                    if(a[size-1] == 0) return 0.0;
                    
                    // REPLICAR PYTHON: ratio = x[0]/x[-1]; return ratio - 1.0
                    return (a[0] / a[size-1]) - 1.0;
                }
                """,
            "fractal": """
                double stat_fractal(const double &x[])
                {
                    int size = ArraySize(x);
                    if(size < 2) return 1.0;
                    
                    // REPLICAR ALGORITMO PYTHON EXACTO
                    double std_dev = stat_std(x);
                    double eps = std_dev / 4.0;
                    if(eps == 0.0) return 1.0;
                    
                    int count = 0;
                    for(int i = 0; i < size-1; i++) {
                        if(MathAbs(x[i+1] - x[i]) > eps) count++;
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
                    
                    // REPLICAR ALGORITMO PYTHON EXACTO
                    double valid_rs[];
                    int valid_count = 0;
                    ArrayResize(valid_rs, n-1);
                    
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
                        for(int j = 1; j < i+1; j++) {
                            if(subseries[j] > max_val) max_val = subseries[j];
                            if(subseries[j] < min_val) min_val = subseries[j];
                        }
                        double r = max_val - min_val;
                        double rs = r / s;
                        if(rs > 0) {
                            valid_rs[valid_count] = rs;
                            valid_count++;
                        }
                    }
                    
                    // Verificar si tenemos suficientes valores válidos
                    if(valid_count == 0) return 0.5;
                    
                    // REPLICAR PYTHON EXACTO: log_rs = np.log(valid_rs[:valid_count])
                    // mean_log_rs = mean_manual(log_rs)
                    double log_rs[];
                    ArrayResize(log_rs, valid_count);
                    for(int i = 0; i < valid_count; i++) {
                        log_rs[i] = MathLog(valid_rs[i]);
                    }
                    double mean_log_rs = stat_mean(log_rs);
                    double log_n = MathLog(n);
                    
                    // Evitar división por valores cercanos a cero
                    if(MathAbs(log_n) < 1e-10) return 0.5;
                    
                    return mean_log_rs / log_n;
                }
                """,
            "autocorr": """
                double stat_autocorr(const double &a[])
                {
                    int n = ArraySize(a);
                    if(n < 2) return 0.0;
                    
                    // REPLICAR PYTHON EXACTO: mean_manual(x) con precisión máxima
                    double mean = stat_mean(a);
                    double num = 0.0, den = 0.0;
                    
                    for(int i = 0; i < n-1; i++)
                    {
                        double d0 = a[i]   - mean;
                        double d1 = a[i+1] - mean;
                        num += d0 * d1;
                        den += d0 * d0;
                    }
                    
                    // USAR MISMA CONDICIÓN Y RETORNO QUE PYTHON
                    return (den != 0.0) ? (num / den) : 0.0;
                }
                """,
            "maxdd": """
                double stat_maxdd(const double &a[])
                {
                    int n = ArraySize(a);
                    if(n == 0) return 0.0;
                    double peak = a[0];
                    double maxdd = 0.0;
                    for(int i = 0; i < n; i++)
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
                    double mean = stat_mean(a);
                    double std = stat_std(a);
                    return std == 0.0 ? 0.0 : mean / std;
                }
            """,
            "fisher": """
                double stat_fisher(const double &a[])
                {
                    int n = ArraySize(a);
                    if(n <= 1) return 0.0;
                    
                    // Usar la función stat_momentum ya definida
                    double momentum = stat_momentum(a);
                    
                    // Aplicar transformación de Fisher - REPLICAR PYTHON EXACTO
                    double x = MathMax(-0.9999, MathMin(0.9999, momentum));
                    return 0.5 * MathLog((1.0 + x) / (1.0 - x));
                }
                """,
            "chande": """
                double stat_chande(const double &a[])
                {
                    int n = ArraySize(a);
                    if(n < 2) return 0.0;
                    
                    // REPLICAR ALGORITMO PYTHON
                    double up = 0.0, down = 0.0;
                    for(int i = 1; i < n; i++)
                    {
                        double diff = a[i] - a[i-1];
                        if(diff > 0) up += diff;
                        else down += MathAbs(diff);  // down -= diff where diff is negative
                    }
                    double sum = up + down;
                    return sum == 0.0 ? 0.0 : (up - down) / sum;
                }
            """,
            "approxentropy": """
                double stat_approxentropy(const double &a[])
                {
                    int n = ArraySize(a);
                    int m = 2;
                    if(n <= m + 1) return 0.0;
                    
                    // REPLICAR PYTHON EXACTO: usar misma función std que Python
                    double sd = stat_std(a);
                    double r = 0.2 * sd;
                    
                    // REPLICAR ALGORITMO PYTHON EXACTO
                    // Para m = 2 - mejorar precisión con double desde el inicio
                    double count_m = 0.0;  // Cambio: usar double para mayor precisión
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
                                if(match_m) count_m += 1.0;
                            }
                        }
                    }
                    double phi1 = count_m > 0.0 ? MathLog(count_m / (double)(n - 1)) : 0.0;
                    
                    // Para m = 3 - mejorar precisión con double desde el inicio
                    double count_m1 = 0.0;  // Cambio: usar double para mayor precisión
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
                                if(match_m1) count_m1 += 1.0;
                            }
                        }
                    }
                    double phi2 = count_m1 > 0.0 ? MathLog(count_m1 / (double)(n - 2)) : 0.0;
                    
                    return phi1 - phi2;
                }
                """,
            "effratio": """
                double stat_effratio(const double &a[])
                {
                    int n = ArraySize(a);
                    if(n < 2) return 0.0;
                    
                    // REPLICAR PYTHON EXACTO: _direction = x[-1] - x[0]
                    double _direction = a[n-1] - a[0];
                    
                    // REPLICAR PYTHON EXACTO: volatility = np.sum(np.abs(np.diff(x)))
                    // np.diff(x) calcula x[i+1] - x[i] (diferencias hacia adelante)
                    double volatility = 0.0;
                    for(int i = 0; i < n-1; i++)  // Cambio: i=0 to n-2, calcular a[i+1] - a[i]
                        volatility += MathAbs(a[i+1] - a[i]);
                    
                    return volatility == 0.0 ? 0.0 : _direction / volatility;
                }
            """,
            "corr": """
            double stat_corr(const double &a[], const double &b[])
            {
                int n = ArraySize(a);
                if(n != ArraySize(b) || n < 2) return 0.0;
                
                    // REPLICAR ALGORITMO PYTHON EXACTO
                double mean_a = stat_mean(a);
                double mean_b = stat_mean(b);
                
                // Calcular covarianza
                double cov = 0.0;
                    for(int i = 0; i < n; i++) {
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
                
                // REPLICAR PYTHON EXACTO: lag = min(5, x.size // 2)
                int lag = MathMin(5, n / 2);
                if(n < lag + 1) return 0.0;
                
                int size = n - lag;
                if(size < 2) return 0.0;  // Necesitamos al menos 2 puntos para correlación
                
                // CÁLCULO DIRECTO DE CORRELACIONES SIN ARRAYS INTERMEDIOS
                // Para mayor precisión y consistencia exacta con Python
                
                // Calcular medias de las secuencias x[:-lag] y x[lag:]
                double mean_x1 = 0.0, mean_y = 0.0;
                for(int i = 0; i < size; i++) {
                    mean_x1 += a[i];           // x[:-lag]
                    mean_y += a[i + lag];      // x[lag:]
                }
                mean_x1 /= (double)size;
                mean_y /= (double)size;
                
                // Calcular correlación positiva: corr(x[:-lag], x[lag:])
                double cov_pos = 0.0, var_x1 = 0.0, var_y = 0.0;
                for(int i = 0; i < size; i++) {
                    double dx1 = a[i] - mean_x1;           // x[:-lag] - media
                    double dy = a[i + lag] - mean_y;       // x[lag:] - media
                    
                    cov_pos += dx1 * dy;
                    var_x1 += dx1 * dx1;
                    var_y += dy * dy;
                }
                
                // Calcular std de forma consistente con stat_std
                double std_x1 = MathSqrt(var_x1 / (double)(size - 1));
                double std_y = MathSqrt(var_y / (double)(size - 1));
                
                double corr_pos = 0.0;
                if(std_x1 > 0.0 && std_y > 0.0) {
                    corr_pos = cov_pos / ((double)size * std_x1 * std_y);
                }
                
                // Calcular correlación negativa: corr(-x[:-lag], x[lag:])
                // Media de -x[:-lag] es simplemente -mean_x1
                double mean_neg_x1 = -mean_x1;
                
                double cov_neg = 0.0, var_neg_x1 = 0.0;
                for(int i = 0; i < size; i++) {
                    double dneg_x1 = (-a[i]) - mean_neg_x1;   // -x[:-lag] - media_neg
                    double dy = a[i + lag] - mean_y;          // x[lag:] - media (mismo que antes)
                    
                    cov_neg += dneg_x1 * dy;
                    var_neg_x1 += dneg_x1 * dneg_x1;
                }
                
                // std de -x[:-lag] debería ser igual a std de x[:-lag]
                double std_neg_x1 = MathSqrt(var_neg_x1 / (double)(size - 1));
                
                double corr_neg = 0.0;
                if(std_neg_x1 > 0.0 && std_y > 0.0) {
                    corr_neg = cov_neg / ((double)size * std_neg_x1 * std_y);
                }
                
                return corr_pos - corr_neg;
            }
            """,
            "jumpvol": """
                double stat_jumpvol(const double &a[])
                {
                    int n = ArraySize(a);
                    if(n < 2) return 0.0;
                    
                    // REPLICAR ALGORITMO PYTHON EXACTO
                    // Calcular log returns
                    double logret[];
                    ArrayResize(logret, n-1);
                    for(int i = 1; i < n; i++) {
                        if(a[i-1] <= 0) logret[i-1] = 0.0;
                        else logret[i-1] = MathLog(a[i-1] / a[i]);
                    }
                    
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
                    
                    return (double)jumps / (n-1);
                }
                """,
            "volskew": """
                double stat_volskew(const double &a[])
                {
                    int n = ArraySize(a);
                    if(n < 2) return 0.0;
                    
                    // REPLICAR PYTHON EXACTO: np.maximum(x[1:] - x[:-1], 0) y np.maximum(x[:-1] - x[1:], 0)
                    double up_moves[], down_moves[];
                    ArrayResize(up_moves, n-1);
                    ArrayResize(down_moves, n-1);
                    
                    // Calcular movimientos directamente como Python con máxima precisión
                    for(int i = 0; i < n-1; i++)
                    {
                        double forward_diff = a[i+1] - a[i];    // x[1:] - x[:-1]
                        double backward_diff = a[i] - a[i+1];   // x[:-1] - x[1:]
                        
                        // REPLICAR PYTHON EXACTO: np.maximum(..., 0)
                        up_moves[i] = (forward_diff > 0.0) ? forward_diff : 0.0;
                        down_moves[i] = (backward_diff > 0.0) ? backward_diff : 0.0;
                    }
                    
                    // Usar stat_std para máxima precisión y consistencia
                    double up_vol = stat_std(up_moves);
                    double down_vol = stat_std(down_moves);
                    
                    // REPLICAR RETORNO PYTHON EXACTO con verificación de precisión
                    double sum = up_vol + down_vol;
                    if(sum == 0.0) return 0.0;
                    
                    return (up_vol - down_vol) / sum;
                }
            """,
        }
        
        code = r"#include <Math\Stat\Math.mqh>"
        code += '\n'
        # Las rutas deben incluir la subcarpeta del tag
        code += rf'#resource "\\Files\\{tag}\\{filename_model_main}" as uchar ExtModel_main[]'
        code += '\n'
        code += rf'#resource "\\Files\\{tag}\\{filename_model_meta}" as uchar ExtModel_meta[]'
        code += '\n\n'
        code += '//+------------------------------------------------------------------+\n'
        code += f'//| SCORE: {best_score}                                        |\n'
        code += '//+------------------------------------------------------------------+\n'
        code += '\n\n'
        code += f'#define DIRECTION            "{str(direction)}"\n'
        code += f'#define MAGIC_NUMBER         {str(model_seed)}\n'
        code += f'#define DECIMAL_PRECISION    {str(decimal_precision)}\n'
        code += f'#define MAIN_THRESHOLD       {str(main_threshold)}\n'
        code += f'#define META_THRESHOLD       {str(meta_threshold)}\n'
        # ───── AGREGAR FUNCIÓN DE RETORNOS ─────
        code += """
// ───── FUNCIÓN PARA CALCULAR RETORNOS LOGARÍTMICOS ─────
void compute_returns(const double &prices[], double &returns[])
{
    int n = ArraySize(prices);
    if(n <= 1) {
        ArrayResize(returns, 0);
        return;
    }
    
    ArrayResize(returns, n - 1);
    for(int i = 0; i < n - 1; i++) {
        if(prices[i] <= 0.0 || prices[i + 1] <= 0.0) {
            returns[i] = 0.0;  // Evitar log(0) o log(negativo)
        } else {
            // REPLICAR PYTHON EXACTO con máxima precisión: np.log(prices[i + 1] / prices[i])
            double ratio = prices[i + 1] / prices[i];
            // Verificar que el ratio sea válido antes de calcular log
            if(ratio > 0.0) {
                returns[i] = MathLog(ratio);
            } else {
                returns[i] = 0.0;
            }
        }
    }
}

// ───── FUNCIÓN PARA DETERMINAR SI UN ESTADÍSTICO USA RETORNOS ─────
bool should_use_returns(string stat_name)
{
    return (stat_name == "mean" || stat_name == "median" || stat_name == "std" || 
            stat_name == "iqr" || stat_name == "mad" || stat_name == "sharpe" || stat_name == "autocorr");
}

"""
        
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
        code += "\n//--- descriptors generados automáticamente ---\n"
        code += "typedef double (*StatFunc)(const double &[]);\n"

        code += "const int      PERIODS_MAIN[] = { " + ", ".join(map(str, main_periods)) + " };\n"
        code += "const StatFunc FUNCS_MAIN  [] = { " + ", ".join(main_funcs) + " };\n"
        
        # ───── AGREGAR ARRAYS PARA IDENTIFICAR ESTADÍSTICOS CON RETORNOS ─────
        main_uses_returns = [_should_use_returns(func.replace("stat_", "")) for func in main_funcs]
        code += "const bool     USES_RETURNS_MAIN[] = { " + ", ".join(["true" if x else "false" for x in main_uses_returns]) + " };\n\n"

        if meta_periods:      # sólo si hay features meta
            code += "const int      PERIODS_META[] = { " + ", ".join(map(str, meta_periods)) + " };\n"
            code += "const StatFunc FUNCS_META  [] = { " + ", ".join(meta_funcs) + " };\n"
            meta_uses_returns = [_should_use_returns(func.replace("stat_", "")) for func in meta_funcs]
            code += "const bool     USES_RETURNS_META[] = { " + ", ".join(["true" if x else "false" for x in meta_uses_returns]) + " };\n\n"

        # ──────────────────────────────────────────────────────────────────
        # 2)  Rutinas compactas de cálculo (una pasada, con soporte para retornos)
        # ──────────────────────────────────────────────────────────────────
        code += """
void fill_arays_main(double &dst[])
{{
    double pr[], returns[];
    for(int k=0; k<ArraySize(PERIODS_MAIN); ++k)
    {{
        int per = PERIODS_MAIN[k];
        CopyClose(_Symbol, _Period, 1, per, pr);
        ArraySetAsSeries(pr, false);
        
        // ───── USAR RETORNOS SI EL ESTADÍSTICO LO REQUIERE ─────
        if(USES_RETURNS_MAIN[k]) {{
            compute_returns(pr, returns);
            if(ArraySize(returns) == 0) {{
                dst[k] = 0.0;  // Valor por defecto si no se pueden calcular retornos
            }} else {{
                dst[k] = NormalizeDouble(FUNCS_MAIN[k](returns), DECIMAL_PRECISION);
            }}
        }} else {{
            dst[k] = NormalizeDouble(FUNCS_MAIN[k](pr), DECIMAL_PRECISION);
        }}
    }}
}}
"""

        if meta_periods:  # Solo agregar si hay features meta
            code += """
void fill_arays_meta(double &dst[])
{{
    double pr[], returns[];
    for(int k=0; k<ArraySize(PERIODS_META); ++k)
    {{
        int per = PERIODS_META[k];
        CopyClose(_Symbol, _Period, 1, per, pr);
        ArraySetAsSeries(pr, false);
        
        // ───── USAR RETORNOS SI EL ESTADÍSTICO LO REQUIERE ─────
        if(USES_RETURNS_META[k]) {{
            compute_returns(pr, returns);
            if(ArraySize(returns) == 0) {{
                dst[k] = 0.0;  // Valor por defecto si no se pueden calcular retornos
            }} else {{
                dst[k] = NormalizeDouble(FUNCS_META[k](returns), DECIMAL_PRECISION);
            }}
        }} else {{
            dst[k] = NormalizeDouble(FUNCS_META[k](pr), DECIMAL_PRECISION);
        }}
    }}
}}
"""

        # Crear subcarpeta por tag para el include y nombrar archivo por seed
        tag_include_dir = os.path.join(include_export_path, str(tag))
        os.makedirs(tag_include_dir, exist_ok=True)
        file_name = os.path.join(tag_include_dir, f"{model_seed}.mqh")
        with open(file_name, "w") as file:
            file.write(code)

    except Exception as e:
        print(f"ERROR EN EXPORTACIÓN: {e}")
        raise