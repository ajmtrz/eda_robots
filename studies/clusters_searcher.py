import os
import glob
import pickle
import json
import math
import random
import numpy as np
from datetime import datetime
from tqdm import tqdm
from numba import njit
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Tuple, List
import optuna
from optuna.pruners import SuccessiveHalvingPruner, HyperbandPruner
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from modules.labeling_lib import get_prices
from modules.labeling_lib import get_clustering_features
from modules.labeling_lib import get_labels_one_direction
from modules.labeling_lib import sliding_window_clustering
from modules.tester_lib import test_model_one_direction_clustering
from modules.export_lib import export_model_to_ONNX
import warnings
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

def fit_final_models(clustered: pd.DataFrame,
                     meta: pd.DataFrame,
                     oos_data: pd.DataFrame,
                     backward_data: pd.DataFrame,
                     hp: Dict[str, Any]) -> Tuple[float, float, Any, Any]:
    """Entrena modelo principal + meta‑modelo y evalúa en OOS y backward.

    Devuelve (R2_forward, R2_backward, model_main, meta_model).
    """
    # ---------- 1) main model_main ----------
    X_main = clustered.drop(columns=['labels', *meta.columns[meta.columns.str.contains('_meta_feature')]])
    y_main = clustered['labels'].astype('int16')

    # ---------- 2) meta‑modelo ----------
    X_meta = meta.loc[:, meta.columns.str.contains('_meta_feature')]
    y_meta = meta['clusters'].astype('int16')
    
    # 3) Split aleatorio (70/30)
    train_X, test_X, train_y, test_y = train_test_split(
        X_main, y_main, train_size=0.7, shuffle=True)
    train_X_m, test_X_m, train_y_m, test_y_m = train_test_split(
        X_meta, y_meta, train_size=0.7, shuffle=True)

    # 4) Hiper‑parámetros CatBoost (con valores por defecto + overrides)
    cat_main_params = dict(
        iterations=hp.get('cat_main_iterations', 500),
        depth=hp.get('cat_main_depth', 6),
        learning_rate=hp.get('cat_main_learning_rate', 0.15),
        l2_leaf_reg=hp.get('cat_main_l2_leaf_reg', 3.0),
        custom_loss=['Accuracy'],
        eval_metric='Accuracy',
        use_best_model=True,
        verbose=False,
        thread_count=-1,
        task_type='CPU',
    )
    model_main = CatBoostClassifier(**cat_main_params)
    model_main.fit(train_X, train_y, eval_set=(test_X, test_y), early_stopping_rounds=25)

    cat_meta_params = dict(
        iterations=hp.get('cat_meta_iterations', 500),
        depth=hp.get('cat_meta_depth', 6),
        learning_rate=hp.get('cat_meta_learning_rate', 0.15),
        l2_leaf_reg=hp.get('cat_meta_l2_leaf_reg', 3.0),
        custom_loss=['F1'],
        eval_metric='F1',
        use_best_model=True,
        verbose=False,
        thread_count=-1,
        task_type='CPU',
    )
    meta_model = CatBoostClassifier(**cat_meta_params)
    meta_model.fit(train_X_m, train_y_m, eval_set=(test_X_m, test_y_m), early_stopping_rounds=15)

    # 5) Evaluación en datos fuera de muestra (forward)
    R2_forward = test_model_one_direction_clustering(
        oos_data,
        [model_main, meta_model],
        hp['full forward'],
        hp['forward'],
        hp['markup'],
        hp['direction'],
        plt=False,
    )
    if math.isnan(R2_forward):
        R2_forward = -1.0
        
    # 6) Evaluación en datos históricos (backward)
    R2_backward = test_model_one_direction_clustering(
        backward_data,
        [model_main, meta_model],
        hp['forward'],
        hp['backward'],
        hp['markup'],
        hp['direction'],
        plt=False,
    )
    if math.isnan(R2_backward):
        R2_backward = -1.0
        
    return R2_forward, R2_backward, model_main, meta_model

def objective(trial: optuna.trial.Trial, base_hp: Dict[str, Any], study=None) -> float:
    '''
    # Añadir estrategia de warm start para mejorar la eficiencia
    if study is not None and len(study.trials) > 5:
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        
        # Con 25% de probabilidad, reutilizar parámetros de uno de los 3 mejores trials
        if random.random() < 0.25 and completed_trials:
            # Ordenar trials por valor
            sorted_trials = sorted(completed_trials, 
                                  key=lambda t: t.value if t.value is not None else float('-inf'),
                                  reverse=True)
            
            # Seleccionar uno de los 3 mejores trials aleatoriamente
            top_n = min(3, len(sorted_trials))
            if top_n > 0:
                reference_trial = sorted_trials[random.randint(0, top_n-1)]
                
                # Aplicar una pequeña perturbación a los valores
                for param_name, param_value in reference_trial.params.items():
                    try:
                        # No reutilizar si ya se ha sugerido este parámetro
                        if param_name in trial.params:
                            continue
                            
                        if isinstance(param_value, int):
                            # Para enteros, añadir un pequeño ruido
                            noise = random.randint(-2, 2)
                            # Garantizar valor mínimo de 1 para parámetros enteros
                            new_value = max(1, param_value + noise)
                            trial.suggest_int(param_name, new_value, new_value)
                        elif isinstance(param_value, float):
                            # Para flotantes, añadir ruido porcentual
                            noise_factor = random.uniform(-0.2, 0.2)
                            # Garantizar que el valor mínimo sea positivo
                            min_allowed = 0.001 if param_name.endswith('learning_rate') else 0.1
                            new_value = max(min_allowed, param_value * (1 + noise_factor))
                            trial.suggest_float(param_name, new_value, new_value)
                    except:
                        # Si falla, continuamos con la sugerencia normal
                        pass
    '''
    # Resto del código original
    hp = base_hp.copy()

    # µ··· Espacio de búsqueda optimizado ···µ
    # Parámetros de clustering más amplios para encontrar patrones más diversos
    hp['n_clusters'] = trial.suggest_int('n_clusters', 5, 50, step=5)  
    hp['window_size'] = trial.suggest_int('window_size', 50, 500, step=10)
    
    # Parámetros de etiquetado más agresivos
    hp['label_min'] = trial.suggest_int('label_min', 1, 5)
    hp['label_max'] = trial.suggest_int('label_max', hp['label_min']+5, 30)
    hp['markup'] = trial.suggest_float("markup", 0.1, 0.4)

    # CatBoost principal - Mayor capacidad de aprendizaje
    hp['cat_main_iterations'] = trial.suggest_int('cat_main_iterations', 300, 2000, step=100)
    hp['cat_main_depth'] = trial.suggest_int('cat_main_depth', 6, 12)
    hp['cat_main_learning_rate'] = trial.suggest_float('cat_main_learning_rate', 0.005, 0.4, log=True)
    hp['cat_main_l2_leaf_reg'] = trial.suggest_float('cat_main_l2_leaf_reg', 0.5, 10.0)

    # CatBoost meta - Enfoque en precisión
    hp['cat_meta_iterations'] = trial.suggest_int('cat_meta_iterations', 200, 1000, step=100)
    hp['cat_meta_depth'] = trial.suggest_int('cat_meta_depth', 5, 10)
    hp['cat_meta_learning_rate'] = trial.suggest_float('cat_meta_learning_rate', 0.01, 0.3, log=True)
    hp['cat_meta_l2_leaf_reg'] = trial.suggest_float('cat_meta_l2_leaf_reg', 0.5, 8.0)

    # Optimización de períodos para el modelo principal
    n_periods_main = trial.suggest_int('n_periods_main', 5, 15)
    main_periods = []
    for i in range(n_periods_main):
        period_main = trial.suggest_int(f'period_main_{i}', 5, 200, log=True)
        main_periods.append(period_main)
    main_periods = sorted(list(set(main_periods)))  # Eliminar duplicados y ordenar
    if len(main_periods) < 3:  # Asegurar mínimo de períodos
        return -np.inf
    hp['periods_main'] = main_periods

    # Optimización de períodos para el meta-modelo
    n_periods_meta = 1 #trial.suggest_int('n_periods_meta', 1, 2)
    meta_periods = []
    for i in range(n_periods_meta):
        period_meta = trial.suggest_int(f'period_meta_{i}', 3, 5)
        meta_periods.append(period_meta)
    meta_periods = sorted(list(set(meta_periods)))  # Eliminar duplicados y ordenar
    hp['periods_meta'] = meta_periods

    # Selección de estadísticas para el modelo principal
    main_stat_choices = [
        "std", "skew", "kurt", "zscore", "mean", "range", "median", 
        "mad", "var", "entropy", "slope", "momentum", "roc", "fractal", "hurst"
    ]
    n_main_stats = trial.suggest_int('n_main_stats', 1, 5)
    selected_main_stats = []
    for i in range(n_main_stats):
        stat = trial.suggest_categorical(f'main_stat_{i}', main_stat_choices)
        selected_main_stats.append(stat)
    selected_main_stats = list(set(selected_main_stats))
    if len(selected_main_stats) == 1 and "fractal" in selected_main_stats:
        remaining_stats = [s for s in main_stat_choices if s != "fractal"]
        additional_stat = trial.suggest_categorical('additional_stat', remaining_stats)
        selected_main_stats.append(additional_stat)
    hp["stats_main"] = selected_main_stats
    #print(f"Main features seleccionadas: {hp['stats_main']}")

    # Selección de estadísticas para el meta-modelo
    meta_stat_choices = [
        "std", "skew", "zscore", "range", "mad", 
        "var", "entropy", "slope", "momentum", "roc"
    ]
    # Seleccionar una única estadística meta
    selected_meta_stat = trial.suggest_categorical('meta_stat', meta_stat_choices)
    hp["stats_meta"] = [selected_meta_stat]
    #print(f"Meta features seleccionadas: {hp['stats_meta']}")

    # Dataset completo
    full_ds = get_clustering_features(get_prices(hp), hp)
    
    # Dividir en períodos de entrenamiento, backward testing y forward testing
    ds_train = full_ds[(full_ds.index > hp['backward']) & (full_ds.index < hp['forward'])]
    ds_backward = full_ds[full_ds.index <= hp['backward']]  # Datos para backward testing
    ds_oos = full_ds[(full_ds.index >= hp['forward']) & (full_ds.index < hp['full forward'])]
    
    # Clustering con ventana deslizante
    data = sliding_window_clustering(
        ds_train,
        n_clusters=hp['n_clusters'],
        window_size=hp['window_size']
    )
    
    best_combined_score = -math.inf
    valid_clusters = 0
    
    # Calcular umbral mínimo adaptativo basado en el tamaño del dataset
    total_samples = len(data)
    min_samples_percent = 0.05  # 5% del total de muestras como mínimo
    min_samples_absolute = 200  # Mínimo absoluto
    min_samples_required = max(min_samples_absolute, int(total_samples * min_samples_percent))
    
    # Evaluar clusters ordenados por tamaño
    cluster_sizes = data['clusters'].value_counts()
    for clust in cluster_sizes.index:
        clustered_data = data[data['clusters'] == clust].copy()
        if len(clustered_data) < min_samples_required:
            continue
            
        valid_clusters += 1
        clustered_data = get_labels_one_direction(
            clustered_data,
            markup=hp['markup'],
            min=hp['label_min'],
            max=hp['label_max'],
            direction=hp['direction'])

        clustered_data = clustered_data.drop(['close', 'clusters'], axis=1)
        meta_data = data.copy()
        meta_data['clusters'] = (meta_data['clusters'] == clust).astype(int)

        # Evaluación en ambos períodos
        R2_forward, R2_backward, model_main, meta_model = fit_final_models(
            clustered_data,
            meta_data.drop(['close'], axis=1),
            ds_oos,
            ds_backward,
            hp
        )

        # Calcular puntuación combinada (puedes ajustar los pesos según necesites)
        # Esto prioriza modelos con buen rendimiento en ambos períodos
        forward_weight = 0.6  # Peso para el período forward
        backward_weight = 0.4  # Peso para el período backward
        
        # Penalización por diferencia excesiva (inconsistencia)
        consistency_penalty = 1.0
        diff = abs(R2_forward - R2_backward)
        if diff > 0.3:  # Si la diferencia es mayor al 30%
            consistency_penalty = 0.7  # Penalización del 30%
        
        # Cálculo de la puntuación combinada
        combined_score = ((R2_forward * forward_weight) + 
                          (R2_backward * backward_weight)) * consistency_penalty
        
        # Si ambos scores son negativos, el modelo es malo en ambos períodos
        # Recordar que fit_final_models asigna -1.0 a R2 cuando es NaN
        if R2_forward <= -1.0 and R2_backward <= -1.0:
            combined_score = -1.0
            
        # Si hay diferencia de signo, el modelo es inconsistente
        if (R2_forward > 0 and R2_backward <= -1.0) or (R2_forward <= -1.0 and R2_backward > 0):
            combined_score *= 0.5  # Penalización adicional

        if combined_score > best_combined_score:
            best_combined_score = combined_score
            # Guardar información del trial actual
            trial.set_user_attr("forward_r2", R2_forward)
            trial.set_user_attr("backward_r2", R2_backward)
            trial.set_user_attr("combined_score", combined_score)
            trial.set_user_attr("cluster_id", clust)
            trial.set_user_attr("stats_main", hp["stats_main"])
            trial.set_user_attr("stats_meta", hp["stats_meta"])
            trial.set_user_attr("periods_main", hp["periods_main"])
            trial.set_user_attr("periods_meta", hp["periods_meta"])
            # Guardar parámetros del trial actual (sin fechas)
            params_to_save = hp.copy()
            params_to_save.pop('backward', None)
            params_to_save.pop('forward', None)
            params_to_save.pop('full forward', None)
            trial.set_user_attr("params", params_to_save)
            # Si existe el estudio, actualizar sus atributos
            if study is not None:
                # Guardar información del mejor modelo encontrado
                study.set_user_attr("best_params", params_to_save)
                study.set_user_attr("best_metrics", {
                    "forward_r2": R2_forward,
                    "backward_r2": R2_backward,
                    "combined_score": combined_score,
                    "cluster_id": clust
                })
                study.set_user_attr("best_combined_score", combined_score)  # Guardar sin penalizaciones
                study.set_user_attr("best_models", [model_main, meta_model])
                study.set_user_attr("best_stats_main", hp["stats_main"])
                study.set_user_attr("best_stats_meta", hp["stats_meta"])
                study.set_user_attr("best_periods_main", hp["periods_main"])
                study.set_user_attr("best_periods_meta", hp["periods_meta"])
                study.set_user_attr("best_trial_number", trial.number)
                
    # Si no hay ningún cluster válido, devolver un valor negativo pero no infinito
    if best_combined_score == -math.inf:
        return -10.0

    # No aplicar penalización adicional por pocos clusters para mantener coherencia
    return best_combined_score

def optimize_and_export(symbol, timeframe, model_number, n_trials):
    """Lanza Optuna, guarda el mejor modelo y lo exporta a ONNX."""

    common_file_folder = r"/mnt/c/Users/Administrador/AppData/Roaming/MetaQuotes/Terminal/Common/Files/"
    mql5_files_folder = r'/mnt/c/Users/Administrador/AppData/Roaming/MetaQuotes/Terminal/6C3C6A11D1C3791DD4DBF45421BF8028/MQL5/Files/'
    mql5_include_folder = r'/mnt/c/Users/Administrador/AppData/Roaming/MetaQuotes/Terminal/6C3C6A11D1C3791DD4DBF45421BF8028/MQL5/Include/ajmtrz/include/Dmitrievsky'

    # Configurar el pruner inteligente
    pruner = SuccessiveHalvingPruner(
        min_resource=1,
        reduction_factor=3,
        min_early_stopping_rate=0
    )

    # Crear el estudio sin persistencia
    study = optuna.create_study(
        direction='maximize',
        pruner=pruner,
        sampler=optuna.samplers.TPESampler()
    )

    base_hp: Dict[str, Any] = {
        'symbol': symbol,
        'timeframe': timeframe,
        'models_export_path': mql5_files_folder,
        'include_export_path': mql5_include_folder,
        'history_path': common_file_folder,
        'stats_main': [],
        'stats_meta': [],
        'best_models': [None, None],
        'model_number': model_number,
        'markup': 0.20,
        'label_min'  : 1,
        'label_max'  : 15,
        'direction': 'buy',
        'n_clusters': 30,
        'window_size': 350,
        'periods_main': [i for i in range(5, 300, 30)],
        'periods_meta': [5],
        'backward': datetime(2020, 1, 2),
        'forward': datetime(2024, 1, 1),
        'full forward': datetime(2026, 1, 1),
    }

    study.optimize(lambda t: objective(t, base_hp, study), n_trials=n_trials, show_progress_bar=False)

    print("\n┌───────────────────────────────────────────────┐")
    print(f"│      MEJOR RESULTADO {model_number} = {study.user_attrs['best_metrics']['combined_score']:.4f}│")
    print("└───────────────────────────────────────────────┘\n")
    print("Parámetros óptimos:\n", study.best_params)
    
    # Verificar si hay métricas disponibles en user_attrs
    if 'best_metrics' in study.user_attrs:
        print(f"Evaluación del mejor modelo para estudio {model_number}:")
        print(f"  R2 Forward: {study.user_attrs['best_metrics']['forward_r2']:.4f}")
        print(f"  R2 Backward: {study.user_attrs['best_metrics']['backward_r2']:.4f}")
        print(f"  Puntuación combinada: {study.user_attrs['best_metrics']['combined_score']:.4f}")
    
    # Verificar que ambos modelos existan antes de exportarlos
    print("Exportando modelos ONNX...")
    export_params = base_hp.copy()
    export_params.update({
        "best_periods_main": study.user_attrs["best_periods_main"],
        "best_periods_meta": study.user_attrs["best_periods_meta"],
        "best_stats_main": study.user_attrs["best_stats_main"],
        "best_stats_meta": study.user_attrs["best_stats_meta"],
        "best_models":  study.user_attrs['best_models']
    })
    export_model_to_ONNX(**export_params)
    
    return {
        "forward_r2": study.user_attrs.get('best_metrics', {}).get('forward_r2', 0),
        "backward_r2": study.user_attrs.get('best_metrics', {}).get('backward_r2', 0),
        "combined_score": study.user_attrs.get('best_metrics', {}).get('combined_score', 0)
    }

if __name__ == "__main__":
    symbol = 'XAUUSD'
    timeframe = 'H1'
    n_trials_per_model = 5
    model_range = range(0, 5)
    
    # Para recopilar resultados globales de todos los modelos
    all_results = {}
    best_models = []
    
    for i in tqdm(model_range, desc=f"Optimizando {symbol}/{timeframe}", unit="modelo"):
        try:
            model_results = optimize_and_export(symbol, timeframe, i, n_trials=n_trials_per_model)
            best_models.append((i, model_results))
            
            # Añadir a resultados globales
            all_results[f"model_{i}"] = {
                "success": True,
                "forward_r2": model_results["forward_r2"],
                "backward_r2": model_results["backward_r2"],
                "combined_score": model_results["combined_score"]
            }
            
        except Exception as e:
            import traceback
            tqdm.write(f"\nError procesando modelo {i}: {str(e)}")
            tqdm.write(traceback.format_exc())
            
            all_results[f"model_{i}"] = {
                "success": False,
                "error": str(e)
            }
            continue
    
    # Resumen final
    print("\n" + "="*50)
    print(f"RESUMEN DE OPTIMIZACIÓN {symbol}/{timeframe}")
    print("="*50)
    
    successful_models = [info for model_main, info in all_results.items() if info.get("success", False)]
    print(f"Modelos completados exitosamente: {len(successful_models)}/{len(model_range)}")
    
    if successful_models:
        # Calcular estadísticas globales
        forward_scores = [info["forward_r2"] for info in successful_models]
        backward_scores = [info["backward_r2"] for info in successful_models]
        combined_scores = [info["combined_score"] for info in successful_models]
        
        print(f"\nEstadísticas de rendimiento:")
        print(f"  Forward R2 promedio: {np.mean(forward_scores):.4f} ± {np.std(forward_scores):.4f}")
        print(f"  Backward R2 promedio: {np.mean(backward_scores):.4f} ± {np.std(backward_scores):.4f}")
        print(f"  Puntuación combinada promedio: {np.mean(combined_scores):.4f} ± {np.std(combined_scores):.4f}")
        
        # Identificar el mejor modelo global basado en la puntuación combinada
        best_model_idx = np.argmax(combined_scores)
        best_model_key = list(all_results.keys())[best_model_idx]
        best_info = all_results[best_model_key]
        
        print(f"\nMejor modelo global: {best_model_key}")
        print(f"  Forward R2: {best_info['forward_r2']:.4f}")
        print(f"  Backward R2: {best_info['backward_r2']:.4f}")
        print(f"  Puntuación combinada: {best_info['combined_score']:.4f}")
    
    print("\nProceso de optimización completado.")