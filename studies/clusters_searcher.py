import os
import math
import random
import numpy as np
from tqdm import tqdm
from numba import njit
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Tuple, List
import optuna
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
                     hp: Dict[str, Any]) -> Tuple[float, Any, Any]:
    """Entrena modelo principal + meta‑modelo y evalúa en OOS.

    Devuelve (R2, model, meta_model).
    """
    # ---------- 1) main model ----------
    X_main = clustered.drop(columns=['labels', *meta.columns[meta.columns.str.contains('_meta_feature')]])
    y_main = clustered['labels'].astype('int16')

    # ---------- 2) meta‑model ----------
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
    model = CatBoostClassifier(**cat_main_params)
    model.fit(train_X, train_y, eval_set=(test_X, test_y), early_stopping_rounds=25)

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

    # 5) Evaluación en datos fuera de muestra
    R2 = test_model_one_direction_clustering(
        oos_data,
        [model, meta_model],
        hp['full forward'],
        hp['forward'],
        hp['markup'],
        hp['direction'],
        plt=False,
    )
    if math.isnan(R2):
        R2 = -1.0
    return R2, model, meta_model

def objective(trial: optuna.trial.Trial, base_hp: Dict[str, Any], study=None) -> float:
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
        period = trial.suggest_int(f'period_main_{i}', 2, 200, log=True)
        main_periods.append(period)
    main_periods = sorted(list(set(main_periods)))  # Eliminar duplicados y ordenar
    if len(main_periods) < 3:  # Asegurar mínimo de períodos
        return -np.inf
    hp['periods'] = main_periods

    # Optimización de períodos para el meta-modelo
    n_periods_meta = 1 #trial.suggest_int('n_periods_meta', 1, 2)
    meta_periods = []
    for i in range(n_periods_meta):
        period = trial.suggest_int(f'period_meta_{i}', 2, 5)
        meta_periods.append(period)
    meta_periods = sorted(list(set(meta_periods)))  # Eliminar duplicados y ordenar
    hp['periods_meta'] = meta_periods

    # Selección de estadísticas con pesos para el modelo principal
    stat_choices = {
        "std": 0.8,
        "skew": 0.6, 
        "kurt": 0.5,
        "zscore": 0.9,
        "mean": 0.7,
        "range": 0.8,
        "median": 0.6,
        "mad": 0.5,
        "var": 0.7,
        "entropy": 0.4,
        "slope": 0.9,
        "momentum": 0.8,
        "roc": 0.7,
        "fractal": 0.6,
        "hurst": 0.5
    }
    
    selected_stats = []
    for stat, weight in stat_choices.items():
        if trial.suggest_float(f"stat_weight_{stat}", 0, 1) > (1 - weight):
            selected_stats.append(stat)
            
    if len(selected_stats) < 3:  # Asegurar mínimo de features
        return -np.inf
    hp["stats_main"] = selected_stats

    # Selección de estadísticas para el meta-modelo
    meta_stat_choices = {
        "std": 0.8,
        "skew": 0.6, 
        "kurt": 0.5,
        "zscore": 0.9,
        "mean": 0.7,
        "range": 0.8,
        "median": 0.6,
        "mad": 0.5,
        "var": 0.7,
        "entropy": 0.4,
        "slope": 0.9,
        "momentum": 0.8,
        "roc": 0.7,
        "fractal": 0.6,
        "hurst": 0.5
    }
    
    selected_meta_stats = []
    for stat, weight in meta_stat_choices.items():
        if trial.suggest_float(f"meta_stat_weight_{stat}", 0, 1) > (1 - weight):
            selected_meta_stats.append(stat)
            
    if len(selected_meta_stats) < 1:  # Asegurar al menos una estadística
        return -np.inf
    hp["stats_meta"] = selected_meta_stats

    # Dataset completo
    full_ds = get_clustering_features(get_prices(hp), hp)
    ds_train = full_ds[(full_ds.index > hp['backward']) & (full_ds.index < hp['forward'])]
    ds_oos = full_ds[(full_ds.index >= hp['forward']) & (full_ds.index < hp['full forward'])]
    
    # Clustering con ventana deslizante
    data = sliding_window_clustering(
        ds_train,
        n_clusters=hp['n_clusters'],
        window_size=hp['window_size']
    )
    
    best_R2 = -math.inf
    valid_clusters = 0
    
    # Evaluar clusters ordenados por tamaño
    cluster_sizes = data['clusters'].value_counts()
    for clust in cluster_sizes.index:
        clustered_data = data[data['clusters'] == clust].copy()
        if len(clustered_data) < 750:  # Aumentar mínimo de muestras
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

        R2, model, meta_model = fit_final_models(
            clustered_data,
            meta_data.drop(['close'], axis=1),
            ds_oos,
            hp
        )

        if R2 > best_R2:
            best_R2 = R2
            best_pack = (model, meta_model)
            
            if study is not None:
                prev_best = study.user_attrs.get("best_r2", -np.inf)
                if best_R2 > prev_best:
                    study.set_user_attr("best_model", best_pack[0])
                    study.set_user_attr("best_meta_model", best_pack[1])
                    study.set_user_attr("best_r2", best_R2)
                    study.set_user_attr("best_stats_main", hp["stats_main"])
                    study.set_user_attr("best_stats_meta", hp["stats_meta"])
                    study.set_user_attr("best_periods", hp["periods"])
                    study.set_user_attr("best_periods_meta", hp["periods_meta"])
                    
    # Penalizar si muy pocos clusters válidos
    if valid_clusters < 3:
        best_R2 *= 0.5

    return best_R2

def optimize_and_export(symbol, timeframe, model_number, n_trials):
    """Lanza Optuna, guarda el mejor modelo y lo exporta a ONNX."""

    common_file_folder = r"/mnt/c/Users/Administrador/AppData/Roaming/MetaQuotes/Terminal/Common/Files/"
    mql5_files_folder = r'/mnt/c/Users/Administrador/AppData/Roaming/MetaQuotes/Terminal/6C3C6A11D1C3791DD4DBF45421BF8028/MQL5/Files/'
    mql5_include_folder = r'/mnt/c/Users/Administrador/AppData/Roaming/MetaQuotes/Terminal/6C3C6A11D1C3791DD4DBF45421BF8028/MQL5/Include/ajmtrz/include/Dmitrievsky'

    base_hp: Dict[str, Any] = {
        'symbol': symbol,
        'timeframe': timeframe,
        'models_export_path': mql5_files_folder,
        'include_export_path': mql5_include_folder,
        'history_path': common_file_folder,
        'best_models': [],
        'stats_main': [],
        'stats_meta': ["std"],  # Por defecto usa std
        'model_number': model_number,
        'markup': 0.20,
        'label_min'  : 1,
        'label_max'  : 15,
        'direction': 'buy',
        'n_clusters': 30,
        'window_size': 350,
        'periods': [i for i in range(5, 300, 30)],
        'periods_meta': [5],
        'backward': datetime(2020, 3, 26),
        'forward': datetime(2024, 1, 1),
        'full forward': datetime(2026, 1, 1),
    }

    study = optuna.create_study(direction='maximize')
    study.optimize(lambda t: objective(t, base_hp, study), n_trials=n_trials, show_progress_bar=False)

    print("\n┌───────────────────────────────────────────────┐")
    print(f"│      MEJOR RESULTADO {model_number} = {study.best_value:.4f}│")
    print("└───────────────────────────────────────────────┘\n")
    print("Parámetros óptimos:\n", study.best_params)

    # Recuperar el mejor modelo y meta‑modelo
    base_hp.update(study.best_params)
    model      = study.user_attrs["best_model"]
    meta_model = study.user_attrs["best_meta_model"]
    best_r2    = study.user_attrs["best_r2"]
    base_hp['stats_main'] = study.user_attrs["best_stats_main"]
    base_hp['stats_meta'] = study.user_attrs["best_stats_meta"]
    base_hp['periods'] = study.user_attrs["best_periods"]
    base_hp['periods_meta'] = study.user_attrs["best_periods_meta"]
    base_hp.pop('best_models', None)
    print("Exportando modelos ONNX… R2 = {:.4f}".format(best_r2))
    export_model_to_ONNX(best_models=[model, meta_model], **base_hp)

if __name__ == "__main__":
    symbol = 'XAUUSD'
    timeframe = 'H1'
    n_trials_per_model = 50
    model_range = range(0, 5)
    for i in tqdm(model_range, desc=f"Optimizando {symbol}/{timeframe}", unit="modelo"):
        try:
            optimize_and_export(symbol, timeframe, i, n_trials=n_trials_per_model)
        except Exception as e:
            tqdm.write(f"\nError procesando modelo {i}: {e}")
            continue
    print("Proceso de optimización completado.") 