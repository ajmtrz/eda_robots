import math
import time
import traceback
import random
import os
import psutil
import inspect
import numpy as np
import pandas as pd
from datetime import datetime
from time import perf_counter
from typing import Dict, Any
import optuna
from optuna.pruners import HyperbandPruner, SuccessiveHalvingPruner
# from optuna.integration import CatBoostPruningCallback
from catboost import CatBoostClassifier, CatBoostRegressor
from mapie.classification import CrossConformalClassifier
from mapie.regression import CrossConformalRegressor
from sklearn.model_selection import TimeSeriesSplit
from modules.labeling_lib import (
    get_prices, get_features, get_labels_random,
    get_labels_trend, get_labels_trend_multi, get_labels_clusters,
    get_labels_multi_window, get_labels_validated_levels,
    get_labels_mean_reversion,
    get_labels_mean_reversion_multi, get_labels_mean_reversion_vol,
    get_labels_filter, get_labels_filter_multi, get_labels_trend_filters,
    get_labels_filter_binary, get_labels_fractal_patterns, get_labels_zigzag,
    clustering_kmeans, clustering_hdbscan,
    clustering_markov, clustering_lgmm,
    wkmeans_clustering,
)
from modules.tester_lib import tester, clear_onnx_session_cache
from modules.export_lib import export_models_to_ONNX, export_dataset_to_csv, export_to_mql5

class StrategySearcher:
    LABEL_FUNCS = {
        # No usan confiabilidad
        "random": get_labels_random,
        "filter": get_labels_filter,
        "filter_binary": get_labels_filter_binary,
        "filter_multi": get_labels_filter_multi,
        # Usan confiabilidad
        "fractal": get_labels_fractal_patterns,
        "trend": get_labels_trend,
        "trend_multi": get_labels_trend_multi,
        "trend_filters": get_labels_trend_filters,
        "clusters": get_labels_clusters,
        "multi_window": get_labels_multi_window,
        "validated_levels": get_labels_validated_levels,
        "zigzag": get_labels_zigzag,
        "mean_rev": get_labels_mean_reversion,
        "mean_rev_multi": get_labels_mean_reversion_multi,
        "mean_rev_vol": get_labels_mean_reversion_vol,
    }
    # Allowed smoothing methods for label functions that support a 'filter' kwarg
    ALLOWED_FILTERS = {
        "trend_filters": {"savgol", "spline", "sma", "ema"},
        "trend_multi": {"savgol", "spline", "sma", "ema"},
        "mean_rev": {"mean", "spline", "savgol"},
        "mean_rev_vol": {"mean", "spline", "savgol"},
    }

    def __init__(
        self,
        symbol: str,
        timeframe: str,
        direction: str,
        train_start: datetime,
        train_end: datetime,
        test_start: datetime,
        test_end: datetime,
        pruner_type: str = 'hyperband',
        n_trials: int = 500,
        n_models: int = 10,
        n_jobs: int = 1,
        models_export_path: str = r'/mnt/c/Users/Administrador/AppData/Roaming/MetaQuotes/Terminal/6C3C6A11D1C3791DD4DBF45421BF8028/MQL5/Files/',
        include_export_path: str = r'/mnt/c/Users/Administrador/AppData/Roaming/MetaQuotes/Terminal/6C3C6A11D1C3791DD4DBF45421BF8028/MQL5/Include/ajmtrz/include/Dmitrievsky',
        history_path: str = r"/mnt/c/Users/Administrador/AppData/Roaming/MetaQuotes/Terminal/Common/Files/",
        search_type: str = 'clusters',
        search_subtype: str = 'kmeans',
        search_filter: str = '',
        label_method: str = 'random',
        label_type: str = 'classification',  # 'classification' o 'regression'
        tag: str = "",
        debug: bool = False,
        decimal_precision: int = 6,
    ):
        self.symbol = symbol
        self.timeframe = timeframe
        self.direction = direction
        if self.direction not in {'buy', 'sell', 'both'}:
            raise ValueError("direction must be 'buy', 'sell', or 'both'")
        self.train_start = train_start
        self.train_end = train_end
        self.test_start = test_start
        self.test_end = test_end
        self.models_export_path = models_export_path
        self.include_export_path = include_export_path
        self.history_path = history_path
        self.search_type = search_type
        self.search_subtype = search_subtype
        self.search_filter = search_filter
        self.label_method = label_method
        self.label_type = label_type
        self.pruner_type = pruner_type
        self.n_trials = n_trials
        self.n_models = n_models
        self.n_jobs = n_jobs
        self.tag = tag
        self.debug = debug
        self.base_df = get_prices(symbol, timeframe, history_path)
        self.decimal_precision = decimal_precision  # NUEVO ATRIBUTO

        # Configuración de logging para optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    # =========================================================================
    # Método principal
    # =========================================================================

    def run_search(self) -> None:
        search_funcs = {
            'reliability': self.search_reliability,
            'clusters': self.search_clusters,
            'mapie': self.search_mapie,
            'causal': self.search_causal,
        }
        
        if self.search_type not in search_funcs:
            raise ValueError(f"Tipo de búsqueda no válido: {self.search_type}")
            
        search_func = search_funcs[self.search_type]
        
        for i in range(self.n_models):
            try:

                # Inicializar estudio de Optuna con objetivo único
                pruners = {
                    'hyperband': HyperbandPruner(max_resource='auto'),
                    'sucessive': SuccessiveHalvingPruner(min_resource='auto')
                }
                study = optuna.create_study(
                    study_name=self.tag,
                    direction='maximize',
                    storage=f"sqlite:///optuna_dbs/{self.tag}.db",
                    load_if_exists=True,
                    pruner=pruners[self.pruner_type],
                    sampler=optuna.samplers.TPESampler(
                        n_startup_trials=int(np.sqrt(self.n_trials)*1.5),
                        multivariate=True, group=True,
                        warn_independent_sampling=False
                    )
                )

                t0 = perf_counter()
                def log_trial(study, trial):
                    def _log_memory() -> float:
                        try:
                            mem = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
                            return mem
                        except Exception:
                            pass
                    try:
                        # Obtener el mejor trial
                        if study.best_trial and study.best_trial.value > 0.0:
                            best_trial = study.best_trial
                            # Si este trial es el mejor, guardar sus modelos
                            if trial.number == best_trial.number:
                                if trial.user_attrs.get('model_paths') is not None:
                                    # Eliminar modelos anteriores
                                    if study.user_attrs.get("best_model_paths"):
                                        for p in study.user_attrs["best_model_paths"]:
                                            if p and os.path.exists(p):
                                                os.remove(p)
                                    if study.user_attrs.get("best_full_ds_with_labels_path"):
                                        if os.path.exists(study.user_attrs["best_full_ds_with_labels_path"]):
                                            os.remove(study.user_attrs["best_full_ds_with_labels_path"])
                                    # Guardar nuevas rutas de modelos
                                    study.set_user_attr("best_score", trial.user_attrs['score'])
                                    study.set_user_attr("best_model_paths", trial.user_attrs['model_paths'])
                                    study.set_user_attr("best_full_ds_with_labels_path", trial.user_attrs['full_ds_with_labels_path'])
                                    study.set_user_attr("best_periods_main", trial.user_attrs.get('feature_main_periods'))
                                    study.set_user_attr("best_stats_main", trial.user_attrs.get('feature_main_stats'))
                                    study.set_user_attr("best_model_cols", trial.user_attrs['model_cols'])
                                    # THRESHOLDS UNIFICADOS: guardar thresholds apropiados
                                    study.set_user_attr("best_main_threshold", trial.user_attrs.get('best_main_threshold', 0.5))
                                    study.set_user_attr("best_meta_threshold", trial.user_attrs.get('best_meta_threshold', 0.5))
                                    # Cambiar acceso directo por .get para evitar error si no existe
                                    study.set_user_attr("best_periods_meta", trial.user_attrs.get('feature_meta_periods'))
                                    study.set_user_attr("best_stats_meta", trial.user_attrs.get('feature_meta_stats'))
                                    # Exportar modelo
                                    export_params = {
                                        "tag": self.tag,
                                        "direction": self.direction,
                                        "label_type": self.label_type,
                                        "models_export_path": self.models_export_path,
                                        "include_export_path": self.include_export_path,
                                        "best_score": study.user_attrs["best_score"],
                                        "best_full_ds_with_labels_path": study.user_attrs["best_full_ds_with_labels_path"],
                                        "best_model_paths": study.user_attrs["best_model_paths"],
                                        "best_model_cols": study.user_attrs["best_model_cols"],
                                        "best_periods_main": study.user_attrs["best_periods_main"],
                                        "best_periods_meta": study.user_attrs["best_periods_meta"],
                                        "best_stats_main": study.user_attrs["best_stats_main"],
                                        "best_stats_meta": study.user_attrs["best_stats_meta"],
                                        "best_main_threshold": study.user_attrs.get("best_main_threshold"),
                                        "best_meta_threshold": study.user_attrs.get("best_meta_threshold"),
                                        "decimal_precision": self.decimal_precision,
                                    }
                                    export_to_mql5(**export_params)

                                    # Eliminar archivos temporales del mejor modelo
                                    for p in study.user_attrs.get("best_model_paths", []):
                                        if p and os.path.exists(p):
                                            os.remove(p)
                                    if os.path.exists(study.user_attrs["best_full_ds_with_labels_path"]):
                                        os.remove(study.user_attrs["best_full_ds_with_labels_path"])
                                    # Parar el algoritmo
                                    if self.debug:
                                        #if trial.number > 1:
                                            study.stop()

                            # Liberar memoria eliminando datos pesados del trial
                            if 'model_paths' in trial.user_attrs and trial.user_attrs['model_paths']:
                                for p in trial.user_attrs['model_paths']:
                                    if p and os.path.exists(p):
                                        os.remove(p)
                            if 'full_ds_with_labels_path' in trial.user_attrs and trial.user_attrs['full_ds_with_labels_path']:
                                if os.path.exists(trial.user_attrs['full_ds_with_labels_path']):
                                    os.remove(trial.user_attrs['full_ds_with_labels_path'])

                        # Log
                        if study.best_trial:
                            best_trial = study.best_trial
                            best_str = f"score={best_trial.value:.6f}"
                        else:
                            best_str = "score=---"
                        elapsed = perf_counter() - t0
                        n_done = trial.number + 1
                        avg_time = elapsed / n_done
                        mem_details = ""
                        if hasattr(self, "_trial_memory"):
                            mem_details = " ".join(
                                f"{name}:{mem:.2f}MB" for name, mem in self._trial_memory
                            )
                        print(
                            f"[{self.tag}] modelo {i}",
                            f"trial {n_done}/{self.n_trials}",
                            f"{best_str}",
                            f"avg={avg_time:.2f}s",
                            f"mem={_log_memory():.2f}MB",
                            mem_details,
                            flush=True,
                        )

                    except Exception as e:
                        print(f"⚠️ ERROR en log_trial: {str(e)}")

                study.optimize(
                    search_func,
                    n_trials=self.n_trials,
                    gc_after_trial=True,
                    show_progress_bar=False,
                    callbacks=[log_trial],
                    n_jobs=self.n_jobs,
                )
                
                # 🔍 DEBUG: Verificar por qué se paró el estudio
                print(f"🔍 DEBUG: Study terminado después de {len(study.trials)} trials")
                print(f"🔍   n_trials configurado: {self.n_trials}")
                print(f"🔍   best_trial: {study.best_trial}")
                if study.best_trial:
                    print(f"🔍   best_score: {study.best_trial.value}")
                print(f"🔍   trials completados: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
                print(f"🔍   trials pruned: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
                print(f"🔍   trials failed: {len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])}")
                
            except Exception as e:
                print(f"\nError procesando modelo {i}:")
                print(f"Error: {str(e)}")
                print("Traceback:")
                print(traceback.format_exc())
                continue

    # =========================================================================
    # Métodos de búsqueda específicos
    # =========================================================================

    def search_reliability(self, trial: optuna.Trial) -> float:
        """
        Búsqueda basada en confiabilidad de patrones (esquema puro de confiabilidad).
        
        Este método implementa el esquema original del artículo MQL5 donde:
        - Meta model: "¿es confiable el patrón?" (1=confiable, 0=no confiable)
        - Main model: "¿buy o sell?" (solo en muestras confiables)
        
        NOTA: Con el Enfoque 4 de confiabilidad implementado, ahora cualquier 
        método de búsqueda (clusters, mapie, causal, etc.) puede usar automáticamente
        etiquetado fractal. Este método se mantiene para compatibilidad y como
        referencia de la implementación original del artículo.
        """
        try:
            hp = self.suggest_all_params(trial)
            
            # 🔍 DEBUG: Supervisar parámetros específicos de confiabilidad
            if self.debug:
                reliability_params = {k: v for k, v in hp.items() if k.startswith('label_')}
                print(f"🔍 DEBUG search_reliability - Parámetros de confiabilidad: {reliability_params}")
                
            full_ds = self.get_labeled_full_data(hp)
            if full_ds is None:
                return -1.0

            # Main base mask
            base_mask = self.get_base_mask(full_ds)
            if not base_mask.any():
                if self.debug:
                    print(f"🔍 DEBUG search_reliability - No hay muestras de trading")
                return -1.0

            # APLICAR FILTROS SECUNDARIOS (opcional)
            if self.search_filter == 'mapie':
                if self.debug:
                    print(f"🔍 DEBUG search_reliability - Aplicando filtrado MAPIE")
                    print(f"🔍   base_mask.sum(): {base_mask.sum()}")
                    print(f"🔍   base_mask.mean(): {base_mask.mean():.3f}")
                
                mapie_scores = self.apply_mapie_filter(trial, full_ds, hp, base_mask)
                mapie_mask = mapie_scores == 1.0
                final_mask = base_mask & mapie_mask
                
                if self.debug:
                    print(f"🔍   mapie_mask.sum(): {mapie_mask.sum()}")
                    print(f"🔍   mapie_mask.mean(): {mapie_mask.mean():.3f}")
                    print(f"🔍   final_mask.sum(): {final_mask.sum()}")
                    print(f"🔍   final_mask.mean(): {final_mask.mean():.3f}")
                    if base_mask.sum() > 0:
                        print(f"🔍   Reducción de muestras: {base_mask.sum()} -> {final_mask.sum()} ({final_mask.sum()/base_mask.sum()*100:.1f}%)")
            elif self.search_filter == 'causal':
                if self.debug:
                    print(f"🔍 DEBUG search_reliability - Aplicando filtrado CAUSAL")
                    print(f"🔍   base_mask.sum(): {base_mask.sum()}")
                    print(f"🔍   base_mask.mean(): {base_mask.mean():.3f}")
                
                causal_scores = self.apply_causal_filter(trial, full_ds, hp, base_mask)
                causal_mask = causal_scores == 1.0
                final_mask = base_mask & causal_mask
                
                if self.debug:
                    print(f"🔍   causal_mask.sum(): {causal_mask.sum()}")
                    print(f"🔍   causal_mask.mean(): {causal_mask.mean():.3f}")
                    print(f"🔍   final_mask.sum(): {final_mask.sum()}")
                    print(f"🔍   final_mask.mean(): {final_mask.mean():.3f}")
                    if base_mask.sum() > 0:
                        print(f"🔍   Reducción de muestras: {base_mask.sum()} -> {final_mask.sum()} ({final_mask.sum()/base_mask.sum()*100:.1f}%)")
            else:
                final_mask = base_mask
                
                if self.debug:
                    print(f"🔍 DEBUG search_reliability - Sin filtrado MAPIE o CAUSAL")
                    print(f"🔍   final_mask.sum(): {final_mask.sum()}")
                    print(f"🔍   final_mask.mean(): {final_mask.mean():.3f}")

            # Crear dataset main con final_mask
            main_feature_cols = [c for c in full_ds.columns if c.endswith('_main_feature')]
            model_main_train_data = full_ds.loc[final_mask, main_feature_cols + ['labels_main']].dropna(subset=main_feature_cols).copy()
            if len(model_main_train_data) < 200:
                if self.debug:
                    print(f"🔍 DEBUG search_reliability - Insuficientes muestras main: {len(model_main_train_data)}")
                return -1.0

            # Crear dataset meta con final_mask + threshold
            if self.label_type == 'regression':
                hp['main_threshold'], final_mask = self.calculate_regression_threshold_cv(
                    full_ds['labels_main'], reliability_mask=final_mask, hp=hp
                )
            full_ds['labels_meta'] = final_mask.astype('int8')
            meta_feature_cols = [c for c in full_ds.columns if c.endswith('_meta_feature')]
            if not meta_feature_cols:  # Fallback: usar main features si no hay meta features
                meta_feature_cols = main_feature_cols
            model_meta_train_data = full_ds[meta_feature_cols + ['labels_meta']].dropna(subset=meta_feature_cols).copy()
            
            if self.debug:
                print(f"🔍 DEBUG search_reliability - Main data shape: {model_main_train_data.shape}")
                print(f"🔍   Meta data shape: {model_meta_train_data.shape}")
                meta_dist = model_meta_train_data['labels_meta'].value_counts()
                print(f"🔍   Meta labels distribution: {meta_dist}")
            
            # Verificar distribución de clases
            if self.label_type == 'classification':
                if set(model_main_train_data['labels_main'].unique()) != {0.0, 1.0}:
                    if self.debug:
                        print(f"🔍   Search reliability - labels_main insuficientes")
                    return -1.0
            elif self.label_type == 'regression':
                if set(model_main_train_data['labels_main'].unique()) == {0.0}:
                    if self.debug:
                        print(f"🔍   Search reliability - labels_main insuficientes")
                    return -1.0
            if set(model_meta_train_data['labels_meta'].unique()) != {0.0, 1.0}:
                if self.debug:
                    print(f"🔍   Search reliability - labels_meta insuficientes")
                return -1.0
                
            # Usar pipeline existente
            score, full_ds_with_labels_path, model_paths, models_cols = self.fit_final_models(
                trial=trial,
                full_ds=full_ds,
                model_main_train_data=model_main_train_data,
                model_meta_train_data=model_meta_train_data,
                hp=hp.copy()
            )
            
            if score is None or model_paths is None or models_cols is None or full_ds_with_labels_path is None:
                return -1.0
                
            trial.set_user_attr('score', score)
            trial.set_user_attr('model_paths', model_paths)
            trial.set_user_attr('model_cols', models_cols)
            trial.set_user_attr('full_ds_with_labels_path', full_ds_with_labels_path)
            trial.set_user_attr('best_meta_threshold', hp.get('meta_threshold', 0.5))
            return trial.user_attrs.get('score', -1.0)
            
        except Exception as e:
            print(f"Error en search_reliability: {str(e)}")
            return -1.0

    def search_clusters(self, trial: optuna.Trial) -> float:
        """Implementa la búsqueda de estrategias usando clustering."""
        def _clustering_method(data, hp):
            try:
                if self.search_subtype == 'kmeans':
                    reliable_data = clustering_kmeans(
                        data,
                        n_clusters=hp['kmeans_n_clusters'],
                        window_size=hp['kmeans_window'],
                        step=hp.get('kmeans_step', None)
                    )
                elif self.search_subtype == 'hdbscan':
                    reliable_data = clustering_hdbscan(
                        data,
                        n_clusters=hp['hdbscan_min_cluster_size']
                    )
                elif self.search_subtype == 'markov':
                    reliable_data = clustering_markov(
                        data,
                        model_type=hp['markov_model'],
                        n_regimes=hp['markov_regimes'],
                        n_iter=hp.get('markov_iter', 100),
                        n_mix=hp.get('markov_mix', 3)
                    )
                elif self.search_subtype == 'lgmm':
                    reliable_data = clustering_lgmm(
                        data,
                        n_components=hp['lgmm_components'],
                        covariance_type=hp.get('lgmm_covariance', 'full'),
                        max_iter=hp.get('lgmm_iter', 100),
                    )
                elif self.search_subtype == 'wkmeans':
                    reliable_data = wkmeans_clustering(
                        data,
                        n_clusters=hp["wk_n_clusters"],
                        window_size=hp["wk_window"],
                        metric="wasserstein",
                        step=hp.get("wk_step", 1),
                        bandwidth=hp.get("wk_bandwidth", 1.0),
                        n_proj=hp.get("wk_proj", 100),
                        max_iter=hp.get("wk_iter", 300),
                    )

                return reliable_data
            
            except Exception as e:
                print(f"Error en _clustering_method: {str(e)}")
                return pd.DataFrame()
        
        try:
            hp = self.suggest_all_params(trial)
            
            # 🔍 DEBUG: Supervisar parámetros específicos de clusters
            if self.debug:
                clust_params = {k: v for k, v in hp.items() if k.startswith('clust_')}
                print(f"🔍 DEBUG search_clusters {self.search_subtype} - Parámetros clusters: {clust_params}")
                
            full_ds = self.get_labeled_full_data(hp)
            if full_ds is None:
                return -1.0
            
            base_mask = self.get_base_mask(full_ds)
            reliable_data = full_ds[base_mask].copy()
            if len(reliable_data) < 200:
                if self.debug:
                    print(f"🔍 DEBUG search_clusters {self.search_subtype} - Insuficientes muestras confiables para hacer clustering")
                return -1.0
            
            # 🔍 DEBUG: Verificar distribución después del filtrado
            if self.debug:
                print(f"🔍 DEBUG search_clusters {self.search_subtype} - Filtrado por {self.label_type}:")
                print(f"🔍   Total muestras: {len(full_ds)}")
                print(f"🔍   Muestras confiables: {base_mask.sum()} ({base_mask.mean():.1%})")
                print(f"🔍   Total muestras confiables: {len(reliable_data)}")
            
            # Aplicar clustering
            reliable_data_clustered = _clustering_method(reliable_data, hp)
            # Propagar clusters a dataset completo
            full_ds.loc[base_mask, 'labels_meta'] = reliable_data_clustered['labels_meta']
            full_ds.loc[~base_mask, 'labels_meta'] = -1  # Muestras no confiables sin cluster

            score, full_ds_with_labels_path, model_paths, models_cols, best_main_threshold = self.evaluate_clusters(trial, full_ds, base_mask, hp)
            if score is None or model_paths is None or models_cols is None or full_ds_with_labels_path is None:
                return -1.0
            
            trial.set_user_attr('score', score)
            trial.set_user_attr('model_paths', model_paths)
            trial.set_user_attr('model_cols', models_cols)
            trial.set_user_attr('full_ds_with_labels_path', full_ds_with_labels_path)
            trial.set_user_attr('best_main_threshold', best_main_threshold)
            trial.set_user_attr('best_meta_threshold', hp.get('meta_threshold', 0.5))

            return trial.user_attrs.get('score', -1.0)
        except Exception as e:
            print(f"Error en search_clusters: {str(e)}")
            return -1.0
        
    def search_mapie(self, trial) -> float:
        """Implementa la búsqueda de estrategias usando conformal prediction (MAPIE) con CatBoost, usando el mismo conjunto de features para ambos modelos."""
        try:
            hp = self.suggest_all_params(trial)

            full_ds = self.get_labeled_full_data(hp)
            if full_ds is None:
                return -1.0

            # Usar esquema de confiabilidad directamente
            if self.debug:
                print(f"🔍 DEBUG search_mapie - Usando esquema de confiabilidad")

            # Main: solo muestras con señales, según label_type
            base_mask = self.get_base_mask(full_ds)

            # 2. Filtro secundario: MAPIE como método principal
            mapie_scores = self.apply_mapie_filter(
                trial=trial, 
                full_ds=full_ds, 
                hp=hp, 
                reliable_mask=base_mask
            )
            
            # 3. Combinación: muestras que pasan AMBOS filtros
            final_mask = base_mask & (mapie_scores == 1.0)

            # Main model filtrado
            main_feature_cols = [col for col in full_ds.columns if col.endswith('_main_feature')]
            model_main_train_data = full_ds[final_mask][main_feature_cols + ['labels_main']].dropna(subset=main_feature_cols).copy()
            if len(model_main_train_data) < 200:
                if self.debug:
                    print(f"🔍 DEBUG search_mapie - Insuficientes muestras main: {len(model_main_train_data)}")
                return -1.0
            
            # Meta model debe usar meta features, si no hay meta features, usar main features (fallback)
            if self.label_type == 'regression':
                hp['main_threshold'], final_mask = self.calculate_regression_threshold_cv(
                    full_ds['labels_main'], reliability_mask=final_mask, hp=hp
                )
            full_ds['labels_meta'] = final_mask.astype('int8')
            meta_feature_cols = [col for col in full_ds.columns if col.endswith('_meta_feature')]
            if not meta_feature_cols:  # Fallback: usar main features si no hay meta features
                meta_feature_cols = main_feature_cols
            model_meta_train_data = full_ds[meta_feature_cols + ['labels_meta']].dropna(subset=meta_feature_cols).copy()

            if self.debug:
                print(f"🔍 DEBUG search_mapie - Main data shape: {model_main_train_data.shape}")
                print(f"🔍   Meta data shape: {model_meta_train_data.shape}")
                meta_dist = model_meta_train_data['labels_meta'].value_counts()
                print(f"🔍   Meta labels distribution: {meta_dist}")

            # Verificar distribución de clases
            if self.label_type == 'classification':
                if set(model_main_train_data['labels_main'].unique()) != {0.0, 1.0}:
                    if self.debug:
                        print(f"🔍   Search mapie - labels_main insuficientes")
                    return -1.0
            elif self.label_type == 'regression':
                if set(model_main_train_data['labels_main'].unique()) == {0.0}:
                    if self.debug:
                        print(f"🔍   Search mapie - labels_main insuficientes")
                    return -1.0
            if set(model_meta_train_data['labels_meta'].unique()) != {0.0, 1.0}:
                if self.debug:
                    print(f"🔍   Search mapie - labels_meta insuficientes")
                return -1.0
            
            score, full_ds_with_labels_path, model_paths, models_cols = self.fit_final_models(
                trial=trial,
                full_ds=full_ds,
                model_main_train_data=model_main_train_data,
                model_meta_train_data=model_meta_train_data,
                hp=hp.copy()
            )
                
            if score is None or model_paths is None or models_cols is None or full_ds_with_labels_path is None:
                return -1.0
            
            trial.set_user_attr('score', score)
            trial.set_user_attr('model_paths', model_paths)
            trial.set_user_attr('model_cols', models_cols)
            trial.set_user_attr('full_ds_with_labels_path', full_ds_with_labels_path)
            trial.set_user_attr('best_meta_threshold', hp.get('meta_threshold', 0.5))

            return trial.user_attrs.get('score', -1.0)
        except Exception as e:
            print(f"Error en search_mapie: {str(e)}")
            return -1.0

    def search_causal(self, trial: optuna.Trial) -> float:
        """Búsqueda basada en detección causal de muestras malas."""
        try:
            hp = self.suggest_all_params(trial)

            full_ds = self.get_labeled_full_data(hp)
            if full_ds is None:
                return -1.0

            # Usar esquema de confiabilidad directamente
            if self.debug:
                print(f"🔍 DEBUG search_causal - Usando esquema de confiabilidad")
            
            # Main base mask
            base_mask = self.get_base_mask(full_ds)

            # 2. Filtro secundario: CAUSAL como método principal
            causal_scores = self.apply_causal_filter(
                trial=trial, 
                full_ds=full_ds, 
                hp=hp, 
                reliable_mask=base_mask
            )
            
            # 3. Combinación: muestras que pasan AMBOS filtros
            final_mask = base_mask & (causal_scores == 1.0)
                
            # Crear dataset main con final_mask
            main_feature_cols = [col for col in full_ds.columns if col.endswith('_main_feature')]
            model_main_train_data = full_ds[final_mask][main_feature_cols + ['labels_main']].dropna(subset=main_feature_cols).copy()
            if len(model_main_train_data) < 200:
                if self.debug:
                    print(f"🔍 DEBUG search_causal - Insuficientes muestras main: {len(model_main_train_data)}")
                return -1.0
            
            # Crear dataset meta con final_mask + threshold
            if self.label_type == 'regression':
                hp['main_threshold'], final_mask = self.calculate_regression_threshold_cv(
                    full_ds['labels_main'], reliability_mask=final_mask, hp=hp
                )
            full_ds['labels_meta'] = final_mask.astype('int8')
            meta_feature_cols = [col for col in full_ds.columns if col.endswith('_meta_feature')]
            if not meta_feature_cols:  # Fallback: usar main features si no hay meta features
                meta_feature_cols = main_feature_cols
            model_meta_train_data = full_ds[meta_feature_cols + ['labels_meta']].dropna(subset=meta_feature_cols).copy()
            
            if self.debug:
                print(f"🔍 DEBUG search_reliability - Main data shape: {model_main_train_data.shape}")
                print(f"🔍   Meta data shape: {model_meta_train_data.shape}")
                meta_dist = model_meta_train_data['labels_meta'].value_counts()
                print(f"🔍   Meta labels distribution: {meta_dist}")
                
            # Verificar distribución de clases
            if self.label_type == 'classification':
                if set(model_main_train_data['labels_main'].unique()) != {0.0, 1.0}:
                    if self.debug:
                        print(f"🔍   Search causal - labels_main insuficientes")
                    return -1.0
            elif self.label_type == 'regression':
                if set(model_main_train_data['labels_main'].unique()) == {0.0}:
                    if self.debug:
                        print(f"🔍   Search causal - labels_main insuficientes")
                    return -1.0
            if set(model_meta_train_data['labels_meta'].unique()) != {0.0, 1.0}:
                if self.debug:
                    print(f"🔍   Search causal - labels_meta insuficientes")
                return -1.0
            
            score, full_ds_with_labels_path, model_paths, models_cols = self.fit_final_models(
                trial=trial,
                full_ds=full_ds,
                model_main_train_data=model_main_train_data,
                model_meta_train_data=model_meta_train_data,
                hp=hp.copy()
            )
                
            if score is None or model_paths is None or models_cols is None or full_ds_with_labels_path is None:
                return -1.0
            
            trial.set_user_attr('score', score)
            trial.set_user_attr('model_paths', model_paths)
            trial.set_user_attr('model_cols', models_cols)
            trial.set_user_attr('full_ds_with_labels_path', full_ds_with_labels_path)
            trial.set_user_attr('best_meta_threshold', hp.get('meta_threshold', 0.5))

            return trial.user_attrs.get('score', -1.0)
        except Exception as e:
            print(f"Error en search_causal: {str(e)}")
            return -1.0

    # =========================================================================
    # Métodos auxiliares
    # =========================================================================
    
    def evaluate_clusters(self, trial: optuna.trial, full_ds: pd.DataFrame, base_mask: pd.Series, hp: Dict[str, Any]) -> tuple[float, tuple, tuple]:
        """Función helper para evaluar clusters y entrenar modelos."""
        try:
            # Esquema tradicional de clusters
            best_score = -math.inf
            best_model_paths = (None, None)
            best_models_cols = (None, None)
            best_full_ds_with_labels_path = None
            best_main_threshold = None

            # 🔍 DEBUG: Supervisar parámetros
            if self.debug:
                validation_params = {k: v for k, v in hp.items() if k.startswith('label_')}
                print(f"🔍 DEBUG evaluate_clusters - Parámetros de validación: {validation_params}")

            # Extraer clusters
            cluster_sizes = full_ds['labels_meta'].value_counts().sort_index()
            if self.debug:
                print(f"🔍 DEBUG: Cluster sizes:\n{cluster_sizes}")
            if -1 in cluster_sizes.index:
                cluster_sizes = cluster_sizes.drop(-1)
            if cluster_sizes.empty:
                if self.debug:
                    print("⚠️ ERROR: No hay clusters")
                return None, None, None, None, None

            # Evaluar cada cluster
            for clust in cluster_sizes.index:
                cluster_mask = full_ds['labels_meta'] == clust
                cluster_mask = cluster_mask & base_mask
                if not cluster_mask.any():
                    if self.debug:
                        print(f"🔍   Cluster {clust} descartado: sin muestras confiables")
                    continue
                
                # APLICAR MAPIE COMO FILTRO SECUNDARIO (opcional)
                if self.search_filter == 'mapie':
                    if self.debug:
                        print(f"🔍 DEBUG evaluate_clusters - Aplicando filtrado MAPIE al cluster {clust}")
                        print(f"🔍   cluster_mask.sum(): {cluster_mask.sum()}")
                        print(f"🔍   cluster_mask.mean(): {cluster_mask.mean():.3f}")
                    
                    mapie_scores = self.apply_mapie_filter(trial, full_ds, hp, cluster_mask)
                    mapie_mask = mapie_scores == 1.0
                    final_mask = cluster_mask & mapie_mask
                    
                    if self.debug:
                        print(f"🔍   mapie_mask.sum(): {mapie_mask.sum()}")
                        print(f"🔍   mapie_mask.mean(): {mapie_mask.mean():.3f}")
                        print(f"🔍   final_mask.sum(): {final_mask.sum()}")
                        print(f"🔍   final_mask.mean(): {final_mask.mean():.3f}")
                        if cluster_mask.sum() > 0:
                            print(f"🔍   Reducción de muestras: {cluster_mask.sum()} -> {final_mask.sum()} ({final_mask.sum()/cluster_mask.sum()*100:.1f}%)")
                elif self.search_filter == 'causal':
                    if self.debug:
                        print(f"🔍 DEBUG evaluate_clusters - Aplicando filtrado CAUSAL al cluster {clust}")
                        print(f"🔍   cluster_mask.sum(): {cluster_mask.sum()}")
                        print(f"🔍   cluster_mask.mean(): {cluster_mask.mean():.3f}")
                    
                    causal_scores = self.apply_causal_filter(trial, full_ds, hp, cluster_mask)
                    causal_mask = causal_scores == 1.0
                    final_mask = cluster_mask & causal_mask
                    
                    if self.debug:
                        print(f"🔍   causal_mask.sum(): {causal_mask.sum()}")
                        print(f"🔍   causal_mask.mean(): {causal_mask.mean():.3f}")
                        print(f"🔍   final_mask.sum(): {final_mask.sum()}")
                        print(f"🔍   final_mask.mean(): {final_mask.mean():.3f}")
                        if cluster_mask.sum() > 0:
                            print(f"🔍   Reducción de muestras: {cluster_mask.sum()} -> {final_mask.sum()} ({final_mask.sum()/cluster_mask.sum()*100:.1f}%)")
                else:
                    final_mask = cluster_mask
                    
                    if self.debug:
                        print(f"🔍 DEBUG evaluate_clusters - Sin filtrado MAPIE o CAUSAL al cluster {clust}")
                        print(f"🔍   final_mask.sum(): {final_mask.sum()}")
                        print(f"🔍   final_mask.mean(): {final_mask.mean():.3f}")
                
                # Crear dataset main con final_mask
                main_feature_cols = [c for c in full_ds.columns if c.endswith('_main_feature')]
                model_main_train_data = full_ds.loc[final_mask, main_feature_cols + ['labels_main']].dropna(subset=main_feature_cols).copy()
                if len(model_main_train_data) < 200:
                    if self.debug:
                        print(f"🔍 DEBUG evaluate_clusters - Insuficientes muestras main: {len(model_main_train_data)}")
                    continue

                # Crear dataset meta con final_mask + threshold
                if self.label_type == 'regression':
                    hp['main_threshold'], final_mask = self.calculate_regression_threshold_cv(
                        full_ds['labels_main'], reliability_mask=final_mask, hp=hp
                    )
                meta_feature_cols = [c for c in full_ds.columns if c.endswith('_meta_feature')]
                model_meta_train_data = full_ds[meta_feature_cols].dropna(subset=meta_feature_cols).copy()
                model_meta_train_data['labels_meta'] = final_mask.loc[model_meta_train_data.index].astype('int8')

                # Verificar que tenemos suficientes muestras para main y meta
                if self.label_type == 'classification':
                    if set(model_main_train_data['labels_main'].unique()) != {0.0, 1.0}:
                        if self.debug:
                            print(f"🔍   Cluster {clust} descartado: labels_main insuficientes")
                        continue
                elif self.label_type == 'regression':
                    if set(model_main_train_data['labels_main'].unique()) == {0.0}:
                        if self.debug:
                            print(f"🔍   Cluster {clust} descartado: labels_main insuficientes")
                        continue
                if set(model_meta_train_data['labels_meta'].unique()) != {0.0, 1.0}:
                    if self.debug:
                        print(f"🔍   Cluster {clust} descartado: labels_meta insuficientes")
                    continue

                # Información de debug para cluster
                if self.debug:
                    print(f"🔍   Evaluando cluster {clust}:")
                    print(f"🔍      Main data shape: {model_main_train_data.shape}")
                    if self.label_type == 'classification':
                        main_dist = model_main_train_data['labels_main'].value_counts()
                        print(f"🔍      Main labels: {main_dist}")
                    else:
                        print(f"🔍      Main labels: {model_main_train_data['labels_main'].min()}, {model_main_train_data['labels_main'].max()}")
                    print(f"🔍      Meta data shape: {model_meta_train_data.shape}")
                    meta_dist = model_meta_train_data['labels_meta'].value_counts()
                    print(f"🔍      Meta labels: {meta_dist}")
                    
                # Entrenar modelos
                score, full_ds_with_labels_path, model_paths, models_cols = self.fit_final_models(
                    trial=trial,
                    full_ds=full_ds,
                    model_main_train_data=model_main_train_data,
                    model_meta_train_data=model_meta_train_data,
                    hp=hp.copy()
                )
                if score is None or model_paths is None or models_cols is None:
                    continue
                if score > best_score:
                    if best_model_paths != (None, None):
                        for p in best_model_paths:
                            if p and os.path.exists(p):
                                os.remove(p)
                    if best_full_ds_with_labels_path and os.path.exists(best_full_ds_with_labels_path):
                        os.remove(best_full_ds_with_labels_path)
                    best_score = score
                    best_main_threshold = hp.get('main_threshold', 0.5)
                    best_model_paths = model_paths
                    best_full_ds_with_labels_path = full_ds_with_labels_path
                    best_models_cols = models_cols

                    if self.debug:
                        print(f"🔍   Nuevo mejor cluster {clust}: score = {score}")
                else:
                    for p in model_paths:
                        if p and os.path.exists(p):
                            os.remove(p)
                    if full_ds_with_labels_path and os.path.exists(full_ds_with_labels_path):
                        os.remove(full_ds_with_labels_path)
            if best_score == -math.inf or best_model_paths == (None, None):
                return None, None, None, None, None
            return best_score, best_full_ds_with_labels_path, best_model_paths, best_models_cols, best_main_threshold
        except Exception as e:
            print(f"⚠️ ERROR en evaluación de clusters: {str(e)}")
            return None, None, None, None, None
    
    def _suggest_catboost(self, group: str, trial: optuna.Trial) -> Dict[str, float]:
        """Devuelve hiperparámetros CatBoost (main|meta) con prefijo `group`."""
        p = {}
        p[f'{group}_iterations']      = trial.suggest_int (f'{group}_iterations',      200, 800, step=50)
        p[f'{group}_depth']           = trial.suggest_int (f'{group}_depth',           4,   8)
        p[f'{group}_learning_rate']   = trial.suggest_float(f'{group}_learning_rate',  1e-3, .3, log=True)
        p[f'{group}_l2_leaf_reg']     = trial.suggest_float(f'{group}_l2_leaf_reg',    1.0,  10.0, log=True)
        p[f'{group}_early_stopping']  = trial.suggest_int (f'{group}_early_stopping',  20,  200,  step=20)
        return p

    # ─────────────────────────  FUNCIÓN PRINCIPAL  ──────────────────────────────
    def _suggest_feature(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Sugiere períodos y estadísticas de la matriz de features
        (y opcionalmente los 'meta-features').

        Compatible con TPESampler multivariante y agrupado:
        - mismos nombres / mismas distribuciones SIEMPRE
        - prefijos consistentes ('feature_main_', 'feature_meta_')
        - espacio de hiperparámetros completamente fijo
        """
        ALL_STATS = (
            "chande",
            "var",
            "hurst",
            "std",
            "approxentropy",
            "range",
            "mean",
            "maxdd",
            "jumpvol",
            "fisher",
            "momentum",
            "volskew",
            "entropy",
            "iqr",
            "effratio",
            "fractal",
            "mad",
            "cv",
            "slope",
            "skew",
            "zscore",
            "median",
            "autocorr",
            "kurt",
            "corrskew",
            "sharpe",
        )
        p: Dict[str, Any] = {}

        # ─── FEATURE MAIN - PERÍODOS ────────────────────────────────────────────────
        n_periods = trial.suggest_int("feature_main_n_periods", 1, 12)
        feature_periods = [
            trial.suggest_int(f"feature_main_period_{i}", 5, 200, log=True)
            for i in range(n_periods)
        ]
        p["feature_main_periods"] = tuple(sorted(set(feature_periods)))

        # ─── FEATURE MAIN - ESTADÍSTICAS ───────────────────────────────────────────
        n_stats = trial.suggest_int("feature_main_n_stats", 1, 6)
        feature_stats = [
            trial.suggest_categorical(f"feature_main_stat_{i}", ALL_STATS)
            for i in range(n_stats)
        ]
        # mantener orden de aparición sin duplicados
        p["feature_main_stats"] = tuple(sorted(dict.fromkeys(feature_stats)))

        # ─── FEATURE META (solo ciertos search_type) ──────────────────────────
        if self.search_type in {"clusters", "markov", "lgmm", "wkmeans"}:
            # períodos meta
            n_meta_periods = trial.suggest_int("feature_meta_n_periods", 1, 3)
            meta_periods = [
                trial.suggest_int(f"feature_meta_period_{i}", 4, 12, log=True)
                for i in range(n_meta_periods)
            ]
            p["feature_meta_periods"] = tuple(sorted(set(meta_periods)))

            # estadísticas meta
            n_meta_stats = trial.suggest_int("feature_meta_n_stats", 1, 3)
            meta_stats = [
                trial.suggest_categorical(f"feature_meta_stat_{i}", ALL_STATS)
                for i in range(n_meta_stats)
            ]
            p["feature_meta_stats"] = tuple(sorted(dict.fromkeys(meta_stats)))

        return p

    # ----------------------------------------------------------------------------- 
    def _suggest_label(self, trial: optuna.Trial) -> Dict[str, float]:
        """Hiperparámetros de etiquetado dependientes de la función label_method."""
        label_search_space = {
            'label_markup':     lambda t: t.suggest_float('label_markup',     0.10, 1.0, log=True),
            'label_n_clusters': lambda t: t.suggest_int('label_n_clusters', 3, 12, log=True),
            'label_polyorder':  lambda t: t.suggest_int('label_polyorder',    1, 5, log=True),
            'label_threshold':  lambda t: t.suggest_float('label_threshold',  0.05, 0.5),
            'label_corr_threshold': lambda t: t.suggest_float('label_corr_threshold', 0.6, 0.9),
            'label_rolling':    lambda t: t.suggest_int  ('label_rolling',    50, 300, log=True),
            'label_rolling2':  lambda t: t.suggest_int  ('label_rolling2',   50, 300, log=True),
            'label_rolling_periods_small': lambda t: [t.suggest_int(f'label_rolling_periods_small_{i}', 4, 60, log=True) for i in range(3)],
            'label_rolling_periods_big': lambda t: [t.suggest_int(f'label_rolling_periods_big_{i}', 200, 600, log=True) for i in range(3)],
            'label_atr_period': lambda t: t.suggest_int  ('label_atr_period', 15, 50, log=True),
            'label_min_val':    lambda t: t.suggest_int  ('label_min_val',    1,  15, log=True),
            'label_max_val':    lambda t: t.suggest_int  ('label_max_val',    15, 30, log=True),
            'label_method_trend':     lambda t: t.suggest_categorical('label_method_trend', ['normal', 'inverse']),
            'label_method_random':     lambda t: t.suggest_categorical('label_method_random', ['first', 'last', 'mean', 'max', 'min', 'random']),
            'label_filter':     lambda t: t.suggest_categorical('label_filter', ['savgol', 'spline', 'sma', 'ema']),
            'label_filter_mean':     lambda t: t.suggest_categorical('label_filter_mean', ['savgol', 'spline', 'mean']),
            'label_window_size': lambda t: t.suggest_int('label_window_size', 4, 60, log=True),
            'label_window_sizes_int': lambda t: [t.suggest_int(f'label_window_sizes_{i}', 4, 60, log=True) for i in range(3)],
            'label_window_sizes_float': lambda t: [t.suggest_float(f'label_window_sizes_{i}', 0.01, 0.1) for i in range(3)],
            'label_min_window': lambda t: t.suggest_int('label_min_window', 2, 20, log=True),
            'label_max_window': lambda t: t.suggest_int('label_max_window', 20, 60, log=True),
            'label_vol_window': lambda t: t.suggest_int('label_vol_window', 20, 100, log=True),
            'label_min_touches': lambda t: t.suggest_int('label_min_touches', 2, 10),
            'label_peak_prominence': lambda t: t.suggest_float('label_peak_prominence', 0.05, 0.5),
            'label_quantiles': lambda t: [t.suggest_float(f'label_quantiles_{i}', 0.25, 0.75) for i in range(2)],
            'label_decay_factor': lambda t: t.suggest_float('label_decay_factor', 0.7, 0.99),
            'label_shift': lambda t: t.suggest_int('label_shift', 0, 10),
        }
        p = {}
        label_func = self.LABEL_FUNCS[self.label_method]
        params_sig = inspect.signature(label_func).parameters
        for name in params_sig:
            if name in label_search_space:
                p[name] = label_search_space[name](trial)
        return p

    # ----------------------------------------------------------------------------- 
    def _suggest_algo_specific(self, trial: optuna.Trial) -> Dict[str, float]:
        """Parámetros exclusivos según self.search_type."""
        p = {}
        if self.search_type == 'clusters':
            if self.search_subtype == 'kmeans':
                p['kmeans_n_clusters'] = trial.suggest_int ('kmeans_n_clusters', 5, 30, log=True)
                p['kmeans_window']     = trial.suggest_int ('kmeans_window',     40, 250, log=True)
                p['kmeans_step']       = trial.suggest_int ('kmeans_step',       10, 50, log=True)
            elif self.search_subtype == 'hdbscan':
                p['hdbscan_min_cluster_size'] = trial.suggest_int ('hdbscan_min_cluster_size', 5, 30, log=True)
            elif self.search_subtype == 'markov':
                p['markov_model']    = trial.suggest_categorical('markov_model', ['GMMHMM', 'HMM'])
                p['markov_regimes']  = trial.suggest_int ('markov_regimes', 3, 8, log=True)
                p['markov_iter']     = trial.suggest_int ('markov_iter',    50, 200, log=True)
                p['markov_mix']      = trial.suggest_int ('markov_mix',     2, 5)
            elif self.search_subtype == 'lgmm':
                p['lgmm_components']  = trial.suggest_int ('lgmm_components',  3, 15, log=True)
                p['lgmm_covariance']  = trial.suggest_categorical('lgmm_covariance', ['full', 'tied', 'diag', 'spherical'])
                p['lgmm_iter']        = trial.suggest_int ('lgmm_iter',        50, 200, log=True)
            elif self.search_subtype == 'wkmeans':
                p['wk_n_clusters']    = trial.suggest_int ('wk_n_clusters',    6, 20, log=True)
                p['wk_bandwidth']     = trial.suggest_float('wk_bandwidth',    0.1, 5.0, log=True)
                p['wk_window']        = trial.suggest_int ('wk_window',        30, 120, log=True)
                p['wk_step']          = trial.suggest_int ('wk_step',          1, 10)
                p['wk_proj']          = trial.suggest_int ('wk_proj',          50, 200, log=True)
                p['wk_iter']          = trial.suggest_int ('wk_iter',          100, 500, log=True)
        elif self.search_type == 'mapie':
            p['mapie_confidence_level'] = trial.suggest_float('mapie_confidence_level', 0.8, 0.95)
            p['mapie_cv']               = trial.suggest_int  ('mapie_cv',               3, 10)
            # Parámetros específicos para regresión MAPIE
            if self.label_type == 'regression':
                p['mapie_threshold_width']     = trial.suggest_int('mapie_threshold_width', 30, 70)
        elif self.search_type == 'causal':
            p['causal_meta_learners'] = trial.suggest_int('causal_meta_learners', 5, 15)
            p['causal_percentile'] = trial.suggest_int('causal_percentile', 60, 90)
            # Parámetros específicos para regresión causal
            if self.label_type == 'regression':
                p['causal_error_threshold'] = trial.suggest_float('causal_error_threshold', 0.1, 2.0, log=True)

        # Parámetros de filtros (independientes del search_type)
        if self.search_filter == 'mapie':
            p['mapie_confidence_level'] = trial.suggest_float('mapie_confidence_level', 0.8, 0.95)
            p['mapie_cv']               = trial.suggest_int  ('mapie_cv',               3, 10)
            # Parámetros específicos para regresión MAPIE
            if self.label_type == 'regression':
                p['mapie_threshold_width']     = trial.suggest_int('mapie_threshold_width', 30, 70)
        elif self.search_filter == 'causal':
            p['causal_meta_learners'] = trial.suggest_int('causal_meta_learners', 5, 15)
            p['causal_percentile'] = trial.suggest_int('causal_percentile', 60, 90)
            # Parámetros específicos para regresión causal
            if self.label_type == 'regression':
                p['causal_error_threshold'] = trial.suggest_float('causal_error_threshold', 0.1, 2.0, log=True)

        # THRESHOLDS UNIFICADOS (preparación para optimización futura)
        if self.label_type == 'classification':
            # Clasificación: ambos thresholds fijos por ahora (futuro optimizables)
            p['main_threshold'] = 0.5  # FUTURO: trial.suggest_float('main_threshold', 0.1, 0.9)
            p['meta_threshold'] = 0.5  # FUTURO: trial.suggest_float('meta_threshold', 0.1, 0.9)
        else:  # regression
            # Regresión: meta threshold fijo por ahora (futuro optimizable), main se calcula dinámicamente
            p['meta_threshold'] = 0.5  # FUTURO: trial.suggest_float('meta_threshold', 0.1, 0.9)
            p['model_main_percentile'] = trial.suggest_float('model_main_percentile', 0.6, 0.8)
            # main_threshold se calcula dinámicamente en calculate_regression_threshold_cv

        return p

    # ========= main entry =========================================================
    def suggest_all_params(self, trial: optuna.Trial) -> Dict[str, float]:
        """Sugiere TODOS los hiperparámetros, agrupados de forma independiente."""
        try:
            params = {}
            # --- CatBoost main & meta ------------------------------------------------
            params.update(self._suggest_catboost('cat_main', trial))
            params.update(self._suggest_catboost('cat_meta', trial))
            # --- Feature engineering -------------------------------------------------
            params.update(self._suggest_feature(trial))
            # --- Labelling -----------------------------------------------------------
            params.update(self._suggest_label(trial))
            # --- Algo-specific -------------------------------------------------------
            params.update(self._suggest_algo_specific(trial))

            if self.debug:
                print(f"🔍 DEBUG suggest_all_params - Parámetros generados:")
                cat_params = {k: v for k, v in params.items() if k.startswith('cat_')}
                feature_params = {k: v for k, v in params.items() if k.startswith('feature_')}
                label_params = {k: v for k, v in params.items() if k.startswith('label_')}
                algo_params = {k: v for k, v in params.items() if not any(k.startswith(p) for p in ['cat_', 'feature_', 'label_'])}
                
                print(f"🔍   CatBoost params: {list(cat_params.keys())}")
                print(f"🔍   Feature params: {list(feature_params.keys())}")
                print(f"🔍   Label params: {list(label_params.keys())}")
                print(f"🔍   Algo params: {list(algo_params.keys())}")
                print(f"🔍   Total params: {len(params)}")

            # Guarda atributos para posteriores análisis
            for k, v in params.items():
                trial.set_user_attr(k, v)
            return params
            
        except Exception as e:   # Optuna gestiona la excepción/prune
            print(f"⚠️ Suggest params error: {e}")
            return None

    def fit_final_models(self, trial: optuna.trial,
                         full_ds: pd.DataFrame,
                         model_main_train_data: pd.DataFrame,
                         model_meta_train_data: pd.DataFrame,
                         hp: Dict[str, Any]) -> tuple[float, tuple, tuple]:
        """Ajusta los modelos finales y devuelve rutas a archivos temporales."""
        try:
            # 🔍 DEBUG: Supervisar parámetros CatBoost
            if self.debug:
                cat_main_params = {k: v for k, v in hp.items() if k.startswith('cat_main_')}
                cat_meta_params = {k: v for k, v in hp.items() if k.startswith('cat_meta_')}
                print(f"🔍 DEBUG fit_final_models - Parámetros CatBoost:")
                print(f"🔍   cat_main_*: {cat_main_params}")
                print(f"🔍   cat_meta_*: {cat_meta_params}")
            
            # Preparar datos del modelo main según label_type
            if self.label_type == 'classification':
                if model_main_train_data.empty:
                    return None, None, None, None
                main_feature_cols = [col for col in model_main_train_data.columns if col != 'labels_main']
                if self.debug:
                    print(f"🔍 DEBUG: Main model data shape: {model_main_train_data.shape}")
                    print(f"🔍 DEBUG: Main feature columns: {main_feature_cols}")
                model_main_train_data, model_main_eval_data = self.get_train_test_data(dataset=model_main_train_data)
                if model_main_train_data is None or model_main_eval_data is None:
                    return None, None, None, None
                X_train_main = model_main_train_data[main_feature_cols].astype('float32')
                y_train_main = model_main_train_data['labels_main'].astype('int8')
                X_val_main = model_main_eval_data[main_feature_cols].astype('float32')
                y_val_main = model_main_eval_data['labels_main'].astype('int8')
                if self.debug:
                    print(f"🔍 DEBUG: X_train_main shape: {X_train_main.shape}, y_train_main shape: {y_train_main.shape}")
                    print(f"🔍 DEBUG: X_val_main shape: {X_val_main.shape}, y_val_main shape: {y_val_main.shape}")

            else:  # regression
                if model_main_train_data.empty:
                    return None, None, None, None
                main_feature_cols = [col for col in model_main_train_data.columns if col != 'labels_main']
                if self.debug:
                    print(f"🔍 DEBUG: Main model data shape: {model_main_train_data.shape}")
                    print(f"🔍 DEBUG: Main feature columns: {main_feature_cols}")
                model_main_train_data, model_main_eval_data = self.get_train_test_data(dataset=model_main_train_data)
                if model_main_train_data is None or model_main_eval_data is None:
                    return None, None, None, None
                X_train_main = model_main_train_data[main_feature_cols].astype('float32')
                y_train_main = model_main_train_data['labels_main'].astype('float32')
                X_val_main = model_main_eval_data[main_feature_cols].astype('float32')
                y_val_main = model_main_eval_data['labels_main'].astype('float32')
                if self.debug:
                    print(f"🔍 DEBUG: X_train_main shape: {X_train_main.shape}, y_train_main shape: {y_train_main.shape}")
                    print(f"🔍 DEBUG: X_val_main shape: {X_val_main.shape}, y_val_main shape: {y_val_main.shape}")
                    print(f"🔍 DEBUG: y_train_main range: [{y_train_main.min():.4f}, {y_train_main.max():.4f}]")
                    print(f"🔍 DEBUG: y_val_main range: [{y_val_main.min():.4f}, {y_val_main.max():.4f}]")
                    print(f"🔍 DEBUG: y_train_main percentiles: {y_train_main.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).to_dict()}")
                    print(f"🔍 DEBUG: y_val_main percentiles: {y_val_main.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).to_dict()}")
                    print(f"🔍 DEBUG: y_train_main > 0: {(y_train_main > 0).sum()}/{len(y_train_main)}")
                    print(f"🔍 DEBUG: y_val_main > 0: {(y_val_main > 0).sum()}/{len(y_val_main)}")
            meta_feature_cols = [col for col in model_meta_train_data.columns if col != 'labels_meta']
            if self.debug:
                print(f"🔍 DEBUG: Meta model data shape: {model_meta_train_data.shape}")
                print(f"🔍 DEBUG: Meta feature columns: {meta_feature_cols}")
            model_meta_train_data, model_meta_eval_data = self.get_train_test_data(dataset=model_meta_train_data)
            if model_meta_train_data is None or model_meta_eval_data is None:
                return None, None, None, None
            X_train_meta = model_meta_train_data[meta_feature_cols].astype('float32')
            y_train_meta = model_meta_train_data['labels_meta'].astype('int8')
            X_val_meta = model_meta_eval_data[meta_feature_cols].astype('float32')
            y_val_meta = model_meta_eval_data['labels_meta'].astype('int8')
            if self.debug:
                print(f"🔍 DEBUG: X_train_meta shape: {X_train_meta.shape}, y_train_meta shape: {y_train_meta.shape}")
                print(f"🔍 DEBUG: X_val_meta shape: {X_val_meta.shape}, y_val_meta shape: {y_val_meta.shape}")

            # Configurar parámetros CatBoost según label_type
            if self.label_type == 'classification':
                cat_main_params = dict(
                    iterations=hp['cat_main_iterations'],
                    depth=hp['cat_main_depth'],
                    learning_rate=hp['cat_main_learning_rate'],
                    l2_leaf_reg=hp['cat_main_l2_leaf_reg'],
                    auto_class_weights='Balanced',
                    eval_metric='Accuracy',
                    store_all_simple_ctr=False,
                    allow_writing_files=False,
                    thread_count=-1,
                    task_type='CPU',
                    verbose=False,
                )
                
                # 🔍 DEBUG: Mostrar configuración final de CatBoost
                if self.debug:
                    print(f"🔍 DEBUG: CatBoost Main (Classification) configuración final:")
                    for k, v in cat_main_params.items():
                        print(f"🔍   {k}: {v}")
                    
                model_main = CatBoostClassifier(**cat_main_params)
            else:  # regression
                cat_main_params = dict(
                    iterations=hp['cat_main_iterations'],
                    depth=hp['cat_main_depth'],
                    learning_rate=hp['cat_main_learning_rate'],
                    l2_leaf_reg=hp['cat_main_l2_leaf_reg'],
                    eval_metric='MAE',
                    loss_function='MAE',
                    store_all_simple_ctr=False,
                    allow_writing_files=False,
                    thread_count=-1,
                    task_type='CPU',
                    verbose=False,
                )
                
                # 🔍 DEBUG: Mostrar configuración final de CatBoost
                if self.debug:
                    print(f"🔍 DEBUG: CatBoost Main (Regression) configuración final:")
                    for k, v in cat_main_params.items():
                        print(f"🔍   {k}: {v}")
                    
                model_main = CatBoostRegressor(**cat_main_params)
            t_train_main_start = time.time()
            model_main.fit(X_train_main, y_train_main, 
                           eval_set=[(X_val_main, y_val_main)],
                           early_stopping_rounds=hp['cat_main_early_stopping'],
                           # callbacks=[CatBoostPruningCallback(trial=trial, metric='Logloss')],
                           use_best_model=True,
                           verbose=False
            )
            t_train_main_end = time.time()

            if self.debug:
                print(f"🔍 DEBUG: Tiempo de entrenamiento modelo main: {t_train_main_end - t_train_main_start:.2f} segundos")
                
                # 🔍 DEBUG: Verificar predicciones del modelo en datos de entrenamiento
                train_pred = model_main.predict(X_train_main)
                val_pred = model_main.predict(X_val_main)
                print(f"🔍 DEBUG: main_threshold: {hp.get('main_threshold', 0.5)}")
                print(f"🔍 DEBUG: Train predictions - min: {train_pred.min():.4f}, max: {train_pred.max():.4f}, mean: {train_pred.mean():.4f}")
                print(f"🔍 DEBUG: Val predictions - min: {val_pred.min():.4f}, max: {val_pred.max():.4f}, mean: {val_pred.mean():.4f}")
                print(f"🔍 DEBUG: Train predictions > threshold: {(train_pred > hp.get('main_threshold', 0.5)).sum()}/{len(train_pred)}")
                print(f"🔍 DEBUG: Val predictions > threshold: {(val_pred > hp.get('main_threshold', 0.5)).sum()}/{len(val_pred)}")
            cat_meta_params = dict(
                iterations=hp['cat_meta_iterations'],
                depth=hp['cat_meta_depth'],
                learning_rate=hp['cat_meta_learning_rate'],
                l2_leaf_reg=hp['cat_meta_l2_leaf_reg'],
                auto_class_weights='Balanced',
                eval_metric='F1',
                store_all_simple_ctr=False,
                allow_writing_files=False,
                thread_count=-1,
                task_type='CPU',
                verbose=False,
            )
            
            # 🔍 DEBUG: Mostrar configuración final de CatBoost Meta
            if self.debug:
                print(f"🔍 DEBUG: CatBoost Meta configuración final:")
                for k, v in cat_meta_params.items():
                    print(f"🔍   {k}: {v}")
                    
            model_meta = CatBoostClassifier(**cat_meta_params)
            t_train_meta_start = time.time()
            model_meta.fit(X_train_meta, y_train_meta, 
                           eval_set=[(X_val_meta, y_val_meta)], 
                           early_stopping_rounds=hp['cat_meta_early_stopping'],
                           # callbacks=[CatBoostPruningCallback(trial=trial, metric='Logloss')],
                           use_best_model=True,
                           verbose=False
            )
            t_train_meta_end = time.time()
            
            # DEBUG: Verificar entrenamiento del modelo meta
            if self.debug:
                print(f"🔍 DEBUG: Tiempo de entrenamiento modelo meta: {t_train_meta_end - t_train_meta_start:.2f} segundos")
                
                # Debug de datos de entrenamiento meta
                print(f"🔍 DEBUG fit_final_models - Datos de entrenamiento meta:")
                print(f"🔍   X_train_meta.shape: {X_train_meta.shape}")
                print(f"🔍   y_train_meta.shape: {y_train_meta.shape}")
                print(f"🔍   y_train_meta.value_counts(): {y_train_meta.value_counts().to_dict()}")
                print(f"🔍   y_train_meta.mean(): {y_train_meta.mean():.6f}")
                
                # Debug de predicciones del modelo meta en validación
                meta_predictions_train = model_meta.predict_proba(X_train_meta)[:, 1]
                meta_predictions_val = model_meta.predict_proba(X_val_meta)[:, 1]
                
                print(f"🔍 DEBUG fit_final_models - Predicciones modelo meta:")
                print(f"🔍   Train predictions - min: {meta_predictions_train.min():.6f}, max: {meta_predictions_train.max():.6f}, mean: {meta_predictions_train.mean():.6f}")
                print(f"🔍   Val predictions - min: {meta_predictions_val.min():.6f}, max: {meta_predictions_val.max():.6f}, mean: {meta_predictions_val.mean():.6f}")
                
                # Debug de métricas de entrenamiento
                train_accuracy = (meta_predictions_train > 0.5) == y_train_meta
                val_accuracy = (meta_predictions_val > 0.5) == y_val_meta
                print(f"🔍   Train accuracy: {train_accuracy.mean():.6f}")
                print(f"🔍   Val accuracy: {val_accuracy.mean():.6f}")
            # DEBUG: Verificar tipos de modelos antes de exportar
            if self.debug:
                print(f"🔍 DEBUG fit_final_models - Tipos de modelos:")
                print(f"🔍   model_main.__class__: {model_main.__class__}")
                print(f"🔍   model_meta.__class__: {model_meta.__class__}")
                print(f"🔍   model_main.get_params(): {model_main.get_params()}")
                print(f"🔍   model_meta.get_params(): {model_meta.get_params()}")
            
            model_main_path, model_meta_path = export_models_to_ONNX(models=(model_main, model_meta))
            
            # DEBUG: Verificar exportación ONNX y predicciones
            if self.debug:
                print(f"🔍 DEBUG fit_final_models - Verificación ONNX:")
                print(f"🔍   model_main_path: {model_main_path}")
                print(f"🔍   model_meta_path: {model_meta_path}")
                
                # Verificar predicciones ONNX del modelo meta
                try:
                    from modules.tester_lib import predict_proba_onnx_models
                    # Convertir DataFrame a numpy array
                    X_val_meta_np = X_val_meta.to_numpy().astype(np.float32)
                    meta_onnx_predictions = predict_proba_onnx_models(model_meta_path, X_val_meta_np)
                    print(f"🔍   ONNX meta predictions - min: {meta_onnx_predictions.min():.6f}, max: {meta_onnx_predictions.max():.6f}, mean: {meta_onnx_predictions.mean():.6f}")
                    
                    # Comparar predicciones originales vs ONNX
                    original_predictions = model_meta.predict_proba(X_val_meta)[:, 1]
                    onnx_predictions = meta_onnx_predictions
                    
                    print(f"🔍   Original vs ONNX comparison:")
                    print(f"🔍     Original - min: {original_predictions.min():.6f}, max: {original_predictions.max():.6f}, mean: {original_predictions.mean():.6f}")
                    print(f"🔍     ONNX - min: {onnx_predictions.min():.6f}, max: {onnx_predictions.max():.6f}, mean: {onnx_predictions.mean():.6f}")
                    
                    # Verificar si hay diferencias significativas
                    diff = np.abs(original_predictions - onnx_predictions)
                    print(f"🔍     Max difference: {diff.max():.6f}")
                    print(f"🔍     Mean difference: {diff.mean():.6f}")
                    
                except Exception as e:
                    if self.debug:
                        print(f"🔍   ERROR verificando ONNX: {e}")

            if self.debug:
                print(f"🔍 DEBUG fit_final_models - Threshold para tester:")
                print(f"🔍   main_threshold: {hp.get('main_threshold', 0.5)}")
                
                # DEBUG: Verificar predicciones del modelo meta en el dataset completo
                print(f"🔍 DEBUG fit_final_models - Verificación predicciones meta en dataset completo:")
                
                # Obtener features meta del dataset completo
                meta_feature_cols = [col for col in full_ds.columns if col.endswith('_meta_feature')]
                if meta_feature_cols:
                    X_full_meta = full_ds[meta_feature_cols].dropna(subset=meta_feature_cols)
                    print(f"🔍   meta_feature_cols: {meta_feature_cols}")
                    print(f"🔍   X_full_meta.shape: {X_full_meta.shape}")
                    
                    # Predicciones originales del modelo meta
                    meta_predictions_full = model_meta.predict_proba(X_full_meta)[:, 1]
                    print(f"🔍   Meta predictions full dataset:")
                    print(f"🔍     min: {meta_predictions_full.min():.6f}")
                    print(f"🔍     max: {meta_predictions_full.max():.6f}")
                    print(f"🔍     mean: {meta_predictions_full.mean():.6f}")
                    print(f"🔍     std: {meta_predictions_full.std():.6f}")
                    
                    # Verificar distribución de predicciones
                    predictions_above_05 = (meta_predictions_full > 0.5).sum()
                    print(f"🔍     predictions > 0.5: {predictions_above_05} ({predictions_above_05/len(meta_predictions_full)*100:.2f}%)")
                    
                    # Predicciones ONNX del dataset completo
                    try:
                        # Convertir DataFrame a numpy array
                        X_full_meta_np = X_full_meta.to_numpy().astype(np.float32)
                        meta_onnx_full = predict_proba_onnx_models(model_meta_path, X_full_meta_np)
                        print(f"🔍   ONNX predictions full dataset:")
                        print(f"🔍     min: {meta_onnx_full.min():.6f}")
                        print(f"🔍     max: {meta_onnx_full.max():.6f}")
                        print(f"🔍     mean: {meta_onnx_full.mean():.6f}")
                        print(f"🔍     std: {meta_onnx_full.std():.6f}")
                        
                        onnx_above_05 = (meta_onnx_full > 0.5).sum()
                        print(f"🔍     ONNX > 0.5: {onnx_above_05} ({onnx_above_05/len(meta_onnx_full)*100:.2f}%)")
                        
                    except Exception as e:
                        print(f"🔍   ERROR en predicciones ONNX full dataset: {e}")
                else:
                    print(f"🔍   No se encontraron meta features en el dataset")
            
            try:
                # Inicializar score con valor por defecto
                score = -1.0
                test_train_time_start = time.time()
                score = tester(
                    dataset=full_ds,
                    model_main=model_main_path,
                    model_meta=model_meta_path,
                    model_main_cols=main_feature_cols,
                    model_meta_cols=meta_feature_cols,
                    direction=self.direction,
                    timeframe=self.timeframe,
                    model_main_threshold=hp.get('main_threshold', 0.5),
                    model_meta_threshold=hp.get('meta_threshold', 0.5),
                    label_type=self.label_type,
                    debug=self.debug,
                )
            except Exception as tester_error:
                if self.debug:
                    print(f"🔍 DEBUG: Error en tester: {tester_error}")
                score = -1.0
            test_train_time_end = time.time()
            if self.debug:
                print(f"🔍 DEBUG: Tiempo de test in-sample: {test_train_time_end - test_train_time_start:.2f} segundos")
                print(f"🔍 DEBUG: Score in-sample: {score}")
            
            if not np.isfinite(score):
                score = -1.0

            # Desplazar columnas OHLCV una posición hacia atrás
            ohlcv_cols = ["open", "high", "low", "close", "volume"]
            for col in ohlcv_cols:
                if col in full_ds.columns:
                    full_ds[col] = full_ds[col].shift(1)
            full_ds = full_ds.iloc[1:]
            full_ds_with_labels_path = export_dataset_to_csv(full_ds, self.decimal_precision)

            if self.debug:
                print(f"🔍   DEBUG: Dataset con shape {full_ds.shape} guardado en {full_ds_with_labels_path}")
                # Resumen de las columnas de etiquetas
                if 'labels_main' in full_ds.columns:
                    labels_main = full_ds['labels_main']
                    if self.label_type == 'classification':
                        main_counts = labels_main.value_counts(dropna=False).to_dict()
                        print(f"🔍      labels_main value_counts: {main_counts}")
                        print(f"🔍      labels_main únicos: {sorted(labels_main.unique())}")
                    else:
                        print(f"🔍      labels_main resumen: min={labels_main.min():.6f}, max={labels_main.max():.6f}, mean={labels_main.mean():.6f}, std={labels_main.std():.6f}")
                else:
                    print(f"🔍      labels_main no encontrada en el dataset")
                if 'labels_meta' in full_ds.columns:
                    labels_meta = full_ds['labels_meta']
                    print(f"🔍      labels_meta resumen: min={labels_meta.min():.6f}, max={labels_meta.max():.6f}, mean={labels_meta.mean():.6f}, std={labels_meta.std():.6f}")
                else:
                    print(f"🔍      labels_meta no encontrada en el dataset")
                print(f"🔍 DEBUG: Modelos guardados en {model_main_path} y {model_meta_path}")
            return score, full_ds_with_labels_path, (model_main_path, model_meta_path), (main_feature_cols, meta_feature_cols)
        except Exception as e:
            print(f"Error en función de entrenamiento y test: {str(e)}")
            return None, None, None, None
        finally:
            clear_onnx_session_cache()

    def apply_mapie_filter(self, trial, full_ds, hp, reliable_mask=None) -> np.ndarray:
        """
        Aplica conformal prediction (MAPIE) para obtener scores de confiabilidad.
        
        Args:
            trial: Optuna trial
            full_ds: Dataset completo con features
            hp: Hiperparámetros
            reliable_mask: Máscara opcional para filtrar muestras confiables
            
        Returns:
            np.ndarray: combined_scores con 1.0 para muestras confiables y precisas, 0.0 para el resto
        """
        try:
            if self.debug:
                print(f"🔍 DEBUG apply_mapie_filter - Iniciando filtrado MAPIE")
                print(f"🔍   full_ds.shape: {full_ds.shape}")
                print(f"🔍   reliable_mask is None: {reliable_mask is None}")
                if reliable_mask is not None:
                    print(f"🔍   reliable_mask.sum(): {reliable_mask.sum()}")
                    print(f"🔍   reliable_mask.mean(): {reliable_mask.mean():.3f}")
            
            if reliable_mask is not None:
                # Usar esquema de confiabilidad
                reliable_data = full_ds[reliable_mask].copy()
                main_feature_cols = [col for col in reliable_data.columns if col.endswith('_main_feature')]
                X = reliable_data[main_feature_cols].dropna(subset=main_feature_cols)
                y = reliable_data.loc[X.index, 'labels_main']
                
                # Verificar alineación de índices
                if len(X) != len(y):
                    if self.debug:
                        print(f"🔍 DEBUG apply_mapie_filter - Desalineación de índices reliable: X={len(X)}, y={len(y)}")
                    return np.zeros(len(full_ds))
                
                if self.debug:
                    print(f"🔍   reliable_data.shape: {reliable_data.shape}")
                    print(f"🔍   X.shape: {X.shape}")
                    print(f"🔍   y.shape: {y.shape}")
                    # Ajuste: para regresión, muestra solo conteo de no-cero y cero; para clasificación, muestra value_counts completo
                    if self.label_type == 'classification':
                        print(f"🔍   y.value_counts(): {y.value_counts().to_dict()}")
                    else:
                        n_nonzero = (y != 0.0).sum()
                        n_zero = (y == 0.0).sum()
                        print(f"🔍   y: n_nonzero={n_nonzero}, n_zero={n_zero}, total={len(y)}")
                        print(f"🔍   y.min(): {y.min()}, y.max(): {y.max()}")
            else:
                # Esquema tradicional
                main_feature_cols = [col for col in full_ds.columns if col.endswith('_main_feature')]
                X = full_ds[main_feature_cols].dropna(subset=main_feature_cols)
                y = full_ds.loc[X.index, 'labels_main']
                
                # Verificar alineación de índices
                if len(X) != len(y):
                    if self.debug:
                        print(f"🔍 DEBUG apply_mapie_filter - Desalineación de índices tradicional: X={len(X)}, y={len(y)}")
                    return np.zeros(len(X))
                
                if self.debug:
                    print(f"🔍   X.shape: {X.shape}")
                    print(f"🔍   y.shape: {y.shape}")
                    # Ajuste: para regresión, muestra solo conteo de no-cero y cero; para clasificación, muestra value_counts completo
                    if self.label_type == 'classification':
                        print(f"🔍   y.value_counts(): {y.value_counts().to_dict()}")
                    else:
                        n_nonzero = (y != 0.0).sum()
                        n_zero = (y == 0.0).sum()
                        print(f"🔍   y: n_nonzero={n_nonzero}, n_zero={n_zero}, total={len(y)}")
                        print(f"🔍   y.min(): {y.min()}, y.max(): {y.max()}")

            # Verificar que tenemos suficientes datos y clases balanceadas
            if len(X) < 100:  # Mínimo requerido para conformal prediction robusta
                if self.debug:
                    print(f"🔍 DEBUG apply_mapie_filter - Datos insuficientes para MAPIE: {len(X)}")
                if reliable_mask is not None:
                    return np.zeros(len(full_ds))
                else:
                    return np.zeros(len(X))
            
            # Verificar datos según label_type
            if self.label_type == 'classification':
                # Clasificación: verificar que tenemos al menos 2 clases
                if len(y.unique()) < 2:
                    if self.debug:
                        print(f"🔍 DEBUG apply_mapie_filter - Clases insuficientes para MAPIE: {y.unique()}")
                    if reliable_mask is not None:
                        return np.zeros(len(full_ds))
                    else:
                        return np.zeros(len(X))
            else:  # regression
                # Regresión: verificar que tenemos suficientes muestras con magnitudes
                non_zero_mask = y != 0.0
                if non_zero_mask.sum() < 50:  # Mínimo para regresión
                    if self.debug:
                        print(f"🔍 DEBUG apply_mapie_filter - Muestras con magnitudes insuficientes: {non_zero_mask.sum()}")
                    if reliable_mask is not None:
                        return np.zeros(len(full_ds))
                    else:
                        return np.zeros(len(X))

            def _randomize_catboost_params(base_params):
                # Pequeñas variaciones aleatorias (±10%) para cada hiperparámetro relevante
                randomized = base_params.copy()
                for k in ['iterations', 'depth', 'l2_leaf_reg']:
                    if k in randomized:
                        jitter = random.uniform(0.7, 1.3)
                        # Mantener enteros para iteraciones y profundidad
                        if k in ['iterations', 'depth']:
                            randomized[k] = max(1, int(round(randomized[k] * jitter)))
                        else:
                            randomized[k] = randomized[k] * jitter
                if 'learning_rate' in randomized:
                    jitter = random.uniform(0.7, 1.3)
                    randomized['learning_rate'] = max(1e-4, randomized['learning_rate'] * jitter)
                return randomized
            # Configurar CatBoost y MAPIE según label_type
            if self.label_type == 'classification':
                catboost_params = dict(
                    iterations=hp['cat_main_iterations'],
                    depth=hp['cat_main_depth'],
                    learning_rate=hp['cat_main_learning_rate'],
                    l2_leaf_reg=hp['cat_main_l2_leaf_reg'],
                    eval_metric='Accuracy',
                    store_all_simple_ctr=False,
                    allow_writing_files=False,
                    thread_count=-1,
                    task_type='CPU',
                    verbose=False,
                )
                catboost_params = _randomize_catboost_params(catboost_params)
                base_estimator = CatBoostClassifier(**catboost_params)
                mapie = CrossConformalClassifier(
                    estimator=base_estimator,
                    confidence_level=hp['mapie_confidence_level'],
                    conformity_score='lac',
                    cv=hp['mapie_cv'],
                )
            else:  # regression
                catboost_params = dict(
                    iterations=hp['cat_main_iterations'],
                    depth=hp['cat_main_depth'],
                    learning_rate=hp['cat_main_learning_rate'],
                    l2_leaf_reg=hp['cat_main_l2_leaf_reg'],
                    eval_metric='MAE',
                    loss_function='MAE',
                    store_all_simple_ctr=False,
                    allow_writing_files=False,
                    thread_count=-1,
                    task_type='CPU',
                    verbose=False,
                )
                catboost_params = _randomize_catboost_params(catboost_params)
                base_estimator = CatBoostRegressor(**catboost_params)
                mapie = CrossConformalRegressor(
                    estimator=base_estimator,
                    confidence_level=hp['mapie_confidence_level'],
                    conformity_score='absolute',  # Método de conformidad
                    method='plus',  # Método de predicción
                    cv=hp['mapie_cv'],
                )
            
            mapie.fit_conformalize(X, y)
            
            # Calcular scores según label_type
            if self.label_type == 'classification':
                # Para clasificación: usar predict_set (conjuntos de predicción)
                predicted, prediction_sets = mapie.predict_set(X)
                # Calcular el tamaño del conjunto por muestra de forma robusta
                prediction_sets_2d = prediction_sets[:, :, 0]
                set_sizes = prediction_sets_2d.sum(axis=1)
                # Scores de confiabilidad: 1.0 si set_size == 1 (alta confianza), 0.0 en caso contrario
                conformal_scores = (set_sizes == 1).astype(float)
                # Scores de precisión: 1.0 si predicted == y (predicción correcta), 0.0 en caso contrario
                precision_scores = (predicted == y.to_numpy()).astype(float)
                # Combinar: solo muestras que son tanto confiables como precisas
                combined_scores = ((conformal_scores == 1.0) & (precision_scores == 1.0)).astype(float)
            else:  # regression
                # Para regresión: predict_interval devuelve (predictions, prediction_intervals)
                predicted, y_prediction_intervals = mapie.predict_interval(X)
                
                # Para regresión: analizar intervalos de confianza
                # Según documentación MAPIE: prediction_intervals tiene shape (n_samples, 2, n_confidence_levels)
                # Necesitamos extraer el primer (y único) nivel de confianza
                if y_prediction_intervals.ndim == 3:
                    # Shape: (n_samples, 2, n_confidence_levels) -> extraer primer nivel
                    intervals = y_prediction_intervals[:, :, 0]  # Shape: (n_samples, 2)
                else:
                    # Shape: (n_samples, 2) -> usar directamente
                    intervals = y_prediction_intervals
                
                interval_width = intervals[:, 1] - intervals[:, 0]
                
                # Criterios de confiabilidad para regresión (optimizados por Optuna)
                # MAPIE solo evalúa confiabilidad del modelo, no magnitud de la señal
                
                # Ancho del intervalo (incertidumbre): usar percentil optimizado por Optuna
                # mapie_threshold_width representa el percentil (15-40) para filtrar intervalos estrechos
                threshold_percentile = hp['mapie_threshold_width']
                
                # Validar que hay variabilidad suficiente en los intervalos
                if np.std(interval_width) < 1e-6:
                    # Si todos los intervalos son prácticamente iguales, aceptar todos
                    width_confidence = np.ones_like(interval_width, dtype=bool)
                    if self.debug:
                        print(f"🔍   WARNING: interval_width sin variabilidad (std={np.std(interval_width):.6f}), aceptando todas las muestras")
                else:
                    # Usar percentil optimizado por Optuna directamente
                    threshold_width_value = np.percentile(interval_width, threshold_percentile)
                    width_confidence = interval_width < threshold_width_value
                    
                    if self.debug:
                        print(f"🔍   threshold_width como percentil: {threshold_percentile:.1f}%")
                        print(f"🔍   threshold_width calculado: {threshold_width_value:.4f}")
                
                # MAPIE solo evalúa confiabilidad del modelo (width_confidence)
                # La magnitud de la señal se evalúa en el backtest, no aquí
                
                # 🔍 DEBUG DETALLADO PARA REGRESIÓN
                if self.debug:
                    print(f"🔍   === DEBUG MAPIE REGRESIÓN ===")
                    print(f"🔍   threshold_width (valor): {threshold_width_value:.4f}")
                    print(f"🔍   interval_width.min(): {interval_width.min():.4f}, interval_width.max(): {interval_width.max():.4f}")
                    print(f"🔍   interval_width.mean(): {interval_width.mean():.4f}, interval_width.std(): {interval_width.std():.4f}")
                    print(f"🔍   predicted.min(): {predicted.min():.4f}, predicted.max(): {predicted.max():.4f}")
                    print(f"🔍   predicted.mean(): {predicted.mean():.4f}, predicted.std(): {predicted.std():.4f}")
                    print(f"🔍   abs(predicted).min(): {abs(predicted).min():.4f}, abs(predicted).max(): {abs(predicted).max():.4f}")
                    print(f"🔍   width_confidence.sum(): {width_confidence.sum()}")
                    print(f"🔍   width_confidence.mean(): {width_confidence.mean():.4f}")
                    
                    # Debug adicional: mostrar algunos ejemplos
                    print(f"🔍   === EJEMPLOS DE INTERVALOS ===")
                    for i in range(min(5, len(intervals))):
                        print(f"🔍     Muestra {i}: intervalo=[{intervals[i,0]:.4f}, {intervals[i,1]:.4f}], width={interval_width[i]:.4f}, predicted={predicted[i]:.4f}")
                        print(f"🔍       width_confidence: {width_confidence[i]}")
                    
                    # Debug: distribución de interval_width
                    width_percentiles = np.percentile(interval_width, [10, 25, 50, 75, 90])
                    print(f"🔍   interval_width percentiles: {width_percentiles}")
                    
                    # Debug: distribución de abs(predicted)
                    pred_percentiles = np.percentile(abs(predicted), [10, 25, 50, 75, 90])
                    print(f"🔍   abs(predicted) percentiles: {pred_percentiles}")
                
                # MAPIE solo evalúa confiabilidad del modelo (width_confidence)
                # La magnitud de la señal se evalúa en el backtest
                combined_scores = width_confidence.astype(float)
            
            if self.debug:
                if self.label_type == 'classification':
                    print(f"🔍   set_sizes.min(): {set_sizes.min()}, set_sizes.max(): {set_sizes.max()}")
                    print(f"🔍   conformal_scores.sum(): {conformal_scores.sum()}")
                    print(f"🔍   precision_scores.sum(): {precision_scores.sum()}")
                else:  # regression
                    print(f"🔍   threshold_width (valor): {threshold_width_value:.4f}")
                    print(f"🔍   interval_width.min(): {interval_width.min():.4f}, interval_width.max(): {interval_width.max():.4f}")
                    print(f"🔍   width_confidence.sum(): {width_confidence.sum()}")
                    print(f"🔍   predicted range: [{predicted.min():.4f}, {predicted.max():.4f}]")
                print(f"🔍   combined_scores.sum(): {combined_scores.sum()}")
                print(f"🔍   combined_scores.mean(): {combined_scores.mean():.3f}")
                
                # Debug adicional para regresión
                if self.label_type == 'regression':
                    print(f"🔍   === RESULTADO FINAL MAPIE ===")
                    print(f"🔍   combined_scores.shape: {combined_scores.shape}")
                    print(f"🔍   combined_scores.dtype: {combined_scores.dtype}")
                    print(f"🔍   combined_scores.unique(): {np.unique(combined_scores)}")
                    print(f"🔍   combined_scores == 1.0: {(combined_scores == 1.0).sum()}")
                    print(f"🔍   combined_scores == 0.0: {(combined_scores == 0.0).sum()}")
                
                print(f"🔍 DEBUG apply_mapie_filter - Filtrado MAPIE completado ({self.label_type})")
            
            if reliable_mask is not None:
                full_combined_scores = np.zeros(len(full_ds))
                full_combined_scores[reliable_mask] = combined_scores
                
                if self.debug:
                    print(f"🔍   reliable_mask.sum(): {reliable_mask.sum()}")
                    print(f"🔍   combined_scores.shape: {combined_scores.shape}")
                
                return full_combined_scores
            else:
                return combined_scores
            
        except Exception as e:
            if self.debug:
                print(f"🔍 DEBUG apply_mapie_filter - ERROR: {str(e)}")
            return np.zeros(len(full_ds))
        
    def apply_causal_filter(self, trial, full_ds, hp, reliable_mask=None) -> np.ndarray:
        """
        Aplica detección causal de muestras malas usando bootstrap OOB.
        
        Args:
            trial: Optuna trial
            full_ds: Dataset completo con features
            hp: Hiperparámetros
            reliable_mask: Máscara opcional para filtrar muestras confiables
            
        Returns:
            np.ndarray: causal_scores con 1.0 para muestras buenas, 0.0 para muestras malas
        """
        try:
            if self.debug:
                print(f"🔍 DEBUG apply_causal_filter - Iniciando filtrado causal ({self.label_type})")
                print(f"🔍   full_ds.shape: {full_ds.shape}")
                print(f"🔍   reliable_mask is None: {reliable_mask is None}")
                if reliable_mask is not None:
                    print(f"🔍   reliable_mask.sum(): {reliable_mask.sum()}")
                    print(f"🔍   reliable_mask.mean(): {reliable_mask.mean():.3f}")
            
            def _bootstrap_oob_identification(X: pd.DataFrame, y: pd.Series, n_models: int = 25):
                oob_counts = pd.Series(0, index=X.index)
                if self.label_type == 'classification':
                    error_counts_0 = pd.Series(0, index=X.index)
                    error_counts_1 = pd.Series(0, index=X.index)
                else:  # regression
                    error_sums = pd.Series(0.0, index=X.index)
                    error_counts = pd.Series(0, index=X.index)
                
                for _ in range(n_models):
                    frac = random.uniform(0.4, 0.6)
                    train_idx = X.sample(frac=frac, replace=True).index
                    val_idx = X.index.difference(train_idx)
                    if len(val_idx) == 0:
                        continue
                    
                    # Diversificar hiperparámetros CatBoost para cada modelo bootstrap
                    def _randomize_catboost_params(base_params):
                        randomized = base_params.copy()
                        for k in ['iterations', 'depth', 'l2_leaf_reg']:
                            if k in randomized:
                                jitter = random.uniform(0.7, 1.3)
                                if k in ['iterations', 'depth']:
                                    randomized[k] = max(1, int(round(randomized[k] * jitter)))
                                else:
                                    randomized[k] = randomized[k] * jitter
                        if 'learning_rate' in randomized:
                            jitter = random.uniform(0.7, 1.3)
                            randomized['learning_rate'] = max(1e-4, randomized['learning_rate'] * jitter)
                        return randomized

                    if self.label_type == 'classification':
                        catboost_params = dict(
                            iterations=hp['cat_main_iterations'],
                            depth=hp['cat_main_depth'],
                            learning_rate=hp['cat_main_learning_rate'],
                            l2_leaf_reg=hp['cat_main_l2_leaf_reg'],
                            eval_metric='Accuracy',
                            store_all_simple_ctr=False,
                            allow_writing_files=False,
                            thread_count=-1,
                            task_type='CPU',
                            verbose=False,
                        )
                        catboost_params = _randomize_catboost_params(catboost_params)
                        model = CatBoostClassifier(**catboost_params)
                        model.fit(X.loc[train_idx], y.loc[train_idx], eval_set=[(X.loc[val_idx], y.loc[val_idx])], verbose=False)
                        pred = (model.predict_proba(X.loc[val_idx])[:, 1] >= 0.5).astype(int)
                        val_y = y.loc[val_idx]
                        val0 = val_idx[val_y == 0]
                        val1 = val_idx[val_y == 1]
                        diff0 = val0[pred[val_y == 0] != 0]
                        diff1 = val1[pred[val_y == 1] != 1]
                        oob_counts.loc[val_idx] += 1
                        error_counts_0.loc[diff0] += 1
                        error_counts_1.loc[diff1] += 1
                    else:  # regression
                        catboost_params = dict(
                            iterations=hp['cat_main_iterations'],
                            depth=hp['cat_main_depth'],
                            learning_rate=hp['cat_main_learning_rate'],
                            l2_leaf_reg=hp['cat_main_l2_leaf_reg'],
                            eval_metric='MAE',
                            loss_function='MAE',
                            store_all_simple_ctr=False,
                            allow_writing_files=False,
                            thread_count=-1,
                            task_type='CPU',
                            verbose=False,
                        )
                        catboost_params = _randomize_catboost_params(catboost_params)
                        model = CatBoostRegressor(**catboost_params)
                        model.fit(X.loc[train_idx], y.loc[train_idx], eval_set=[(X.loc[val_idx], y.loc[val_idx])], verbose=False)
                        pred = model.predict(X.loc[val_idx])
                        val_y = y.loc[val_idx]
                        errors = np.abs(pred - val_y)
                        oob_counts.loc[val_idx] += 1
                        error_sums.loc[val_idx] += errors
                        error_counts.loc[val_idx] += 1
                
                if self.label_type == 'classification':
                    return error_counts_0, error_counts_1, oob_counts
                else:  # regression
                    return error_sums, error_counts, oob_counts
            
            def _optimize_bad_samples_threshold(err0, err1, oob, fracs=[0.5, 0.6, 0.7, 0.8]):
                to_mark_0 = (err0 / oob.replace(0, 1)).fillna(0)
                to_mark_1 = (err1 / oob.replace(0, 1)).fillna(0)
                best_f = None
                best_s = np.inf
                for frac in fracs:
                    thr0 = np.percentile(to_mark_0[to_mark_0 > 0], 75) * frac if len(to_mark_0[to_mark_0 > 0]) else 0
                    thr1 = np.percentile(to_mark_1[to_mark_1 > 0], 75) * frac if len(to_mark_1[to_mark_1 > 0]) else 0
                    marked0 = to_mark_0[to_mark_0 > thr0].index
                    marked1 = to_mark_1[to_mark_1 > thr1].index
                    all_bad = pd.Index(marked0).union(marked1)
                    good_mask = ~to_mark_0.index.isin(all_bad)
                    ratios = []
                    for idx in to_mark_0[good_mask].index:
                        if to_mark_0[idx] > 0:
                            ratios.append(to_mark_0[idx])
                        if to_mark_1[idx] > 0:
                            ratios.append(to_mark_1[idx])
                    mean_err = np.mean(ratios) if ratios else 1.0
                    if mean_err < best_s:
                        best_s = mean_err
                        best_f = frac
                return best_f
            
            if reliable_mask is not None:
                # Usar esquema de confiabilidad
                reliable_data = full_ds[reliable_mask].copy()
                main_feature_cols = [col for col in reliable_data.columns if col.endswith('_main_feature')]
                X = reliable_data[main_feature_cols].dropna(subset=main_feature_cols)
                y = reliable_data.loc[X.index, 'labels_main']
                
                # Verificar alineación de índices
                if len(X) != len(y):
                    if self.debug:
                        print(f"🔍 DEBUG apply_causal_filter - Desalineación de índices reliable: X={len(X)}, y={len(y)}")
                    return np.zeros(len(full_ds))
                
                if self.debug:
                    print(f"🔍   reliable_data.shape: {reliable_data.shape}")
                    print(f"🔍   X.shape: {X.shape}")
                    print(f"🔍   y.shape: {y.shape}")
                    # Ajuste: para regresión, muestra solo conteo de no-cero y cero; para clasificación, muestra value_counts completo
                    if self.label_type == 'classification':
                        print(f"🔍   y.value_counts(): {y.value_counts().to_dict()}")
                    else:
                        n_nonzero = (y != 0.0).sum()
                        n_zero = (y == 0.0).sum()
                        print(f"🔍   y: n_nonzero={n_nonzero}, n_zero={n_zero}, total={len(y)}")
                        print(f"🔍   y.min(): {y.min()}, y.max(): {y.max()}")
            else:
                # Esquema tradicional
                main_feature_cols = [col for col in full_ds.columns if col.endswith('_main_feature')]
                X = full_ds[main_feature_cols].dropna(subset=main_feature_cols)
                y = full_ds.loc[X.index, 'labels_main']
                
                # Verificar alineación de índices
                if len(X) != len(y):
                    if self.debug:
                        print(f"🔍 DEBUG apply_causal_filter - Desalineación de índices tradicional: X={len(X)}, y={len(y)}")
                    return np.zeros(len(X))
                
                if self.debug:
                    print(f"🔍   X.shape: {X.shape}")
                    print(f"🔍   y.shape: {y.shape}")
                    # Ajuste: para regresión, muestra solo conteo de no-cero y cero; para clasificación, muestra value_counts completo
                    if self.label_type == 'classification':
                        print(f"🔍   y.value_counts(): {y.value_counts().to_dict()}")
                    else:
                        n_nonzero = (y != 0.0).sum()
                        n_zero = (y == 0.0).sum()
                        print(f"🔍   y: n_nonzero={n_nonzero}, n_zero={n_zero}, total={len(y)}")
                        print(f"🔍   y.min(): {y.min()}, y.max(): {y.max()}")
            
            # Verificar que tenemos suficientes datos
            if len(X) < 100:  # Mínimo requerido para análisis causal robusto
                if self.debug:
                    print(f"🔍 DEBUG apply_causal_filter - Datos insuficientes para análisis causal: {len(X)}")
                if reliable_mask is not None:
                    return np.zeros(len(full_ds))
                else:
                    return np.zeros(len(X))
            
            # Verificar que tenemos al menos 2 clases para clasificación
            if self.label_type == 'classification' and len(y.unique()) != 2:
                if self.debug:
                    print(f"🔍 DEBUG apply_causal_filter - Clases insuficientes para análisis causal: {y.unique()}")
                if reliable_mask is not None:
                    return np.zeros(len(full_ds))
                else:
                    return np.zeros(len(X))
            
            # Aplicar detección causal
            if self.label_type == 'classification':
                err0, err1, oob = _bootstrap_oob_identification(X, y, n_models=hp['causal_meta_learners'])
                best_frac = _optimize_bad_samples_threshold(err0, err1, oob, fracs=[hp['causal_percentile']/100])
                
                to_mark_0 = (err0 / oob.replace(0, 1)).fillna(0)
                to_mark_1 = (err1 / oob.replace(0, 1)).fillna(0)
                thr0 = np.percentile(to_mark_0[to_mark_0 > 0], hp['causal_percentile']) * best_frac if len(to_mark_0[to_mark_0 > 0]) else 0
                thr1 = np.percentile(to_mark_1[to_mark_1 > 0], hp['causal_percentile']) * best_frac if len(to_mark_1[to_mark_1 > 0]) else 0
                marked0 = to_mark_0[to_mark_0 > thr0].index
                marked1 = to_mark_1[to_mark_1 > thr1].index
                all_bad = pd.Index(marked0).union(marked1)
            else:  # regression
                error_sums, error_counts, oob = _bootstrap_oob_identification(X, y, n_models=hp['causal_meta_learners'])
                
                # Calcular errores promedio por muestra
                avg_errors = (error_sums / error_counts.replace(0, 1)).fillna(0)
                
                # Aplicar umbrales para regresión
                error_threshold = hp['causal_error_threshold']
                percentile_threshold = np.percentile(avg_errors[avg_errors > 0], hp['causal_percentile']) if len(avg_errors[avg_errors > 0]) > 0 else 0
                
                # Usar el umbral más estricto entre los dos
                final_threshold = max(error_threshold, percentile_threshold)
                
                # Identificar muestras malas
                all_bad = avg_errors[avg_errors > final_threshold].index
                
                if self.debug:
                    print(f"🔍   error_threshold: {error_threshold:.4f}")
                    print(f"🔍   percentile_threshold: {percentile_threshold:.4f}")
                    print(f"🔍   final_threshold: {final_threshold:.4f}")
                    print(f"🔍   avg_errors.describe(): {avg_errors.describe()}")
            
            # Crear scores: 1.0 para muestras buenas, 0.0 para muestras malas
            causal_scores = np.ones(len(X))
            causal_scores[X.index.isin(all_bad)] = 0.0
            
            if self.debug:
                print(f"🔍   all_bad.shape: {len(all_bad)}")
                print(f"🔍   causal_scores.sum(): {causal_scores.sum()}")
                print(f"🔍   causal_scores.mean(): {causal_scores.mean():.3f}")
                print(f"🔍 DEBUG apply_causal_filter - Filtrado causal completado ({self.label_type})")
            
            if reliable_mask is not None:
                # Mapear scores de vuelta al dataset completo
                full_causal_scores = np.zeros(len(full_ds))
                full_causal_scores[reliable_mask] = causal_scores
                
                if self.debug:
                    print(f"🔍   reliable_mask.sum(): {reliable_mask.sum()}")
                    print(f"🔍   causal_scores.shape: {causal_scores.shape}")
                
                return full_causal_scores
            else:
                return causal_scores
            
        except Exception as e:
            if self.debug:
                print(f"🔍 DEBUG apply_causal_filter - ERROR: {str(e)}")
            return np.zeros(len(full_ds))

    def apply_labeling(self, dataset: pd.DataFrame, hp: dict) -> pd.DataFrame:
        """Apply the selected labeling function dynamically.

        Returns an empty DataFrame if labeling fails or results in no rows.
        """
        label_func = self.LABEL_FUNCS[self.label_method]
        params = inspect.signature(label_func).parameters
        kwargs = {}

        # 🔍 DEBUG: Supervisar función de etiquetado y parámetros esperados
        if self.debug:
            print(f"🔍 DEBUG apply_labeling - Función: {label_func.__name__}")
            print(f"🔍   Parámetros esperados por función: {list(params.keys())}")
            label_params_in_hp = {k: v for k, v in hp.items() if k.startswith('label_')}
            print(f"🔍   Parámetros label_* en hp: {list(label_params_in_hp.keys())}")

        for name, param in params.items():
            if name == 'dataset':
                continue
            elif name == 'direction':
                # Mapeo consistente de string a int para todas las funciones de etiquetado
                direction_map = {"buy": 0, "sell": 1, "both": 2}
                kwargs['direction'] = direction_map.get(self.direction, 2)
                if self.debug:
                    print(f"🔍   Mapeando direction: '{self.direction}' -> {kwargs['direction']}")
            elif name == 'label_type':
                # Pasar label_type desde self.label_type
                kwargs['label_type'] = self.label_type
                if self.debug:
                    print(f"🔍   Mapeando label_type: '{self.label_type}' -> {kwargs['label_type']}")
            # ✅ SIMPLIFICADO: Pasar parámetros directamente sin conversiones
            elif name in hp:
                kwargs[name] = hp[name]
                if self.debug:
                    print(f"🔍   Mapeando: {name} = {hp[name]}")
            elif param.default is not inspect.Parameter.empty:
                kwargs[name] = param.default
                if self.debug:
                    print(f"🔍   Default: {name} = {param.default}")

        # 🔍 DEBUG: Supervisar parámetros finales que se pasan a la función
        if self.debug:
            print(f"🔍   Parámetros finales para {label_func.__name__}: {list(kwargs.keys())}")
            critical_params = ['label_markup', 'label_min_val', 'label_max_val', 'label_threshold', 'label_rolling']
            for cp in critical_params:
                if cp in kwargs:
                    print(f"🔍   {cp}: {kwargs[cp]}")

        # ── Validaciones simples ───────────────────────────────────────────
        try:
            if self.debug:
                print(f"🔍 DEBUG apply_labeling - Validaciones iniciales:")
                print(f"🔍   Dataset shape: {dataset.shape}")
                print(f"🔍   Dataset columns: {list(dataset.columns)}")
            
            if len(dataset) < 2:
                if self.debug:
                    print(f"🔍 DEBUG apply_labeling - FALLO: Dataset muy pequeño ({len(dataset)} < 2)")
                return pd.DataFrame()

            polyorder = kwargs.get('label_polyorder', 2)
            if len(dataset) <= polyorder:
                if self.debug:
                    print(f"🔍 DEBUG apply_labeling - FALLO: Dataset <= polyorder ({len(dataset)} <= {polyorder})")
                return pd.DataFrame()

            # Ajuste automático para savgol_filter y similares
            filter_val = kwargs.get('label_filter')
            filter_mean_val = kwargs.get('label_filter_mean')
            # Detectar parámetros de ventana relevantes
            window_keys = [k for k in kwargs if any(x in k for x in ('rolling', 'window', 'window_size'))]
            if (filter_val == 'savgol' or filter_mean_val == 'savgol') and window_keys:
                for k in window_keys:
                    v = kwargs[k]
                    if isinstance(v, list) or isinstance(v, tuple):
                        new_v = []
                        for val in v:
                            win = int(val)
                            win = max(win, polyorder + 1)
                            if win % 2 == 0:
                                win += 1
                            win = min(win, len(dataset))
                            if win % 2 == 0:
                                win = max(win - 1, polyorder + 1)
                            if win <= polyorder:
                                if self.debug:
                                    print(f"🔍 DEBUG apply_labeling - FALLO: Ventana savgol inválida ({win} <= {polyorder})")
                                return pd.DataFrame()
                            new_v.append(win)
                        kwargs[k] = new_v
                    else:
                        win = int(v)
                        win = max(win, polyorder + 1)
                        if win % 2 == 0:
                            win += 1
                        win = min(win, len(dataset))
                        if win % 2 == 0:
                            win = max(win - 1, polyorder + 1)
                        if win <= polyorder:
                            if self.debug:
                                print(f"🔍 DEBUG apply_labeling - FALLO: Ventana savgol inválida ({win} <= {polyorder})")
                            return pd.DataFrame()
                        kwargs[k] = win

            # Clamp window/rolling parameters to dataset length
            for k, v in list(kwargs.items()):
                if isinstance(v, (int, float)) and any(x in k for x in ('rolling', 'window', 'period', 'span', 'max_val')):
                    iv = max(int(v), 1)
                    kwargs[k] = min(iv, max(len(dataset) - 1, 1))
                elif isinstance(v, list) and any(x in k for x in ('rolling', 'window', 'period')):
                    kwargs[k] = [min(max(int(val), 1), max(len(dataset) - 1, 1)) for val in v]

            # Check for negative or too large window/rolling/period parameters
            for k, v in list(kwargs.items()):
                if isinstance(v, int) and any(x in k for x in ('rolling', 'window', 'period', 'span', 'max_val')):
                    if v <= 0 or v >= len(dataset):
                        if self.debug:
                            print(f"🔍 DEBUG apply_labeling - FALLO: parámetro '{k}'={v} inválido para dataset de tamaño {len(dataset)}")
                        return pd.DataFrame()
                elif isinstance(v, list) and any(x in k for x in ('rolling', 'window', 'period')):
                    if any((val <= 0 or val >= len(dataset)) for val in v):
                        if self.debug:
                            print(f"🔍 DEBUG apply_labeling - FALLO: lista '{k}' contiene valores inválidos para dataset de tamaño {len(dataset)}")
                        return pd.DataFrame()

            # ✅ SIMPLIFICADO: Validaciones directas con nombres originales
            if 'label_min_val' in kwargs and 'label_max_val' in kwargs and kwargs['label_min_val'] > kwargs['label_max_val']:
                kwargs['label_min_val'] = kwargs['label_max_val']

            if 'label_max_val' in kwargs and len(dataset) <= kwargs['label_max_val']:
                if self.debug:
                    print(f"🔍 DEBUG apply_labeling - FALLO: Dataset <= label_max_val ({len(dataset)} <= {kwargs['label_max_val']})")
                return pd.DataFrame()

            if self.debug:
                print(f"🔍 DEBUG apply_labeling - Validaciones pasadas, llamando a {label_func.__name__}")
                print(f"🔍   kwargs finales: {list(kwargs.keys())}")

            df = label_func(dataset, **kwargs)

            if self.debug:
                print(f"🔍 DEBUG apply_labeling - Resultado de {label_func.__name__}:")
                print(f"🔍   df is None: {df is None}")
                print(f"🔍   df.empty: {df.empty if df is not None else 'N/A'}")
                print(f"🔍   df.shape: {df.shape if df is not None else 'N/A'}")
                if df is not None and not df.empty:
                    print(f"🔍   df.columns: {list(df.columns)}")

            if df is None or df.empty:
                if self.debug:
                    print(f"🔍 DEBUG apply_labeling - FALLO: función de etiquetado devolvió DataFrame vacío o None")
                return pd.DataFrame()

            if 'labels_main' in df.columns:
                if self.debug:
                    print(f"🔍   Etiquetado exitoso: {len(df)} filas con labels_main")
                    if self.label_type == 'classification':
                        print(f"🔍   df['labels_main'].value_counts(): {df['labels_main'].value_counts()}")
                    else:
                        print(f"🔍   df['labels_main'].min(): {df['labels_main'].min()}, df['labels_main'].max(): {df['labels_main'].max()}")
                return df
            else:
                if self.debug:
                    print(f"🔍 DEBUG apply_labeling - FALLO: No se encontró columna 'labels_main' en el resultado")
                return pd.DataFrame()
        except Exception as e:
            if self.debug:
                print(f"🔍 DEBUG apply_labeling - EXCEPCIÓN: {e}")
                print(f"🔍   Traceback: {traceback.format_exc()}")
            return pd.DataFrame()

    def get_labeled_full_data(self, hp):
        try:
            if hp is None:
                return None

            if self.debug:
                print(f"🔍 DEBUG: base_df.shape = {self.base_df.shape}")
                print(f"🔍 DEBUG: train_start = {self.train_start}, train_end = {self.train_end}")
                print(f"🔍 DEBUG: test_start = {self.test_start}, test_end = {self.test_end}")

            # ──────────────────────────────────────────────────────────────
            # 1) Calcular el colchón de barras necesario
            main_periods = hp.get('feature_main_periods', ())
            meta_periods = hp.get('feature_meta_periods', ())
            
            # 🔍 DEBUG: Supervisar parámetros de features
            if self.debug:
                print(f"🔍 DEBUG get_labeled_full_data - Parámetros de features:")
                feature_params = {k: v for k, v in hp.items() if k.startswith('feature_')}
                for k, v in feature_params.items():
                    if isinstance(v, (list, tuple)) and len(v) > 3:
                        print(f"🔍   {k}: {type(v).__name__}[{len(v)}] = {v[:3]}...")
                    else:
                        print(f"🔍   {k}: {v}")
            
            if not main_periods:
                return None
                
            pad = int(max(main_periods) + max(meta_periods) if meta_periods else max(main_periods))
            if self.debug:
                print(f"🔍 DEBUG: pad = {pad}")

            # ──────────────────────────────────────────────────────────────
            # 2) Paso típico de la serie (modal → inmune a huecos)
            idx = self.base_df.index.sort_values()
            if pad > 0 and len(idx) > 1:
                bar_delta = idx.to_series().diff().dropna().mode().iloc[0]
            else:
                bar_delta = pd.Timedelta(0)
            if self.debug:
                print(f"🔍 DEBUG: bar_delta = {bar_delta}")

            # ──────────────────────────────────────────────────────────────
            # 3) Rango extendido para calcular features "con contexto"
            start_ext = min(self.train_start, self.test_start) - pad * bar_delta
            if start_ext < idx[0]:
                start_ext = idx[0]

            end_ext = max(self.train_end, self.test_end)
            if self.debug:
                print(f"🔍 DEBUG: start_ext = {start_ext}, end_ext = {end_ext}")

            # ──────────────────────────────────────────────────────────────
            # 4) Obtener features de todo el rango extendido
            full_ds = self.base_df.loc[start_ext:end_ext].copy()
            if self.debug:
                print(f"🔍 DEBUG: full_ds.shape = {full_ds.shape}")
            
            # 🔍 DEBUG: Supervisar paso de parámetros a get_features
            if self.debug:
                print(f"🔍 DEBUG: Llamando get_features con:")
                print(f"🔍   feature_main_periods: {hp.get('feature_main_periods', 'N/A')}")
                print(f"🔍   feature_main_stats: {hp.get('feature_main_stats', 'N/A')}")
                print(f"🔍   feature_meta_periods: {hp.get('feature_meta_periods', 'N/A')}")
                print(f"🔍   feature_meta_stats: {hp.get('feature_meta_stats', 'N/A')}")
            if not isinstance(hp.get('feature_main_periods', []), (list, tuple)):
                hp['feature_main_periods'] = tuple(hp['feature_main_periods'])
            if not isinstance(hp.get('feature_main_stats', []), (list, tuple)):
                hp['feature_main_stats'] = tuple(hp['feature_main_stats'])
            if not isinstance(hp.get('feature_meta_periods', []), (list, tuple)):
                hp['feature_meta_periods'] = tuple(hp['feature_meta_periods'])
            if not isinstance(hp.get('feature_meta_stats', []), (list, tuple)):
                hp['feature_meta_stats'] = tuple(hp['feature_meta_stats'])
            full_ds = get_features(data=full_ds, hp=hp, decimal_precision=self.decimal_precision)
            if self.debug:
                print(f"🔍 DEBUG: full_ds.shape después de get_features = {full_ds.shape}")
                feature_cols = [c for c in full_ds.columns if 'feature' in c]
                print(f"🔍   Columnas de features generadas: {len(feature_cols)}")
                main_features = [c for c in feature_cols if '_main_feature' in c]
                meta_features = [c for c in feature_cols if '_meta_feature' in c]
                print(f"🔍   Main features: {len(main_features)}, Meta features: {len(meta_features)}")

            # ✅ SIMPLIFICADO: Pasar hp directamente sin conversiones
            # 🔍 DEBUG: Supervisar paso de parámetros a apply_labeling
            if self.debug:
                print(f"🔍 DEBUG: Llamando apply_labeling con label_method='{self.label_method}'")
                label_params = {k: v for k, v in hp.items() if k.startswith('label_')}
                print(f"🔍   Parámetros label_* disponibles: {list(label_params.keys())}")

            full_ds = self.apply_labeling(full_ds, hp)
            if self.debug:
                print(f"🔍 DEBUG: full_ds.shape después de apply_labeling = {full_ds.shape}")

            # y recortar exactamente al rango que interesa
            full_ds = full_ds.loc[
                min(self.train_start, self.test_start):
                max(self.train_end,   self.test_end)
            ]
            if self.debug:
                print(f"🔍 DEBUG: full_ds.shape después de recorte a rango de interés = {full_ds.shape}")

            if full_ds.empty:
                return None

            # ──────────────────────────────────────────────────────────────
            # 5) Comprobaciones de calidad de features
            feature_cols = full_ds.columns[full_ds.columns.str.contains('feature')]
            if feature_cols.empty:
                return None

            problematic = self.check_constant_features(full_ds, list(feature_cols))
            if problematic:
                if self.debug:
                    print(f"🔍 DEBUG: Columnas problemáticas eliminadas: {len(problematic)}")
                full_ds = full_ds.drop(columns=problematic)
                feature_cols = [c for c in feature_cols if c not in problematic]
                if not feature_cols:
                    return None

            # --- 5b) Reconstruir hp manteniendo el orden original ---------
            main_periods_ordered = []
            main_stats_ordered   = []
            meta_periods_ordered = []
            meta_stats_ordered   = []

            seen_main_periods = set()
            seen_main_stats   = set()
            seen_meta_periods = set()
            seen_meta_stats   = set()

            for col in feature_cols:
                if col.endswith('_main_feature'):
                    # Remover '_main_feature'
                    col_parts = col[:-13]
                    # Dividir en período y estadística
                    parts = col_parts.split('_')
                    p = int(parts[0])
                    stat = '_'.join(parts[1:])

                    # periodo main
                    if p not in seen_main_periods:
                        main_periods_ordered.append(p)
                        seen_main_periods.add(p)

                    # estadístico main
                    if stat not in seen_main_stats:
                        main_stats_ordered.append(stat)
                        seen_main_stats.add(stat)

                elif col.endswith('_meta_feature'):
                    # Remover '_meta_feature'
                    col_parts = col[:-13]
                    # Dividir en período y estadística
                    parts = col_parts.split('_')
                    p = int(parts[0])
                    stat = '_'.join(parts[1:])

                    # periodo meta
                    if p not in seen_meta_periods:
                        meta_periods_ordered.append(p)
                        seen_meta_periods.add(p)

                    # estadístico meta
                    if stat not in seen_meta_stats:
                        meta_stats_ordered.append(stat)
                        seen_meta_stats.add(stat)

            # -------- aplicar a hp con los nombres nuevos ----------------
            hp['feature_main_periods'] = tuple(main_periods_ordered)
            hp['feature_main_stats']   = tuple(main_stats_ordered)
            hp['feature_meta_periods'] = tuple(meta_periods_ordered)
            hp['feature_meta_stats']   = tuple(meta_stats_ordered)
            if self.debug:
                print(f"🔍 DEBUG: feature_main_periods después de reconstruir: {hp['feature_main_periods']}")
                print(f"🔍 DEBUG: feature_main_stats después de reconstruir: {hp['feature_main_stats']}")
                print(f"🔍 DEBUG: feature_meta_periods después de reconstruir: {hp['feature_meta_periods']}")
                print(f"🔍 DEBUG: feature_meta_stats después de reconstruir: {hp['feature_meta_stats']}")

            # Verificar que tenemos al menos períodos y stats main
            main_periods = hp.get('feature_main_periods', ())
            main_stats = hp.get('feature_main_stats', ())
            if len(main_periods) == 0 or len(main_stats) == 0:
                return None

            # 🔍 DEBUG: Mostrar primera fila del dataset con características
            if self.debug and not full_ds.empty:
                print(f"🔍 DEBUG: Primera fila del dataset con características:")
                print(f"🔍   Índice: {full_ds.index[0]}")
                print(f"🔍   Muestra: {full_ds.iloc[0].to_dict()}")

            # ──────────────────────────────────────────────────────────────
            # Chequeo de integridad: asegurar que todos los índices de base_df en el rango de full_ds están en full_ds
            idx_full = full_ds.index
            start, end = idx_full[0], idx_full[-1]
            base_range = self.base_df.loc[start:end]
            missing_idx = base_range.index.difference(idx_full)
            if self.debug:
                if missing_idx.empty:
                    print(f"🔍 DEBUG: Chequeo de integridad OK: todos los índices de base_df[{start} - {end}] ({len(base_range)}) están en full_ds ({len(idx_full)})")
                else:
                    print(f"🔍 DEBUG: ERROR de integridad: faltan {len(missing_idx)} índices de base_df en full_ds")
                    print(f"🔍   Ejemplo de índices faltantes: {list(missing_idx[:5])}")
            if not missing_idx.empty:
                raise ValueError(f"Integridad de full_ds fallida: faltan {len(missing_idx)} índices de base_df en full_ds en el rango de interés.")

            return full_ds

        except Exception as e:
            print(f"🔍 DEBUG: ERROR en get_labeled_full_data: {str(e)}")
            return None

    def get_train_test_data(self, dataset) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Genera los DataFrames de entrenamiento y prueba a partir del DataFrame completo."""
        if dataset is None or dataset.empty:
            return None, None
        
        # Máscaras de train / test
        test_mask  = (dataset.index >= self.test_start)  & (dataset.index <= self.test_end)
        train_mask = (dataset.index >= self.train_start) & (dataset.index <= self.train_end)

        if not test_mask.any() or not train_mask.any():
            return None, None

        # Evitar solapamiento
        if self.test_start <= self.train_end and self.test_end >= self.train_start:
            train_mask &= ~test_mask
            if self.debug:
                print(f"🔍 DEBUG: train_mask.sum() después de evitar solapamiento = {train_mask.sum()}")

        # DataFrames finales, ordenados cronológicamente
        train_data = dataset[train_mask].sort_index().copy()
        test_data  = dataset[test_mask].sort_index().copy()

        if self.debug:
            print(f"🔍 DEBUG: train_data.shape final = {train_data.shape}")
            print(f"🔍 DEBUG: test_data.shape final = {test_data.shape}")
        
        return train_data, test_data

    def get_base_mask(self, full_ds: pd.DataFrame) -> pd.Series:
        """
        Obtiene máscara de trading según label_type.
        
        Args:
            full_ds: Dataset completo con labels_main
            hp: Hiperparámetros para calcular threshold dinámico (opcional)
            use_cv: Si usar validación cruzada temporal para regresión
            
        Returns:
            pd.Series: Máscara booleana para muestras de trading
        """
        if self.label_type == 'classification':
            base_mask = full_ds['labels_main'] != 2.0
        else:
            # Aplicar filtro según dirección
            if self.direction == 'buy':
                base_mask = full_ds['labels_main'] > 0.0
            elif self.direction == 'sell':
                base_mask = full_ds['labels_main'] < 0.0
            else:  # 'both'
                base_mask = full_ds['labels_main'] != 0.0

        if self.debug:
            print(f"🔍   Main labels - total_samples: {len(full_ds)}")
            print(f"🔍   Main labels - trading_samples: {base_mask.sum()}")
            print(f"🔍   Main labels - reduction: {len(full_ds)} -> {base_mask.sum()}")

        return base_mask

    def check_constant_features(self, X: pd.DataFrame, feature_cols: list, std_epsilon: float = 1e-6) -> list:
        """Return the list of columns that may cause numerical instability.
        
        Args:
            X: DataFrame con los datos
            feature_cols: Lista con nombres de las columnas
            std_epsilon: Umbral para considerar una columna como constante
            
        Returns:
            list: Lista de columnas problemáticas
        """
        problematic_cols = []
        
        # 1) Verificar columnas con nan/inf
        for col in feature_cols:
            series = X[col]
            if not np.isfinite(series).all():
                problematic_cols.append(col)
                
        # 2) Verificar columnas (casi) constantes
        stds = X[feature_cols].std(axis=0, skipna=True)
        for col, std in stds.items():
            if std < std_epsilon:
                problematic_cols.append(col)
                
        return problematic_cols

    def calculate_regression_threshold(self, labels_main: pd.Series, hp: Dict[str, Any]) -> tuple[float, int]:
        """
        Calcula el threshold dinámico para regresión según la dirección.
        
        Esta función centraliza la lógica de cálculo de threshold para regresión,
        asegurando consistencia en todo el pipeline y facilitando el mantenimiento.
        
        Parameters
        ----------
        labels_main : pd.Series
            Etiquetas de regresión (valores continuos)
        hp : Dict[str, Any]
            Hiperparámetros con 'main_threshold'
        
        Returns
        -------
        tuple[float, int]
            (threshold_value, significant_samples)
            threshold_value: Umbral calculado dinámicamente
            significant_samples: Número de muestras que superan el umbral
        """
        if self.direction == 'buy':
            # Solo valores positivos significativos
            positive_labels = labels_main[labels_main > 0]
            if len(positive_labels) > 0:
                # Usar percentil optimizado por Optuna
                percentile = hp['model_main_percentile']
                threshold_value = positive_labels.quantile(percentile)
                significant_samples = (labels_main > threshold_value).sum()
                if self.debug:
                    print(f"🔍   buy: threshold={threshold_value:.4f} (q{percentile:.2f}), positive_q{percentile:.2f}={positive_labels.quantile(percentile):.4f}")
            else:
                threshold_value = 0.0
                significant_samples = 0
                if self.debug:
                    print(f"🔍   buy: no positive labels found")
                
        elif self.direction == 'sell':
            # Solo valores negativos significativos
            negative_labels = labels_main[labels_main < 0]
            if len(negative_labels) > 0:
                # Usar percentil optimizado por Optuna (invertido para valores negativos)
                percentile = hp['model_main_percentile']
                threshold_value = abs(negative_labels.quantile(percentile))
                significant_samples = (labels_main < -threshold_value).sum()
                if self.debug:
                    print(f"🔍   sell: threshold={threshold_value:.4f} (q{percentile:.2f}), negative_q{percentile:.2f}={negative_labels.quantile(percentile):.4f}")
            else:
                threshold_value = 0.0
                significant_samples = 0
                if self.debug:
                    print(f"🔍   sell: no negative labels found")
                
        else:  # 'both'
            # Valores absolutos significativos
            # Usar percentil optimizado por Optuna
            percentile = hp['model_main_percentile']
            abs_labels = abs(labels_main)
            threshold_value = abs_labels.quantile(percentile)
            significant_samples = (abs(labels_main) > threshold_value).sum()
            if self.debug:
                print(f"🔍   both: threshold={threshold_value:.4f} (q{percentile:.2f}), abs_q{percentile:.2f}={abs_labels.quantile(percentile):.4f}")
        
        return threshold_value, significant_samples

    def calculate_regression_threshold_cv(self, labels_main: pd.Series, hp: Dict[str, Any], reliability_mask: pd.Series = None, n_splits: int = 5) -> tuple[float, pd.Series]:
        """
        Calcula threshold usando validación cruzada temporal.
        
        Parameters
        ----------
        labels_main : pd.Series
            Etiquetas de regresión (valores continuos)
        hp : Dict[str, Any]
            Hiperparámetros con 'main_threshold'
        reliability_mask : pd.Series, optional
            Máscara de confiabilidad para filtrar muestras en cálculos internos
        n_splits : int
            Número de splits para validación cruzada temporal
            
        Returns
        -------
        tuple[float, pd.Series]
            (threshold_value, threshold_mask)
            threshold_value: Umbral optimizado por CV
            threshold_mask: Máscara que indica qué muestras superan el umbral
        """
        
        # Determinar qué datos usar para el cálculo del threshold
        if reliability_mask is not None:
            # Usar solo las muestras filtradas
            data_for_calculation = labels_main[reliability_mask]
            if self.debug:
                print(f"🔍   CV: Usando {len(data_for_calculation)} muestras filtradas para cálculo")
        else:
            # Usar todas las muestras
            data_for_calculation = labels_main
            if self.debug:
                print(f"🔍   CV: Usando {len(data_for_calculation)} muestras totales para cálculo")
        
        # Calcular threshold usando validación cruzada temporal
        if len(data_for_calculation) < n_splits * 2:
            if self.debug:
                print(f"🔍   CV: Datos insuficientes para {n_splits} splits, usando threshold original")
            threshold_value, _ = self.calculate_regression_threshold(data_for_calculation, hp)
        else:
            tscv = TimeSeriesSplit(n_splits=n_splits)
            thresholds = []
            performances = []
            
            if self.debug:
                print(f"🔍   CV: Iniciando validación cruzada temporal con {n_splits} splits")
            
            for fold, (train_idx, val_idx) in enumerate(tscv.split(data_for_calculation)):
                train_labels = data_for_calculation.iloc[train_idx]
                val_labels = data_for_calculation.iloc[val_idx]
                
                # Calcular threshold en train
                train_threshold, _ = self.calculate_regression_threshold(train_labels, hp)
                
                # Validar en val
                val_performance = self._evaluate_threshold_performance(val_labels, train_threshold)
                
                thresholds.append(train_threshold)
                performances.append(val_performance)
                
                if self.debug:
                    print(f"🔍   CV Fold {fold+1}: threshold={train_threshold:.4f}, performance={val_performance:.4f}")
            
            # Seleccionar threshold con mejor performance promedio
            best_idx = np.argmax(performances)
            threshold_value = thresholds[best_idx]
            avg_performance = np.mean(performances)
            
            if self.debug:
                print(f"🔍   CV: Mejor threshold={threshold_value:.4f}, avg_performance={avg_performance:.4f}")
        
        # Crear threshold_mask aplicando el threshold
        if self.direction == 'buy':
            threshold_mask = labels_main > threshold_value
        elif self.direction == 'sell':
            threshold_mask = labels_main < -threshold_value
        else:  # 'both'
            threshold_mask = abs(labels_main) > threshold_value
        
        # Si se proporcionó reliability_mask, aplicar solo a esas muestras
        if reliability_mask is not None:
            threshold_mask = threshold_mask & reliability_mask
        
        return threshold_value, threshold_mask

    def _evaluate_threshold_performance(self, labels: pd.Series, threshold: float) -> float:
        """
        Evalúa la calidad del threshold usando métricas de trading.
        
        Parameters
        ----------
        labels : pd.Series
            Etiquetas de validación
        threshold : float
            Threshold a evaluar
            
        Returns
        -------
        float
            Score de performance del threshold
        """
        if self.direction == 'buy':
            significant_labels = labels[labels > threshold]
            if len(significant_labels) == 0:
                return 0.0
            consistency = (significant_labels > 0).mean()
        elif self.direction == 'sell':
            significant_labels = labels[labels < -threshold]
            if len(significant_labels) == 0:
                return 0.0
            consistency = (significant_labels < 0).mean()
        else:  # 'both'
            significant_labels = labels[abs(labels) > threshold]
            if len(significant_labels) == 0:
                return 0.0
            consistency = 0.5  # Neutral para 'both'
        
        # Métricas de calidad
        avg_magnitude = abs(significant_labels).mean()
        signal_strength = avg_magnitude * consistency
        
        return signal_strength