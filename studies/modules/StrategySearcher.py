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
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from catboost import CatBoostClassifier
from mapie.classification import CrossConformalClassifier
from modules.labeling_lib import (
    get_prices, get_features, get_labels_trend, 
    get_labels_trend_multi, get_labels_clusters,
    get_labels_multi_window, get_labels_validated_levels,
    get_labels_mean_reversion, get_labels_random,
    get_labels_mean_reversion_multi, get_labels_mean_reversion_vol,
    get_labels_filter, get_labels_filter_multi, get_labels_trend_filters,
    get_labels_filter_binary, get_labels_fractal_patterns, get_labels_zigzag,
    clustering_kmeans, clustering_hdbscan, clustering_markov, clustering_lgmm
)
from modules.tester_lib import tester, clear_onnx_session_cache, run_monkey_test
from modules.export_lib import export_models_to_ONNX, export_dataset_to_csv, export_to_mql5

class StrategySearcher:
    LABEL_FUNCS = {
        "random": get_labels_random,
        "filter": get_labels_filter,
        "filter_binary": get_labels_filter_binary,
        "filter_multi": get_labels_filter_multi,
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
        label_method: str = 'random',
        tag: str = "",
        debug: bool = False,
        decimal_precision: int = 6,
        monkey_n_simulations: int = 5000,
        monkey_alpha: float = 0.01,
        monkey_min_percentile: float = 99.0,
        monkey_min_zscore: float = 2.5,
        monkey_block_multiplier: float = 1.5,
        monkey_wfv_windows: int = 3,
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
        self.label_method = label_method
        self.pruner_type = pruner_type
        self.n_trials = n_trials
        self.n_models = n_models
        self.n_jobs = n_jobs
        self.tag = tag
        self.debug = debug
        self.base_df = get_prices(symbol, timeframe, history_path)
        self.decimal_precision = decimal_precision  # NUEVO ATRIBUTO
        self.monkey_n_simulations = monkey_n_simulations
        self.monkey_alpha = monkey_alpha
        self.monkey_min_percentile = monkey_min_percentile
        self.monkey_min_zscore = monkey_min_zscore
        self.monkey_block_multiplier = monkey_block_multiplier
        self.monkey_wfv_windows = monkey_wfv_windows

        # Configuración de logging para optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    # =========================================================================
    # Método principal
    # =========================================================================

    def run_search(self) -> None:
        search_funcs = {
            'reliability': self.search_reliability,
            'clusters': self.search_clusters
        }
        
        if self.search_type not in search_funcs:
            raise ValueError(f"Tipo de búsqueda no válido: {self.search_type}")
            
        search_func = search_funcs[self.search_type]
        
        for i in range(self.n_models):
            try:

                # Inicializar estudio de Optuna con objetivo único
                pruners = {
                    'hyperband': HyperbandPruner(max_resource='auto'),
                    'successive': SuccessiveHalvingPruner(min_resource='auto')
                }
                # Asegurar carpeta para bases de datos de Optuna
                try:
                    os.makedirs("optuna_dbs", exist_ok=True)
                except Exception as e:
                    print(f"🔍 DEBUG: No se pudo crear carpeta optuna_dbs: {e}")

                db_url = f"sqlite:///optuna_dbs/{self.tag}.db"
                study = optuna.create_study(
                    study_name=self.tag,
                    direction='maximize',
                    storage=db_url,
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
                                    study.set_user_attr('monkey_p_value', trial.user_attrs.get('monkey_p_value'))
                                    study.set_user_attr('monkey_percentile', trial.user_attrs.get('monkey_percentile'))
                                    study.set_user_attr('monkey_pass', trial.user_attrs.get('monkey_pass'))
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
                                        "models_export_path": self.models_export_path,
                                        "include_export_path": self.include_export_path,
                                        "best_score": study.user_attrs["best_score"],
                                        "monkey_p_value": study.user_attrs.get('monkey_p_value'),
                                        "monkey_percentile": study.user_attrs.get('monkey_percentile'),
                                        "monkey_pass": study.user_attrs.get('monkey_pass'),
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
                    callbacks=[log_trial]
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
        try:
            hp = self.suggest_all_params(trial)
            
            # 🔍 DEBUG: Supervisar parámetros específicos de confiabilidad
            if self.debug:
                reliability_params = {k: v for k, v in hp.items() if k.startswith('label_')}
                print(f"🔍 DEBUG search_reliability - Parámetros de confiabilidad: {reliability_params}")
                
            full_ds_is, full_ds_oos = self.get_labeled_full_data(hp=hp)
            if full_ds_is is None or full_ds_oos is None:
                return -1.0

            # Main base mask
            base_mask = full_ds_is['labels_main'].isin([0.0, 1.0])
            if not base_mask.any():
                if self.debug:
                    print(f"🔍 DEBUG search_reliability - No hay muestras de trading")
                return -1.0

            # Filtro MAPIE
            if self.debug:
                print(f"🔍 DEBUG search_reliability - Aplicando filtrado MAPIE")
                print(f"🔍   base_mask.sum(): {base_mask.sum()}")
                print(f"🔍   base_mask.mean(): {base_mask.mean():.3f}")
            
            mapie_scores = self.apply_mapie_filter(trial=trial, full_ds=full_ds_is, hp=hp, reliable_mask=base_mask)
            mapie_mask = mapie_scores == 1.0
            
            if self.debug:
                print(f"🔍   mapie_mask.sum(): {mapie_mask.sum()}")
                print(f"🔍   mapie_mask.mean(): {mapie_mask.mean():.3f}")

            # Filtro CAUSAL
            if self.debug:
                print(f"🔍 DEBUG search_reliability - Aplicando filtrado CAUSAL")
                print(f"🔍   base_mask.sum(): {base_mask.sum()}")
                print(f"🔍   base_mask.mean(): {base_mask.mean():.3f}")
            
            causal_scores = self.apply_causal_filter(trial=trial, full_ds=full_ds_is, hp=hp, reliable_mask=base_mask)
            causal_mask = causal_scores == 1.0
            
            if self.debug:
                print(f"🔍   causal_mask.sum(): {causal_mask.sum()}")
                print(f"🔍   causal_mask.mean(): {causal_mask.mean():.3f}")

            # Crear máscara final
            mapie_causal_mask = (base_mask & mapie_mask & causal_mask).astype(bool)
            if self.debug:
                print(f"🔍   mapie_causal_mask.sum(): {mapie_causal_mask.sum()}")
                print(f"🔍   mapie_causal_mask.mean(): {mapie_causal_mask.mean():.3f}")
                print(f"🔍   Reducción de muestras: {base_mask.sum()} -> {mapie_causal_mask.sum()} ({mapie_causal_mask.sum()/base_mask.sum()*100:.1f}%)")

            # Crear dataset main con base_mask
            main_feature_cols = [c for c in full_ds_is.columns if c.endswith('_main_feature')]
            model_main_train_data = full_ds_is.loc[base_mask, main_feature_cols + ['labels_main']].dropna(subset=main_feature_cols).copy()
            if len(model_main_train_data) < 200:
                if self.debug:
                    print(f"🔍 DEBUG search_reliability - Insuficientes muestras main: {len(model_main_train_data)}")
                return -1.0
            if set(model_main_train_data['labels_main'].unique()) != {0.0, 1.0}:
                if self.debug:
                    print(f"🔍   Search reliability - labels_main insuficientes")
                return -1.0
            if self.debug:
                print(f"🔍 DEBUG search_reliability - Main data shape: {model_main_train_data.shape}")
                print(f"🔍   Main data shape: {model_main_train_data.shape}")
            if self.debug:
                main_dist = model_main_train_data['labels_main'].value_counts()
                print(f"🔍   Main labels distribution: {main_dist}")

            # Crear dataset meta con final_mask
            meta_feature_cols = [c for c in full_ds_is.columns if c.endswith('_meta_feature')]
            if not meta_feature_cols:  # Fallback: usar main features si no hay meta features
                meta_feature_cols = main_feature_cols
            model_meta_train_data = full_ds_is[meta_feature_cols].dropna(subset=meta_feature_cols).copy()
            meta_mask = self.create_oof_meta_mask(
                model_main_train_data=model_main_train_data, model_meta_train_data=model_meta_train_data, hp=hp)
            meta_mask_full = meta_mask.reindex(full_ds_is.index).fillna(False)
            final_mask = (mapie_causal_mask & meta_mask_full).astype(bool)
            model_meta_train_data['labels_meta'] = (
                final_mask.reindex(model_meta_train_data.index).fillna(False).astype('int8')
            )
            if model_meta_train_data is None or model_meta_train_data.empty:
                return -1.0
            if set(model_meta_train_data['labels_meta'].unique()) != {0.0, 1.0}:
                if self.debug:
                    print(f"🔍   Search reliability - labels_meta insuficientes")
                return -1.0
            if self.debug:
                print(f"🔍 DEBUG search_reliability - Meta data shape: {model_meta_train_data.shape}")
                print(f"🔍   Meta data shape: {model_meta_train_data.shape}")
            if self.debug:
                meta_dist = model_meta_train_data['labels_meta'].value_counts()
                print(f"🔍   Meta labels distribution: {meta_dist}")
                
            # Usar pipeline existente
            score, full_ds_with_labels_path, model_paths, models_cols, monkey_info = self.fit_final_models(
                trial=trial,
                full_ds_is=full_ds_is,
                full_ds_oos=full_ds_oos,
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
            trial.set_user_attr('best_main_threshold', hp['main_threshold'])
            trial.set_user_attr('best_meta_threshold', hp['meta_threshold'])
            if isinstance(monkey_info, dict) and monkey_info:
                trial.set_user_attr('monkey_p_value', monkey_info.get('monkey_p_value'))
                trial.set_user_attr('monkey_percentile', monkey_info.get('monkey_percentile'))
                trial.set_user_attr('monkey_pass', monkey_info.get('monkey_pass'))

            return trial.user_attrs.get('score', -1.0)
            
        except Exception as e:
            print(f"Error en search_reliability: {str(e)}")
            return -1.0

    def search_clusters(self, trial: optuna.Trial) -> float:
        def _clustering_method(data, hp):
            try:
                if self.search_subtype == 'kmeans':
                    reliable_data = clustering_kmeans(
                        data,
                        n_clusters=hp['cluster_kmeans_n_clusters'],
                        window_size=hp['cluster_kmeans_window'],
                        step=hp.get('cluster_kmeans_step', None)
                    )
                elif self.search_subtype == 'hdbscan':
                    reliable_data = clustering_hdbscan(
                        data,
                        n_clusters=hp['cluster_hdbscan_min_cluster_size']
                    )
                elif self.search_subtype == 'markov':
                    reliable_data = clustering_markov(
                        data,
                        model_type=hp['cluster_markov_model'],
                        n_regimes=hp['cluster_markov_regimes'],
                        n_iter=hp.get('cluster_markov_iter', 100),
                        n_mix=hp.get('cluster_markov_mix', 3)
                    )
                elif self.search_subtype == 'lgmm':
                    reliable_data = clustering_lgmm(
                        data,
                        n_components=hp['cluster_lgmm_components'],
                        covariance_type=hp.get('cluster_lgmm_covariance', 'full'),
                        max_iter=hp.get('cluster_lgmm_iter', 100),
                    )

                return reliable_data
            
            except Exception as e:
                print(f"Error en _clustering_method: {str(e)}")
                return pd.DataFrame()
        
        try:
            hp = self.suggest_all_params(trial)
            
            # 🔍 DEBUG: Supervisar parámetros específicos de clusters
            if self.debug:
                cluster_params = {k: v for k, v in hp.items() if k.startswith('cluster_')}
                print(f"🔍 DEBUG search_clusters {self.search_subtype} - Parámetros clusters: {cluster_params}")
                
            full_ds_is, full_ds_oos = self.get_labeled_full_data(hp=hp)
            if full_ds_is is None or full_ds_oos is None:
                return -1.0
            
            base_mask = full_ds_is['labels_main'].isin([0.0, 1.0])
            reliable_data = full_ds_is[base_mask].copy()
            if len(reliable_data) < 200:
                if self.debug:
                    print(f"🔍 DEBUG search_clusters {self.search_subtype} - Insuficientes muestras confiables para hacer clustering")
                return -1.0
            
            # 🔍 DEBUG: Verificar distribución después del filtrado
            if self.debug:
                print(f"🔍 DEBUG search_clusters {self.search_subtype}:")
                print(f"🔍   Total muestras: {len(full_ds_is)}")
                print(f"🔍   Muestras confiables: {base_mask.sum()} ({base_mask.mean():.1%})")
                print(f"🔍   Total muestras confiables: {len(reliable_data)}")
            
            # Aplicar clustering
            reliable_data_clustered = _clustering_method(reliable_data, hp)
            # Propagar clusters a dataset completo
            full_ds_is.loc[base_mask, 'labels_meta'] = reliable_data_clustered['labels_meta']
            full_ds_is.loc[~base_mask, 'labels_meta'] = -1  # Muestras no confiables sin cluster

            score, full_ds_with_labels_path, model_paths, models_cols, monkey_info = self.evaluate_clusters(
                trial=trial, full_ds_is=full_ds_is, full_ds_oos=full_ds_oos, hp=hp)
            if score is None or model_paths is None or models_cols is None or full_ds_with_labels_path is None:
                return -1.0
            
            trial.set_user_attr('score', score)
            trial.set_user_attr('model_paths', model_paths)
            trial.set_user_attr('model_cols', models_cols)
            trial.set_user_attr('full_ds_with_labels_path', full_ds_with_labels_path)
            trial.set_user_attr('best_main_threshold', hp['main_threshold'])
            trial.set_user_attr('best_meta_threshold', hp['meta_threshold'])
            if isinstance(monkey_info, dict) and monkey_info:
                trial.set_user_attr('monkey_p_value', monkey_info.get('monkey_p_value'))
                trial.set_user_attr('monkey_percentile', monkey_info.get('monkey_percentile'))
                trial.set_user_attr('monkey_pass', monkey_info.get('monkey_pass'))

            return trial.user_attrs.get('score', -1.0)
        except Exception as e:
            print(f"Error en search_clusters: {str(e)}")
            return -1.0
        
    # =========================================================================
    # Métodos auxiliares
    # =========================================================================
    
    def evaluate_clusters(self, trial: optuna.trial, full_ds_is: pd.DataFrame, full_ds_oos: pd.DataFrame, hp: Dict[str, Any]) -> tuple[float, tuple, tuple, str, dict]:
        """Función helper para evaluar clusters y entrenar modelos."""
        try:
            # Esquema tradicional de clusters
            best_score = -math.inf
            best_model_paths = (None, None)
            best_models_cols = (None, None)
            best_full_ds_with_labels_path = None
            best_monkey_info: dict | None = None

            # 🔍 DEBUG: Supervisar parámetros
            if self.debug:
                validation_params = {k: v for k, v in hp.items() if k.startswith('label_')}
                print(f"🔍 DEBUG evaluate_clusters - Parámetros de validación: {validation_params}")

            # Extraer clusters
            cluster_sizes = full_ds_is['labels_meta'].value_counts().sort_index()
            if self.debug:
                print(f"🔍 DEBUG: Cluster sizes:\n{cluster_sizes}")
            if -1 in cluster_sizes.index:
                cluster_sizes = cluster_sizes.drop(-1)
            if cluster_sizes.empty:
                if self.debug:
                    print("⚠️ ERROR: No hay clusters")
                return None, None, None, None

            # Evaluar cada cluster
            for clust in cluster_sizes.index:
                cluster_mask = full_ds_is['labels_meta'] == clust
                if not cluster_mask.any():
                    if self.debug:
                        print(f"🔍   Cluster {clust} descartado: sin muestras confiables")
                    continue
                
                # Filtro MAPIE
                if self.debug:
                    print(f"🔍 DEBUG evaluate_clusters - Aplicando filtrado MAPIE al cluster {clust}")
                    print(f"🔍   cluster_mask.sum(): {cluster_mask.sum()}")
                    print(f"🔍   cluster_mask.mean(): {cluster_mask.mean():.3f}")
                
                mapie_scores = self.apply_mapie_filter(trial=trial, full_ds=full_ds_is, hp=hp, reliable_mask=cluster_mask)
                mapie_mask = mapie_scores == 1.0
                
                if self.debug:
                    print(f"🔍   mapie_mask.sum(): {mapie_mask.sum()}")
                    print(f"🔍   mapie_mask.mean(): {mapie_mask.mean():.3f}")
                
                # Filtro CAUSAL
                if self.debug:
                    print(f"🔍 DEBUG evaluate_clusters - Aplicando filtrado CAUSAL al cluster {clust}")
                    print(f"🔍   cluster_mask.sum(): {cluster_mask.sum()}")
                    print(f"🔍   cluster_mask.mean(): {cluster_mask.mean():.3f}")
                
                causal_scores = self.apply_causal_filter(trial=trial, full_ds=full_ds_is, hp=hp, reliable_mask=cluster_mask)
                causal_mask = causal_scores == 1.0
                
                if self.debug:
                    print(f"🔍   causal_mask.sum(): {causal_mask.sum()}")
                    print(f"🔍   causal_mask.mean(): {causal_mask.mean():.3f}")
                
                # Crear máscara final
                mapie_causal_mask = (cluster_mask & mapie_mask & causal_mask).astype(bool)
                if self.debug:
                    print(f"🔍   mapie_causal_mask.sum(): {mapie_causal_mask.sum()}")
                    print(f"🔍   mapie_causal_mask.mean(): {mapie_causal_mask.mean():.3f}")
                    print(f"🔍   Reducción de muestras: {cluster_mask.sum()} -> {mapie_causal_mask.sum()} ({mapie_causal_mask.sum()/cluster_mask.sum()*100:.1f}%)")
                
                # Crear dataset main con cluster_mask
                main_feature_cols = [c for c in full_ds_is.columns if c.endswith('_main_feature')]
                model_main_train_data = full_ds_is.loc[cluster_mask, main_feature_cols + ['labels_main']].dropna(subset=main_feature_cols).copy()
                if len(model_main_train_data) < 200:
                    if self.debug:
                        print(f"🔍 DEBUG evaluate_clusters - Insuficientes muestras main: {len(model_main_train_data)}")
                    continue
                if set(model_main_train_data['labels_main'].unique()) != {0.0, 1.0}:
                    if self.debug:
                        print(f"🔍   Cluster {clust} descartado: labels_main insuficientes")
                    continue
                if self.debug:
                    print(f"🔍 DEBUG evaluate_clusters - Main data shape: {model_main_train_data.shape}")
                    print(f"🔍   Main data shape: {model_main_train_data.shape}")
                if self.debug:
                    main_dist = model_main_train_data['labels_main'].value_counts()
                    print(f"🔍   Main labels distribution: {main_dist}")

                # Crear dataset meta con final_mask
                meta_feature_cols = [c for c in full_ds_is.columns if c.endswith('_meta_feature')]
                if not meta_feature_cols:  # Fallback: usar main features si no hay meta features
                    meta_feature_cols = main_feature_cols
                model_meta_train_data = full_ds_is[meta_feature_cols].dropna(subset=meta_feature_cols).copy()
                meta_mask = self.create_oof_meta_mask(
                    model_main_train_data=model_main_train_data, model_meta_train_data=model_meta_train_data, hp=hp)
                meta_mask_full = meta_mask.reindex(full_ds_is.index).fillna(False)
                final_mask = (mapie_causal_mask & meta_mask_full).astype(bool)
                model_meta_train_data['labels_meta'] = (
                    final_mask.reindex(model_meta_train_data.index).fillna(False).astype('int8')
                )
                if model_meta_train_data is None or model_meta_train_data.empty:
                    continue
                if set(model_meta_train_data['labels_meta'].unique()) != {0.0, 1.0}:
                    if self.debug:
                        print(f"🔍   Cluster {clust} descartado: labels_meta insuficientes")
                    continue
                if self.debug:
                    print(f"🔍 DEBUG evaluate_clusters - Meta data shape: {model_meta_train_data.shape}")
                    print(f"🔍   Meta data shape: {model_meta_train_data.shape}")
                if self.debug:
                    meta_dist = model_meta_train_data['labels_meta'].value_counts()
                    print(f"🔍   Meta labels distribution: {meta_dist}")
                    
                # Entrenar modelos
                score, full_ds_with_labels_path, model_paths, models_cols, monkey_info = self.fit_final_models(
                    trial=trial,
                    full_ds_is=full_ds_is,
                    full_ds_oos=full_ds_oos,
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
                    best_model_paths = model_paths
                    best_full_ds_with_labels_path = full_ds_with_labels_path
                    best_models_cols = models_cols
                    best_monkey_info = monkey_info

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
            return best_score, best_full_ds_with_labels_path, best_model_paths, best_models_cols, (best_monkey_info or {})
        except Exception as e:
            print(f"⚠️ ERROR en evaluación de clusters: {str(e)}")
            return None, None, None, None
    
    def _suggest_catboost(self, group: str, trial: optuna.Trial) -> Dict[str, float]:
        """Devuelve hiperparámetros CatBoost (main|meta) con prefijo `group`."""
        p = {}
        # Rango más acotado para promover entrenamientos estables (menos under/overfitting) y menor diversidad
        p[f'{group}_iterations']      = trial.suggest_int (f'{group}_iterations',      400, 800, step=50)
        p[f'{group}_depth']           = trial.suggest_int (f'{group}_depth',           5,   10)
        p[f'{group}_learning_rate']   = trial.suggest_float(f'{group}_learning_rate',  1e-2, 0.10, log=True)
        p[f'{group}_l2_leaf_reg']     = trial.suggest_float(f'{group}_l2_leaf_reg',    2.0,  5.0,  log=True)
        p[f'{group}_early_stopping']  = trial.suggest_int (f'{group}_early_stopping',  50,  150,  step=20)
        return p

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
        n_periods = trial.suggest_int("feature_main_n_periods", 1, 15)
        feature_periods = [
            trial.suggest_int(f"feature_main_period_{i}", 5, 200, log=True)
            for i in range(n_periods)
        ]
        p["feature_main_periods"] = tuple(sorted(set(feature_periods)))

        # ─── FEATURE MAIN - ESTADÍSTICAS ───────────────────────────────────────────
        n_stats = trial.suggest_int("feature_main_n_stats", 1, 5)
        feature_stats = [
            trial.suggest_categorical(f"feature_main_stat_{i}", ALL_STATS)
            for i in range(n_stats)
        ]
        # mantener orden de aparición sin duplicados
        p["feature_main_stats"] = tuple(sorted(dict.fromkeys(feature_stats)))

        # ─── FEATURE META (solo ciertos search_type) ──────────────────────────
        if self.search_type in {"reliability", "clusters"}:
            # períodos meta
            n_meta_periods = trial.suggest_int("feature_meta_n_periods", 1, 5)
            meta_periods = [
                trial.suggest_int(f"feature_meta_period_{i}", 5, 15, log=True)
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
            'label_markup':     lambda t: t.suggest_float('label_markup',     0.15, 1.50, log=True),
            'label_n_clusters': lambda t: t.suggest_int('label_n_clusters', 4, 20, log=True),
            'label_polyorder':  lambda t: t.suggest_int('label_polyorder',    2, 4),
            'label_threshold':  lambda t: t.suggest_float('label_threshold',  0.20, 0.60),
            'label_corr_threshold': lambda t: t.suggest_float('label_corr_threshold', 0.70, 0.90),
            'label_rolling':    lambda t: t.suggest_int  ('label_rolling',    50, 400, log=True),
            'label_rolling2':  lambda t: t.suggest_int  ('label_rolling2',   50, 400, log=True),
            'label_rolling_periods_small': lambda t: [t.suggest_int(f'label_rolling_periods_small_{i}', 10, 60, log=True) for i in range(3)],
            'label_rolling_periods_big': lambda t: [t.suggest_int(f'label_rolling_periods_big_{i}', 150, 800, log=True) for i in range(3)],
            'label_atr_period': lambda t: t.suggest_int  ('label_atr_period', 14, 28, log=True),
            'label_min_val':    lambda t: t.suggest_int  ('label_min_val',    5,  20, log=True),
            'label_max_val':    lambda t: t.suggest_int  ('label_max_val',    20, 60, log=True),
            'label_method_trend':     lambda t: t.suggest_categorical('label_method_trend', ['normal', 'inverse']),
            'label_method_random':     lambda t: t.suggest_categorical('label_method_random', ['first', 'last', 'mean', 'max', 'min', 'random']),
            'label_filter':     lambda t: t.suggest_categorical('label_filter', ['savgol', 'spline', 'sma', 'ema']),
            'label_filter_mean':     lambda t: t.suggest_categorical('label_filter_mean', ['savgol', 'spline', 'mean']),
            'label_window_size': lambda t: t.suggest_int('label_window_size', 20, 120, log=True),
            'label_window_sizes_int': lambda t: [t.suggest_int(f'label_window_sizes_{i}', 20, 150, log=True) for i in range(3)],
            'label_window_sizes_float': lambda t: [t.suggest_float(f'label_window_sizes_{i}', 0.10, 0.60) for i in range(3)],
            'label_min_window': lambda t: t.suggest_int('label_min_window', 6, 30, log=True),
            'label_max_window': lambda t: t.suggest_int('label_max_window', 30, 120, log=True),
            'label_vol_window': lambda t: t.suggest_int('label_vol_window', 20, 150, log=True),
            'label_min_touches': lambda t: t.suggest_int('label_min_touches', 2, 6),
            'label_peak_prominence': lambda t: t.suggest_float('label_peak_prominence', 0.08, 0.35),
            'label_quantiles': lambda t: [t.suggest_float(f'label_quantiles_{i}', 0.30, 0.70) for i in range(2)],
            'label_decay_factor': lambda t: t.suggest_float('label_decay_factor', 0.90, 0.99),
            'label_shift': lambda t: t.suggest_int('label_shift', 0, 5),
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
                p['cluster_kmeans_n_clusters'] = trial.suggest_int ('cluster_kmeans_n_clusters', 5, 30, log=True)
                p['cluster_kmeans_window']     = trial.suggest_int ('cluster_kmeans_window',     40, 300, log=True)
                p['cluster_kmeans_step']       = trial.suggest_int ('cluster_kmeans_step',       3,  40)
            elif self.search_subtype == 'hdbscan':
                p['cluster_hdbscan_min_cluster_size'] = trial.suggest_int ('cluster_hdbscan_min_cluster_size', 5, 100, log=True)
            elif self.search_subtype == 'markov':
                p['cluster_markov_model']    = trial.suggest_categorical('cluster_markov_model', ['GMMHMM', 'HMM'])
                p['cluster_markov_regimes']  = trial.suggest_int ('cluster_markov_regimes', 2, 10, log=True)
                p['cluster_markov_iter']     = trial.suggest_int ('cluster_markov_iter',    30, 300, log=True)
                p['cluster_markov_mix']      = trial.suggest_int ('cluster_markov_mix',     2, 5)
            elif self.search_subtype == 'lgmm':
                p['cluster_lgmm_components']  = trial.suggest_int ('cluster_lgmm_components',  2, 20, log=True)
                p['cluster_lgmm_covariance']  = trial.suggest_categorical('cluster_lgmm_covariance', ['full', 'tied', 'diag', 'spherical'])
                p['cluster_lgmm_iter']        = trial.suggest_int ('cluster_lgmm_iter',        50, 500, log=True)

        p['mapie_confidence_level'] = trial.suggest_float('mapie_confidence_level', 0.75, 0.98)
        p['mapie_cv']               = trial.suggest_int  ('mapie_cv',               2, 8)
        p['causal_meta_learners'] = trial.suggest_int('causal_meta_learners', 5, 20)
        p['causal_percentile'] = trial.suggest_int('causal_percentile', 60, 95)
        p['oof_resid_percentile'] = trial.suggest_int('oof_resid_percentile', 75, 95)
        p['meta_threshold'] = 0.5 # trial.suggest_float('meta_threshold', 0.3, 0.7)
        p['main_threshold'] = 0.5 # trial.suggest_float('main_threshold', 0.3, 0.7)

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
                         full_ds_is: pd.DataFrame,
                         full_ds_oos: pd.DataFrame,
                         model_main_train_data: pd.DataFrame,
                         model_meta_train_data: pd.DataFrame,
                         hp: Dict[str, Any]) -> tuple[float, str, tuple, tuple, dict]:
        """Ajusta los modelos finales y devuelve rutas a archivos temporales."""
        try:
            # 🔍 DEBUG: Supervisar parámetros CatBoost
            if self.debug:
                cat_main_params = {k: v for k, v in hp.items() if k.startswith('cat_main_')}
                cat_meta_params = {k: v for k, v in hp.items() if k.startswith('cat_meta_')}
                print(f"🔍 DEBUG fit_final_models - Parámetros CatBoost:")
                print(f"🔍   cat_main_*: {cat_main_params}")
                print(f"🔍   cat_meta_*: {cat_meta_params}")
            
            # Preparar datos del modelo main
            if model_main_train_data.empty:
                return None, None, None, None, None
            main_feature_cols = [col for col in model_main_train_data.columns if col != 'labels_main']
            if self.debug:
                print(f"🔍 DEBUG: Main model data shape: {model_main_train_data.shape}")
                print(f"🔍 DEBUG: Main feature columns: {main_feature_cols}")

            # Dividir train/test
            train_df_main, val_df_main = train_test_split(
                model_main_train_data,
                test_size=0.2,
                stratify=model_main_train_data['labels_main'],
                shuffle=True,
                random_state=None
            )
            if train_df_main is None or val_df_main is None or train_df_main.empty or val_df_main.empty:
                return None, None, None, None, None
            X_train_main = train_df_main[main_feature_cols].astype('float32')
            y_train_main = train_df_main['labels_main'].astype('int8')
            X_val_main = val_df_main[main_feature_cols].astype('float32')
            y_val_main = val_df_main['labels_main'].astype('int8')
            if self.debug:
                print(f"🔍 DEBUG: X_train_main shape: {X_train_main.shape}, y_train_main shape: {y_train_main.shape}")
                print(f"🔍 DEBUG: X_val_main shape: {X_val_main.shape}, y_val_main shape: {y_val_main.shape}")

            # Configurar parámetros CatBoost
            cat_main_params = dict(
                iterations=hp['cat_main_iterations'],
                depth=hp['cat_main_depth'],
                learning_rate=hp['cat_main_learning_rate'],
                l2_leaf_reg=hp['cat_main_l2_leaf_reg'],
                auto_class_weights='Balanced',
                eval_metric='Accuracy',
                store_all_simple_ctr=False,
                allow_writing_files=False,
                thread_count=self.n_jobs,
                task_type='CPU',
                verbose=False,
            )
            
            # 🔍 DEBUG: Mostrar configuración final de CatBoost
            if self.debug:
                print(f"🔍 DEBUG: CatBoost Main (Classification) configuración final:")
                for k, v in cat_main_params.items():
                    print(f"🔍   {k}: {v}")
                
            model_main = CatBoostClassifier(**cat_main_params)

            t_train_main_start = time.time()
            model_main.fit(X_train_main, y_train_main, 
                           eval_set=[(X_val_main, y_val_main)],
                           early_stopping_rounds=hp['cat_main_early_stopping'],
                           # callbacks=[CatBoostPruningCallback(trial=trial, metric='Logloss')],
                           use_best_model=True,
                           verbose=False
            )
            t_train_main_end = time.time()
            
            # Preparar datos del modelo meta
            if model_meta_train_data.empty:
                return None, None, None, None, None
            meta_feature_cols = [col for col in model_meta_train_data.columns if col != 'labels_meta']
            if self.debug:
                print(f"🔍 DEBUG: Meta model data shape: {model_meta_train_data.shape}")
                print(f"🔍 DEBUG: Meta feature columns: {meta_feature_cols}")

            # Dividir train/test
            train_df_meta, val_df_meta = train_test_split(
                model_meta_train_data,
                test_size=0.2,
                stratify=model_meta_train_data['labels_meta'],
                shuffle=True,
                random_state=None
            )
            if train_df_meta is None or val_df_meta is None or train_df_meta.empty or val_df_meta.empty:
                return None, None, None, None, None
            X_train_meta = train_df_meta[meta_feature_cols].astype('float32')
            y_train_meta = train_df_meta['labels_meta'].astype('int8')
            X_val_meta = val_df_meta[meta_feature_cols].astype('float32')
            y_val_meta = val_df_meta['labels_meta'].astype('int8')
            if self.debug:
                print(f"🔍 DEBUG: X_train_meta shape: {X_train_meta.shape}, y_train_meta shape: {y_train_meta.shape}")
                print(f"🔍 DEBUG: X_val_meta shape: {X_val_meta.shape}, y_val_meta shape: {y_val_meta.shape}")

            cat_meta_params = dict(
                iterations=hp['cat_meta_iterations'],
                depth=hp['cat_meta_depth'],
                learning_rate=hp['cat_meta_learning_rate'],
                l2_leaf_reg=hp['cat_meta_l2_leaf_reg'],
                auto_class_weights='Balanced',
                eval_metric='F1',
                store_all_simple_ctr=False,
                allow_writing_files=False,
                thread_count=self.n_jobs,
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
            if self.debug:
                print(f"🔍 DEBUG: Tiempo de entrenamiento modelo main: {t_train_main_end - t_train_main_start:.2f} segundos")
                print(f"🔍 DEBUG: Tiempo de entrenamiento modelo meta: {t_train_meta_end - t_train_meta_start:.2f} segundos")

            model_main_path, model_meta_path = export_models_to_ONNX(models=(model_main, model_meta))
            
            try:
                # Inicializar score con valor por defecto
                score = -1.0
                test_train_time_start = time.time()
                
                score, equity_curve, returns_series, pos_series = tester(
                    dataset=full_ds_oos,
                    model_main=model_main_path,
                    model_meta=model_meta_path,
                    model_main_cols=main_feature_cols,
                    model_meta_cols=meta_feature_cols,
                    direction=self.direction,
                    model_main_threshold=hp.get('main_threshold', 0.5),
                    model_meta_threshold=hp.get('meta_threshold', 0.5),
                    debug=self.debug,
                    plot=False
                )
            except Exception as tester_error:
                if self.debug:
                    print(f"🔍 DEBUG: Error en tester: {tester_error}")
                score, equity_curve, returns_series, pos_series = -1.0, None, None, None
            test_train_time_end = time.time()
            if self.debug:
                print(f"🔍 DEBUG: Tiempo de test in-sample: {test_train_time_end - test_train_time_start:.2f} segundos")
                print(f"🔍 DEBUG: Score in-sample: {score}")
            
            if score < 0.0 or not np.isfinite(score):
                if self.debug:
                    print(f"🔍 DEBUG: Score < 0.0 ({score:.6f}), retornando -1.0")
                return None, None, None, None, None
            
            # 1) Determinar best actual de Optuna (solo score óptimo de estudio)
            current_best = None
            try:
                if trial.study is not None and hasattr(trial.study, 'best_value'):
                    current_best = trial.study.best_value
            except Exception:
                current_best = None

            # 2) Decidir si ejecutar Monkey Test
            must_run_monkey = False
            if score > 0.0:
                if current_best is None:
                    must_run_monkey = True
                else:
                    must_run_monkey = score > float(current_best)

            monkey_pass = None
            monkey_p_value = None
            monkey_percentile = None
            if must_run_monkey and equity_curve is not None and len(equity_curve) > 1:
                try:
                    test_monkey_time_start = time.time()
                    # Series OOS
                    price_series = full_ds_oos['open'].to_numpy(dtype='float64') if 'open' in full_ds_oos.columns else None
                    rs = returns_series if returns_series is not None else None
                    ps = pos_series if pos_series is not None else None
                    if rs is None or ps is None or price_series is None:
                        raise RuntimeError("Series para Monkey Test no disponibles")

                    # Detectar segmentos OOS contiguos por timestamp
                    idx_vals = full_ds_oos.index.values.astype('datetime64[ns]').astype('int64')
                    segments = []
                    if idx_vals.size <= 1:
                        segments = [(0, idx_vals.size - 1)]
                    else:
                        diffs = np.diff(idx_vals)
                        # Paso esperado: moda, fallback mediana
                        try:
                            vals, counts = np.unique(diffs, return_counts=True)
                            expected = vals[np.argmax(counts)] if len(vals) > 0 else int(np.median(diffs))
                        except Exception:
                            expected = int(np.median(diffs)) if len(diffs) > 0 else 0
                        threshold = int(expected * 1.5) if expected > 0 else 0
                        breaks = np.where(diffs > threshold)[0]
                        start = 0
                        for bi in breaks:
                            end = bi
                            segments.append((start, end))
                            start = bi + 1
                        segments.append((start, idx_vals.size - 1))

                    # Repartir ventanas por segmento
                    seg_lens = [end - start + 1 for (start, end) in segments]
                    total_len = sum(seg_lens)
                    m = int(max(1, self.monkey_wfv_windows))
                    per_segment_windows = []
                    if total_len > 0:
                        base = [max(1, (l * m) // total_len) for l in seg_lens]
                        # Ajustar suma a m
                        diff_needed = m - sum(base)
                        if diff_needed != 0 and len(base) > 0:
                            order = np.argsort([-l for l in seg_lens]) if diff_needed > 0 else np.argsort([l for l in seg_lens])
                            k = 0
                            while diff_needed != 0 and k < 1000:
                                i = order[abs(k) % len(base)]
                                if diff_needed > 0:
                                    base[i] += 1
                                    diff_needed -= 1
                                else:
                                    if base[i] > 1:
                                        base[i] -= 1
                                        diff_needed += 1
                                k += 1
                        per_segment_windows = base
                    else:
                        per_segment_windows = [1 for _ in segments]

                    # Ejecutar por segmento → subventanas
                    global_pvals = []
                    global_percs = []
                    global_zs = []
                    segments_info = []

                    for (seg_idx, (s, e)) in enumerate(segments):
                        if e < s:
                            continue
                        wcount = max(1, per_segment_windows[seg_idx] if seg_idx < len(per_segment_windows) else 1)
                        # Crear subventanas de índices [s..e]
                        windows = np.array_split(np.arange(s, e + 1), wcount)
                        seg_pvals = []
                        seg_percs = []
                        seg_zs = []
                        windows_info = []
                        for w in windows:
                            if w.size < 5:
                                continue
                            res = run_monkey_test(
                                actual_returns=rs[w],
                                price_series=price_series[w],
                                pos_series=ps[w],
                                direction=self.direction,
                                n_simulations=self.monkey_n_simulations,
                                block_multiplier=self.monkey_block_multiplier,
                            )
                            p = float(res.get('p_value', 1.0))
                            q = float(res.get('percentile', 0.0))
                            mu0 = float(res.get('monkey_sharpes_mean', 0.0))
                            sd0 = float(res.get('monkey_sharpes_std', 0.0))
                            sr = float(res.get('actual_sharpe', 0.0))
                            z = (sr - mu0) / sd0 if sd0 > 0.0 else 0.0
                            seg_pvals.append(p)
                            seg_percs.append(q)
                            seg_zs.append(z)
                            global_pvals.append(p)
                            global_percs.append(q)
                            global_zs.append(z)
                            windows_info.append({'p_value': p, 'percentile': q, 'zscore': z, 'size': int(w.size)})

                        # Holm por segmento
                        if len(seg_pvals) == 0:
                            seg_holm = 1.0
                        else:
                            order = np.argsort(seg_pvals)
                            sorted_p = np.array(seg_pvals)[order]
                            m_eff = len(sorted_p)
                            holm_adj = [min(1.0, p * (m_eff - j + 1)) for j, p in enumerate(sorted_p, start=1)]
                            seg_holm = float(np.max(holm_adj))

                        segments_info.append({
                            'segment_index': int(seg_idx),
                            'start': str(full_ds_oos.index[s]) if s < len(full_ds_oos.index) else None,
                            'end': str(full_ds_oos.index[e]) if e < len(full_ds_oos.index) else None,
                            'holm_p_value': seg_holm,
                            'min_percentile': float(min(seg_percs)) if len(seg_percs) else 0.0,
                            'min_zscore': float(min(seg_zs)) if len(seg_zs) else 0.0,
                            'windows': windows_info,
                        })

                    # Holm global sobre todas las subventanas
                    if len(global_pvals) == 0:
                        raise RuntimeError("Monkey OOS sin ventanas evaluables")
                    order = np.argsort(global_pvals)
                    sorted_p = np.array(global_pvals)[order]
                    m_eff = len(sorted_p)
                    holm_adj = [min(1.0, p * (m_eff - j + 1)) for j, p in enumerate(sorted_p, start=1)]
                    holm_max = float(np.max(holm_adj))

                    monkey_p_value = holm_max
                    monkey_percentile = float(np.min(global_percs))
                    min_z = float(np.min(global_zs))

                    # Regla de aprobación: todos los segmentos deben ser significativos + tamaño de efecto y percentil globales
                    segments_pass = all(seg['holm_p_value'] < self.monkey_alpha for seg in segments_info) if len(segments_info) > 0 else False
                    cond_p = (monkey_p_value < self.monkey_alpha) and segments_pass
                    cond_q = monkey_percentile >= self.monkey_min_percentile
                    cond_z = min_z >= self.monkey_min_zscore
                    monkey_pass = bool(cond_p and cond_q and cond_z)

                    test_monkey_time_end = time.time()
                    if self.debug:
                        print(f"🔍 DEBUG: Tiempo de test Monkey (segmentado+Holm): {test_monkey_time_end - test_monkey_time_start:.2f} s")
                        print(f"🔍 DEBUG: Holm global: {monkey_p_value:.6f} (alpha={self.monkey_alpha})")
                        print(f"🔍 DEBUG: Percentil mínimo global: {monkey_percentile:.3f} (req={self.monkey_min_percentile})")
                        print(f"🔍 DEBUG: Z-score mínimo global: {min_z:.3f} (req={self.monkey_min_zscore})")
                        for seg in segments_info:
                            print(f"🔍 DEBUG: Segmento {seg['segment_index']} [{seg['start']} → {seg['end']}] holm_p={seg['holm_p_value']:.6f}, min_q={seg['min_percentile']:.3f}, min_z={seg['min_zscore']:.3f}")
                        print(f"🔍 DEBUG: Monkey pass: {monkey_pass}")
                except Exception as e_monkey:
                    if self.debug:
                        print(f"🔍 DEBUG: Error en Monkey Test: {e_monkey}")
                    monkey_pass = False
                    monkey_p_value = 1.0
                    monkey_percentile = 0.0

                # 3) Si falla, forzar score -1.0 para evitar autoengaño
                if not monkey_pass:
                    if self.debug:
                        print(f"🔍 DEBUG: Monkey Test NO superado (holm_p={monkey_p_value:.4f}, perc_min={monkey_percentile:.2f}) → score := -1.0")
                    score = -1.0

            monkey_info = {}
            if monkey_pass is not None:
                # Añadir detalles si existen (segmentos_info en el scope local)
                try:
                    details = {'segments': segments_info}
                except Exception:
                    details = {}
                monkey_info = {
                    'monkey_pass': monkey_pass,
                    'monkey_p_value': monkey_p_value,
                    'monkey_percentile': monkey_percentile,
                    'n_simulations': int(self.monkey_n_simulations),
                    'alpha': float(self.monkey_alpha),
                    'min_percentile': float(self.monkey_min_percentile),
                    'min_zscore': float(self.monkey_min_zscore),
                    'block_multiplier': float(self.monkey_block_multiplier),
                    **details,
                }

            # Desplazar columnas OHLCV una posición hacia atrás
            full_ds = pd.concat([full_ds_is, full_ds_oos]).sort_index()
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
                    main_counts = labels_main.value_counts(dropna=False).to_dict()
                    print(f"🔍      labels_main value_counts: {main_counts}")
                    print(f"🔍      labels_main únicos: {sorted(labels_main.unique())}")
                else:
                    print(f"🔍      labels_main no encontrada en el dataset")
                if 'labels_meta' in full_ds.columns:
                    labels_meta = full_ds['labels_meta']
                    print(f"🔍      labels_meta resumen: min={labels_meta.min():.6f}, max={labels_meta.max():.6f}, mean={labels_meta.mean():.6f}, std={labels_meta.std():.6f}")
                else:
                    print(f"🔍      labels_meta no encontrada en el dataset")
                print(f"🔍 DEBUG: Modelos guardados en {model_main_path} y {model_meta_path}")
            return score, full_ds_with_labels_path, (model_main_path, model_meta_path), (main_feature_cols, meta_feature_cols), monkey_info
        except Exception as e:
            print(f"Error en función de entrenamiento y test: {str(e)}")
            return None, None, None, None, None
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
            np.ndarray: reliability_scores con 1.0 para muestras confiables y precisas, 0.0 para el resto
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
                    print(f"🔍   y.value_counts(): {y.value_counts().to_dict()}")
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
                    print(f"🔍   y.value_counts(): {y.value_counts().to_dict()}")

            # Verificar que tenemos suficientes datos y clases balanceadas
            if len(X) < 100:  # Mínimo requerido para conformal prediction robusta
                if self.debug:
                    print(f"🔍 DEBUG apply_mapie_filter - Datos insuficientes para MAPIE: {len(X)}")
                if reliable_mask is not None:
                    return np.zeros(len(full_ds))
                else:
                    return np.zeros(len(X))
            
            # Verificar datos
            if len(y.unique()) < 2:
                if self.debug:
                    print(f"🔍 DEBUG apply_mapie_filter - Clases insuficientes para MAPIE: {y.unique()}")
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
            # Configurar CatBoost y MAPIE
            catboost_params = dict(
                iterations=hp['cat_main_iterations'],
                depth=hp['cat_main_depth'],
                learning_rate=hp['cat_main_learning_rate'],
                l2_leaf_reg=hp['cat_main_l2_leaf_reg'],
                auto_class_weights='Balanced',
                eval_metric='Accuracy',
                store_all_simple_ctr=False,
                allow_writing_files=False,
                thread_count=self.n_jobs,
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
            
            mapie.fit_conformalize(X, y)
            
            predicted, prediction_sets = mapie.predict_set(X)
            prediction_sets_2d = prediction_sets[:, :, 0]
            set_sizes = prediction_sets_2d.sum(axis=1)
            conformal_scores = (set_sizes == 1).astype(float)
            precision_scores = (predicted == y.to_numpy()).astype(float)
            combined_scores = ((conformal_scores == 1.0) & (precision_scores == 1.0)).astype(float)
            
            if self.debug:
                print(f"🔍   set_sizes.min(): {set_sizes.min()}, set_sizes.max(): {set_sizes.max()}")
                print(f"🔍   conformal_scores.sum(): {(set_sizes == 1).sum()}")
                print(f"🔍   conformal_scores.mean(): {conformal_scores.mean():.3f}")
                print(f"🔍   precision_scores.sum(): {precision_scores.sum()}")
                print(f"🔍   precision_scores.mean(): {precision_scores.mean():.3f}")
                print(f"🔍   reliability_scores.sum(): {combined_scores.sum()}")
                print(f"🔍   reliability_scores.mean(): {combined_scores.mean():.3f}")
                print(f"🔍 DEBUG apply_mapie_filter - Filtrado MAPIE completado")
            
            if reliable_mask is not None:
                full_reliability_scores = np.zeros(len(full_ds))
                full_reliability_scores[reliable_mask] = combined_scores
                
                if self.debug:
                    print(f"🔍   reliable_mask.sum(): {reliable_mask.sum()}")
                    print(f"🔍   reliability_scores.shape: {combined_scores.shape}")
                
                return full_reliability_scores
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
                print(f"🔍 DEBUG apply_causal_filter - Iniciando filtrado causal")
                print(f"🔍   full_ds.shape: {full_ds.shape}")
                print(f"🔍   reliable_mask is None: {reliable_mask is None}")
                if reliable_mask is not None:
                    print(f"🔍   reliable_mask.sum(): {reliable_mask.sum()}")
                    print(f"🔍   reliable_mask.mean(): {reliable_mask.mean():.3f}")
            
            def _bootstrap_oob_identification(X: pd.DataFrame, y: pd.Series, n_models: int = 25):
                oob_counts = pd.Series(0, index=X.index)
                error_counts_0 = pd.Series(0, index=X.index)
                error_counts_1 = pd.Series(0, index=X.index)
                
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

                    catboost_params = dict(
                        iterations=hp['cat_main_iterations'],
                        depth=hp['cat_main_depth'],
                        learning_rate=hp['cat_main_learning_rate'],
                        l2_leaf_reg=hp['cat_main_l2_leaf_reg'],
                        auto_class_weights='Balanced',
                        eval_metric='Accuracy',
                        store_all_simple_ctr=False,
                        allow_writing_files=False,
                        thread_count=self.n_jobs,
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
                    
                return error_counts_0, error_counts_1, oob_counts
            
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
                    print(f"🔍   y.value_counts(): {y.value_counts().to_dict()}")
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
                    print(f"🔍   y.value_counts(): {y.value_counts().to_dict()}")
            
            # Verificar que tenemos suficientes datos
            if len(X) < 100:  # Mínimo requerido para análisis causal robusto
                if self.debug:
                    print(f"🔍 DEBUG apply_causal_filter - Datos insuficientes para análisis causal: {len(X)}")
                if reliable_mask is not None:
                    return np.zeros(len(full_ds))
                else:
                    return np.zeros(len(X))
            
            # Verificar que tenemos al menos 2 clases para clasificación
            if len(y.unique()) != 2:
                if self.debug:
                    print(f"🔍 DEBUG apply_causal_filter - Clases insuficientes para análisis causal: {y.unique()}")
                if reliable_mask is not None:
                    return np.zeros(len(full_ds))
                else:
                    return np.zeros(len(X))
            
            # Aplicar detección causal
            err0, err1, oob = _bootstrap_oob_identification(X, y, n_models=hp['causal_meta_learners'])
            best_frac = _optimize_bad_samples_threshold(err0, err1, oob, fracs=[hp['causal_percentile']/100])
            
            to_mark_0 = (err0 / oob.replace(0, 1)).fillna(0)
            to_mark_1 = (err1 / oob.replace(0, 1)).fillna(0)
            thr0 = np.percentile(to_mark_0[to_mark_0 > 0], hp['causal_percentile']) * best_frac if len(to_mark_0[to_mark_0 > 0]) else 0
            thr1 = np.percentile(to_mark_1[to_mark_1 > 0], hp['causal_percentile']) * best_frac if len(to_mark_1[to_mark_1 > 0]) else 0
            marked0 = to_mark_0[to_mark_0 > thr0].index
            marked1 = to_mark_1[to_mark_1 > thr1].index
            all_bad = pd.Index(marked0).union(marked1)
            
            # Crear scores: 1.0 para muestras buenas, 0.0 para muestras malas
            causal_scores = np.ones(len(X))
            causal_scores[X.index.isin(all_bad)] = 0.0
            
            if self.debug:
                print(f"🔍   all_bad.shape: {len(all_bad)}")
                print(f"🔍   causal_scores.sum(): {causal_scores.sum()}")
                print(f"🔍   causal_scores.mean(): {causal_scores.mean():.3f}")
                print(f"🔍 DEBUG apply_causal_filter - Filtrado causal completado")
            
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
                    print(f"🔍   df['labels_main'].value_counts(): {df['labels_main'].value_counts()}")
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

    def get_labeled_full_data(self, hp) -> tuple[pd.DataFrame, pd.DataFrame]:
        try:
            if hp is None:
                return None, None

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
                return None, None
                
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
            # end_ext = max(self.train_end, self.test_end)
            full_ds = self.base_df.loc[start_ext:].copy()
            if self.debug:
                print(f"🔍 DEBUG: full_ds.shape ANTES de recortar a rango de interés = {full_ds.shape}")
            
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

            # 4) Obtener features
            full_ds = get_features(data=full_ds, hp=hp, decimal_precision=self.decimal_precision)
            if self.debug:
                feature_cols = [c for c in full_ds.columns if 'feature' in c]
                print(f"🔍   Columnas de features generadas: {len(feature_cols)}")
                main_features = [c for c in feature_cols if '_main_feature' in c]
                meta_features = [c for c in feature_cols if '_meta_feature' in c]
                print(f"🔍   Main features: {len(main_features)}, Meta features: {len(meta_features)}")

            # 5) Aplicar labeling
            if self.debug:
                print(f"🔍 DEBUG: Llamando apply_labeling con label_method='{self.label_method}'")
                label_params = {k: v for k, v in hp.items() if k.startswith('label_')}
                print(f"🔍   Parámetros label_* disponibles: {list(label_params.keys())}")
            full_ds = self.apply_labeling(full_ds, hp)
            if self.debug:
                print(f"🔍 DEBUG: full_ds.shape DESPUES de apply_labeling = {full_ds.shape}")

            # ──────────────────────────────────────────────────────────────
            # 6) Comprobaciones de calidad de features
            full_ds = full_ds.loc[min(self.train_start, self.test_start):]
            full_ds = full_ds.loc[full_ds.first_valid_index():full_ds.last_valid_index()]
            feature_cols = full_ds.columns[full_ds.columns.str.contains('feature')]
            if feature_cols.empty:
                return None, None
            problematic = self.check_constant_features(full_ds, list(feature_cols))
            if problematic:
                if self.debug and not full_ds.empty:
                    print(f"🔍 DEBUG: Primera fila del dataset con características:")
                    print(f"🔍   Índice: {full_ds.index[0]}")
                    print(f"🔍   Muestra: {full_ds.iloc[0].to_dict()}")
                    print(f"🔍 DEBUG: Última fila del dataset con características:")
                    print(f"🔍   Índice: {full_ds.index[-1]}")
                    print(f"🔍   Muestra: {full_ds.iloc[-1].to_dict()}")
                if self.debug:
                    print(f"🔍 DEBUG: Columnas problemáticas eliminadas: {len(problematic)}")
                full_ds = full_ds.drop(columns=problematic)
                feature_cols = [c for c in feature_cols if c not in problematic]
                if not feature_cols:
                    return None, None

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
                return None, None

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

            # 7) Devolver IS y OOS
            # Recortar a rangos: IS y OOS
            is_mask = (full_ds.index >= self.train_start) & (full_ds.index <= self.train_end)
            oos_mask = (~is_mask) & (full_ds.index >= self.test_start)
            full_ds_is = full_ds.loc[is_mask]
            full_ds_oos = full_ds.loc[oos_mask]
            if self.debug:
                print(f"🔍 DEBUG: full_ds.shape DESPUES de recortar a rango de interés = {full_ds.shape}")
                print(f"🔍 DEBUG: full_ds_is.shape = {full_ds_is.shape}, rango: {full_ds_is.index.min() if len(full_ds_is)>0 else None} → {full_ds_is.index.max() if len(full_ds_is)>0 else None}")
                print(f"🔍 DEBUG: full_ds_oos.shape = {full_ds_oos.shape}, rango: {full_ds_oos.index.min() if len(full_ds_oos)>0 else None} → {full_ds_oos.index.max() if len(full_ds_oos)>0 else None}")

            if full_ds_is.empty or full_ds_oos.empty:
                return None, None
            
            # 8) Devolver IS y OOS
            return full_ds_is, full_ds_oos

        except Exception as e:
            print(f"🔍 DEBUG: ERROR en get_labeled_full_data: {str(e)}")
            return None, None

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
            try:
                print(f"🔍 DEBUG: train_data rango: {train_data.index.min()} → {train_data.index.max()}")
                print(f"🔍 DEBUG: test_data  rango: {test_data.index.min()} → {test_data.index.max()}")
            except Exception:
                pass
        
        return train_data, test_data

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
    
    def _compute_main_oof_predictions_cls(
        self,
        X_main: pd.DataFrame,
        y_main: pd.Series,
        hp: Dict[str, Any],
    ) -> pd.Series:
        """Devuelve probabilidades OOF del modelo main (classification) usando TimeSeriesSplit."""
        if X_main is None or y_main is None or X_main.empty or y_main.empty:
            return pd.Series(dtype='float32')

        Xm = X_main.astype('float32')
        ym = y_main.astype('int8')
        default_splits = 5
        n_splits = int(hp.get('oof_n_splits', default_splits))
        n_splits = max(2, min(n_splits, max(2, len(Xm) - 1)))
        tscv = TimeSeriesSplit(n_splits=n_splits)

        cat_params = dict(
            iterations=hp['cat_main_iterations'],
            depth=hp['cat_main_depth'],
            learning_rate=hp['cat_main_learning_rate'],
            l2_leaf_reg=hp['cat_main_l2_leaf_reg'],
            auto_class_weights='Balanced',
            eval_metric='Accuracy',
            store_all_simple_ctr=False,
            allow_writing_files=False,
            thread_count=self.n_jobs,
            task_type='CPU',
            verbose=False,
        )

        oof = pd.Series(np.nan, index=ym.index, dtype='float32')
        for tr_idx, va_idx in tscv.split(Xm):
            X_tr, X_va = Xm.iloc[tr_idx], Xm.iloc[va_idx]
            y_tr = ym.iloc[tr_idx]
            model = CatBoostClassifier(**cat_params)
            model.fit(
                X_tr,
                y_tr,
                eval_set=[(X_va, ym.iloc[va_idx])],
                early_stopping_rounds=hp.get('cat_main_early_stopping', 80),
                use_best_model=True,
                verbose=False,
            )
            # Probabilidad de la clase positiva (1)
            oof.iloc[va_idx] = model.predict_proba(X_va)[:, 1].astype('float32')

        if oof.isna().any():
            oof = oof.fillna(oof.median())
        return oof
    
    def create_oof_meta_mask(
            self, 
            model_main_train_data: pd.DataFrame, 
            model_meta_train_data: pd.DataFrame, 
            hp: Dict[str, Any]
            ) -> pd.Series:
        """
        Crea labels meta a partir de predicciones OOF del main.
        - Regression: usa residuales |pred - y| mas magnitud
        - Classification: usa error de probabilidad |p - y| y confianza max(p, 1-p)
        """
        main_feature_cols = [col for col in model_main_train_data.columns if col not in ['labels_main']]
    
        # 1) Probabilidades OOF del main (clase 1)
        main_oof_proba = self._compute_main_oof_predictions_cls(
            X_main=model_main_train_data[main_feature_cols],
            y_main=model_main_train_data['labels_main'],
            hp=hp,
        )
        # 2) Definir umbral operativo como confianza mínima
        # Confianza = max(p, 1-p)
        confidence = main_oof_proba.copy()
        confidence = pd.Series(np.maximum(confidence.values, 1.0 - confidence.values), index=confidence.index)
        
        # 3) Confiabilidad por residuo de probabilidad
        y_true = model_main_train_data['labels_main'].astype('float32')
        prob_resid = (main_oof_proba - y_true).abs()
        tau = float(np.nanpercentile(prob_resid.values, hp['oof_resid_percentile'])) if len(prob_resid) > 0 else float(prob_resid.median() if len(prob_resid) else 0.0)
        reliability_mask = (prob_resid <= tau)
        magnitude_mask = (confidence >= hp['main_threshold'])
        labels_meta = (reliability_mask & magnitude_mask).astype('int8')

        # Construir máscara booleana alineada con meta_features_frame
        meta_X = model_meta_train_data.copy()
        mask = pd.Series(False, index=meta_X.index)
        common_idx = labels_meta.index.intersection(meta_X.index)
        if len(common_idx) > 0:
            mask.loc[common_idx] = labels_meta.loc[common_idx] == 1

        if self.debug:
            n_true = mask.sum()
            n_total = len(mask)
            print(f"🔍 DEBUG: create_meta_labels (cls) - {n_true}/{n_total} muestras con etiqueta meta=1.0 ({n_true/n_total:.2%})")
            print(f"🔍   main_oof_proba range: [{main_oof_proba.min():.3f}, {main_oof_proba.max():.3f}]")
            print(f"🔍   confidence range: [{confidence.min():.3f}, {confidence.max():.3f}]")
            print(f"🔍   prob_resid range: [{prob_resid.min():.3f}, {prob_resid.max():.3f}]")
            print(f"🔍   tau: {tau:.3f}")

        return mask