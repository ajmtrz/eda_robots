import gc
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
##from optuna.integration import CatBoostPruningCallback
from sklearn.model_selection import train_test_split, ParameterSampler
from catboost import CatBoostClassifier
from mapie.classification import CrossConformalClassifier
from modules.labeling_lib import (
    get_prices, get_features,
    get_labels_one_direction,
    get_labels_trend, get_labels_trend_with_profit,
    get_labels_trend_with_profit_multi, get_labels_clusters,
    get_labels_multi_window, get_labels_validated_levels,
    get_labels_filter_ZZ, get_labels_mean_reversion,
    get_labels_mean_reversion_multi, get_labels_mean_reversion_v,
    get_labels_filter, get_labels_multiple_filters,
    get_labels_filter_bidirectional, get_labels_filter_one_direction,
    get_labels_trend_one_direction, get_labels_filter_flat,
    sliding_window_clustering, clustering_simple,
    markov_regime_switching_simple, markov_regime_switching_advanced,
    lgmm_clustering, wkmeans_clustering
)
from modules.tester_lib import (
    tester,
    walk_forward_robust_score,
)
from modules.export_lib import export_models_to_ONNX, export_to_mql5

class StrategySearcher:
    LABEL_FUNCS = {
        "random": get_labels_one_direction,
        "trend": get_labels_trend,
        "trend_profit": get_labels_trend_with_profit,
        "trend_multi": get_labels_trend_with_profit_multi,
        "clusters": get_labels_clusters,
        "multi_window": get_labels_multi_window,
        "validated_levels": get_labels_validated_levels,
        "zigzag": get_labels_filter_ZZ,
        "mean_rev": get_labels_mean_reversion,
        "mean_rev_multi": get_labels_mean_reversion_multi,
        "mean_rev_vol": get_labels_mean_reversion_v,
        "filter": get_labels_filter,
        "multi_filter": get_labels_multiple_filters,
        "filter_bidirectional": get_labels_filter_bidirectional,
        "filter_one": get_labels_filter_one_direction,
        "trend_one": get_labels_trend_one_direction,
        "filter_flat": get_labels_filter_flat,
    }
    # Allowed smoothing methods for label functions that support a 'method' kwarg
    ALLOWED_METHODS = {
        "trend_profit": {"savgol", "spline", "sma", "ema"},
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
        search_subtype: str = 'simple',
        label_method: str = 'atr',
        tag: str = "",
        debug: bool = False,
        wkmeans_params: dict | None = None,
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
        self.wkmeans_params = wkmeans_params or {}
        self._wkmeans_clusters = None

        # Configuración de logging para optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    # =========================================================================
    # Método principal
    # =========================================================================

    def run_search(self) -> None:
        search_funcs = {
            'clusters': self.search_clusters,
            'markov': self.search_markov,
            'lgmm': self.search_lgmm,
            'mapie': self.search_mapie,
            'causal': self.search_causal,
            'wkmeans' : self.search_wkmeans,
        }
        
        if self.search_type not in search_funcs:
            raise ValueError(f"Tipo de búsqueda no válido: {self.search_type}")

        search_func = search_funcs[self.search_type]

        for i in range(self.n_models):
            if self.search_type == "wkmeans":
                param_dist = {
                    "n_clusters": list(range(2, 13)),
                    "window_size": list(range(20, 366)),
                    "step": list(range(1, 16)),
                    "max_iter": list(range(50, 301, 25)),
                }
                if self.search_subtype == "mmd":
                    param_dist["bandwidth"] = list(np.logspace(-2, np.log10(5), 20))
                if self.search_subtype == "sliced_w":
                    param_dist["n_proj"] = list(range(5, 401, 5))
                wk = next(ParameterSampler(param_dist, n_iter=1, random_state=i))
                ds_full = self.base_df.loc[
                    min(self.train_start, self.test_start):
                    max(self.train_end, self.test_end)
                ]
                self._wkmeans_clusters = wkmeans_clustering(
                    ds_full,
                    n_clusters=wk.get("n_clusters", 4),
                    window=wk.get("window_size", 60),
                    metric=self.search_subtype,
                    step=wk.get("step", 1),
                    bandwidth=wk.get("bandwidth") if self.search_subtype == "mmd" else None,
                    n_proj=wk.get("n_proj") if self.search_subtype == "sliced_w" else None,
                    max_iter=wk.get("max_iter", 100),
                )[["labels_meta"]]
                self.wkmeans_params = wk

            try:
                # Generar un seed único para este modelo
                model_seed = int(time.time() * 1000) + i

                # Inicializar estudio de Optuna con objetivos múltiples
                pruners = {
                    'hyperband': HyperbandPruner(max_resource='auto'),
                    'halving': SuccessiveHalvingPruner(min_resource='auto')
                }
                study = optuna.create_study(
                    directions=['maximize', 'maximize'],
                    storage=f"sqlite:///optuna_dbs/optuna_{self.tag}.db",
                    load_if_exists=True,
                    pruner=pruners[self.pruner_type],
                    sampler=optuna.samplers.TPESampler(
                        n_startup_trials=int(np.sqrt(self.n_trials)),
                        multivariate=True
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
                        # Obtener el mejor trial según criterio maximin
                        if study.best_trials:
                            best_trial = max(study.best_trials, 
                                          key=lambda t: min(t.values[0], t.values[1]))
                            
                            # Si este trial es el mejor, guardar sus modelos
                            if trial.number == best_trial.number:
                                if trial.user_attrs.get('model_paths') is not None:
                                    # Eliminar modelos anteriores
                                    if study.user_attrs.get("best_model_paths"):
                                        for p in study.user_attrs["best_model_paths"]:
                                            if p and os.path.exists(p):
                                                os.remove(p)
                                    # Guardar nuevas rutas de modelos
                                    study.set_user_attr("best_model_paths", trial.user_attrs['model_paths'])
                                    study.set_user_attr("best_scores", trial.user_attrs['scores'])
                                    study.set_user_attr("best_periods_main", trial.user_attrs['periods_main'])
                                    study.set_user_attr("best_stats_main", trial.user_attrs['stats_main'])
                                    study.set_user_attr("best_model_cols", trial.user_attrs['model_cols'])
                                    # Cambia acceso directo por .get para evitar error si no existe
                                    study.set_user_attr("best_periods_meta", trial.user_attrs.get('periods_meta'))
                                    study.set_user_attr("best_stats_meta", trial.user_attrs.get('stats_meta'))
                            # Liberar memoria eliminando datos pesados del trial
                            if 'model_paths' in trial.user_attrs:
                                if trial.user_attrs['model_paths'] != study.user_attrs.get('best_model_paths'):
                                    for p in trial.user_attrs['model_paths']:
                                        if p and os.path.exists(p):
                                            os.remove(p)
                                trial.set_user_attr("model_paths", None)

                        # Log
                        if study.best_trials:
                            best_trial = max(study.best_trials, key=lambda t: min(*t.values))
                            best_str = f"ins={best_trial.values[0]:.6f} oos={best_trial.values[1]:.6f}"
                        else:
                            best_str = "ins=--- oos=---"
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

                # Verificar y exportar el mejor modelo
                best_model_paths = study.user_attrs.get("best_model_paths", None)
                if best_model_paths is None or not best_model_paths:
                    print(f"⚠️ ERROR: best_model_paths VACÍO")
                    continue
                
                export_params = {
                    "symbol": self.symbol,
                    "timeframe": self.timeframe,
                    "direction": self.direction,
                    "label_method": self.label_method,
                    "models_export_path": self.models_export_path,
                    "include_export_path": self.include_export_path,
                    "search_type": self.search_type,
                    "search_subtype": self.search_subtype,
                    "best_model_seed": model_seed,
                    "best_scores": study.user_attrs["best_scores"],
                    "best_model_paths": study.user_attrs["best_model_paths"],
                    "best_model_cols": study.user_attrs["best_model_cols"],
                    "best_periods_main": study.user_attrs["best_periods_main"],
                    "best_periods_meta": study.user_attrs["best_periods_meta"],
                    "best_stats_main": study.user_attrs["best_stats_main"],
                    "best_stats_meta": study.user_attrs["best_stats_meta"],
                }
                
                export_to_mql5(**export_params)

                # Eliminar archivos temporales del mejor modelo
                for p in study.user_attrs.get("best_model_paths", []):
                    if p and os.path.exists(p):
                        os.remove(p)
                
            except Exception as e:
                print(f"\nError procesando modelo {i}:")
                print(f"Error: {str(e)}")
                print("Traceback:")
                print(traceback.format_exc())
                continue
            finally:
                # Liberar memoria
                gc.collect()

    # =========================================================================
    # Métodos de búsqueda específicos
    # =========================================================================

    def search_markov(self, trial: optuna.Trial) -> tuple[float, float]:
        """Implementa la búsqueda de estrategias usando modelos markovianos."""
        try:
            # Obtener todos los parámetros de una vez
            hp = self.suggest_all_params(trial)

            # Obtener datos de entrenamiento y prueba
            ds_train, ds_test = self.get_train_test_data(hp)
            if ds_train is None or ds_test is None:
                return -1.0, -1.0
            
            # Markov
            if self.search_subtype == 'simple':
                ds_train = markov_regime_switching_simple(
                    ds_train,
                    model_type=hp['model_type'],
                    n_regimes=hp['n_regimes'],
                    n_iter=hp['n_iter'],
                    n_mix=hp['n_mix'] if hp['model_type'] == 'VARHMM' else 3
                )
            elif self.search_subtype == 'advanced':
                ds_train = markov_regime_switching_advanced(
                    ds_train,
                    model_type=hp['model_type'],
                    n_regimes=hp['n_regimes'],
                    n_iter=hp['n_iter'],
                    n_mix=hp['n_mix'] if hp['model_type'] == 'VARHMM' else 3
                )

            scores, model_paths, models_cols = self.evaluate_clusters(ds_train, ds_test, hp)
            if scores is None or model_paths is None or models_cols is None:
                return -1.0, -1.0

            trial.set_user_attr('scores', scores)
            trial.set_user_attr('model_paths', model_paths)
            trial.set_user_attr('model_cols', models_cols)

            return trial.user_attrs.get('scores', (-1.0, -1.0))
            
        except Exception as e:
            print(f"Error en search_markov: {str(e)}")
            return -1.0, -1.0

    def search_clusters(self, trial: optuna.Trial) -> tuple[float, float]:
        """Implementa la búsqueda de estrategias usando clustering."""
        try:
            # Obtener todos los parámetros de una vez
            hp = self.suggest_all_params(trial)

            # Obtener datos de entrenamiento y prueba
            ds_train, ds_test = self.get_train_test_data(hp)
            if ds_train is None or ds_test is None:
                return -1.0, -1.0
            
            # Clustering
            if self.search_subtype == 'simple':
                ds_train = clustering_simple(
                    ds_train,
                    min_cluster_size=hp['n_clusters']
                )
            elif self.search_subtype == 'advanced':
                ds_train = sliding_window_clustering(
                    ds_train,
                    n_clusters=hp['n_clusters'],
                    window_size=hp['window_size'],
                    step=hp.get('step', None),
                )
            scores, model_paths, models_cols = self.evaluate_clusters(ds_train, ds_test, hp)
            if scores is None or model_paths is None or models_cols is None:
                return -1.0, -1.0

            trial.set_user_attr('scores', scores)
            trial.set_user_attr('model_paths', model_paths)
            trial.set_user_attr('model_cols', models_cols)

            return trial.user_attrs.get('scores', (-1.0, -1.0))
        
        except Exception as e:
            print(f"Error en search_clusters: {str(e)}")
            return -1.0, -1.0

    def search_lgmm(self, trial: optuna.Trial) -> tuple[float, float]:
        """Búsqueda basada en GaussianMixture para etiquetar clusters."""
        try:
            hp = self.suggest_all_params(trial)

            ds_train, ds_test = self.get_train_test_data(hp)
            if ds_train is None or ds_test is None:
                return -1.0, -1.0

            ds_train = lgmm_clustering(
                ds_train,
                n_components=hp['n_components'],
                covariance_type=hp['covariance_type'],
                max_iter=hp['max_iter'],
            )

            scores, model_paths, models_cols = self.evaluate_clusters(ds_train, ds_test, hp)
            if scores is None or model_paths is None or models_cols is None:
                return -1.0, -1.0

            trial.set_user_attr('scores', scores)
            trial.set_user_attr('model_paths', model_paths)
            trial.set_user_attr('model_cols', models_cols)

            return trial.user_attrs.get('scores', (-1.0, -1.0))

        except Exception as e:
            print(f"Error en search_lgmm: {str(e)}")
            return -1.0, -1.0

    def search_mapie(self, trial) -> tuple[float, float]:
        """Implementa la búsqueda de estrategias usando conformal prediction (MAPIE) con CatBoost, usando el mismo conjunto de features para ambos modelos."""
        try:
            hp = self.suggest_all_params(trial)
            ds_train, ds_test = self.get_train_test_data(hp)
            if ds_train is None or ds_test is None:
                return -1.0, -1.0
            
            # Selección de features: todas las columnas *_feature
            feature_cols = [col for col in ds_train.columns if col.endswith('_feature')]
            X = ds_train[feature_cols]
            y = ds_train['labels_main'] if 'labels_main' in ds_train.columns else ds_train['labels']

            # CatBoost como estimador base para MAPIE
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
            base_estimator = CatBoostClassifier(**catboost_params)

            mapie = CrossConformalClassifier(
                estimator=base_estimator,
                confidence_level=hp.get('mapie_confidence_level', 0.9),
                cv=hp.get('mapie_cv', 5),
            )
            mapie.fit_conformalize(X, y)
            predicted, y_prediction_sets = mapie.predict_set(X)
            y_prediction_sets = np.squeeze(y_prediction_sets, axis=-1)
            set_sizes = np.sum(y_prediction_sets, axis=1)
            ds_train['conformal_labels'] = 0.0
            ds_train.loc[set_sizes == 1, 'conformal_labels'] = 1.0
            ds_train['meta_labels'] = 0.0
            ds_train.loc[predicted == y, 'meta_labels'] = 1.0

            # Ambos modelos usan el mismo conjunto de features
            # Main: solo donde meta_labels==1
            model_main_data = ds_train[ds_train['meta_labels'] == 1][feature_cols + ['labels_main']].copy()
            # Meta: todas las filas, target = conformal_labels
            model_meta_data = ds_train[feature_cols].copy()
            model_meta_data['labels_meta'] = ds_train['conformal_labels']

            # Llamar a fit_final_models
            scores, model_paths, models_cols = self.fit_final_models(
                model_main_data=model_main_data,
                model_meta_data=model_meta_data,
                ds_train=ds_train,
                ds_test=ds_test,
                hp=hp.copy()
            )
            if scores is None or model_paths is None or models_cols is None:
                return -1.0, -1.0

            trial.set_user_attr('scores', scores)
            trial.set_user_attr('model_paths', model_paths)
            trial.set_user_attr('model_cols', models_cols)

            return trial.user_attrs.get('scores', (-1.0, -1.0))
        except Exception as e:
            print(f"Error en search_mapie: {str(e)}")
            return -1.0, -1.0

    def search_causal(self, trial: optuna.Trial) -> tuple[float, float]:
        """Búsqueda basada en detección causal de muestras malas."""
        try:
            hp = self.suggest_all_params(trial)

            ds_train, ds_test = self.get_train_test_data(hp)
            if ds_train is None or ds_test is None:
                return -1.0, -1.0

            feature_cols = [c for c in ds_train.columns if c.endswith('_feature')]
            X = ds_train[feature_cols]
            y = ds_train['labels_main']

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

            err0, err1, oob = _bootstrap_oob_identification(X, y, n_models=hp.get('n_meta_learners', 5))
            best_frac = _optimize_bad_samples_threshold(err0, err1, oob)
            to_mark_0 = (err0 / oob.replace(0, 1)).fillna(0)
            to_mark_1 = (err1 / oob.replace(0, 1)).fillna(0)
            thr0 = np.percentile(to_mark_0[to_mark_0 > 0], 75) * best_frac if len(to_mark_0[to_mark_0 > 0]) else 0
            thr1 = np.percentile(to_mark_1[to_mark_1 > 0], 75) * best_frac if len(to_mark_1[to_mark_1 > 0]) else 0
            marked0 = to_mark_0[to_mark_0 > thr0].index
            marked1 = to_mark_1[to_mark_1 > thr1].index
            all_bad = pd.Index(marked0).union(marked1)
            ds_train['meta_labels'] = 1.0
            ds_train.loc[ds_train.index.isin(all_bad), 'meta_labels'] = 0.0

            model_main_data = ds_train[ds_train['meta_labels'] == 1.0][feature_cols + ['labels_main']].copy()
            model_meta_data = ds_train[feature_cols].copy()
            model_meta_data['labels_meta'] = ds_train['meta_labels']

            # Llamar a fit_final_models
            scores, model_paths, models_cols = self.fit_final_models(
                model_main_data=model_main_data,
                model_meta_data=model_meta_data,
                ds_train=ds_train,
                ds_test=ds_test,
                hp=hp.copy()
            )
            if scores is None or model_paths is None or models_cols is None:
                return -1.0, -1.0

            trial.set_user_attr('scores', scores)
            trial.set_user_attr('model_paths', model_paths)
            trial.set_user_attr('model_cols', models_cols)

            return trial.user_attrs.get('scores', (-1.0, -1.0))

        except Exception as e:
            print(f"Error en search_causal: {str(e)}")
            return -1.0, -1.0

    def search_wkmeans(self, trial: optuna.Trial) -> tuple[float, float]:
        """
        Implementa la búsqueda de estrategias utilizando WK-means / MMDK-means
        para detectar y etiquetar regímenes de mercado desde labeling_lib.wkmeans_clustering.
        Se apoya en evaluate_clusters exactamente igual que el resto de métodos.
        """
        try:
            # 1) Hiper-parámetros sugeridos de forma coherente con el estudio Optuna
            hp = self.suggest_all_params(trial)

            # 2) Cargar datos de entrenamiento / prueba
            ds_train, ds_test = self.get_train_test_data(hp)
            if ds_train is None or ds_test is None:
                return -1.0, -1.0

            # (b) Lanzar error si no hay clusters precomputados
            if self._wkmeans_clusters is None:
                raise RuntimeError("[wkmeans] Error: self._wkmeans_clusters es None. La clusterización debe realizarse una sola vez por modelo antes de los trials de Optuna.")

            # 4) Etiquetado de regímenes con WK-means calculados previamente
            labels = self._wkmeans_clusters.loc[ds_train.index]
            ds_train = ds_train.join(labels, how="left")

            # (a) Chequeo de NaNs tras el join
            if ds_train['labels_meta'].isna().any():
                n_nans = ds_train['labels_meta'].isna().sum()
                print(f"⚠️ WARNING: {n_nans} filas con labels_meta NaN tras el join. Se eliminarán del dataset de entrenamiento.")
                ds_train = ds_train[ds_train['labels_meta'].notna()].copy()
                if ds_train.empty:
                    print("⚠️ ERROR: El dataset de entrenamiento quedó vacío tras eliminar NaNs en labels_meta.")
                    return -1.0, -1.0

            # 5) Entrenar modelos y obtener scores (mismo helper que el resto)
            scores, model_paths, model_cols = self.evaluate_clusters(ds_train, ds_test, hp)
            if scores is None or model_paths is None or model_cols is None:
                return -1.0, -1.0

            # 6) Registrar resultados en el trial para Optuna
            trial.set_user_attr("scores", scores)
            trial.set_user_attr("model_paths", model_paths)
            trial.set_user_attr("model_cols", model_cols)

            return trial.user_attrs.get("scores", (-1.0, -1.0))

        except Exception as e:
            print(f"Error en search_wkmeans: {str(e)}")
            return -1.0, -1.0

    # =========================================================================
    # Métodos auxiliares
    # =========================================================================
    
    def evaluate_clusters(self, ds_train: pd.DataFrame, ds_test: pd.DataFrame, hp: Dict[str, Any]) -> tuple[float, float]:
        """Función helper para evaluar clusters y entrenar modelos."""
        try:
            best_scores = (-math.inf, -math.inf)
            best_model_paths = (None, None)
            best_models_cols = (None, None)
            
            cluster_sizes = ds_train['labels_meta'].value_counts()
            if getattr(self, "debug", False):
                print(f"🔍 DEBUG: Cluster sizes:\n{cluster_sizes}")

            # Verificar que hay clusters
            if cluster_sizes.empty:
                print("⚠️ ERROR: No hay clusters")
                return None, None, None

            # Filtrar el cluster -1 (inválido) si existe
            if -1 in cluster_sizes.index:
                cluster_sizes = cluster_sizes.drop(-1)
                if cluster_sizes.empty:
                    return None, None, None

            # Evaluar cada cluster
            for clust in cluster_sizes.index:
                
                # Main data

                # Filtrar datos
                model_main_data = ds_train.loc[ds_train["labels_meta"] == clust].copy()

                # Preparar features
                main_feature_cols = ds_train.columns[ds_train.columns.str.contains('_feature') & \
                                                       ~ds_train.columns.str.contains('_meta_feature')]
                model_main_data = model_main_data[main_feature_cols.tolist() + ['labels_main']]

                # Verificar
                if 'label_max' in hp and len(model_main_data) <= hp["label_max"]:
                    continue
                if model_main_data is None or model_main_data.empty:
                    continue
                if (model_main_data['labels_main'].value_counts() < 2).any():
                    continue

                # Meta data

                # Preparar features
                meta_feature_cols = ds_train.columns[ds_train.columns.str.contains('_meta_feature')]
                model_meta_data = ds_train.loc[:, meta_feature_cols].copy()
                model_meta_data['labels_meta'] = (ds_train['labels_meta'] == clust).astype('int8')

                # Verificar
                if (model_meta_data['labels_meta'].value_counts() < 2).any():
                    continue

                # ── Evaluación en ambos períodos ──────────────────────────────
                # Llamar a fit_final_models
                scores, model_paths, models_cols = self.fit_final_models(
                    model_main_data=model_main_data,
                    model_meta_data=model_meta_data,
                    ds_train=ds_train,
                    ds_test=ds_test,
                    hp=hp.copy()
                )
                if scores is None or model_paths is None or models_cols is None:
                    return None, None, None

                # ── Actualizar mejores modelos y scores usando maximin method ─────────────────────
                if min(scores) > min(best_scores):
                    # Remove previous best temporary files
                    if best_model_paths != (None, None):
                        for p in best_model_paths:
                            if p and os.path.exists(p):
                                os.remove(p)
                    best_scores = scores
                    best_model_paths = model_paths
                    best_models_cols = models_cols
                else:
                    # Remove non-best temporary files
                    for p in model_paths:
                        if p and os.path.exists(p):
                            os.remove(p)

            # Verificar que encontramos algún cluster válido
            if best_scores == (-math.inf, -math.inf) or best_model_paths == (None, None):
                return None, None, None

            return best_scores, best_model_paths, best_models_cols

        except Exception as e:
            print(f"⚠️ ERROR en evaluación de clusters: {str(e)}")
            return None, None, None
    
    def suggest_all_params(self, trial: 'optuna.Trial') -> dict:
        try:
            MAX_MAIN_PERIODS = 15
            MAX_META_PERIODS = 3
            MAX_MAIN_STATS = 5
            MAX_META_STATS = 3
            all_stats = [
                "std", "skew", "zscore", "range", "mad", "entropy",
                "slope", "momentum", "autocorr", "maxdd", "hurst", "corrskew",
                "sharpe", "fisher", "chande", "var", "effratio", "kurt",
                "jump_vol", "fractal", "volskew", "approxentropy",
                "mean", "median", "iqr", "cv",
            ]
            params = {
                'cat_main_iterations': trial.suggest_int('cat_main_iterations', 100, 1000),
                'cat_main_depth': trial.suggest_int('cat_main_depth', 3, 10),
                'cat_main_learning_rate': trial.suggest_float('cat_main_learning_rate', 0.01, 0.3, log=True),
                'cat_main_l2_leaf_reg': trial.suggest_float('cat_main_l2_leaf_reg', 0.1, 10.0, log=True),
                'cat_main_early_stopping': trial.suggest_int('cat_main_early_stopping', 20, 200),
                'cat_meta_iterations': trial.suggest_int('cat_meta_iterations', 100, 1000),
                'cat_meta_depth': trial.suggest_int('cat_meta_depth', 3, 10),
                'cat_meta_learning_rate': trial.suggest_float('cat_meta_learning_rate', 0.01, 0.3, log=True),
                'cat_meta_l2_leaf_reg': trial.suggest_float('cat_meta_l2_leaf_reg', 0.1, 10.0, log=True),
                'cat_meta_early_stopping': trial.suggest_int('cat_meta_early_stopping', 20, 200),
                'max_main_periods': trial.suggest_int('max_main_periods', 3, MAX_MAIN_PERIODS, log=True),
                'max_main_stats': trial.suggest_int('max_main_stats', 1, MAX_MAIN_STATS, log=True),
            }
            # ---------- Parámetros de etiquetado dinámicos ----------
            label_func = self.LABEL_FUNCS.get(self.label_method, get_labels_one_direction)
            label_params = inspect.signature(label_func).parameters

            # Si el método de etiquetado es 'random', permitir elegir el método determinista
            if 'method' in label_params:
                params['method'] = trial.suggest_categorical('random_method', ['first', 'last', 'mean', 'max', 'min'])

            if 'markup' in label_params:
                params['markup'] = trial.suggest_float('markup', 0.2, 2.0, log=True)

            if 'min_val' in label_params: 
                params['min_val'] = trial.suggest_int('min_val', 1, 5)

            if 'max_val' in label_params:
                params['max_val'] = trial.suggest_int('max_val', 6, 20)

            if 'atr_period' in label_params:
                params['atr_period'] = trial.suggest_int('atr_period', 5, 50, log=True)

            if 'polyorder' in label_params:
                params['polyorder'] = trial.suggest_int('polyorder', 2, 5)

            if 'rolling' in label_params:
                params['rolling'] = trial.suggest_int('rolling', 20, 400, log=True)
                # Ajuste para cumplir la condición rolling > polyorder
                if 'polyorder' in label_params and params['rolling'] <= params['polyorder']:
                    params['rolling'] = params['polyorder'] + 2 + (params['polyorder'] % 2)

            if 'threshold' in label_params:
                params['threshold'] = trial.suggest_float('threshold', 0.1, 1.0)

            if 'vol_window' in label_params:
                params['vol_window'] = trial.suggest_int('vol_window', 20, 100, log=True)

            if 'num_clusters' in label_params:
                params['num_clusters'] = trial.suggest_int('num_clusters', 5, 50)

            if 'window_size' in label_params:
                params['window_size'] = trial.suggest_int('window_size', 10, 100)

            if 'threshold_pct' in label_params:
                params['threshold_pct'] = trial.suggest_float('threshold_pct', 0.005, 0.05)

            if 'min_touches' in label_params:
                params['min_touches'] = trial.suggest_int('min_touches', 2, 5)

            if 'peak_prominence' in label_params:
                params['peak_prominence'] = trial.suggest_float('peak_prominence', 0.05, 0.5)

            if 'decay_factor' in label_params:
                params['decay_factor'] = trial.suggest_float('decay_factor', 0.8, 1.0)

            if 'shift' in label_params:
                params['shift'] = trial.suggest_int('shift', -5, 5)

            if 'volatility_window' in label_params:
                params['volatility_window'] = trial.suggest_int('volatility_window', 10, 50)

            if 'rolling1' in label_params:
                params['rolling1'] = trial.suggest_int('rolling1', 20, 400, log=True)

            if 'rolling2' in label_params:
                params['rolling2'] = trial.suggest_int('rolling2', 20, 400, log=True)

            if 'method' in label_params:
                params['method'] = trial.suggest_categorical('method', ['savgol', 'spline', 'sma', 'ema', 'mean'])

            if 'rolling_periods' in label_params:
                rps = [trial.suggest_int(f'rolling_period_{i}', 20, 400, log=True) for i in range(3)]
                rps = list(dict.fromkeys(rps))
                params['rolling_periods'] = tuple(rps)

            if 'window_sizes' in label_params:
                ws = [trial.suggest_int(f'window_size_{i}', 10, 100, step=10) for i in range(3)]
                ws = list(dict.fromkeys(ws))
                params['window_sizes'] = tuple(ws)

            if 'window' in label_params:
                max_possible = min(400, max(len(self.base_df) - 1, 20))
                params['window'] = trial.suggest_int('window', 20, max_possible)

            if 'windows' in label_params:
                wnds = [trial.suggest_float(f'window_{i}', 0.1, 1.0) for i in range(3)]
                wnds = list(dict.fromkeys(wnds))
                params['windows'] = tuple(wnds)

            if 'quantiles' in label_params:
                q_low = trial.suggest_float('q_low', 0.3, 0.49)
                q_high = trial.suggest_float('q_high', 0.51, 0.7)
                params['quantiles'] = [q_low, q_high]
            # ---------- PERÍODOS MAIN ----------
            periods_main = [
                trial.suggest_int(f'period_main_{i}', 3, 200, log=True)
                for i in range(MAX_MAIN_PERIODS)
            ]
            periods_main = list(dict.fromkeys(periods_main))
            if len(periods_main) > params['max_main_periods']:
                periods_main = random.sample(periods_main, params['max_main_periods'])
            params['periods_main'] = tuple(periods_main[:params['max_main_periods']])
            # ---------- STATS MAIN ----------
            stats_main = [trial.suggest_categorical(f'stat_main_{i}', all_stats)
                        for i in range(MAX_MAIN_STATS)]
            stats_main = list(dict.fromkeys(stats_main))
            params['stats_main'] = tuple(stats_main[:params['max_main_stats']])
            # ---------- Hiperparámetros meta solo si no es mapie o causal ----------
            if self.search_type in ['clusters', 'markov', 'lgmm', 'wkmeans']:
                params['max_meta_periods'] = trial.suggest_int('max_meta_periods', 1, MAX_META_PERIODS, log=True)
                params['max_meta_stats'] = trial.suggest_int('max_meta_stats', 1, MAX_META_STATS, log=True)
                periods_meta = [
                    trial.suggest_int(f'period_meta_{i}', 3, 7)
                    for i in range(MAX_META_PERIODS)
                ]
                periods_meta = list(dict.fromkeys(periods_meta))
                if len(periods_meta) > params['max_meta_periods']:
                    periods_meta = random.sample(periods_meta, params['max_meta_periods'])
                params['periods_meta'] = tuple(periods_meta[:params['max_meta_periods']])
                stats_meta = [trial.suggest_categorical(f'stat_meta_{i}', all_stats)
                            for i in range(MAX_META_STATS)]
                stats_meta = list(dict.fromkeys(stats_meta))
                params['stats_meta'] = tuple(stats_meta[:params['max_meta_stats']])
            # ---------- Otros hiperparámetros específicos ----------
            if self.search_type == 'markov':
                params.update({
                    'model_type': trial.suggest_categorical('model_type', ['GMMHMM', 'HMM', 'VARHMM']),
                    'n_regimes': trial.suggest_int('n_regimes', 2, 10, log=True),
                    'n_iter': trial.suggest_int('n_iter', 50, 200, step=10),
                    'n_mix': trial.suggest_int('n_mix', 1, 5, log=True)
                })
            elif self.search_type == 'clusters':
                params.update({
                    'n_clusters': trial.suggest_int('n_clusters', 5, 50, step=5)
                })
                if self.search_subtype == 'advanced':
                    params.update({
                        'window_size': trial.suggest_int('window_size', 100, 500, step=50)
                    })
            elif self.search_type == 'lgmm':
                params.update({
                    'n_components': trial.suggest_int('n_components', 2, 10),
                    'covariance_type': trial.suggest_categorical('covariance_type', ['full', 'diag']),
                    'max_iter': trial.suggest_int('max_iter', 50, 300, step=10),
                })
            elif self.search_type == 'mapie':
                params.update({
                    'mapie_confidence_level': trial.suggest_float('mapie_confidence_level', 0.7, 0.99),
                    'mapie_cv': trial.suggest_int('mapie_cv', 3, 10),
                })
            elif self.search_type == 'causal':
                params.update({
                    'n_meta_learners': trial.suggest_int('n_meta_learners', 5, 30),
                })
            # Actualizar trial.params con los valores procesados
            for key, value in params.items():
                trial.set_user_attr(key, value)

            return params
        except Exception as e:
            print(f"⚠️ ERROR en suggest_all_params: {str(e)}")
            return None

    def fit_final_models(self, model_main_data: pd.DataFrame,
                        model_meta_data: pd.DataFrame,
                        ds_train: pd.DataFrame,
                        ds_test: pd.DataFrame,
                        hp: Dict[str, Any]) -> tuple[tuple[float, float], tuple[str, str], tuple[str, str]]:
        """Ajusta los modelos finales y devuelve rutas a archivos temporales."""
        try:
            # ---------- 1) MODEL MAIN ----------

            # Remove neutral labels before training
            if 'labels_main' in model_main_data.columns:
                model_main_data = model_main_data[model_main_data['labels_main'].isin([0.0, 1.0])]
            if model_main_data.empty:
                return None, None, None

            # Get feature columns
            main_feature_cols = [col for col in model_main_data.columns if col != 'labels_main']
            if getattr(self, "debug", False):
                print(f"🔍 DEBUG: Main model data shape: {model_main_data.shape}")
                print(f"🔍 DEBUG: Main feature columns: {main_feature_cols}")
            X_main = model_main_data[main_feature_cols]
            y_main = model_main_data['labels_main'].astype('int8')
            
            # División de datos para el modelo principal según fechas
            X_train_main, X_val_main, y_train_main, y_val_main = train_test_split(
                X_main, y_main,
                test_size=0.2,
                shuffle=True
            )
            
            # ── descartar clusters problemáticos ────────────────────────────
            if len(y_train_main.value_counts()) < 2 or len(y_val_main.value_counts()) < 2:
                return None, None, None

            # ---------- 2) MODEL META ----------
            meta_feature_cols = [col for col in model_meta_data.columns if col != 'labels_meta']
            if getattr(self, "debug", False):
                print(f"🔍 DEBUG: Meta model data shape: {model_meta_data.shape}")
                print(f"🔍 DEBUG: Meta feature columns: {meta_feature_cols}")
            X_meta = model_meta_data[meta_feature_cols]
            y_meta = model_meta_data['labels_meta'].astype('int8')

            # División de datos para el modelo principal según fechas
            X_train_meta, X_val_meta, y_train_meta, y_val_meta = train_test_split(
                X_meta, y_meta,
                test_size=0.2,
                shuffle=True
            )
            
            # ── descartar clusters problemáticos ────────────────────────────
            if len(y_train_meta.value_counts()) < 2 or len(y_val_meta.value_counts()) < 2:
                return None, None, None

            # Main model
            cat_main_params = dict(
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
            model_main = CatBoostClassifier(**cat_main_params)
            t_train_main_start = time.time()
            model_main.fit(X_train_main, y_train_main, 
                           eval_set=[(X_val_main, y_val_main)],
                           early_stopping_rounds=hp['cat_main_early_stopping'],
                           use_best_model=True,
                           verbose=False
            )
            t_train_main_end = time.time()
            if getattr(self, "debug", False):
                print(f"🔍 DEBUG: Tiempo de entrenamiento modelo main: {t_train_main_end - t_train_main_start:.2f} segundos")

            # Meta-modelo
            cat_meta_params = dict(
                iterations=hp['cat_meta_iterations'],
                depth=hp['cat_meta_depth'],
                learning_rate=hp['cat_meta_learning_rate'],
                l2_leaf_reg=hp['cat_meta_l2_leaf_reg'],
                eval_metric='F1',
                store_all_simple_ctr=False,
                allow_writing_files=False,
                thread_count=-1,
                task_type='CPU',
                verbose=False,
            )
            model_meta = CatBoostClassifier(**cat_meta_params)
            t_train_meta_start = time.time()
            model_meta.fit(X_train_meta, y_train_meta, 
                           eval_set=[(X_val_meta, y_val_meta)], 
                           early_stopping_rounds=hp['cat_meta_early_stopping'],
                           use_best_model=True,
                           verbose=False
            )
            t_train_meta_end = time.time()
            if getattr(self, "debug", False):
                print(f"🔍 DEBUG: Tiempo de entrenamiento modelo meta: {t_train_meta_end - t_train_meta_start:.2f} segundos")

            # ── EVALUACIÓN ───────────────────────────────────────────────

            # Verificar orden exacto
            if not (ds_test[main_feature_cols].columns == main_feature_cols).all():
                print("⚠️ ERROR: El orden de las columnas main no coincide en test")
                return None, None, None
            if not (ds_test[meta_feature_cols].columns == meta_feature_cols).all():
                print("⚠️ ERROR: El orden de las columnas meta no coincide en test")
                return None, None, None
            
            # Extraer datos para evaluación
            ds_train_eval_main = ds_train[main_feature_cols].to_numpy()
            ds_train_eval_meta = ds_train[meta_feature_cols].to_numpy()
            ds_test = ds_test[["open", "high", "low", "close", "volume"]]
            close_train_eval = ds_train['close'].to_numpy()
            
            # Evaluación in-sample y out-of-sample
            model_main_path, model_meta_path = export_models_to_ONNX(models=(model_main, model_meta))

            test_train_time_start = time.time()
            score_ins = tester(
                ds_main=ds_train_eval_main,
                ds_meta=ds_train_eval_meta,
                close=close_train_eval,
                model_main=model_main_path,
                model_meta=model_meta_path,
                direction=self.direction,
                plot=False,
                prd='insample',
            )
            test_train_time_end = time.time()
            if getattr(self, "debug", False):
                print(f"🔍 DEBUG: Tiempo de test in-sample: {test_train_time_end - test_train_time_start:.2f} segundos")

            test_test_time_start = time.time()
            score_oos = walk_forward_robust_score(
                ds_test=ds_test,
                model_main=model_main_path,
                model_meta=model_meta_path,
                model_main_cols=main_feature_cols,
                model_meta_cols=meta_feature_cols,
                hp=hp,
                direction=self.direction,
                n_splits=3,
                agg='min',
                plot=False,
            )
            test_test_time_end = time.time()
            if getattr(self, "debug", False):
                print(f"🔍 DEBUG: Tiempo de test out-of-sample: {test_test_time_end - test_test_time_start:.2f} segundos")

            # Manejar valores inválidos
            if not np.isfinite(score_ins) or not np.isfinite(score_oos):
                score_ins = -1.0
                score_oos = -1.0

            if getattr(self, "debug", False):
                print(f"🔍 DEBUG: Modelos guardados en {model_main_path} y {model_meta_path}")

            return (score_ins, score_oos), (model_main_path, model_meta_path), (main_feature_cols, meta_feature_cols)

        except Exception as e:
            print(f"Error en función de entrenamiento y test: {str(e)}")
            return None, None, None
        
    def apply_labeling(self, dataset: pd.DataFrame, hp: dict) -> pd.DataFrame:
        """Apply the selected labeling function dynamically.

        Returns an empty DataFrame if labeling fails or results in no rows.
        """
        label_func = self.LABEL_FUNCS.get(self.label_method, get_labels_one_direction)
        params = inspect.signature(label_func).parameters
        kwargs = {}

        for name, param in params.items():
            if name == 'dataset':
                continue
            if name == 'direction':
                kwargs['direction'] = self.direction if self.direction != 'both' else 'both'
            elif name in hp:
                kwargs[name] = hp[name]
            elif param.default is not inspect.Parameter.empty:
                kwargs[name] = param.default

        # ── Validaciones simples ───────────────────────────────────────────
        try:
            if len(dataset) < 2:
                return pd.DataFrame()

            polyorder = kwargs.get('polyorder', 2)
            if len(dataset) <= polyorder:
                return pd.DataFrame()

            # Ajuste automático para savgol_filter y similares
            method = kwargs.get('method')
            allowed = self.ALLOWED_METHODS.get(self.label_method)
            # Detectar parámetros de ventana relevantes
            window_keys = [k for k in kwargs if any(x in k for x in ('rolling', 'window', 'window_size'))]
            if method == 'savgol' and window_keys:
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
                        print(f"⚠️ ERROR en apply_labeling: parámetro '{k}'={v} inválido para dataset de tamaño {len(dataset)}")
                        return pd.DataFrame()
                elif isinstance(v, list) and any(x in k for x in ('rolling', 'window', 'period')):
                    if any((val <= 0 or val >= len(dataset)) for val in v):
                        print(f"⚠️ ERROR en apply_labeling: lista '{k}' contiene valores inválidos para dataset de tamaño {len(dataset)}")
                        return pd.DataFrame()

            if 'min_val' in kwargs and 'max_val' in kwargs and kwargs['min_val'] > kwargs['max_val']:
                kwargs['min_val'] = kwargs['max_val']

            if 'max_val' in hp and len(dataset) <= hp['max_val']:
                return pd.DataFrame()

            method = kwargs.get('method')
            allowed = self.ALLOWED_METHODS.get(self.label_method)
            if method and allowed and method not in allowed:
                return pd.DataFrame()

            df = label_func(dataset, **kwargs)

            if df is None or df.empty:
                return pd.DataFrame()

            if 'labels_main' in df.columns:
                return df
            if 'labels' in df.columns:
                return df.rename(columns={'labels': 'labels_main'})
            return pd.DataFrame()
        except Exception as e:
            print(f"⚠️ ERROR en apply_labeling: {e}")
            return pd.DataFrame()

    def get_train_test_data(self, hp):
        try:
            if hp is None:
                return None, None

            if getattr(self, "debug", False):
                print(f"🔍 DEBUG: base_df.shape = {self.base_df.shape}")
                print(f"🔍 DEBUG: train_start = {self.train_start}, train_end = {self.train_end}")
                print(f"🔍 DEBUG: test_start = {self.test_start}, test_end = {self.test_end}")

            # ──────────────────────────────────────────────────────────────
            # 1) Calcular el colchón de barras necesario
            pad = max(hp.get('periods_main', ()) + hp.get('periods_meta', ()), default=0)
            pad = int(pad)
            if getattr(self, "debug", False):
                print(f"🔍 DEBUG: pad = {pad}")

            # ──────────────────────────────────────────────────────────────
            # 2) Paso típico de la serie (mediana -> inmune a huecos)
            idx = self.base_df.index.sort_values()
            if pad == 0 or len(idx) < 2:
                bar_delta = pd.Timedelta(0)
            else:
                bar_delta = idx.to_series().diff().dropna().median()
            if getattr(self, "debug", False):
                print(f"🔍 DEBUG: bar_delta = {bar_delta}")

            # ──────────────────────────────────────────────────────────────
            # 3) Rango extendido para calcular features "con contexto"
            start_ext = min(self.train_start, self.test_start) - pad * bar_delta
            if start_ext < idx[0]:                      # evita pedir antes de que existan datos
                start_ext = idx[0]

            end_ext = max(self.train_end, self.test_end)
            if getattr(self, "debug", False):
                print(f"🔍 DEBUG: start_ext = {start_ext}, end_ext = {end_ext}")

            # ──────────────────────────────────────────────────────────────
            # 4) Obtener features de todo el rango extendido
            hp_tuple = tuple(sorted(hp.items()))
            ds_slice = self.base_df.loc[start_ext:end_ext].copy()
            if getattr(self, "debug", False):
                print(f"🔍 DEBUG: ds_slice.shape = {ds_slice.shape}")
            
            full_ds = get_features(ds_slice, dict(hp_tuple))
            
            if getattr(self, "debug", False):
                print(f"🔍 DEBUG: full_ds.shape después de get_features = {full_ds.shape}")

            full_ds = self.apply_labeling(full_ds, hp)
            if getattr(self, "debug", False):
                print(f"🔍 DEBUG: full_ds.shape después de apply_labeling = {full_ds.shape}")

            # y recortar exactamente al rango que interesa
            full_ds = full_ds.loc[
                min(self.train_start, self.test_start):
                max(self.train_end,   self.test_end)
            ]
            if getattr(self, "debug", False):
                print(f"🔍 DEBUG: full_ds.shape después de recorte = {full_ds.shape}")

            if full_ds.empty:
                return None, None

            # ──────────────────────────────────────────────────────────────
            # 5) Comprobaciones de calidad de features
            feature_cols = full_ds.columns[full_ds.columns.str.contains('feature')]
            if feature_cols.empty:
                return None, None

            problematic = self.check_constant_features(full_ds, list(feature_cols))
            if problematic:
                if getattr(self, "debug", False):
                    print(f"🔍 DEBUG: Columnas problemáticas eliminadas: {len(problematic)}")
                full_ds.drop(columns=problematic, inplace=True)
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
                if col.endswith('_meta_feature'):
                    p_str, stat = col[:-13].split('_', 1)
                    p = int(p_str)

                    # periodo meta
                    if p not in seen_meta_periods:
                        meta_periods_ordered.append(p)
                        seen_meta_periods.add(p)

                    # estadístico meta
                    if stat not in seen_meta_stats:
                        meta_stats_ordered.append(stat)
                        seen_meta_stats.add(stat)

                elif col.endswith('_feature'):
                    p_str, stat = col[:-8].split('_', 1)
                    p = int(p_str)

                    # periodo main
                    if p not in seen_main_periods:
                        main_periods_ordered.append(p)
                        seen_main_periods.add(p)

                    # estadístico main
                    if stat not in seen_main_stats:
                        main_stats_ordered.append(stat)
                        seen_main_stats.add(stat)

            # -------- aplicar a hp ----------------------------------------
            hp['periods_main'] = tuple(main_periods_ordered)
            hp['stats_main']   = tuple(main_stats_ordered)
            hp['periods_meta'] = tuple(meta_periods_ordered)
            hp['stats_meta']   = tuple(meta_stats_ordered)

            if (hp.get('periods_main') and not hp['stats_main']) or (hp.get('stats_main') and not hp['periods_main']):
                return None, None
            if (hp.get('periods_meta') and not hp['stats_meta']) or (hp.get('stats_meta') and not hp['periods_meta']):
                return None, None

            # ──────────────────────────────────────────────────────────────
            # 6) Máscaras de train / test
            test_mask  = (full_ds.index >= self.test_start)  & (full_ds.index <= self.test_end)
            train_mask = (full_ds.index >= self.train_start) & (full_ds.index <= self.train_end)

            if not test_mask.any() or not train_mask.any():
                print("⚠️ ERROR: Períodos sin datos")
                return None, None

            # Evitar solapamiento
            if self.test_start <= self.train_end and self.test_end >= self.train_start:
                train_mask &= ~test_mask
                if getattr(self, "debug", False):
                    print(f"🔍 DEBUG: train_mask.sum() después de evitar solapamiento = {train_mask.sum()}")

            # ──────────────────────────────────────────────────────────────
            # 7) DataFrames finales, ordenados cronológicamente
            train_data = full_ds[train_mask].sort_index().copy()
            test_data  = full_ds[test_mask].sort_index().copy()

            if getattr(self, "debug", False):
                print(f"🔍 DEBUG: train_data.shape final = {train_data.shape}")
                print(f"🔍 DEBUG: test_data.shape final = {test_data.shape}")

            if len(train_data) < 100 or len(test_data) < 50:
                print("⚠️ ERROR: Datasets demasiado pequeños")
                return None, None

            return train_data, test_data

        except Exception as e:
            print(f"⚠️ ERROR en get_train_test_data: {str(e)}")
            return None, None

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
