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
from optuna.integration import CatBoostPruningCallback
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
from modules.tester_lib import tester, clear_onnx_cache
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
    # Allowed smoothing methods for label functions that support a 'filter' kwarg
    ALLOWED_FILTERS = {
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
            try:
                # Generar un seed único para este modelo
                model_seed = int(time.time() * 1000) + np.random.randint(10, 100)

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
                        n_startup_trials=int(np.sqrt(self.n_trials)),
                        multivariate=True, group=True,
                        warn_independent_sampling=True
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
                        if study.best_trial and study.best_trial.value > -1.0:
                            best_trial = study.best_trial
                            # Si este trial es el mejor, guardar sus modelos
                            if trial.number == best_trial.number:
                                if trial.user_attrs.get('model_paths') is not None:
                                    # Eliminar modelos anteriores
                                    if study.user_attrs.get("best_model_paths"):
                                        for p in study.user_attrs["best_model_paths"]:
                                            if p and os.path.exists(p):
                                                os.remove(p)
                                    # Guardar nuevas rutas de modelos
                                    study.set_user_attr("best_score", trial.user_attrs['score'])
                                    study.set_user_attr("best_model_paths", trial.user_attrs['model_paths'])
                                    study.set_user_attr("best_periods_main", trial.user_attrs['periods_main'])
                                    study.set_user_attr("best_stats_main", trial.user_attrs['stats_main'])
                                    study.set_user_attr("best_model_cols", trial.user_attrs['model_cols'])
                                    # Cambia acceso directo por .get para evitar error si no existe
                                    study.set_user_attr("best_periods_meta", trial.user_attrs.get('periods_meta'))
                                    study.set_user_attr("best_stats_meta", trial.user_attrs.get('stats_meta'))
                                    # Exportar modelo
                                    export_params = {
                                        "tag": self.tag,
                                        "direction": self.direction,
                                        "best_model_seed": model_seed,
                                        "models_export_path": self.models_export_path,
                                        "include_export_path": self.include_export_path,
                                        "best_score": study.user_attrs["best_score"],
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
                                    # Parar el algoritmo
                                    if self.debug:
                                        study.stop()

                            # Liberar memoria eliminando datos pesados del trial
                            if 'model_paths' in trial.user_attrs and trial.user_attrs['model_paths']:
                                for p in trial.user_attrs['model_paths']:
                                    if p and os.path.exists(p):
                                        os.remove(p)

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
                
            except Exception as e:
                print(f"\nError procesando modelo {i}:")
                print(f"Error: {str(e)}")
                print("Traceback:")
                print(traceback.format_exc())
                continue

    # =========================================================================
    # Métodos de búsqueda específicos
    # =========================================================================

    def search_markov(self, trial: optuna.Trial) -> float:
        """Implementa la búsqueda de estrategias usando modelos markovianos."""
        try:
            hp = self.suggest_all_params(trial)
            full_ds = self.get_labeled_full_data(hp)
            if full_ds is None:
                return -1.0
            if self.search_subtype == 'simple':
                full_ds = markov_regime_switching_simple(
                    full_ds,
                    model_type=hp['model_type'],
                    n_regimes=hp['n_regimes'],
                    n_iter=hp['n_iter'],
                    n_mix=hp['n_mix'] if hp['model_type'] == 'VARHMM' else 3
                )
            elif self.search_subtype == 'advanced':
                full_ds = markov_regime_switching_advanced(
                    full_ds,
                    model_type=hp['model_type'],
                    n_regimes=hp['n_regimes'],
                    n_iter=hp['n_iter'],
                    n_mix=hp['n_mix'] if hp['model_type'] == 'VARHMM' else 3
                )
            score, model_paths, models_cols = self.evaluate_clusters(trial, full_ds, hp)
            if score is None or model_paths is None or models_cols is None:
                return -1.0
            trial.set_user_attr('score', score)
            trial.set_user_attr('model_paths', model_paths)
            trial.set_user_attr('model_cols', models_cols)
            return trial.user_attrs.get('score', -1.0)
        except Exception as e:
            print(f"Error en search_markov: {str(e)}")
            return -1.0

    def search_clusters(self, trial: optuna.Trial) -> float:
        """Implementa la búsqueda de estrategias usando clustering."""
        try:
            hp = self.suggest_all_params(trial)
            full_ds = self.get_labeled_full_data(hp)
            if full_ds is None:
                return -1.0
            if self.search_subtype == 'simple':
                full_ds = clustering_simple(
                    full_ds,
                    min_cluster_size=hp['n_clusters']
                )
            elif self.search_subtype == 'advanced':
                full_ds = sliding_window_clustering(
                    full_ds,
                    n_clusters=hp['n_clusters'],
                    window_size=hp['window_size'],
                    step=hp.get('step', None),
                )
            score, model_paths, models_cols = self.evaluate_clusters(trial, full_ds, hp)
            if score is None or model_paths is None or models_cols is None:
                return -1.0
            trial.set_user_attr('score', score)
            trial.set_user_attr('model_paths', model_paths)
            trial.set_user_attr('model_cols', models_cols)
            return trial.user_attrs.get('score', -1.0)
        except Exception as e:
            print(f"Error en search_clusters: {str(e)}")
            return -1.0

    def search_lgmm(self, trial: optuna.Trial) -> float:
        """Búsqueda basada en GaussianMixture para etiquetar clusters."""
        try:
            hp = self.suggest_all_params(trial)
            full_ds = self.get_labeled_full_data(hp)
            if full_ds is None:
                return -1.0
            full_ds = lgmm_clustering(
                full_ds,
                n_components=hp['n_components'],
                covariance_type=hp['covariance_type'],
                max_iter=hp['max_iter'],
            )
            score, model_paths, models_cols = self.evaluate_clusters(trial, full_ds, hp)
            if score is None or model_paths is None or models_cols is None:
                return -1.0
            trial.set_user_attr('score', score)
            trial.set_user_attr('model_paths', model_paths)
            trial.set_user_attr('model_cols', models_cols)
            return trial.user_attrs.get('score', -1.0)
        except Exception as e:
            print(f"Error en search_lgmm: {str(e)}")
            return -1.0

    def search_mapie(self, trial) -> float:
        """Implementa la búsqueda de estrategias usando conformal prediction (MAPIE) con CatBoost, usando el mismo conjunto de features para ambos modelos."""
        try:
            hp = self.suggest_all_params(trial)
            full_ds = self.get_labeled_full_data(hp)
            if full_ds is None:
                return -1.0
            feature_cols = [col for col in full_ds.columns if col.endswith('_feature')]
            X = full_ds[feature_cols]
            y = full_ds['labels_main'] if 'labels_main' in full_ds.columns else full_ds['labels']
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
            full_ds['conformal_labels'] = 0.0
            full_ds.loc[set_sizes == 1, 'conformal_labels'] = 1.0
            full_ds['meta_labels'] = 0.0
            full_ds.loc[predicted == y, 'meta_labels'] = 1.0
            model_main_train_data = full_ds[full_ds['meta_labels'] == 1][feature_cols + ['labels_main']].copy()
            model_meta_train_data = full_ds[feature_cols].copy()
            model_meta_train_data['labels_meta'] = full_ds['conformal_labels']
            score, model_paths, models_cols = self.fit_final_models(
                trial=trial,
                full_ds=full_ds,
                model_main_train_data=model_main_train_data,
                model_meta_train_data=model_meta_train_data,
                hp=hp.copy()
            )
            if score is None or model_paths is None or models_cols is None:
                return -1.0
            trial.set_user_attr('score', score)
            trial.set_user_attr('model_paths', model_paths)
            trial.set_user_attr('model_cols', models_cols)
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
            feature_cols = [c for c in full_ds.columns if c.endswith('_feature')]
            X = full_ds[feature_cols]
            y = full_ds['labels_main']
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
            full_ds['meta_labels'] = 1.0
            full_ds.loc[full_ds.index.isin(all_bad), 'meta_labels'] = 0.0
            model_main_train_data = full_ds[full_ds['meta_labels'] == 1.0][feature_cols + ['labels_main']].copy()
            model_meta_train_data = full_ds[feature_cols].copy()
            model_meta_train_data['labels_meta'] = full_ds['meta_labels']
            score, model_paths, models_cols = self.fit_final_models(
                trial=trial,
                full_ds=full_ds,
                model_main_train_data=model_main_train_data,
                model_meta_train_data=model_meta_train_data,
                hp=hp.copy()
            )
            if score is None or model_paths is None or models_cols is None:
                return -1.0
            trial.set_user_attr('score', score)
            trial.set_user_attr('model_paths', model_paths)
            trial.set_user_attr('model_cols', models_cols)
            return trial.user_attrs.get('score', -1.0)
        except Exception as e:
            print(f"Error en search_causal: {str(e)}")
            return -1.0

    def search_wkmeans(self, trial: optuna.Trial) -> float:
        """
        Implementa la búsqueda de estrategias utilizando WK-means / MMDK-means
        para detectar y etiquetar regímenes de mercado desde labeling_lib.wkmeans_clustering.
        Se apoya en evaluate_clusters exactamente igual que el resto de métodos.
        """
        try:
            hp = self.suggest_all_params(trial)
            full_ds = self.get_labeled_full_data(hp)
            if full_ds is None:
                return -1.0
            full_ds = wkmeans_clustering(
                full_ds,
                n_clusters=hp["n_clusters"],
                window_size=hp["window_size"],
                metric=self.search_subtype,
                step=hp["step"],
                bandwidth=hp["bandwidth"] if self.search_subtype == "mmd" else None,
                n_proj=hp["n_proj"] if self.search_subtype == "sliced_w" else None,
                max_iter=hp["max_iter"],
            )
            score, model_paths, model_cols = self.evaluate_clusters(trial, full_ds, hp)
            if score is None or model_paths is None or model_cols is None:
                return -1.0
            trial.set_user_attr("score", score)
            trial.set_user_attr("model_paths", model_paths)
            trial.set_user_attr("model_cols", model_cols)
            return trial.user_attrs.get("score", -1.0)
        except Exception as e:
            print(f"Error en search_wkmeans: {str(e)}")
            return -1.0

    # =========================================================================
    # Métodos auxiliares
    # =========================================================================
    
    def evaluate_clusters(self, trial: optuna.trial, full_ds: pd.DataFrame, hp: Dict[str, Any]) -> tuple[float, tuple, tuple]:
        """Función helper para evaluar clusters y entrenar modelos."""
        try:
            best_score = -math.inf
            best_model_paths = (None, None)
            best_models_cols = (None, None)
            cluster_sizes = full_ds['labels_meta'].value_counts()
            if self.debug:
                print(f"🔍 DEBUG: Cluster sizes:\n{cluster_sizes}")
            if cluster_sizes.empty:
                print("⚠️ ERROR: No hay clusters")
                return None, None, None
            if -1 in cluster_sizes.index:
                cluster_sizes = cluster_sizes.drop(-1)
                if cluster_sizes.empty:
                    return None, None, None
            for clust in cluster_sizes.index:
                model_main_train_data = full_ds.loc[full_ds["labels_meta"] == clust].copy()
                main_feature_cols = full_ds.columns[full_ds.columns.str.contains('_feature') & \
                                                       ~full_ds.columns.str.contains('_meta_feature')]
                model_main_train_data = model_main_train_data[main_feature_cols.tolist() + ['labels_main']]
                if model_main_train_data is None or model_main_train_data.empty:
                    continue
                if 'label_max' in hp and len(model_main_train_data) <= hp["label_max"]:
                    continue
                if (model_main_train_data['labels_main'].value_counts() < 2).any():
                    continue
                meta_feature_cols = full_ds.columns[full_ds.columns.str.contains('_meta_feature')]
                model_meta_train_data = full_ds.loc[:, meta_feature_cols].copy()
                model_meta_train_data['labels_meta'] = (full_ds['labels_meta'] == clust).astype('int8')
                if (model_meta_train_data['labels_meta'].value_counts() < 2).any():
                    continue
                score, model_paths, models_cols = self.fit_final_models(
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
                    best_score = score
                    best_model_paths = model_paths
                    best_models_cols = models_cols
                else:
                    for p in model_paths:
                        if p and os.path.exists(p):
                            os.remove(p)
            if best_score == -math.inf or best_model_paths == (None, None):
                return None, None, None
            return best_score, best_model_paths, best_models_cols
        except Exception as e:
            print(f"⚠️ ERROR en evaluación de clusters: {str(e)}")
            return None, None, None
    
    def suggest_all_params(self, trial: 'optuna.Trial') -> dict:
        try:
            # ========== CONSTANTES OPTIMIZADAS ==========
            MAX_MAIN_PERIODS = 12  # Reducido para mejor exploración
            MAX_META_PERIODS = 3
            MAX_MAIN_STATS = 4     # Reducido para evitar sobreajuste
            MAX_META_STATS = 3
            
            # Estadísticas agrupadas por funcionalidad para mejor coherencia
            MOMENTUM_STATS = ["momentum", "slope", "hurst", "autocorr", "effratio"]
            VOLATILITY_STATS = ["std", "range", "mad", "var", "maxdd", "jump_vol", "volskew"]
            DISTRIBUTION_STATS = ["skew", "kurt", "entropy", "zscore", "corrskew", "fisher"]
            SIMPLE_STATS = ["mean", "median", "iqr", "cv", "sharpe", "chande"]
            COMPLEXITY_STATS = ["fractal", "approxentropy"]
            
            all_stats = MOMENTUM_STATS + VOLATILITY_STATS + DISTRIBUTION_STATS + SIMPLE_STATS + COMPLEXITY_STATS

            params = {}
            
            # ========== PARÁMETROS CATBOOST MAIN (AGRUPADOS) ==========
            # Usar rangos más coherentes y distribuciones logarítmicas donde corresponda
            catboost_complexity = trial.suggest_float('catboost_complexity', 0.1, 1.0)  # Factor de complejidad
            
            # Iterations correlacionadas con complejidad
            base_iterations = trial.suggest_int('base_iterations', 200, 800, step=50)
            params['cat_main_iterations'] = int(base_iterations * (0.8 + 0.4 * catboost_complexity))
            
            # Profundidad correlacionada con complejidad
            params['cat_main_depth'] = trial.suggest_int('cat_main_depth', 4, 8)
            
            # Learning rate inversamente correlacionado con iterations
            lr_factor = trial.suggest_float('lr_factor', 0.5, 2.0, log=True)
            base_lr = 0.1 / np.sqrt(base_iterations / 300)  # Inversamente proporcional
            params['cat_main_learning_rate'] = base_lr * lr_factor
            
            # Regularización correlacionada con complejidad
            params['cat_main_l2_leaf_reg'] = trial.suggest_float('cat_main_l2_leaf_reg', 1.0, 20.0, log=True)
            
            # Early stopping proporcional a iterations
            early_stopping_ratio = trial.suggest_float('early_stopping_ratio', 0.1, 0.4)
            params['cat_main_early_stopping'] = max(20, int(params['cat_main_iterations'] * early_stopping_ratio))

            # ========== PARÁMETROS CATBOOST META (COHERENTES CON MAIN) ==========
            # Factor de escalamiento para meta vs main
            meta_scale_factor = trial.suggest_float('meta_scale_factor', 0.6, 1.2)
            
            params['cat_meta_iterations'] = max(100, int(params['cat_main_iterations'] * meta_scale_factor))
            params['cat_meta_depth'] = max(3, min(10, params['cat_main_depth'] + trial.suggest_int('meta_depth_offset', -2, 2)))
            
            # Learning rate del meta correlacionado con el main
            meta_lr_factor = trial.suggest_float('meta_lr_factor', 0.8, 1.5)
            params['cat_meta_learning_rate'] = params['cat_main_learning_rate'] * meta_lr_factor
            
            # Regularización del meta
            meta_reg_factor = trial.suggest_float('meta_reg_factor', 0.5, 2.0, log=True)
            params['cat_meta_l2_leaf_reg'] = params['cat_main_l2_leaf_reg'] * meta_reg_factor
            
            params['cat_meta_early_stopping'] = max(20, int(params['cat_meta_iterations'] * early_stopping_ratio))

            # ========== PARÁMETROS DE FEATURES (OPTIMIZADOS) ==========
            # Usar distribución más coherente para períodos y stats
            feature_complexity = trial.suggest_float('feature_complexity', 0.3, 1.0)  # Factor de complejidad de features
            
            params['max_main_periods'] = trial.suggest_int('max_main_periods', 3, int(MAX_MAIN_PERIODS * feature_complexity) + 2)
            params['max_main_stats'] = trial.suggest_int('max_main_stats', 2, int(MAX_MAIN_STATS * feature_complexity) + 1)

            # ========== PARÁMETROS DE ETIQUETADO DINÁMICOS (MEJORADOS) ==========
            label_func = self.LABEL_FUNCS.get(self.label_method, get_labels_one_direction)
            label_params = inspect.signature(label_func).parameters

            # Parámetros básicos de etiquetado
            if 'method' in label_params:
                params['method'] = trial.suggest_categorical('method', ['first', 'last', 'mean', 'max', 'min', 'random'])

            if 'filter' in label_params:
                params['filter'] = trial.suggest_categorical('filter', ['savgol', 'spline', 'sma', 'ema', 'mean'])

            # Markup con distribución logarítmica más coherente
            if 'markup' in label_params:
                params['markup'] = trial.suggest_float('markup', 0.2, 2.0, log=True)

            # Parámetros de valor con mejor distribución
            if 'min_val' in label_params: 
                params['min_val'] = trial.suggest_int('min_val', 1, 4)

            if 'max_val' in label_params:
                params['max_val'] = trial.suggest_int('max_val', 
                                                     params.get('min_val', 1) + 2, 
                                                     max(params.get('min_val', 1) + 5, 15))

            # Parámetros de período con distribución logarítmica
            if 'atr_period' in label_params:
                params['atr_period'] = trial.suggest_int('atr_period', 10, 50, log=True)

            if 'polyorder' in label_params:
                params['polyorder'] = trial.suggest_int('polyorder', 2, 4)

            # Rolling con distribución logarítmica y dependencia de polyorder
            if 'rolling' in label_params:
                min_rolling = max(50, params.get('polyorder', 2) * 10)
                params['rolling'] = trial.suggest_int('rolling', min_rolling, 300, log=True)

            # Threshold con distribución uniforme más apropiada
            if 'threshold' in label_params:
                params['threshold'] = trial.suggest_float('threshold', 0.2, 0.8)

            # Parámetros de clustering con distribución logarítmica
            if 'l_n_clusters' in label_params:
                params['l_n_clusters'] = trial.suggest_int('l_n_clusters', 8, 40, log=True)

            if 'l_window_size' in label_params:
                params['l_window_size'] = trial.suggest_int('l_window_size', 20, 80, log=True)

            if 'vol_window' in label_params:
                params['vol_window'] = trial.suggest_int('vol_window', 20, 100, log=True)

            # Window sizes múltiples con mejor distribución
            if 'window_sizes' in label_params:
                base_window = trial.suggest_int('base_window_size', 20, 80, log=True)
                ws = [
                    base_window,
                    int(base_window * trial.suggest_float('window_ratio_2', 1.2, 2.0)),
                    int(base_window * trial.suggest_float('window_ratio_3', 2.1, 4.0))
                ]
                params['window_sizes'] = tuple(sorted(set(ws)))

            # Threshold percentage más coherente
            if 'threshold_pct' in label_params:
                params['threshold_pct'] = trial.suggest_float('threshold_pct', 0.01, 0.03, log=True)

            if 'min_touches' in label_params:
                params['min_touches'] = trial.suggest_int('min_touches', 2, 4)

            if 'peak_prominence' in label_params:
                params['peak_prominence'] = trial.suggest_float('peak_prominence', 0.1, 0.4)

            if 'decay_factor' in label_params:
                params['decay_factor'] = trial.suggest_float('decay_factor', 0.85, 0.98)

            if 'shift' in label_params:
                params['shift'] = trial.suggest_int('shift', -3, 3)

            # Rolling parameters correlacionados
            if 'rolling1' in label_params:
                base_rolling = trial.suggest_int('base_rolling', 50, 250, log=True)
                params['rolling1'] = base_rolling

            if 'rolling2' in label_params:
                rolling2_factor = trial.suggest_float('rolling2_factor', 1.2, 3.0)
                params['rolling2'] = int(params.get('rolling1', 100) * rolling2_factor)

            # Rolling periods múltiples con mejor correlación
            if 'rolling_periods' in label_params:
                base_rp = trial.suggest_int('base_rolling_period', 30, 150, log=True)
                rps = [
                    base_rp,
                    int(base_rp * trial.suggest_float('rp_factor_2', 1.5, 2.5)),
                    int(base_rp * trial.suggest_float('rp_factor_3', 3.0, 5.0))
                ]
                params['rolling_periods'] = tuple(sorted(set(rps)))

            # Quantiles con mejor distribución
            if 'quantiles' in label_params:
                q_spread = trial.suggest_float('quantile_spread', 0.15, 0.25)
                q_center = 0.5
                params['quantiles'] = [q_center - q_spread, q_center + q_spread]

            # ========== PERÍODOS MAIN (OPTIMIZADOS) ==========
            # Usar distribución más inteligente para períodos
            period_strategy = trial.suggest_categorical('period_strategy', ['geometric', 'arithmetic', 'mixed'])
            
            if period_strategy == 'geometric':
                # Distribución geométrica para mejor cobertura
                base_period = trial.suggest_int('base_period', 5, 15)
                growth_factor = trial.suggest_float('period_growth_factor', 1.3, 2.0)
                periods_main = [int(base_period * (growth_factor ** i)) for i in range(MAX_MAIN_PERIODS)]
                periods_main = [min(p, 200) for p in periods_main if p <= 200]
            elif period_strategy == 'arithmetic':
                # Distribución aritmética
                start_period = trial.suggest_int('start_period', 5, 20)
                step_period = trial.suggest_int('period_step', 10, 25)
                periods_main = [start_period + i * step_period for i in range(MAX_MAIN_PERIODS)]
                periods_main = [p for p in periods_main if p <= 200]
            else:  # mixed
                # Combinación de períodos cortos, medios y largos - usando nombres únicos para cada período
                short_period_1 = trial.suggest_int('short_period_1', 5, 20)
                short_period_2 = trial.suggest_int('short_period_2', 5, 20)
                short_period_3 = trial.suggest_int('short_period_3', 5, 20)
                medium_period_1 = trial.suggest_int('medium_period_1', 21, 60, log=True)
                medium_period_2 = trial.suggest_int('medium_period_2', 21, 60, log=True)
                medium_period_3 = trial.suggest_int('medium_period_3', 21, 60, log=True)
                long_period_1 = trial.suggest_int('long_period_1', 61, 200, log=True)
                long_period_2 = trial.suggest_int('long_period_2', 61, 200, log=True)
                long_period_3 = trial.suggest_int('long_period_3', 61, 200, log=True)
                
                short_periods = [short_period_1, short_period_2, short_period_3]
                medium_periods = [medium_period_1, medium_period_2, medium_period_3]
                long_periods = [long_period_1, long_period_2, long_period_3]
                periods_main = short_periods + medium_periods + long_periods
            
            # Eliminar duplicados y limitar
            periods_main = sorted(list(set(periods_main)))
            if len(periods_main) > params['max_main_periods']:
                # Selección más inteligente en lugar de random
                step = len(periods_main) // params['max_main_periods']
                periods_main = periods_main[::step][:params['max_main_periods']]
            params['periods_main'] = tuple(periods_main)

            # ========== STATS MAIN (OPTIMIZADOS POR GRUPOS) ==========
            # Selección más inteligente de estadísticas por grupos
            stats_selection_strategy = trial.suggest_categorical('stats_strategy', ['balanced', 'momentum_focused', 'volatility_focused', 'distribution_focused'])
            
            # Pre-selección de estadísticas usando distribuciones categóricas fijas
            momentum_stat_choice = trial.suggest_categorical('momentum_stat', MOMENTUM_STATS)
            volatility_stat_choice = trial.suggest_categorical('volatility_stat', VOLATILITY_STATS)
            distribution_stat_choice = trial.suggest_categorical('distribution_stat', DISTRIBUTION_STATS)
            simple_stat_choice = trial.suggest_categorical('simple_stat', SIMPLE_STATS)
            
            # Estadísticas adicionales con distribuciones fijas
            momentum_stat_2 = trial.suggest_categorical('momentum_stat_2', MOMENTUM_STATS)
            momentum_stat_3 = trial.suggest_categorical('momentum_stat_3', MOMENTUM_STATS)
            volatility_stat_2 = trial.suggest_categorical('volatility_stat_2', VOLATILITY_STATS)
            volatility_stat_3 = trial.suggest_categorical('volatility_stat_3', VOLATILITY_STATS)
            distribution_stat_2 = trial.suggest_categorical('distribution_stat_2', DISTRIBUTION_STATS)
            distribution_stat_3 = trial.suggest_categorical('distribution_stat_3', DISTRIBUTION_STATS)
            simple_stat_2 = trial.suggest_categorical('simple_stat_2', SIMPLE_STATS)
            
            if stats_selection_strategy == 'balanced':
                # Una estadística de cada grupo principal
                selected_stats = []
                if len(selected_stats) < params['max_main_stats']:
                    selected_stats.append(momentum_stat_choice)
                if len(selected_stats) < params['max_main_stats']:
                    selected_stats.append(volatility_stat_choice)
                if len(selected_stats) < params['max_main_stats']:
                    selected_stats.append(distribution_stat_choice)
                if len(selected_stats) < params['max_main_stats']:
                    selected_stats.append(simple_stat_choice)
            elif stats_selection_strategy == 'momentum_focused':
                # Foco en momentum + algunas de otros grupos
                selected_stats = [momentum_stat_choice, momentum_stat_2, momentum_stat_3][:min(3, params['max_main_stats'])]
                remaining = params['max_main_stats'] - len(selected_stats)
                if remaining > 0:
                    additional_stats = [volatility_stat_choice, distribution_stat_choice][:remaining]
                    selected_stats.extend(additional_stats)
            elif stats_selection_strategy == 'volatility_focused':
                # Foco en volatilidad
                selected_stats = [volatility_stat_choice, volatility_stat_2, volatility_stat_3][:min(3, params['max_main_stats'])]
                remaining = params['max_main_stats'] - len(selected_stats)
                if remaining > 0:
                    additional_stats = [momentum_stat_choice, simple_stat_choice][:remaining]
                    selected_stats.extend(additional_stats)
            else:  # distribution_focused
                selected_stats = [distribution_stat_choice, distribution_stat_2, distribution_stat_3][:min(3, params['max_main_stats'])]
                remaining = params['max_main_stats'] - len(selected_stats)
                if remaining > 0:
                    additional_stats = [momentum_stat_choice, volatility_stat_choice][:remaining]
                    selected_stats.extend(additional_stats)
            
            params['stats_main'] = tuple(list(dict.fromkeys(selected_stats))[:params['max_main_stats']])

            # ========== HIPERPARÁMETROS META (COHERENTES) ==========
            if self.search_type in ['clusters', 'markov', 'lgmm', 'wkmeans']:
                # Meta periods correlacionados con feature complexity
                params['max_meta_periods'] = trial.suggest_int('max_meta_periods', 1, MAX_META_PERIODS)
                params['max_meta_stats'] = trial.suggest_int('max_meta_stats', 1, MAX_META_STATS)
                
                # Períodos meta más cortos y coherentes
                meta_base_period = trial.suggest_int('meta_base_period', 4, 6)
                periods_meta = [meta_base_period + i for i in range(MAX_META_PERIODS)]
                periods_meta = list(dict.fromkeys(periods_meta))
                if len(periods_meta) > params['max_meta_periods']:
                    periods_meta = periods_meta[:params['max_meta_periods']]
                params['periods_meta'] = tuple(sorted(periods_meta))
                
                # Stats meta seleccionadas de forma coherente - distribuciones fijas
                META_STATS_OPTIONS = SIMPLE_STATS + MOMENTUM_STATS[:2]  # Lista fija para evitar dynamic value space
                meta_stat_1 = trial.suggest_categorical('meta_stat_1', META_STATS_OPTIONS)
                meta_stat_2 = trial.suggest_categorical('meta_stat_2', META_STATS_OPTIONS)
                meta_stat_3 = trial.suggest_categorical('meta_stat_3', META_STATS_OPTIONS)
                
                meta_stats = [meta_stat_1, meta_stat_2, meta_stat_3][:MAX_META_STATS]
                params['stats_meta'] = tuple(list(dict.fromkeys(meta_stats))[:params['max_meta_stats']])
            else:
                params['periods_meta'] = tuple()
                params['stats_meta'] = tuple()

            # ========== PARÁMETROS ESPECÍFICOS DEL ALGORITMO (OPTIMIZADOS) ==========
            if self.search_type == 'markov':
                # Parámetros markov con distribuciones mejoradas
                params.update({
                    'model_type': trial.suggest_categorical('model_type', ['GMMHMM', 'HMM', 'VARHMM']),
                    'n_regimes': trial.suggest_int('n_regimes', 3, 8, log=True),
                    'n_iter': trial.suggest_int('n_iter', 80, 150, step=10),
                    'n_mix': trial.suggest_int('n_mix', 1, 4)
                })
                
            elif self.search_type == 'clusters':
                cluster_complexity = trial.suggest_float('cluster_complexity', 0.5, 1.0)
                base_clusters = trial.suggest_int('base_clusters', 8, 30, log=True)
                params['n_clusters'] = int(base_clusters * cluster_complexity)
                
                if self.search_subtype == 'advanced':
                    # Window size correlacionado con número de clusters
                    cluster_window_factor = trial.suggest_float('cluster_window_factor', 5, 15)
                    params['window_size'] = int(params['n_clusters'] * cluster_window_factor)
                    
            elif self.search_type == 'lgmm':
                # Parámetros LGMM más coherentes
                params.update({
                    'n_components': trial.suggest_int('n_components', 3, 15, log=True),
                    'covariance_type': trial.suggest_categorical('covariance_type', ['full', 'diag']),
                    'max_iter': trial.suggest_int('max_iter', 80, 200, step=20),
                })
                
            elif self.search_type == "wkmeans":
                # Parámetros wkmeans optimizados
                wk_complexity = trial.suggest_float('wk_complexity', 0.6, 1.0)
                base_wk_clusters = trial.suggest_int('base_wk_clusters', 6, 20, log=True)
                
                params.update({
                    "n_clusters": int(base_wk_clusters * wk_complexity),
                    "window_size": trial.suggest_int("window_size", 40, 250, log=True),
                    "step": trial.suggest_int("step", 1, 15),
                    "max_iter": trial.suggest_int("max_iter", 150, 400, step=25),
                    "bandwidth": trial.suggest_float("bandwidth", 0.1, 5.0, log=True),
                    "n_proj": trial.suggest_int("n_proj", 50, 300, log=True),
                })
                
            elif self.search_type == 'mapie':
                # Parámetros MAPIE optimizados
                params.update({
                    'mapie_confidence_level': trial.suggest_float('mapie_confidence_level', 0.8, 0.95),
                    'mapie_cv': trial.suggest_int('mapie_cv', 4, 8),
                })
                
            elif self.search_type == 'causal':
                # Parámetros causales con mejor distribución
                params.update({
                    'n_meta_learners': trial.suggest_int('n_meta_learners', 10, 25),
                })

            # ========== GUARDAR ATRIBUTOS DEL TRIAL ==========
            for key, value in params.items():
                trial.set_user_attr(key, value)

            if self.debug:
                print("🔍 DEBUG: Parámetros optimizados sugeridos:")
                print(f"  🎯 Feature complexity: {feature_complexity:.3f}")
                print(f"  🎯 CatBoost complexity: {catboost_complexity:.3f}")
                print(f"  📊 Periods strategy: {period_strategy}")
                print(f"  📊 Stats strategy: {stats_selection_strategy}")
                for key, value in sorted(params.items()):
                    if isinstance(value, (tuple, list)) and len(value) > 0:
                        print(f"  {key}: {value}")
                    elif not isinstance(value, (tuple, list)):
                        print(f"  {key}: {value}")

            return params
            
        except Exception as e:
            print(f"⚠️ ERROR en suggest_all_params optimizado: {str(e)}")
            return None

    def fit_final_models(self, trial: optuna.trial,
                         full_ds: pd.DataFrame,
                         model_main_train_data: pd.DataFrame,
                         model_meta_train_data: pd.DataFrame,
                         hp: Dict[str, Any]) -> tuple[float, tuple, tuple]:
        """Ajusta los modelos finales y devuelve rutas a archivos temporales."""
        try:
            if 'labels_main' in model_main_train_data.columns:
                model_main_train_data = model_main_train_data[model_main_train_data['labels_main'].isin([0.0, 1.0])]
            if model_main_train_data.empty:
                return None, None, None
            main_feature_cols = [col for col in model_main_train_data.columns if col != 'labels_main']
            if self.debug:
                print(f"🔍 DEBUG: Main model data shape: {model_main_train_data.shape}")
                print(f"🔍 DEBUG: Main feature columns: {main_feature_cols}")
            model_main_train_data, model_main_eval_data = self.get_train_test_data(dataset=model_main_train_data)
            if model_main_train_data is None or model_main_eval_data is None:
                return None, None, None
            X_train_main = model_main_train_data[main_feature_cols]
            y_train_main = model_main_train_data['labels_main'].astype('int8')
            X_val_main = model_main_eval_data[main_feature_cols]
            y_val_main = model_main_eval_data['labels_main'].astype('int8')
            if self.debug:
                print(f"🔍 DEBUG: X_train_main shape: {X_train_main.shape}, y_train_main shape: {y_train_main.shape}")
                print(f"🔍 DEBUG: X_val_main shape: {X_val_main.shape}, y_val_main shape: {y_val_main.shape}")
            if len(y_train_main.value_counts()) < 2 or len(y_val_main.value_counts()) < 2:
                return None, None, None
            meta_feature_cols = [col for col in model_meta_train_data.columns if col != 'labels_meta']
            if self.debug:
                print(f"🔍 DEBUG: Meta model data shape: {model_meta_train_data.shape}")
                print(f"🔍 DEBUG: Meta feature columns: {meta_feature_cols}")
            model_meta_train_data, model_meta_eval_data = self.get_train_test_data(dataset=model_meta_train_data)
            if model_meta_train_data is None or model_meta_eval_data is None:
                return None, None, None
            X_train_meta = model_meta_train_data[meta_feature_cols]
            y_train_meta = model_meta_train_data['labels_meta'].astype('int8')
            X_val_meta = model_meta_eval_data[meta_feature_cols]
            y_val_meta = model_meta_eval_data['labels_meta'].astype('int8')
            if self.debug:
                print(f"🔍 DEBUG: X_train_meta shape: {X_train_meta.shape}, y_train_meta shape: {y_train_meta.shape}")
                print(f"🔍 DEBUG: X_val_meta shape: {X_val_meta.shape}, y_val_meta shape: {y_val_meta.shape}")
            if len(y_train_meta.value_counts()) < 2 or len(y_val_meta.value_counts()) < 2:
                return None, None, None
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
                           callbacks=[CatBoostPruningCallback(trial=trial, metric='Logloss')],
                           use_best_model=True,
                           verbose=False
            )
            t_train_main_end = time.time()
            if self.debug:
                print(f"🔍 DEBUG: Tiempo de entrenamiento modelo main: {t_train_main_end - t_train_main_start:.2f} segundos")
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
                           callbacks=[CatBoostPruningCallback(trial=trial, metric='Logloss')],
                           use_best_model=True,
                           verbose=False
            )
            t_train_meta_end = time.time()
            if self.debug:
                print(f"🔍 DEBUG: Tiempo de entrenamiento modelo meta: {t_train_meta_end - t_train_meta_start:.2f} segundos")
            model_main_path, model_meta_path = export_models_to_ONNX(models=(model_main, model_meta))
            
            # Inicializar score con valor por defecto
            score = -1.0
            
            test_train_time_start = time.time()
            try:
                score = tester(
                    dataset=full_ds,
                    model_main=model_main_path,
                    model_meta=model_meta_path,
                    model_main_cols=main_feature_cols,
                    model_meta_cols=meta_feature_cols,
                    direction=self.direction,
                    timeframe=self.timeframe,
                    print_metrics=True if self.debug else False,
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
                
            if self.debug:
                print(f"🔍 DEBUG: Modelos guardados en {model_main_path} y {model_meta_path}")
            return score, (model_main_path, model_meta_path), (main_feature_cols, meta_feature_cols)
        except Exception as e:
            print(f"Error en función de entrenamiento y test: {str(e)}")
            return None, None, None
        finally:
            clear_onnx_cache()
        
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
            filter = kwargs.get('filter')
            allowed = self.ALLOWED_FILTERS.get(self.label_method)
            # Detectar parámetros de ventana relevantes
            window_keys = [k for k in kwargs if any(x in k for x in ('rolling', 'window', 'window_size'))]
            if filter == 'savgol' and window_keys:
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

            filter = kwargs.get('filter')
            allowed = self.ALLOWED_FILTERS.get(self.label_method)
            if filter and allowed and filter not in allowed:
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
            pad = int(max(hp['periods_main']) + max(hp.get('periods_meta', [0])))
            if self.debug:
                print(f"🔍 DEBUG: pad = {pad}")

            # ──────────────────────────────────────────────────────────────
            # 2) Paso típico de la serie (modal -> inmune a huecos)
            idx = self.base_df.index.sort_values()
            bar_delta = idx.to_series().diff().dropna().mode().iloc[0] \
                if pad and len(idx) > 1 else pd.Timedelta(0)
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
            hp_tuple = tuple(sorted(hp.items()))
            full_ds = self.base_df.loc[start_ext:end_ext].copy()
            if self.debug:
                print(f"🔍 DEBUG: full_ds.shape = {full_ds.shape}")
            
            full_ds = get_features(full_ds, dict(hp_tuple))
            if self.debug:
                print(f"🔍 DEBUG: full_ds.shape después de get_features = {full_ds.shape}")

            full_ds = self.apply_labeling(full_ds, hp)
            if self.debug:
                print(f"🔍 DEBUG: full_ds.shape después de apply_labeling = {full_ds.shape}")

            # y recortar exactamente al rango que interesa
            full_ds = full_ds.loc[
                min(self.train_start, self.test_start):
                max(self.train_end,   self.test_end)
            ]
            if self.debug:
                print(f"🔍 DEBUG: full_ds.shape después de recorte = {full_ds.shape}")

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
                full_ds.drop(columns=problematic, inplace=True)
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
            if self.debug:
                print(f"🔍 DEBUG: hp = {hp}")

            # Verificar que tenemos al menos períodos y stats main
            if not hp['periods_main'] or not hp['stats_main']:
                return None
            # Guardar dataset completo a disco
            if self.debug:
                if self.tag:
                    data_dir = "./data"
                    os.makedirs(data_dir, exist_ok=True)
                    dataset_filename = f"{self.tag}.csv"
                    dataset_path = os.path.join(data_dir, dataset_filename)
                    full_ds.to_csv(dataset_path, index=True, float_format='%.4f')
                    print(f"🔍 DEBUG: Dataset guardado en {dataset_path}")

            return full_ds

        except Exception as e:
            print(f"⚠️ ERROR en get_labeled_full_data: {str(e)}")
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