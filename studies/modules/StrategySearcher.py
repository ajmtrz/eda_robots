import gc
import math
import time
import traceback
import random
import os
import psutil
import numpy as np
import pandas as pd
from datetime import datetime
from time import perf_counter
from typing import Dict, Any
import optuna
from optuna.pruners import HyperbandPruner, SuccessiveHalvingPruner
#from optuna.integration import CatBoostPruningCallback
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool
from mapie.classification import CrossConformalClassifier
from modules.labeling_lib import (
    get_prices, get_features,
    get_labels_one_direction, get_labels,
    sliding_window_clustering, clustering_simple,
    markov_regime_switching_simple, markov_regime_switching_advanced
)
from modules.tester_lib import (
    tester,
    tester_one_direction,
    robust_oos_score_one_direction,
    _ONNX_CACHE
)
from modules.export_lib import export_model_to_ONNX

class StrategySearcher:
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
        labels_deterministic: bool = False,
        tag: str = "",
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
        self.labels_deterministic = labels_deterministic
        self.pruner_type = pruner_type
        self.n_trials = n_trials
        self.n_models = n_models
        self.n_jobs = n_jobs
        self.tag = tag
        self.base_df = get_prices(symbol, timeframe, history_path)

        # Configuración de sklearn y optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    def _log_memory(self, msg: str = "") -> None:
        """Helper to print current RSS memory usage in MB."""
        try:
            mem = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
            print(f"{msg}RSS={mem:.2f} MB", flush=True)
        except Exception:
            pass

    # =========================================================================
    # Métodos de búsqueda principales
    # =========================================================================

    def run_search(self) -> None:
        search_funcs = {
            'clusters': self.search_clusters,
            'markov': self.search_markov,
            'mapie': self.search_mapie,
            'causal': self.search_causal,
        }
        
        if self.search_type not in search_funcs:
            raise ValueError(f"Tipo de búsqueda no válido: {self.search_type}")
            
        search_func = search_funcs[self.search_type]
        
        for i in range(self.n_models):
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
                    pruner=pruners[self.pruner_type],
                    sampler=optuna.samplers.TPESampler(
                        n_startup_trials=int(np.sqrt(self.n_trials)),
                        multivariate=True
                    )
                )

                t0 = perf_counter()
                def log_trial(study, trial):
                    try:
                        # Obtener el mejor trial según criterio maximin
                        if study.best_trials:
                            best_trial = max(study.best_trials, 
                                          key=lambda t: min(t.values[0], t.values[1]))
                            
                            # Si este trial es el mejor, guardar sus modelos
                            if trial.number == best_trial.number:
                                if trial.user_attrs.get('models') is not None:
                                    study.set_user_attr("best_models", trial.user_attrs['models'])
                                    study.set_user_attr("best_scores", trial.user_attrs['scores'])
                                    study.set_user_attr("best_periods_main", trial.user_attrs['periods_main'])
                                    study.set_user_attr("best_periods_meta", trial.user_attrs['periods_meta'])
                                    study.set_user_attr("best_stats_main", trial.user_attrs['stats_main'])
                                    study.set_user_attr("best_stats_meta", trial.user_attrs['stats_meta'])

                        # Log
                        if study.best_trials:
                            best_trial = max(study.best_trials, key=lambda t: min(*t.values))
                            best_str = f"ins={best_trial.values[0]:.4f} oos={best_trial.values[1]:.4f}"
                        else:
                            best_str = "ins=--- oos=---"
                        elapsed = perf_counter() - t0
                        n_done = trial.number + 1
                        avg_time = elapsed / n_done
                        print(
                            f"[{self.tag}] modelo {i} "
                            f"trial {n_done}/{self.n_trials} "
                            f"{best_str} "
                            f"avg={avg_time:6.2f}s",
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
                best_models = study.user_attrs.get("best_models", None)
                if best_models is None or not best_models:
                    print(f"⚠️ ERROR: best_models VACÍO")
                    continue
                
                export_params = {
                    "symbol": self.symbol,
                    "timeframe": self.timeframe,
                    "direction": self.direction,
                    "models_export_path": self.models_export_path,
                    "include_export_path": self.include_export_path,
                    "search_type": self.search_type,
                    "search_subtype": self.search_subtype,
                    "best_model_seed": model_seed,
                    "best_models": study.user_attrs["best_models"],
                    "best_scores": study.user_attrs["best_scores"],
                    "best_periods_main": study.user_attrs["best_periods_main"],
                    "best_periods_meta": study.user_attrs["best_periods_meta"],
                    "best_stats_main": study.user_attrs["best_stats_main"],
                    "best_stats_meta": study.user_attrs["best_stats_meta"],
                }
                
                export_model_to_ONNX(**export_params)
                
            except Exception as e:
                print(f"\nError procesando modelo {i}:")
                print(f"Error: {str(e)}")
                print("Traceback:")
                print(traceback.format_exc())
                continue

    # =========================================================================
    # Métodos de búsqueda específicos
    # =========================================================================
    
    def _evaluate_clusters(self, ds_train: pd.DataFrame, ds_test: pd.DataFrame, hp: Dict[str, Any]) -> tuple[float, float]:
        """Función helper para evaluar clusters y entrenar modelos."""
        try:
            best_models = (None, None)
            best_scores = (-math.inf, -math.inf)
            cluster_sizes = ds_train['labels_meta'].value_counts()

            # Verificar que hay clusters
            if cluster_sizes.empty:
                print("⚠️ ERROR: No hay clusters")
                return None, None

            # Filtrar el cluster -1 (inválido) si existe
            if -1 in cluster_sizes.index:
                cluster_sizes = cluster_sizes.drop(-1)
                if cluster_sizes.empty:
                    return None, None

            # Evaluar cada cluster
            for clust in cluster_sizes.index:
                model_main_data = ds_train.loc[ds_train["labels_meta"] == clust].copy()

                if len(model_main_data) <= hp["label_max"]:
                    continue
                if self.direction == 'both':
                    model_main_data = get_labels(
                        model_main_data,
                        markup=hp['markup'],
                        max=hp['label_max']
                    )
                    model_main_data = model_main_data.rename(columns={'labels': 'labels_main'})
                else:
                    model_main_data = get_labels_one_direction(
                        model_main_data,
                        markup=hp['markup'],
                        max_val=hp['label_max'],
                        direction=self.direction,
                        atr_period=hp['atr_period'],
                        deterministic=self.labels_deterministic,
                    )
                main_feature_cols = model_main_data.columns[model_main_data.columns.str.contains('_feature') & \
                                                       ~model_main_data.columns.str.contains('_meta_feature')]
                model_main_data = model_main_data[main_feature_cols.tolist() + ['labels_main']]

                if (model_main_data['labels_main'].value_counts() < 2).any():
                    continue

                # Meta data
                meta_feature_cols = ds_train.filter(like='_meta_feature').columns
                model_meta_data = ds_train.loc[:, meta_feature_cols].copy()
                model_meta_data['labels_meta'] = (ds_train['labels_meta'] == clust).astype('int8')

                if (model_meta_data['labels_meta'].value_counts() < 2).any():
                    continue

                # ── Evaluación en ambos períodos ──────────────────────────────
                scores, models = self.fit_final_models(
                    model_main_data=model_main_data, 
                    model_meta_data=model_meta_data,
                    ds_train=ds_train, 
                    ds_test=ds_test, 
                    hp=hp.copy()
                )
                if scores is None or models is None:
                    continue

                # Verificar scores
                if not all(np.isfinite(scores)):
                    continue

                # Aplicar criterio maximin: maximizar el peor valor entre ins/oos
                if min(scores) > min(best_scores):
                    best_scores = scores
                    best_models = models

            # Verificar que encontramos algún cluster válido
            if best_scores == (-math.inf, -math.inf) or best_models == (None, None):
                return None, None

            return best_scores, best_models

        except Exception as e:
            print(f"⚠️ ERROR en evaluación de clusters: {str(e)}")
            return None, None
        finally:
            try:
                del model_main_data
            except Exception:
                pass
            try:
                del model_meta_data
            except Exception:
                pass
            gc.collect()
    
    def check_constant_features(self, X: np.ndarray, feature_cols: list, std_epsilon: float = 1e-12) -> bool:
        """Verifica si hay columnas que podrían causar inestabilidad numérica.
        
        Args:
            X: Array numpy con los datos
            feature_cols: Lista con nombres de las columnas
            std_epsilon: Umbral para considerar una columna como constante
            
        Returns:
            bool: True si hay columnas problemáticas, False en caso contrario
        """
        problematic_cols = []
        
        # 1) Verificar columnas con nan/inf
        for i, col in enumerate(feature_cols):
            if not np.isfinite(X[:, i]).all():
                problematic_cols.append(f"{col} (contiene valores inf/nan)")
                
        # 2) Verificar columnas (casi) constantes
        stds = np.nanstd(X, axis=0)
        for i, (std, col) in enumerate(zip(stds, feature_cols)):
            if std < std_epsilon:
                problematic_cols.append(f"{col} (desviación estándar {std:.2e})")
                
        if problematic_cols:
            # print("\n⚠️ Columnas problemáticas detectadas:")
            # for col in problematic_cols:
            #     print(f"  - {col}")
            return True
            
        return False
    
    def suggest_all_params(self, trial: 'optuna.Trial') -> dict:
        try:
            MAX_MAIN_PERIODS = 15
            MAX_META_PERIODS = 3
            MAX_MAIN_STATS = 5
            MAX_META_STATS = 3
            all_stats = [
                "std", "skew", "zscore", "range", "mad", "entropy",
                "slope", "momentum", "autocorr", "max_dd", "hurst", "corr_skew",
                "sharpe", "fisher", "chande", "var", "eff_ratio", "kurt",
                "jump_vol", "fractal", "vol_skew", "approx_entropy",
            ]
            params = {
                'markup': trial.suggest_float("markup", 0.1, 1.0, log=True),
                'label_max': trial.suggest_int('label_max', 1, 15, log=True),
                'atr_period': trial.suggest_int('atr_period', 5, 50, log=True),
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
            # ---------- PERÍODOS MAIN ----------
            periods_main = [trial.suggest_int(f'period_main_{i}', 3, 200, log=True)
                            for i in range(MAX_MAIN_PERIODS)]
            periods_main = sorted(set(periods_main))
            params['periods_main'] = tuple(periods_main[:params['max_main_periods']])
            # ---------- STATS MAIN ----------
            stats_main = [trial.suggest_categorical(f'stat_main_{i}', all_stats)
                        for i in range(MAX_MAIN_STATS)]
            stats_main = list(dict.fromkeys(stats_main))
            params['stats_main'] = tuple(stats_main[:params['max_main_stats']])
            # ---------- Hiperparámetros meta solo si no es mapie ----------
            if self.search_type in ['clusters', 'markov']:
                params['max_meta_periods'] = trial.suggest_int('max_meta_periods', 1, MAX_META_PERIODS, log=True)
                params['max_meta_stats'] = trial.suggest_int('max_meta_stats', 1, MAX_META_STATS, log=True)
                periods_meta = [trial.suggest_int(f'period_meta_{i}', 3, 7)
                                for i in range(MAX_META_PERIODS)]
                periods_meta = sorted(set(periods_meta))
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
            elif self.search_type == 'mapie':
                params.update({
                    'mapie_confidence_level': trial.suggest_float('mapie_confidence_level', 0.7, 0.99),
                    'mapie_cv': trial.suggest_int('mapie_cv', 3, 10),
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
                        hp: Dict[str, Any]) -> tuple[tuple[float, float], object, object]:
        """Ajusta los modelos finales."""
        try:
            # ---------- 1) main model_main ----------
            # Get feature columns and rename them to follow f%d pattern
            main_feature_cols = [col for col in model_main_data.columns if col != 'labels_main']
            X_main = model_main_data[main_feature_cols]
            y_main = model_main_data['labels_main'].astype('int16')
            
            # División de datos para el modelo principal según fechas
            X_train_main, X_val_main, y_train_main, y_val_main = train_test_split(
                X_main, y_main, 
                test_size=0.2,
                shuffle=True
            )
            
            # ── descartar clusters problemáticos ────────────────────────────
            if len(y_train_main.value_counts()) < 2 or len(y_val_main.value_counts()) < 2:
                return None, None
            
            # ---------- 2) meta‑modelo ----------
            meta_feature_cols = [col for col in model_meta_data.columns if col != 'labels_meta']
            X_meta = model_meta_data[meta_feature_cols]
            y_meta = model_meta_data['labels_meta'].astype('int16')

            # División de datos para el modelo principal según fechas
            X_train_meta, X_val_meta, y_train_meta, y_val_meta = train_test_split(
                X_meta, y_meta, 
                test_size=0.2,
                shuffle=True
            )
            
            # ── descartar clusters problemáticos ────────────────────────────
            if len(y_train_meta.value_counts()) < 2 or len(y_val_meta.value_counts()) < 2:
                return None, None

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
            model_main.fit(X_train_main, y_train_main, 
                           eval_set=Pool(X_val_main, y_val_main), 
                           early_stopping_rounds=hp['cat_main_early_stopping'],
                           use_best_model=True,
                           verbose=False
            )

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
            model_meta.fit(X_train_meta, y_train_meta, 
                           eval_set=Pool(X_val_meta, y_val_meta), 
                           early_stopping_rounds=hp['cat_meta_early_stopping'],
                           use_best_model=True,
                           verbose=False
            )

            # ── evaluación ───────────────────────────────────────────────
            # Preparar datasets de entrenamiento y prueba
            n_test = len(ds_test)
            cut_idx   = ds_train.index[ds_train.index < self.test_start].max()
            if pd.isna(cut_idx):
                return None, None
                
            ds_train_eval_sample = ds_train.loc[:cut_idx].tail(n_test)
            if len(ds_train_eval_sample) != n_test:
                return None, None

            # Verificar orden exacto
            if not (ds_test[main_feature_cols].columns == main_feature_cols).all():
                print("⚠️ ERROR: El orden de las columnas main no coincide en test")
                return None, None
            if not (ds_test[meta_feature_cols].columns == meta_feature_cols).all():
                print("⚠️ ERROR: El orden de las columnas meta no coincide en test")
                return None, None
                
            ds_train_eval_main = ds_train_eval_sample[main_feature_cols].to_numpy()
            ds_train_eval_meta = ds_train_eval_sample[meta_feature_cols].to_numpy()
            ds_test_eval_main = ds_test[main_feature_cols].to_numpy()
            ds_test_eval_meta = ds_test[meta_feature_cols].to_numpy()
            close_train_eval = ds_train_eval_sample['close'].to_numpy()
            close_test_eval = ds_test['close'].to_numpy()
            if self.direction == 'both':
                df_ins = pd.DataFrame({
                    'close': close_train_eval,
                    'labels_main': model_main.predict_proba(ds_train_eval_main)[:, 1],
                    'labels_meta': model_meta.predict_proba(ds_train_eval_meta)[:, 1],
                })
                df_oos = pd.DataFrame({
                    'close': close_test_eval,
                    'labels_main': model_main.predict_proba(ds_test_eval_main)[:, 1],
                    'labels_meta': model_meta.predict_proba(ds_test_eval_meta)[:, 1],
                })
                score_ins = tester(df_ins, plot=False)
                score_oos = tester(df_oos, plot=False)
            else:
                score_ins = tester_one_direction(
                    ds_main=ds_train_eval_main,
                    ds_meta=ds_train_eval_meta,
                    close=close_train_eval,
                    model_main=model_main,
                    model_meta=model_meta,
                    direction=self.direction,
                    plot=False,
                    prd='insample',
                )
                score_oos = tester_one_direction(
                    ds_main=ds_test_eval_main,
                    ds_meta=ds_test_eval_meta,
                    close=close_test_eval,
                    model_main=model_main,
                    model_meta=model_meta,
                    direction=self.direction,
                    plot=False,
                    prd='outofsample',
                )

            # Manejar valores inválidos
            if not np.isfinite(score_ins) or not np.isfinite(score_oos):
                score_ins = -1.0
                score_oos = -1.0

            return (score_ins, score_oos), (model_main, model_meta)
        
        except Exception as e:
            print(f"Error en función de entrenamiento y test: {str(e)}")
            return None, None
        finally:
            try:
                del X_main, X_meta, X_train_main, X_val_main, y_train_main, y_val_main
                del X_train_meta, X_val_meta, y_train_meta, y_val_meta
            except Exception:
                pass
            try:
                del ds_train_eval_sample, ds_train_eval_main, ds_train_eval_meta
                del ds_test_eval_main, ds_test_eval_meta
            except Exception:
                pass
            try:
                del model_main_data, model_meta_data
            except Exception:
                pass
            _ONNX_CACHE.clear()
            gc.collect()

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

            scores, models = self._evaluate_clusters(ds_train, ds_test, hp)
            if scores is None or models is None:
                return -1.0, -1.0
            else:
                trial.set_user_attr('models', models)
                trial.set_user_attr('scores', scores)

            return scores[0], scores[1]
            
        except Exception as e:
            print(f"Error en search_markov: {str(e)}")
            return -1.0, -1.0
        finally:
            try:
                del ds_train, ds_test
            except Exception:
                pass
            gc.collect()
            self._log_memory(f"[{self.tag}] markov trial end ")

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
            scores, models = self._evaluate_clusters(ds_train, ds_test, hp)
            if scores is None or models is None:
                return -1.0, -1.0
            else:
                trial.set_user_attr('models', models)
                trial.set_user_attr('scores', scores)

            return scores[0], scores[1]
        
        except Exception as e:
            print(f"Error en search_clusters: {str(e)}")
            return -1.0, -1.0
        finally:
            try:
                del ds_train, ds_test
            except Exception:
                pass
            gc.collect()
            self._log_memory(f"[{self.tag}] clusters trial end ")

    def search_mapie(self, trial) -> tuple[float, float]:
        """Implementa la búsqueda de estrategias usando conformal prediction (MAPIE) con CatBoost, usando el mismo conjunto de features para ambos modelos."""
        try:
            hp = self.suggest_all_params(trial)
            ds_train, ds_test = self.get_train_test_data(hp)
            if ds_train is None or ds_test is None:
                return -1.0, -1.0

            # Etiquetado según la dirección seleccionada
            if self.direction == 'both':
                ds_train = get_labels(
                    ds_train,
                    markup=hp['markup'],
                    max=hp['label_max']
                )
                ds_train = ds_train.rename(columns={'labels': 'labels_main'})
            else:
                ds_train = get_labels_one_direction(
                    ds_train,
                    markup=hp['markup'],
                    max_val=hp['label_max'],
                    direction=self.direction,
                    atr_period=hp['atr_period'],
                    deterministic=self.labels_deterministic,
                )
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
            scores, models = self.fit_final_models(
                model_main_data=model_main_data,
                model_meta_data=model_meta_data,
                ds_train=ds_train,
                ds_test=ds_test,
                hp=hp.copy()
            )
            if scores is None or models is None:
                return -1.0, -1.0
            trial.set_user_attr('models', models)
            trial.set_user_attr('scores', scores)
            return scores[0], scores[1]
        except Exception as e:
            print(f"Error en search_mapie: {str(e)}")
            return -1.0, -1.0
        finally:
            try:
                del ds_train, ds_test
                del model_main_data, model_meta_data, X, y
            except Exception:
                pass
            gc.collect()
            self._log_memory(f"[{self.tag}] mapie trial end ")

    def search_causal(self, trial: optuna.Trial) -> tuple[float, float]:
        """Búsqueda basada en detección causal de muestras malas."""
        try:
            hp = self.suggest_all_params(trial)

            ds_train, ds_test = self.get_train_test_data(hp)
            if ds_train is None or ds_test is None:
                return -1.0, -1.0

            # Etiquetado según la dirección
            if self.direction == 'both':
                ds_train = get_labels(
                    ds_train,
                    markup=hp['markup'],
                    max=hp['label_max']
                )
                ds_train = ds_train.rename(columns={'labels': 'labels_main'})
            else:
                ds_train = get_labels_one_direction(
                    ds_train,
                    markup=hp['markup'],
                    max_val=hp['label_max'],
                    direction=self.direction,
                    atr_period=hp['atr_period'],
                    deterministic=self.labels_deterministic,
                )

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
                        thread_count=-1,
                        task_type='CPU',
                        verbose=False,
                    )
                    model = CatBoostClassifier(**catboost_params)
                    model.fit(X.loc[train_idx], y.loc[train_idx])
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

            scores, models = self.fit_final_models(
                model_main_data=model_main_data,
                model_meta_data=model_meta_data,
                ds_train=ds_train,
                ds_test=ds_test,
                hp=hp.copy()
            )
            if scores is None or models is None:
                return -1.0, -1.0

            score_ins, score_oos_raw = scores
            model_main, model_meta = models

            main_feature_cols = [c for c in model_main_data.columns if c != 'labels_main']
            meta_feature_cols = [c for c in model_meta_data.columns if c != 'labels_meta']
            ds_test_main = ds_test[main_feature_cols].to_numpy()
            ds_test_meta = ds_test[meta_feature_cols].to_numpy()
            close_test = ds_test['close'].to_numpy()

            score_oos = robust_oos_score_one_direction(
                ds_main=ds_test_main,
                ds_meta=ds_test_meta,
                close=close_test,
                model_main=model_main,
                model_meta=model_meta,
                direction=self.direction,
                plot=False,
                prd='outofsample'
            )

            trial.set_user_attr('models', models)
            trial.set_user_attr('scores', (score_ins, score_oos))
            trial.set_user_attr('oos_raw', score_oos_raw)

            return score_ins, score_oos

        except Exception as e:
            print(f"Error en search_causal: {str(e)}")
            return -1.0, -1.0
        finally:
            try:
                del ds_train, ds_test
                del model_main_data, model_meta_data, X, y
            except Exception:
                pass
            gc.collect()
            self._log_memory(f"[{self.tag}] causal trial end ")

    # =========================================================================
    # Métodos auxiliares
    # =========================================================================
    
    # ---------------------------------------------------------------------
    def get_train_test_data(self, hp):
        try:
            if hp is None:
                return None, None

            # ──────────────────────────────────────────────────────────────
            # 1) Calcular el colchón de barras necesario
            pad = max(hp.get('periods_main', ()) + hp.get('periods_meta', ()), default=0)
            pad = int(pad)

            # ──────────────────────────────────────────────────────────────
            # 2) Paso típico de la serie (mediana -> inmune a huecos)
            idx = self.base_df.index.sort_values()
            if pad == 0 or len(idx) < 2:
                bar_delta = pd.Timedelta(0)
            else:
                bar_delta = idx.to_series().diff().dropna().median()

            # ──────────────────────────────────────────────────────────────
            # 3) Rango extendido para calcular features “con contexto”
            start_ext = min(self.train_start, self.test_start) - pad * bar_delta
            if start_ext < idx[0]:                      # evita pedir antes de que existan datos
                start_ext = idx[0]

            end_ext = max(self.train_end, self.test_end)

            # ──────────────────────────────────────────────────────────────
            # 4) Obtener features de todo el rango extendido
            hp_tuple = tuple(sorted(hp.items()))
            ds_slice = self.base_df.loc[start_ext:end_ext].copy()
            full_ds = get_features(ds_slice, dict(hp_tuple))

            # y recortar exactamente al rango que interesa
            full_ds = full_ds.loc[
                min(self.train_start, self.test_start):
                max(self.train_end,   self.test_end)
            ]

            if full_ds.empty:
                return None, None

            # ──────────────────────────────────────────────────────────────
            # 5) Comprobaciones de calidad de features
            feature_cols = full_ds.columns[full_ds.columns.str.contains('feature')]
            if feature_cols.empty:
                return None, None

            if self.check_constant_features(full_ds[feature_cols].to_numpy('float32'), feature_cols):
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

            # ──────────────────────────────────────────────────────────────
            # 7) DataFrames finales, ordenados cronológicamente
            train_data = full_ds[train_mask].sort_index()
            test_data  = full_ds[test_mask].sort_index()

            if len(train_data) < 100 or len(test_data) < 50:
                print("⚠️ ERROR: Datasets demasiado pequeños")
                return None, None

            return train_data, test_data

        except Exception as e:
            print(f"⚠️ ERROR en get_train_test_data: {str(e)}")
            return None, None
        finally:
            gc.collect()
