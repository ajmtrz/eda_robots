import gc
import math
import time
import weakref
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
from modules.labeling_lib import (
    get_prices, get_features, get_labels_one_direction,
    sliding_window_clustering, clustering_simple,
    markov_regime_switching_simple, markov_regime_switching_advanced
)
from modules.tester_lib import (
    tester_one_direction,
    robust_oos_score_one_direction,
    _ONNX_CACHE
)
from modules.export_lib import export_model_to_ONNX

_BASES = weakref.WeakValueDictionary()

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
        n_jobs: int = -1,
        models_export_path: str = r'/mnt/c/Users/Administrador/AppData/Roaming/MetaQuotes/Terminal/6C3C6A11D1C3791DD4DBF45421BF8028/MQL5/Files/',
        include_export_path: str = r'/mnt/c/Users/Administrador/AppData/Roaming/MetaQuotes/Terminal/6C3C6A11D1C3791DD4DBF45421BF8028/MQL5/Include/ajmtrz/include/Dmitrievsky',
        history_path: str = r"/mnt/c/Users/Administrador/AppData/Roaming/MetaQuotes/Terminal/Common/Files/",
        search_type: str = 'clusters',
        search_subtype: str = 'simple',
        labels_deterministic: bool = True,
        tag: str = "",
    ):
        self.symbol = symbol
        self.timeframe = timeframe
        self.direction = direction
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
        
        key = (symbol, timeframe)
        if key in _BASES:
            self.base_df = _BASES[key]
        else:
            self.base_df = get_prices(symbol, timeframe, history_path)
            self.base_df = self.base_df[~self.base_df.index.duplicated()].sort_index()
            _BASES[key] = self.base_df

        # cache de features POR INSTANCIA --------------------
        self._feature_cache: dict[tuple, pd.DataFrame] = {}
        
        # Configuración de sklearn y optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    # =========================================================================
    # Métodos de búsqueda principales
    # =========================================================================

    def run_search(self) -> None:
        search_funcs = {
            'clusters': self.search_clusters,
            'markov': self.search_markov,
        }
        
        if self.search_type not in search_funcs:
            raise ValueError(f"Tipo de búsqueda no válido: {self.search_type}")
            
        search_func = search_funcs[self.search_type]
        
        for i in range(self.n_models):
            try:
                # Limpiar caché antes de cada modelo
                self.clear_cache()
                
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
                        best_str = f"ins={best_trial.values[0]:.4f} oos={best_trial.values[1]:.4f}"
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

                        # Forzar limpieza de memoria
                        gc.collect()
                    except Exception as e:
                        print(f"⚠️ ERROR en log_trial: {str(e)}")
                        gc.collect()

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
                import traceback
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
                meta_labels = (ds_train['labels_meta'] == clust).astype(np.int8).to_numpy()

                model_meta_data = ds_train.loc[:, meta_feature_cols].copy()
                model_meta_data['labels_meta'] = meta_labels

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
            return None
    
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
    
    def suggest_all_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Sugiere todos los parámetros de una vez para permitir muestreo multivariado."""
        try:
            MAX_MAIN_PERIODS = 15
            MAX_META_PERIODS = 3
            MAX_MAIN_STATS = 5
            MAX_META_STATS = 3
            # Todas las estadísticas disponibles
            all_stats = [
                "std", "skew", "zscore", "range", "mad", "entropy",
                "slope", "momentum", "autocorr", "max_dd", "hurst", "corr_skew",
                "sharpe", "fisher", "chande", "var", "eff_ratio", "kurt",
                "jump_vol", "fractal", "vol_skew", "approx_entropy",
            ]
            # Parámetros base comunes
            params = {
                'markup': trial.suggest_float("markup", 0.1, 1.0, log=True),  # Multiplicativo - mantiene log
                'label_max': trial.suggest_int('label_max', 1, 15, log=True),  # Períodos - mantiene log
                'atr_period': trial.suggest_int('atr_period', 5, 50, log=True),  # Períodos - mantiene log
                
                # Parámetros de CatBoost main
                'cat_main_iterations': trial.suggest_int('cat_main_iterations', 100, 1000),  # Lineal - quita log
                'cat_main_depth': trial.suggest_int('cat_main_depth', 3, 10),  # Discreto - quita log
                'cat_main_learning_rate': trial.suggest_float('cat_main_learning_rate', 0.01, 0.3, log=True),  # Multiplicativo - mantiene log
                'cat_main_l2_leaf_reg': trial.suggest_float('cat_main_l2_leaf_reg', 0.1, 10.0, log=True),  # Regularización - mantiene log
                'cat_main_early_stopping': trial.suggest_int('cat_main_early_stopping', 20, 200),  # Lineal - quita log
                
                # Parámetros de CatBoost meta
                'cat_meta_iterations': trial.suggest_int('cat_meta_iterations', 100, 1000),  # Lineal - quita log
                'cat_meta_depth': trial.suggest_int('cat_meta_depth', 3, 10),  # Discreto - quita log
                'cat_meta_learning_rate': trial.suggest_float('cat_meta_learning_rate', 0.01, 0.3, log=True),  # Multiplicativo - mantiene log
                'cat_meta_l2_leaf_reg': trial.suggest_float('cat_meta_l2_leaf_reg', 0.1, 10.0, log=True),  # Regularización - mantiene log
                'cat_meta_early_stopping': trial.suggest_int('cat_meta_early_stopping', 20, 200),  # Lineal - quita log
                
                # Períodos y estadísticas en el formato esperado
                'max_main_periods': trial.suggest_int('max_main_periods', 3, MAX_MAIN_PERIODS, log=True),
                'max_meta_periods': trial.suggest_int('max_meta_periods', 1, MAX_META_PERIODS, log=True),
                'max_main_stats': trial.suggest_int('max_main_stats', 1, MAX_MAIN_STATS, log=True),
                'max_meta_stats': trial.suggest_int('max_meta_stats', 1, MAX_META_STATS, log=True),
            }

            # ---------- PERÍODOS MAIN ----------
            periods_main = [trial.suggest_int(f'period_main_{i}', 3, 200, log=True)
                            for i in range(MAX_MAIN_PERIODS)]
            periods_main = sorted(set(periods_main))
            params['periods_main'] = tuple(periods_main[:params['max_main_periods']])

            # ---------- PERÍODOS META ----------
            periods_meta = [trial.suggest_int(f'period_meta_{i}', 3, 7)
                            for i in range(MAX_META_PERIODS)]
            periods_meta = sorted(set(periods_meta))
            params['periods_meta'] = tuple(periods_meta[:params['max_meta_periods']])

            # ---------- STATS MAIN ----------
            stats_main = [trial.suggest_categorical(f'stat_main_{i}', all_stats)
                        for i in range(MAX_MAIN_STATS)]
            stats_main = list(dict.fromkeys(stats_main))
            params['stats_main'] = tuple(stats_main[:params['max_main_stats']])

            # ---------- STATS META ----------
            stats_meta = [trial.suggest_categorical(f'stat_meta_{i}', all_stats)
                        for i in range(MAX_META_STATS)]
            stats_meta = list(dict.fromkeys(stats_meta))
            params['stats_meta'] = tuple(stats_meta[:params['max_meta_stats']])

            # Parámetros específicos según el tipo de búsqueda
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
            main_feature_cols = sorted([col for col in model_main_data.columns if col != 'labels_main'])
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
            meta_feature_cols = sorted([col for col in model_meta_data.columns if col != 'labels_meta'])
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
                #store_all_simple_ctr=False,
                thread_count=self.n_jobs,
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
                #store_all_simple_ctr=False,
                thread_count=self.n_jobs,
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
            ds_train_eval_main = ds_train_eval_sample[main_feature_cols].to_numpy()
            ds_train_eval_meta = ds_train_eval_sample[meta_feature_cols].to_numpy()
            ds_test_eval_main = ds_test[main_feature_cols].to_numpy()
            ds_test_eval_meta = ds_test[meta_feature_cols].to_numpy()
            close_train_eval = ds_train_eval_sample['close'].to_numpy()
            close_test_eval = ds_test['close'].to_numpy()

            score_ins = tester_one_direction(
                ds_main=ds_train_eval_main,
                ds_meta=ds_train_eval_meta,
                close=close_train_eval,
                model_main=model_main,
                model_meta=model_meta,
                direction=self.direction,
                plt=False,
                prd='insample',
                # n_sim=100,
                # agg="q05"
            )
            score_oos = tester_one_direction(
                ds_main=ds_test_eval_main,
                ds_meta=ds_test_eval_meta,
                close=close_test_eval,
                model_main=model_main,
                model_meta=model_meta,
                direction=self.direction,
                plt=False,
                prd='outofsample'
                # n_sim=100,
                # agg="q05"
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
            full_ds = self._cached_features(start_ext, end_ext, hp_tuple)

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
        
    # ------- cache interno, sin LRU global ------------------
    def _cached_features(self, start, end, hp_tuple):
        k = (start, end, hp_tuple)
        if k not in self._feature_cache:
            hp = dict(hp_tuple)
            ds = self.base_df.loc[start:end].copy()
            self._feature_cache[k] = get_features(ds, hp)
        return self._feature_cache[k]

    def clear_cache(self):
        """Vacía SOLO las features derivadas; mantiene base_df viva."""
        self._feature_cache.clear()
        gc.collect()