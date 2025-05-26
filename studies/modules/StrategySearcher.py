import gc
import math
import time
import numpy as np
import pandas as pd
from datetime import datetime
from time import perf_counter
from typing import Dict, Any
import optuna
from optuna.pruners import HyperbandPruner, SuccessiveHalvingPruner
#from optuna.integration import CatBoostPruningCallback
from sklearn import set_config
from sklearn.model_selection import train_test_split
from functools import lru_cache
from catboost import CatBoostClassifier, Pool
from modules.labeling_lib import (
    get_prices, get_features, get_labels_one_direction,
    sliding_window_clustering, clustering_simple,
    markov_regime_switching_simple, markov_regime_switching_advanced
)
from modules.tester_lib import (
    test_model_one_direction,
    robust_oos_score_one_direction,
    _ONNX_CACHE
)
from modules.export_lib import export_model_to_ONNX

class StrategySearcher:
    """Clase unificada para búsqueda de estrategias de trading.
    
    Esta clase implementa diferentes métodos de búsqueda de estrategias de trading
    utilizando técnicas de machine learning y optimización bayesiana.
    
    Attributes:
        base_df (pd.DataFrame): DataFrame con datos históricos
        best_models (list): Lista de los mejores modelos encontrados
        model_range (list): Rango de modelos a optimizar
    """
    
    _FEATURE_CACHE = {}
    _BASE_DF_ID = 0

    @staticmethod
    @lru_cache(maxsize=128)
    def _cached_features(df_id: int,
                        start: datetime,
                        end: datetime,
                        hp_tuple: tuple) -> pd.DataFrame:
        """Obtiene características del DataFrame con caché LRU.
        
        El caché almacena hasta 128 resultados diferentes, eliminando automáticamente
        los más antiguos cuando se alcanza el límite. Cada resultado es un DataFrame
        completo con las características calculadas.
        
        Args:
            df_id: ID del DataFrame base
            start: Fecha de inicio
            end: Fecha de fin
            hp_tuple: Tupla de tuplas con los hiperparámetros que afectan a las características

        Returns:
            pd.DataFrame: DataFrame con las características calculadas
        """
        # Convertir la tupla de tuplas en diccionario
        hp = dict(hp_tuple)
        hp.update({
            'periods_main': list(hp.get('periods_main', ())),
            'periods_meta': list(hp.get('periods_meta', ())),
            'stats_main': list(hp.get('stats_main', ())),
            'stats_meta': list(hp.get('stats_meta', ())),
        })
        return get_features(StrategySearcher._FEATURE_CACHE[df_id].loc[start:end].copy(), hp)

    @staticmethod
    def clear_cache():
        """Limpia el caché de características."""
        StrategySearcher._cached_features.cache_clear()
        gc.collect()

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
        tag: str = "",
    ):
        """Inicializa el buscador de estrategias.
        
        Args:
            symbol: Símbolo del activo financiero
            timeframe: Timeframe de los datos
            direction: Dirección de trading ('buy' o 'sell')
            train_start: Fecha de inicio del entrenamiento
            train_end: Fecha de fin del entrenamiento
            test_start: Fecha de inicio de prueba
            test_end: Fecha de fin de prueba
            pruner_type: Tipo de pruner ('hyperband' o 'halving')
            n_trials: Número de trials para la optimización
            models_export_path: Ruta para exportar modelos
            include_export_path: Ruta para archivos include
            history_path: Ruta para datos históricos
            search_type: Tipo de búsqueda a realizar ('clusters' o 'markov')
            search_subtype: Subtipo de búsqueda ('simple' u otros según el tipo)
            tag: Etiqueta opcional para identificar la búsqueda
        """
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
        self.pruner_type = pruner_type
        self.n_trials = n_trials
        self.n_models = n_models
        self.n_jobs = n_jobs
        self.tag = tag
        
        # Cargar datos históricos
        self.base_df = get_prices(self.symbol, self.timeframe, self.history_path)
        StrategySearcher._FEATURE_CACHE[StrategySearcher._BASE_DF_ID] = self.base_df
        
        # Configuración de sklearn y optuna
        set_config(enable_metadata_routing=True, skip_parameter_validation=True)
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        # Resultados
        self.best_models = (None, None)

    # =========================================================================
    # Métodos de búsqueda principales
    # =========================================================================

    def run_search(self) -> None:
        """Ejecuta la búsqueda de estrategias."""
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
                StrategySearcher._FEATURE_CACHE[StrategySearcher._BASE_DF_ID] = self.base_df
                
                # Generar un seed único para este modelo
                model_seed = int(time.time() * 1000) + i

                # Inicializar estudio de Optuna con objetivos múltiples
                pruners = {
                    'hyperband': HyperbandPruner(max_resource='auto'),
                    'halving': SuccessiveHalvingPruner(min_resource='auto', reduction_factor=3)
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
                    lambda t: search_func(t, study),
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
    
    def _evaluate_clusters(self, ds_train: pd.DataFrame, ds_test: pd.DataFrame, hp: Dict[str, Any], trial: optuna.Trial, study: optuna.study.Study) -> tuple[float, float]:
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
                    #print("⚠️ ERROR: Solo hay clusters inválidos")
                    return None, None

            # Evaluar cada cluster
            for clust in cluster_sizes.index:
                # Main data
                main_cols = [c for c in ds_train.columns if '_feature' in c and '_meta_feature' not in c]
                meta_cols = [c for c in ds_train.columns if '_meta_feature' in c]

                if not main_cols or not meta_cols:
                    print(f"⚠️ ERROR: No hay características para el cluster {clust}")
                    continue

                ohlc_cols = ["open", "high", "low", "close"]
                present = [c for c in ohlc_cols if c in ds_train.columns]
                
                if not present:
                    print(f"⚠️ ERROR: No hay datos OHLC para el cluster {clust}")
                    continue

                main_data = ds_train.loc[
                    ds_train["labels_meta"] == clust,
                    present + main_cols
                ].copy()
                
                if len(main_data) <= hp['label_max']:
                    print(f"⚠️ ERROR: Cluster {clust} demasiado pequeño")
                    continue

                main_data = get_labels_one_direction(
                    main_data,
                    markup=hp['markup'],
                    max_val=hp['label_max'],
                    direction=self.direction,
                    atr_period=hp['atr_period']
                )
                
                if (main_data['labels_main'].value_counts() < 2).any():
                    print(f"⚠️ ERROR: Cluster {clust} con clases insuficientes")
                    continue

                # Meta data
                meta_data = ds_train[meta_cols].copy()
                meta_data['labels_meta'] = (ds_train['labels_meta'] == clust).astype(int)
                
                if (meta_data['labels_meta'].value_counts() < 2).any():
                    print(f"⚠️ ERROR: Meta datos del cluster {clust} con clases insuficientes")
                    continue

                # ── Evaluación en ambos períodos ──────────────────────────────
                scores, models = self.fit_final_models(
                    main_data, meta_data, ds_train, ds_test, hp.copy()
                )
                if scores is None or models is None:
                    print(f"⚠️ ERROR: Fit fallido para el cluster {clust}")
                    continue

                # Verificar scores
                if not all(np.isfinite(scores)):
                    print(f"⚠️ ERROR: Scores no finitos para el cluster {clust}")
                    continue

                # Aplicar criterio maximin: maximizar el peor valor entre ins/oos
                if min(scores) > min(best_scores):
                    best_scores = scores
                    best_models = models

            # Verificar que encontramos algún cluster válido
            if best_scores == (-math.inf, -math.inf) or best_models == (None, None):
                print("⚠️ ERROR: No se encontraron clusters válidos")
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
            print("\n⚠️ Columnas problemáticas detectadas:")
            for col in problematic_cols:
                print(f"  - {col}")
            return True
            
        return False
    
    def suggest_all_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Sugiere todos los parámetros de una vez para permitir muestreo multivariado."""
        try:
            # Todas las estadísticas disponibles
            all_stats = [
                "std", "skew", "zscore", "range", "mad", "entropy",
                "slope", "momentum", "autocorr", "max_dd",
                "sharpe", "fisher", "chande", "var", "eff_ratio", 
                "jump_vol", "fractal", "vol_skew", "corr_skew",
                "approx_entropy", "hurst", "kurt"
            ]                # Parámetros base comunes
            params = {
                'markup': trial.suggest_float("markup", 0.1, 1.0, log=True),  # Multiplicativo - mantiene log
                'label_max': trial.suggest_int('label_max', 2, 30, log=True),  # Períodos - mantiene log
                'atr_period': trial.suggest_int('atr_period', 5, 100, log=True),  # Períodos - mantiene log
                
                # Parámetros de CatBoost main
                'cat_main_iterations': trial.suggest_int('cat_main_iterations', 100, 1000),  # Lineal - quita log
                'cat_main_depth': trial.suggest_int('cat_main_depth', 3, 10),  # Discreto - quita log
                'cat_main_learning_rate': trial.suggest_float('cat_main_learning_rate', 0.01, 0.3, log=True),  # Multiplicativo - mantiene log
                'cat_main_l2_leaf_reg': trial.suggest_float('cat_main_l2_leaf_reg', 0.1, 10.0, log=True),  # Regularización - mantiene log
                'cat_main_early_stopping': trial.suggest_int('cat_main_early_stopping', 20, 200),  # Lineal - quita log
                'cat_main_bootstrap_type': trial.suggest_categorical('cat_main_bootstrap_type', ['Bayesian', 'Bernoulli', 'MVS']),
                'cat_main_bagging_temperature': trial.suggest_float('cat_main_bagging_temperature', 0, 10),
                'cat_main_subsample': trial.suggest_float('cat_main_subsample', 0.5, 1),
                
                # Parámetros de CatBoost meta
                'cat_meta_iterations': trial.suggest_int('cat_meta_iterations', 100, 1000),  # Lineal - quita log
                'cat_meta_depth': trial.suggest_int('cat_meta_depth', 3, 10),  # Discreto - quita log
                'cat_meta_learning_rate': trial.suggest_float('cat_meta_learning_rate', 0.01, 0.3, log=True),  # Multiplicativo - mantiene log
                'cat_meta_l2_leaf_reg': trial.suggest_float('cat_meta_l2_leaf_reg', 0.1, 10.0, log=True),  # Regularización - mantiene log
                'cat_meta_early_stopping': trial.suggest_int('cat_meta_early_stopping', 20, 200),  # Lineal - quita log
                'cat_meta_bootstrap_type': trial.suggest_categorical('cat_meta_bootstrap_type', ['Bayesian', 'Bernoulli', 'MVS']),
                'cat_meta_bagging_temperature': trial.suggest_float('cat_meta_bagging_temperature', 0, 10),
                'cat_meta_subsample': trial.suggest_float('cat_meta_subsample', 0.5, 1),
                
                # Períodos y estadísticas en el formato esperado
                # Definir todos los períodos posibles de una vez
                'max_main_periods': trial.suggest_int('max_main_periods', 3, 15, log=True),
                'max_meta_periods': trial.suggest_int('max_meta_periods', 1, 3, log=True),
                'max_main_stats': trial.suggest_int('max_main_stats', 1, 5, log=True),
                'max_meta_stats': trial.suggest_int('max_meta_stats', 1, 3, log=True),
            }

            # Main periods - sugerir como array discreto
            params['periods_main'] = tuple(sorted(set(
                trial.suggest_int('periods_main', 4, 100, log=True)
                for _ in range(params['max_main_periods'])
            )))

            # Meta periods - sugerir como array discreto  
            params['periods_meta'] = tuple(sorted(set(
                trial.suggest_int('periods_meta', 4, 8, log=True)
                for _ in range(params['max_meta_periods'])
            )))

            # Sugerir estadísticas como arrays completos
            params['stats_main'] = tuple(set(
                trial.suggest_categorical('stats_main', all_stats)
                for _ in range(params['max_main_stats'])
            ))

            params['stats_meta'] = tuple(set(
                trial.suggest_categorical('stats_meta', all_stats)
                for _ in range(params['max_meta_stats'])
            ))

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
    
    def fit_final_models(self, main_data: pd.DataFrame,
                        meta_data: pd.DataFrame,
                        ds_train: pd.DataFrame,
                        ds_test: pd.DataFrame,
                        hp: Dict[str, Any]) -> tuple[tuple[float, float], object, object]:
        """Ajusta los modelos finales."""
        try:
            # ---------- 1) main model_main ----------
            # Get feature columns and rename them to follow f%d pattern
            main_feature_cols = main_data.columns[main_data.columns.str.contains('_feature') & ~main_data.columns.str.contains('_meta_feature')]
            X_main = main_data[main_feature_cols]
            X_main.columns = [f'f{i}' for i in range(len(main_feature_cols))]
            y_main = main_data['labels_main'].astype('int16')
            # División de datos para el modelo principal según fechas
            X_train_main, X_val_main, y_train_main, y_val_main = train_test_split(
                X_main, y_main, 
                test_size=0.3,
                shuffle=True
            )
            
            # ── descartar clusters problemáticos ────────────────────────────
            if len(y_train_main.value_counts()) < 2 or len(y_val_main.value_counts()) < 2:
                return None, None
            
            # ---------- 2) meta‑modelo ----------
            meta_feature_cols = meta_data.columns[meta_data.columns.str.contains('_meta_feature')]
            X_meta = meta_data[meta_feature_cols]
            X_meta.columns = [f'f{i}' for i in range(len(meta_feature_cols))]
            y_meta = meta_data['labels_meta'].astype('int16')
            
            # División de datos para el modelo principal según fechas
            X_train_meta, X_val_meta, y_train_meta, y_val_meta = train_test_split(
                X_meta, y_meta, 
                test_size=0.3,
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
                bootstrap_type=hp['cat_main_bootstrap_type'],
                eval_metric='Accuracy',
                store_all_simple_ctr=False,
                thread_count=self.n_jobs,
                task_type='CPU',
                verbose=False,
            )
            
            # Añadir parámetros específicos según el tipo de bootstrap
            if hp['cat_main_bootstrap_type'] == 'Bayesian':
                cat_main_params['bagging_temperature'] = hp['cat_main_bagging_temperature']
            elif hp['cat_main_bootstrap_type'] == 'Bernoulli':
                cat_main_params['subsample'] = hp['cat_main_subsample']

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
                bootstrap_type=hp['cat_meta_bootstrap_type'],
                eval_metric='F1',
                store_all_simple_ctr=False,
                thread_count=self.n_jobs,
                task_type='CPU',
                verbose=False,
            )
            
            # Añadir parámetros específicos según el tipo de bootstrap
            if hp['cat_meta_bootstrap_type'] == 'Bayesian':
                cat_meta_params['bagging_temperature'] = hp['cat_meta_bagging_temperature']
            elif hp['cat_meta_bootstrap_type'] == 'Bernoulli':
                cat_meta_params['subsample'] = hp['cat_meta_subsample']

            model_meta = CatBoostClassifier(**cat_meta_params)
            model_meta.fit(X_train_meta, y_train_meta, 
                           eval_set=Pool(X_val_meta, y_val_meta), 
                           early_stopping_rounds=hp['cat_meta_early_stopping'],
                           use_best_model=True,
                           verbose=False
            )

            # ── evaluación ───────────────────────────────────────────────
            ds_train_eval = ds_train.sample(n=len(ds_test), replace=False)
            r2_ins = robust_oos_score_one_direction(
                dataset=ds_train_eval,
                model_main=model_main,
                model_meta=model_meta,
                direction=self.direction,
                n_sim=100,
                agg="q05"
            )
            r2_oos = robust_oos_score_one_direction(
                dataset=ds_test,
                model_main=model_main,
                model_meta=model_meta,
                direction=self.direction,
                n_sim=100,
                agg="q05"
            )

            # Manejar valores inválidos
            if not np.isfinite(r2_ins) or not np.isfinite(r2_oos):
                r2_ins = -1.0
                r2_oos = -1.0

            return (r2_ins, r2_oos), (model_main, model_meta)
        
        except Exception as e:
            print(f"Error en función de entrenamiento y test: {str(e)}")
            return None, None
        finally:
            _ONNX_CACHE.clear()
            gc.collect()

    def search_markov(self, trial: optuna.Trial, study: optuna.study.Study) -> tuple[float, float]:
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
            # Mostrar el número de clusters encontrados
            cluster_sizes = ds_train['labels_meta'].value_counts()
            print(f"Clusters encontrados: {cluster_sizes}")
            scores, models = self._evaluate_clusters(ds_train, ds_test, hp, trial, study)
            if scores is None or models is None:
                return -1.0, -1.0
            else:
                trial.set_user_attr('models', models)
                trial.set_user_attr('scores', scores)

            return scores
            
        except Exception as e:
            print(f"Error en search_markov: {str(e)}")
            return -1.0, -1.0

    def search_clusters(self, trial: optuna.Trial, study: optuna.study.Study) -> tuple[float, float]:
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
            
            scores, models = self._evaluate_clusters(ds_train, ds_test, hp, trial, study)
            if scores is None or models is None:
                return -1.0, -1.0
            else:
                trial.set_user_attr('models', models)
                trial.set_user_attr('scores', scores)

            return scores
        
        except Exception as e:
            print(f"Error en search_clusters: {str(e)}")
            return -1.0, -1.0

    # =========================================================================
    # Métodos auxiliares
    # =========================================================================
    
    # ---------------------------------------------------------------------
    def get_train_test_data(self, hp):
        """Obtiene los datos de entrenamiento y prueba."""
        try:
            if hp is None:
                print("⚠️ ERROR: Parámetros inválidos")
                return None, None

            # Asegurarnos de que tenemos todos los datos necesarios
            # Convertir el diccionario en una tupla de tuplas para que sea hasheable
            hp_tuple = tuple(sorted(hp.items()))
            full_ds = StrategySearcher._cached_features(
                StrategySearcher._BASE_DF_ID,
                min(self.train_start, self.test_start),
                max(self.train_end, self.test_end),
                hp_tuple,
            )
            if full_ds is None or full_ds.empty:
                print("⚠️ ERROR: Dataset vacío")
                return None, None
            
            # Verificar calidad de los datasets
            feature_cols = full_ds.columns[full_ds.columns.str.contains('feature')]
            if feature_cols.empty:
                print("⚠️ ERROR: No hay columnas de características")
                return None, None
                
            if self.check_constant_features(full_ds[feature_cols].to_numpy('float32'), feature_cols):
                print("⚠️ ERROR: Características constantes detectadas")
                return None, None
            
            # Obtener datasets de entrenamiento y prueba
            test_mask = (full_ds.index >= self.test_start) & (full_ds.index <= self.test_end)
            train_mask = (full_ds.index >= self.train_start) & (full_ds.index <= self.train_end)
            
            # Verificar que hay datos en ambos períodos
            if not any(test_mask) or not any(train_mask):
                print("⚠️ ERROR: Períodos sin datos")
                return None, None
            
            # Excluir el período de test del train si hay solapamiento
            if (self.test_start <= self.train_end and self.test_end >= self.train_start):
                train_mask = train_mask & ~test_mask
            
            train_data = full_ds[train_mask]
            test_data = full_ds[test_mask]
            
            # Verificar tamaño mínimo de datos
            if len(train_data) < 100 or len(test_data) < 50:
                print("⚠️ ERROR: Datasets demasiado pequeños")
                return None, None
            
            return train_data, test_data
            
        except Exception as e:
            print(f"⚠️ ERROR en get_train_test_data: {str(e)}")
            return None, None