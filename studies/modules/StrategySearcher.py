import gc
import math
import time
import numpy as np
import pandas as pd
from datetime import datetime
from time import perf_counter
from typing import Dict, Any
import optuna
from optuna.pruners import HyperbandPruner
from optuna.integration import CatBoostPruningCallback
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
                        hkey: tuple):
        """Obtiene características del DataFrame con caché LRU.
        
        El caché almacena hasta 128 resultados diferentes, eliminando automáticamente
        los más antiguos cuando se alcanza el límite. Cada resultado es un DataFrame
        completo con las características calculadas.
        
        Args:
            df_id: ID del DataFrame base
            start: Fecha de inicio
            end: Fecha de fin
            hkey: Tupla con los hiperparámetros que afectan a las características
            
        Returns:
            pd.DataFrame: DataFrame con las características calculadas
        """
        hp_subset = dict(hkey)
        hp_full = {
            'periods_main': list(hp_subset['periods_main']),
            'periods_meta': list(hp_subset['periods_meta']),
            'stats_main'  : list(hp_subset['stats_main']),
            'stats_meta'  : list(hp_subset['stats_meta']),
        }
        return get_features(StrategySearcher._FEATURE_CACHE[df_id].loc[start:end].copy(), hp_full)
    
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
        self.best_models = []

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
                study = optuna.create_study(
                    directions=['maximize', 'maximize'],  # Maximizar ambos scores
                    pruner=HyperbandPruner(
                        max_resource='auto'
                    ),
                    sampler=optuna.samplers.TPESampler(
                        n_startup_trials=int(np.sqrt(self.n_trials)),
                        multivariate=True
                    )
                )

                t0 = perf_counter()
                def log_trial(study, trial):
                    # Obtener el Pareto front
                    pareto_front = study.best_trials
                    if pareto_front:
                        # Calcular el mejor balance entre in-sample y out-of-sample
                        best_trial = max(pareto_front, 
                                       key=lambda t: min(t.values[0], t.values[1]))
                        best_str = f"ins={best_trial.values[0]:.4f} oos={best_trial.values[1]:.4f}"
                    else:
                        best_str = "nan"
                    
                    elapsed = perf_counter() - t0
                    n_done = trial.number + 1
                    avg_time = elapsed / n_done
                    print(
                        f"[{self.tag}] modelo {i} "
                        f"trial {n_done}/{self.n_trials} "
                        f"best={best_str} "
                        f"avg={avg_time:6.2f}s",
                        flush=True,
                    )

                study.optimize(
                    lambda t: search_func(t, study),
                    n_trials=self.n_trials,
                    gc_after_trial=True,
                    show_progress_bar=False,
                    callbacks=[log_trial],
                )

                # Seleccionar el mejor trial del Pareto front
                pareto_front = study.best_trials
                if not pareto_front:
                    print(f"⚠️ ERROR: No se encontraron trials válidos")
                    continue

                # Seleccionar el trial que mejor balancea in-sample y out-of-sample
                best_trial = max(pareto_front, 
                               key=lambda t: min(t.values[0], t.values[1]))

                # Verificar y exportar el mejor modelo
                best_models = best_trial.user_attrs.get("best_models", [])
                if not (best_models and len(best_models) == 2 and all(best_models)):
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
                    "best_models": best_models,
                    "best_model_seed": model_seed,
                    "best_scores": best_trial.values,
                    "best_periods_main": best_trial.user_attrs.get("best_periods_main"),
                    "best_periods_meta": best_trial.user_attrs.get("best_periods_meta"),
                    "best_stats_main": best_trial.user_attrs.get("best_stats_main"),
                    "best_stats_meta": best_trial.user_attrs.get("best_stats_meta"),
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
            if ds_train is None or ds_test is None:
                print("⚠️ ERROR: Datasets inválidos")
                return -1.0, -1.0
                
            if hp is None:
                print("⚠️ ERROR: Parámetros inválidos")
                return -1.0, -1.0

            best_scores = (-math.inf, -math.inf)
            cluster_sizes = ds_train['labels_meta'].value_counts()

            # Verificar que hay clusters
            if cluster_sizes.empty:
                print("⚠️ ERROR: No hay clusters")
                return -1.0, -1.0

            # Filtrar el cluster -1 (inválido) si existe
            if -1 in cluster_sizes.index:
                cluster_sizes = cluster_sizes.drop(-1)
                if cluster_sizes.empty:
                    print("⚠️ ERROR: Solo hay clusters inválidos")
                    return -1.0, -1.0
        
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
                res = self.fit_final_models(
                    main_data, meta_data, ds_train, ds_test, hp.copy(), trial
                )
                if res is None:
                    print(f"⚠️ ERROR: Fit fallido para el cluster {clust}")
                    continue

                scores, m_main, m_meta = res

                # Verificar scores
                if not all(np.isfinite(scores)):
                    print(f"⚠️ ERROR: Scores no finitos para el cluster {clust}")
                    continue

                # no guardes modelos si el score no mejora
                if min(scores) > min(best_scores):
                    best_scores = scores
                    # guarda métrica + modelos ganadores
                    self.save_best_trial(
                        trial, study,
                        metrics={"score": min(scores)},
                        models=[m_main, m_meta],
                    )
                else:
                    # no mejora → descartar enseguida
                    del m_main, m_meta
                    gc.collect()

            # Verificar que encontramos algún cluster válido
            if best_scores == (-math.inf, -math.inf):
                print("⚠️ ERROR: No se encontraron clusters válidos")
                return -1.0, -1.0

            return best_scores
            
        except Exception as e:
            print(f"⚠️ ERROR en _evaluate_clusters: {str(e)}")
            return -1.0, -1.0
    
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
            # Definir todos los períodos posibles de una vez
            max_main_periods = trial.suggest_int('max_main_periods', 3, 20, log=True)
            max_meta_periods = trial.suggest_int('max_meta_periods', 1, 5, log=True)
            
            # Sugerir todos los períodos main de una vez y asegurar que sean únicos
            main_periods = sorted(list(set([
                trial.suggest_int('period_main_0', 2, 200, log=True),
                trial.suggest_int('period_main_1', 2, 200, log=True),
                trial.suggest_int('period_main_2', 2, 200, log=True),
                trial.suggest_int('period_main_3', 2, 200, log=True),
                trial.suggest_int('period_main_4', 2, 200, log=True),
                trial.suggest_int('period_main_5', 2, 200, log=True),
                trial.suggest_int('period_main_6', 2, 200, log=True),
                trial.suggest_int('period_main_7', 2, 200, log=True),
                trial.suggest_int('period_main_8', 2, 200, log=True),
                trial.suggest_int('period_main_9', 2, 200, log=True),
                trial.suggest_int('period_main_10', 2, 200, log=True),
                trial.suggest_int('period_main_11', 2, 200, log=True),
                trial.suggest_int('period_main_12', 2, 200, log=True),
                trial.suggest_int('period_main_13', 2, 200, log=True),
                trial.suggest_int('period_main_14', 2, 200, log=True),
                trial.suggest_int('period_main_15', 2, 200, log=True),
                trial.suggest_int('period_main_16', 2, 200, log=True),
                trial.suggest_int('period_main_17', 2, 200, log=True),
                trial.suggest_int('period_main_18', 2, 200, log=True),
                trial.suggest_int('period_main_19', 2, 200, log=True),
            ])))[:max_main_periods]
            
            # Sugerir todos los períodos meta de una vez y asegurar que sean únicos
            meta_periods = sorted(list(set([
                trial.suggest_int('period_meta_0', 2, 5, log=True),
                trial.suggest_int('period_meta_1', 2, 5, log=True),
                trial.suggest_int('period_meta_2', 2, 5, log=True),
                trial.suggest_int('period_meta_3', 2, 5, log=True),
                trial.suggest_int('period_meta_4', 2, 5, log=True),
            ])))[:max_meta_periods]

            # Verificar períodos main
            if not main_periods or len(main_periods) < 3:
                print("⚠️ ERROR: Períodos main insuficientes")
                return None
            
            # Verificar períodos meta
            if not meta_periods or len(meta_periods) < 1:
                print("⚠️ ERROR: Períodos meta insuficientes")
                return None

            # Parámetros base comunes
            params = {
                'markup': trial.suggest_float("markup", 0.1, 1.0, log=True),
                'label_max': trial.suggest_int('label_max', 2, 30, log=True),
                'atr_period': trial.suggest_int('atr_period', 5, 100, log=True),
                
                # Parámetros de CatBoost main
                'cat_main_iterations': trial.suggest_int('cat_main_iterations', 100, 1000, log=True),
                'cat_main_depth': trial.suggest_int('cat_main_depth', 3, 10, log=True),
                'cat_main_learning_rate': trial.suggest_float('cat_main_learning_rate', 0.01, 0.3, log=True),
                'cat_main_l2_leaf_reg': trial.suggest_float('cat_main_l2_leaf_reg', 0.1, 10.0, log=True),
                'cat_main_early_stopping': trial.suggest_int('cat_main_early_stopping', 20, 200, log=True),
                'cat_main_bootstrap_type': trial.suggest_categorical('cat_main_bootstrap_type', ['Bayesian', 'Bernoulli', 'MVS']),
                'cat_main_bagging_temperature': trial.suggest_float('cat_main_bagging_temperature', 0, 10),
                'cat_main_subsample': trial.suggest_float('cat_main_subsample', 0.1, 1),
                
                # Parámetros de CatBoost meta
                'cat_meta_iterations': trial.suggest_int('cat_meta_iterations', 100, 1000, log=True),
                'cat_meta_depth': trial.suggest_int('cat_meta_depth', 3, 10, log=True),
                'cat_meta_learning_rate': trial.suggest_float('cat_meta_learning_rate', 0.01, 0.3, log=True),
                'cat_meta_l2_leaf_reg': trial.suggest_float('cat_meta_l2_leaf_reg', 0.1, 10.0, log=True),
                'cat_meta_early_stopping': trial.suggest_int('cat_meta_early_stopping', 20, 200, log=True),
                'cat_meta_bootstrap_type': trial.suggest_categorical('cat_meta_bootstrap_type', ['Bayesian', 'Bernoulli', 'MVS']),
                'cat_meta_bagging_temperature': trial.suggest_float('cat_meta_bagging_temperature', 0, 10),
                'cat_meta_subsample': trial.suggest_float('cat_meta_subsample', 0.1, 1),
                
                # Períodos
                'max_main_periods': max_main_periods,
                'max_meta_periods': max_meta_periods,
                'n_periods_main': len(main_periods),
                'n_periods_meta': len(meta_periods),
                'periods_main': main_periods,
                'periods_meta': meta_periods,
            }

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

            # Estadísticas
            stat_choices = [
                "std", "skew", "kurt", "zscore", "range", "mad", "entropy", 
                "slope", "momentum", "fractal", "hurst", "autocorr", "max_dd", 
                "sharpe", "fisher", "chande", "var", "approx_entropy", 
                "eff_ratio", "corr_skew", "jump_vol", "vol_skew", "hurst"
            ]
            
            # Sugerir todas las estadísticas de una vez
            main_stats = [stat for i, stat in enumerate(stat_choices) 
                         if trial.suggest_categorical(f'main_stat_{i}', [0, 1]) == 1]
            meta_stats = [stat for i, stat in enumerate(stat_choices) 
                         if trial.suggest_categorical(f'meta_stat_{i}', [0, 1]) == 1]
            
            # Verificar estadísticas
            if not main_stats or not meta_stats:
                print("⚠️ ERROR: Estadísticas insuficientes")
                return None
            
            params["stats_main"] = main_stats[:params['n_periods_main']]
            params["stats_meta"] = meta_stats[:params['n_periods_meta']]

            return params
            
        except Exception as e:
            print(f"⚠️ ERROR en suggest_all_params: {str(e)}")
            return None

    def save_best_trial(
            self,
            trial: optuna.Trial,
            study: optuna.study.Study,
            *,
            metrics: dict,
            models: list | None = None,
    ):
        """Registra métricas y conserva solo los modelos del mejor score.
        
        Args:
            trial: Trial actual de Optuna
            study: Estudio de Optuna
            metrics: Diccionario con métricas a guardar
            models: Lista de modelos [model_main, model_meta] o None
        """
        try:
            # Verificar que las métricas sean válidas
            if not metrics or not isinstance(metrics, dict):
                print("⚠️ ERROR: Métricas inválidas")
                return

            # Verificar que las métricas sean finitas y guardarlas en el trial para el logging
            for k, v in metrics.items():
                if not np.isfinite(v):
                    print(f"⚠️ ERROR: Métrica {k} no es finita")
                    return
                trial.set_user_attr(k, v)

            # Verificar que los modelos sean válidos
            if models is not None:
                if not isinstance(models, list) or len(models) != 2:
                    print("⚠️ ERROR: Formato de modelos inválido")
                    return
                if not all(models):
                    print("⚠️ ERROR: Algunos modelos son None")
                    return

                # Si este trial es el mejor hasta ahora, guardar todo en el estudio
                if study is not None:
                    pareto_front = study.best_trials
                    if pareto_front:
                        best_trial = max(pareto_front, 
                                      key=lambda t: min(t.values[0], t.values[1]))
                        if trial.number == best_trial.number:
                            # Liberar modelos anteriores si existen
                            old_models = study.user_attrs.get("best_models")
                            if old_models:
                                try:
                                    old_models.clear()
                                except Exception as e:
                                    print(f"⚠️ ERROR al limpiar modelos antiguos: {str(e)}")
                                finally:
                                    gc.collect()

                            # Guardar en el estudio todos los atributos del mejor trial
                            try:
                                study.set_user_attr("best_models", models)
                                study.set_user_attr("best_metrics", metrics)
                                study.set_user_attr("best_periods_main", trial.params['periods_main'])
                                study.set_user_attr("best_periods_meta", trial.params['periods_meta'])
                                study.set_user_attr("best_stats_main", trial.params['stats_main'])
                                study.set_user_attr("best_stats_meta", trial.params['stats_meta'])
                                study.set_user_attr("best_trial_number", trial.number)
                            except Exception as e:
                                print(f"⚠️ ERROR al guardar atributos: {str(e)}")
                                if "best_models" in study.user_attrs:
                                    del study.user_attrs["best_models"]
                                gc.collect()
                        
        except Exception as e:
            print(f"⚠️ ERROR en save_best_trial: {str(e)}")
            gc.collect()
    
    def fit_final_models(self, main_data: pd.DataFrame,
                        meta_data: pd.DataFrame,
                        ds_train: pd.DataFrame,
                        ds_test: pd.DataFrame,
                        hp: Dict[str, Any],
                        trial: optuna.Trial) -> tuple[tuple[float, float], object, object]:
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
                return None
            
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
                return None

            # Main model
            cat_main_params = dict(
                iterations=hp['cat_main_iterations'],
                depth=hp['cat_main_depth'],
                learning_rate=hp['cat_main_learning_rate'],
                l2_leaf_reg=hp['cat_main_l2_leaf_reg'],
                bootstrap_type=hp['cat_main_bootstrap_type'],
                eval_metric='F1',
                store_all_simple_ctr=False,
                verbose=False,
                thread_count=self.n_jobs,
                task_type='CPU'
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
                verbose=False,
                thread_count=self.n_jobs,
                task_type='CPU'
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
            r2_ins = test_model_one_direction(
                dataset=ds_train_eval,
                model_main=model_main,
                model_meta=model_meta,
                direction=self.direction,
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

            return (r2_ins, r2_oos), model_main, model_meta
        except Exception as e:
            print(f"Error en fit_final_models: {str(e)}")
            return (-1.0, -1.0), None, None
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
            
            regime_counts = ds_train['labels_meta'].value_counts()
            if len(regime_counts) < 2:
                return -1.0, -1.0
            
            # Evaluar los clusters y obtener los scores
            scores = self._evaluate_clusters(ds_train, ds_test, hp, trial, study)
            if scores is None:
                return -1.0, -1.0
                
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
            
            return self._evaluate_clusters(ds_train, ds_test, hp, trial, study)
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

            hkey = tuple(sorted({
                'periods_main': tuple(hp['periods_main']),
                'periods_meta': tuple(hp['periods_meta']),
                'stats_main'  : tuple(hp['stats_main']),
                'stats_meta'  : tuple(hp['stats_meta']),
            }.items()))

            # Asegurarnos de que tenemos todos los datos necesarios
            full_ds = StrategySearcher._cached_features(
                StrategySearcher._BASE_DF_ID,
                min(self.train_start, self.test_start),
                max(self.train_end, self.test_end),
                hkey,
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