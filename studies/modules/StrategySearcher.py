import gc
import copy
import math
import time
import numpy as np
import pandas as pd
from datetime import datetime
from time import perf_counter
from typing import Dict, Any, Tuple, List, Optional
import optuna
from optuna.pruners import HyperbandPruner
from sklearn.ensemble import VotingClassifier
from modules.export_lib import XGBWithEval
from modules.export_lib import CatWithEval
from optuna.integration import CatBoostPruningCallback
from optuna.integration import XGBoostPruningCallback
from sklearn import set_config
from sklearn.model_selection import train_test_split
from functools import lru_cache, partial
from weakref import WeakValueDictionary

from modules.labeling_lib import (
    get_prices, get_features, get_labels_one_direction,
    sliding_window_clustering, clustering_simple,
    markov_regime_switching_simple, markov_regime_switching_sliding
)
from modules.tester_lib import robust_oos_score_one_direction, _ONNX_CACHE
from modules.export_lib import (
    export_model_to_ONNX, XGBWithEval, CatWithEval
)

class StrategySearcher:
    """Clase unificada para búsqueda de estrategias de trading.
    
    Esta clase implementa diferentes métodos de búsqueda de estrategias de trading
    utilizando técnicas de machine learning y optimización bayesiana.
    
    Attributes:
        base_hp (dict): Hiperparámetros base para la búsqueda
        base_df (pd.DataFrame): DataFrame con datos históricos
        all_results (dict): Resultados de todas las búsquedas realizadas
        best_models (list): Lista de los mejores modelos encontrados
        model_range (list): Rango de modelos a optimizar
    """
    
    _DF_REGISTRY = WeakValueDictionary()

    @staticmethod
    @lru_cache()
    def _cached_features(df_id: int,
                        start: datetime,
                        end: datetime,
                        hkey: tuple):
        
        hp_subset = dict(hkey)
        hp_full = {
            'periods_main': list(hp_subset['periods_main']),
            'periods_meta': list(hp_subset['periods_meta']),
            'stats_main'  : list(hp_subset['stats_main']),
            'stats_meta'  : list(hp_subset['stats_meta']),
        }
        df = StrategySearcher._DF_REGISTRY[df_id]
        return get_features(df.loc[start:end].copy(), hp_full)
    
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
        model_seed: Optional[int] = None,
        search_type: str = 'clusters',
        subtype: str = 'simple',
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
            model_seed: Semilla para reproducibilidad
        """
        self.search_type = search_type
        self.subtype = subtype
        self.n_trials = n_trials
        self.n_models = n_models
        self.n_jobs = n_jobs
        self.tag = tag
        self.base_hp = {
            'symbol': symbol,
            'timeframe': timeframe,
            'direction': direction,
            'train_start': train_start,
            'train_end': train_end,
            'test_start': test_start,
            'test_end': test_end,
            'model_seed': model_seed,
            'models_export_path': models_export_path,
            'include_export_path': include_export_path,
            'history_path': history_path,
            'periods_main': [],
            'periods_meta': [],
            'stats_main': [],
            'stats_meta': [],
            'best_models': [],
            'markup': 0.20,
            'label_min': 1,
            'label_max': 15,
            'n_clusters': 30,
        }
        
        # Cargar datos históricos
        self.base_df = get_prices(self.base_hp)
        StrategySearcher._DF_REGISTRY[id(self.base_df)] = self.base_df
        
        # Configuración de sklearn y optuna
        set_config(enable_metadata_routing=True, skip_parameter_validation=True)
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        # Resultados
        self.all_results = {}
        self.best_models = []

    # =========================================================================
    # Métodos de búsqueda principales
    # =========================================================================

    def run_search(self) -> None:
        """Ejecuta la búsqueda de estrategias.
        """
        search_funcs = {
            'clusters': self.search_clusters,
            'markov': self.search_markov,
        }
        
        if self.search_type not in search_funcs:
            raise ValueError(f"Tipo de búsqueda no válido: {self.search_type}")
            
        search_func = search_funcs[self.search_type]
        
        for i in range(self.n_models):
            try:
                # Inicializar estudio de Optuna
                study = optuna.create_study(
                    direction='maximize',
                    pruner=HyperbandPruner(),
                    sampler=optuna.samplers.TPESampler(
                        n_startup_trials=int(np.sqrt(self.n_trials)),
                    )
                )

                t0 = perf_counter()
                def log_trial(study, trial):
                    best = study.best_value
                    best_str  = f"{best:.4f}" if best is not None else "nan"
                    elapsed   = perf_counter() - t0
                    n_done    = trial.number + 1
                    avg_time  = elapsed / n_done
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

                # Verificar y exportar el mejor modelo
                best_models = study.user_attrs.get("best_models", [])
                if not (best_models and len(best_models) == 2 and all(best_models)):
                    print("⚠️  Error: best_models incorrecto")
                    continue

                export_params = self.base_hp.copy()
                export_params.update({
                    "best_trial": study.user_attrs["best_trial_number"],
                    "best_score": study.user_attrs["best_score"],
                    "best_periods_main": study.user_attrs.get("best_periods_main"),
                    "best_periods_meta": study.user_attrs.get("best_periods_meta"),
                    "best_stats_main": study.user_attrs.get("best_stats_main"),
                    "best_stats_meta": study.user_attrs.get("best_stats_meta"),
                    "best_models": best_models,
                    "search_type": self.search_type,
                    "subtype": self.subtype,
                })
                
                export_model_to_ONNX(**export_params)
                
                self.show_best_summary(study)
                
                model_results = {
                    "r2_ins": study.user_attrs.get("best_metrics", {}).get("r2_ins", float("nan")),
                    "r2_oos": study.user_attrs.get("best_metrics", {}).get("r2_oos", float("nan")),
                    "score": study.user_attrs.get("best_metrics", {}).get("score", float("nan")),
                }
                
                self.best_models.append((i, model_results))
                
                self.all_results[f"model_{i}"] = {
                    "success": True,
                    "r2_ins": model_results["r2_ins"],
                    "r2_oos": model_results["r2_oos"],
                    "score": model_results["score"]
                }
                
            except Exception as e:
                import traceback
                print(f"\nError procesando modelo {i}:")
                print(f"Error: {str(e)}")
                print("Traceback:")
                print(traceback.format_exc())
                
                self.all_results[f"model_{i}"] = {
                    "success": False,
                    "error": str(e)
                }
                continue
        
        self._print_summary()

    # =========================================================================
    # Métodos de visualización y reportes
    # =========================================================================
    
    def show_best_summary(self, study: optuna.study.Study) -> None:
        """Muestra resumen del mejor trial encontrado.
        
        Args:
            study: Estudio de Optuna con los resultados
        """
        m = study.user_attrs.get("best_metrics", {})
        f2 = m.get("r2_ins", float("nan"))
        b2 = m.get("r2_oos", float("nan"))
        c2 = m.get("score", float("nan"))
        best_trial = study.user_attrs.get("best_trial_number", "-")

        lines = [
            "┌" + "─" * 55 + "┐",
            f"│  MODELO {self.base_hp['model_seed']} TRIAL ÓPTIMO #{best_trial}",
            "├" + "─" * 55 + "┤",
            f"│  R² Forward : {f2:10.4f}  │  R² Backward : {b2:10.4f} │",
            f"│  Combined     : {c2:10.4f}                            │",
            "├" + "─" * 55 + "┤"
        ]
        print("\n".join(lines))

    def _print_summary(self) -> None:
        """Imprime un resumen final de la optimización."""
        print("\n" + "="*50)
        print(f"RESUMEN DE OPTIMIZACIÓN {self.base_hp['symbol']}/{self.base_hp['timeframe']}")
        print("="*50)
        
        successful_models = [info for model_key, info in self.all_results.items() if info.get("success", False)]
        print(f"Modelos completados exitosamente: {len(successful_models)}/{self.n_models}")
        
        if successful_models:
            # Calcular estadísticas globales
            r2_ins_scores = [info["r2_ins"] for info in successful_models]
            r2_oos_scores = [info["r2_oos"] for info in successful_models]
            scores = [info["score"] for info in successful_models]
            
            print(f"\nEstadísticas de rendimiento:")
            print(f"  R2 INS promedio: {np.mean(r2_ins_scores):.4f} ± {np.std(r2_ins_scores):.4f}")
            print(f"  R2 OOS promedio: {np.mean(r2_oos_scores):.4f} ± {np.std(r2_oos_scores):.4f}")
            print(f"  Puntuación combinada promedio: {np.mean(scores):.4f} ± {np.std(scores):.4f}")

            # Identificar el mejor modelo global
            successful = [(k, v) for k, v in self.all_results.items() if v.get("success", False)]
            scores = [v["score"] for _, v in successful]
            best_model_key, best_info = successful[int(np.argmax(scores))]
            
            print(f"\nMejor modelo global: {best_model_key}")
            print(f"  R2 INS: {best_info['r2_ins']:.4f}")
            print(f"  R2 OOS: {best_info['r2_oos']:.4f}")
            print(f"  Puntuación combinada: {best_info['score']:.4f}")
        
        print("\nProceso de optimización completado.")

    # =========================================================================
    # Métodos de búsqueda específicos
    # =========================================================================
    
    def _evaluate_clusters(self, ds_train: pd.DataFrame, ds_test: pd.DataFrame, hp: Dict[str, Any], trial: optuna.Trial, study: optuna.study.Study) -> float:
        """Función helper para evaluar clusters y entrenar modelos.
        
        Args:
            ds_train: Dataset de entrenamiento con labels_meta
            ds_test: Dataset de prueba
            hp: Hiperparámetros
            trial: Trial actual de Optuna
            study: Estudio de Optuna
            
        Returns:
            float: Mejor puntuación encontrada
        """
        best_score = -math.inf
        cluster_sizes = ds_train['labels_meta'].value_counts()

        # Filtrar el cluster -1 (inválido) si existe
        if -1 in cluster_sizes.index:
            cluster_sizes = cluster_sizes.drop(-1)
        
        # Evaluar cada cluster
        for clust in cluster_sizes.index:
            # Main data
            main_cols = [c for c in ds_train.columns if '_feature' in c and '_meta_feature' not in c]
            meta_cols = [c for c in ds_train.columns if '_meta_feature' in c]

            ohlc_cols = ["open", "high", "low", "close"]
            present   = [c for c in ohlc_cols if c in ds_train.columns]

            main_data = ds_train.loc[
                ds_train["labels_meta"] == clust,
                present + main_cols
            ].copy()
            if len(main_data) <= hp['label_max']:
               continue

            main_data = get_labels_one_direction(
                main_data,
                markup=hp['markup'],
                max_val=hp['label_max'],
                direction=hp['direction'],
                atr_period=hp['atr_period'],
                deterministic=True
            )
            if (main_data['labels_main'].value_counts() < 2).any():
                continue

            # Meta data
            meta_data = ds_train[meta_cols].copy()
            meta_data['labels_meta'] = (ds_train['labels_meta'] == clust).astype(int)
            if (meta_data['labels_meta'].value_counts() < 2).any():
                continue

            # ── Evaluación en ambos períodos ──────────────────────────────
            res = self.fit_final_models(
                main_data, meta_data, ds_train, ds_test, hp.copy(), trial
            )
            if res is None:
                continue

            metrics, m_main, m_meta = res

            # no guardes modelos si el score no mejora
            if metrics["score"] > best_score and metrics["score"] > -1.0:
                best_score = metrics["score"]
                # guarda métrica + modelos ganadores
                self.save_best_trial(
                    trial, study,
                    metrics=metrics,
                    models=[m_main, m_meta],
                )
            else:
                # no mejora → descartar enseguida
                del m_main, m_meta
                gc.collect()

        return best_score if best_score != -math.inf else -1.0
    
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
            return True
            
        return False
    
    def search_clusters(self, trial: optuna.Trial, study: optuna.study.Study) -> float:
        """Implementa la búsqueda de estrategias usando clustering.
        
        Args:
            trial: Trial actual de Optuna
            study: Estudio de Optuna
            
        Returns:
            float: Mejor puntuación encontrada
        """
        # Parámetros base a optimizar
        hp = self.common_hyper_params(trial)

        # Obtener datos de entrenamiento y prueba
        ds_train, ds_test = self.get_train_test_data(hp)
        if ds_train is None or ds_test is None:
            return -1

        # Parámetros a optimizar
        hp['n_clusters'] = trial.suggest_int('n_clusters', 5, 50, step=5)
        if self.subtype != 'simple':
            hp['window_size'] = trial.suggest_int('window_size', 100, 500, step=50)
            #hp['step'] = trial.suggest_int('step', 1, hp['window_size'], step=10)
        
        # Clustering
        if self.subtype == 'simple':
            ds_train = clustering_simple(
                ds_train,
                min_cluster_size=hp['n_clusters']
            )
        elif self.subtype == 'sliding':
            ds_train = sliding_window_clustering(
                ds_train,
                n_clusters=hp['n_clusters'],
                window_size=hp['window_size'],
                step=hp.get('step', None),
            )
            
        return self._evaluate_clusters(ds_train, ds_test, hp, trial, study)

    def search_markov(self, trial: optuna.Trial, study: optuna.study.Study) -> float:
        """Implementa la búsqueda de estrategias usando modelos markovianos.
        
        Args:
            trial: Trial actual de Optuna
            study: Estudio de Optuna
            
        Returns:
            float: Mejor puntuación encontrada
        """
        # Parámetros base a optimizar
        hp = self.common_hyper_params(trial)

        # Obtener datos de entrenamiento y prueba
        ds_train, ds_test = self.get_train_test_data(hp)
        if ds_train is None or ds_test is None:
            return -1
        
        # Parámetros base a optimizar
        hp['model_type'] = trial.suggest_categorical('model_type', ['GMMHMM', 'HMM', 'VARHMM'])
        hp['n_regimes'] = trial.suggest_int('n_regimes', 3, 15, step=1)
        hp['n_iter'] = trial.suggest_int('n_iter', 10, 500, step=10)
        if self.subtype != 'simple':
            hp['window_size'] = trial.suggest_int('window_size', hp['n_regimes'] * 10, len(ds_train)//10, step=50)
            #hp['step'] = trial.suggest_int('step', max(1, hp['window_size']//10), hp['window_size'], step=10)
        
        # Markov
        if self.subtype == 'simple':
            ds_train = markov_regime_switching_simple(
                ds_train,
                model_type=hp['model_type'],
                n_regimes=hp['n_regimes'],
                n_iter=hp['n_iter']
            )
        elif self.subtype == 'sliding':
            ds_train = markov_regime_switching_sliding(
                ds_train,
                model_type=hp['model_type'],
                n_regimes=hp['n_regimes'],
                n_iter=hp['n_iter'],
                window_size=hp['window_size'],
                step=hp.get('step', None)
            )
            
        return self._evaluate_clusters(ds_train, ds_test, hp, trial, study)

    # =========================================================================
    # Métodos auxiliares
    # =========================================================================
    
    # ---------------------------------------------------------------------
    def get_train_test_data(self, hp):
        hkey = tuple(sorted({
            'periods_main': tuple(hp['periods_main']),
            'periods_meta': tuple(hp['periods_meta']),
            'stats_main'  : tuple(hp['stats_main']),
            'stats_meta'  : tuple(hp['stats_meta']),
        }.items()))

        full_ds = StrategySearcher._cached_features(
            id(self.base_df),
            self.base_hp['train_start'],
            self.base_hp['test_end'],
            hkey,
        )
        
        # Verificar calidad de los datasets
        feature_cols = full_ds.columns[full_ds.columns.str.contains('feature')]
        if self.check_constant_features(full_ds[feature_cols].to_numpy('float32'), feature_cols):
            return None, None
        
        # Obtener datasets de entrenamiento y prueba
        test_mask  = (full_ds.index >= hp["test_start"]) & (full_ds.index <= hp["test_end"])
        train_mask = (full_ds.index >= hp["train_start"]) & (full_ds.index <= hp["train_end"]) & ~test_mask
        return full_ds[train_mask], full_ds[test_mask]
    
    def calc_score(self, fwd: float, bwd: float, eps: float = 1e-9) -> float:
        """Calcula la puntuación combinada.
        
        Args:
            fwd: Puntuación forward
            bwd: Puntuación backward
            eps: Valor epsilon para evitar división por cero
            
        Returns:
            float: Puntuación combinada
        """
        if (fwd is None or bwd is None or
            not np.isfinite(fwd) or not np.isfinite(bwd) or
            fwd <= 0 or bwd <= 0):
            return -1.0
        mean = 0.4 * fwd + 0.6 * bwd
        if fwd < bwd * 0.8:
            mean *= 0.8
        delta  = abs(fwd - bwd) / max(abs(fwd), abs(bwd), eps)
        score  = mean * (1.0 - delta)
        return score
    
    def sample_cat_params(self, trial: optuna.Trial, prefix: str) -> Dict[str, Any]:
        """Muestra parámetros para CatBoost.
        
        Args:
            trial: Trial actual de Optuna
            prefix: Prefijo para los parámetros
            
        Returns:
            Dict[str, Any]: Parámetros de CatBoost
        """
        iterations = trial.suggest_int(f"{prefix}_iterations", 100, 500, step=50)
        depth      = trial.suggest_int(f"{prefix}_depth", 3, 6)
        lr         = trial.suggest_float(f"{prefix}_learning_rate", 0.1, 0.3, log=True)
        l2         = trial.suggest_float(f"{prefix}_l2_leaf_reg", 1.0, 5.0, log=True)
        es_rounds  = trial.suggest_int(f"{prefix}_early_stopping", 50, 100, step=10)
        return {
            f"{prefix}_iterations": iterations,
            f"{prefix}_depth": depth,
            f"{prefix}_learning_rate": lr,
            f"{prefix}_l2_leaf_reg": l2,
            f"{prefix}_early_stopping": es_rounds,
        }
    
    def sample_xgb_params(self, trial: optuna.Trial, prefix: str) -> Dict[str, Any]:
        """Muestra parámetros para XGBoost.
        
        Args:
            trial: Trial actual de Optuna
            prefix: Prefijo para los parámetros
            
        Returns:
            Dict[str, Any]: Parámetros de XGBoost
        """
        n_estimators = trial.suggest_int(f"{prefix}_estimators", 100, 500, step=50)
        max_depth = trial.suggest_int(f"{prefix}_max_depth", 3, 6)
        lr           = trial.suggest_float(f"{prefix}_learning_rate", 0.1, 0.3, log=True)
        reg_lambda   = trial.suggest_float(f"{prefix}_reg_lambda", 1.0, 5.0, log=True)
        es_rounds    = trial.suggest_int(f"{prefix}_early_stopping", 50, 100, step=10)
        return {
            f"{prefix}_estimators": n_estimators,
            f"{prefix}_max_depth": max_depth,
            f"{prefix}_learning_rate": lr,
            f"{prefix}_reg_lambda": reg_lambda,
            f"{prefix}_early_stopping": es_rounds,
        }
    
    def common_hyper_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Obtiene hiperparámetros comunes.
        
        Args:
            trial: Trial actual de Optuna
            
        Returns:
            Dict[str, Any]: Hiperparámetros comunes
        """
        hp = {k: copy.deepcopy(v) for k, v in self.base_hp.items() if k != 'base_df'}
        # Optimización de hiperparámetros comunes
        hp['markup'] = trial.suggest_float("markup", 0.1, 1.0, step=0.1)
        hp['label_max'] = trial.suggest_int('label_max', 2, 6, step=1, log=True)
        hp['atr_period'] = trial.suggest_int('atr_period', 5, 50, step=5)
        hp.update(self.sample_cat_params(trial, "cat_main"))
        hp.update(self.sample_cat_params(trial, "cat_meta"))
        hp.update(self.sample_xgb_params(trial, "xgb_main"))
        hp.update(self.sample_xgb_params(trial, "xgb_meta"))

        # Optimización de períodos para el modelo principal
        n_periods_main = trial.suggest_int('n_periods_main', 5, 15, log=True)
        main_periods = []
        for i in range(n_periods_main):
            period_main = trial.suggest_int(f'period_main_{i}', 2, 200, log=True)
            main_periods.append(period_main)
        main_periods = sorted(list(set(main_periods)))
        hp['periods_main'] = main_periods
        # Selección de estadísticas para el modelo principal
        main_stat_choices = [
            "std", "skew", "kurt", "zscore", "range", "mad", "entropy", 
            "slope", "momentum", "fractal", "hurst", "autocorr", "max_dd", 
            "sharpe", "fisher", "chande", "var", "approx_entropy", 
            "eff_ratio", "corr_skew", "jump_vol", "vol_skew", "hurst"
        ]
        n_main_stats = trial.suggest_int('n_main_stats', 1, 5, log=True)
        selected_main_stats = []
        for i in range(n_main_stats):
            stat = trial.suggest_categorical(f'main_stat_{i}', main_stat_choices)
            selected_main_stats.append(stat)
        selected_main_stats = list(set(selected_main_stats))
        hp["stats_main"] = selected_main_stats
        #print(f"Main features seleccionadas: {hp['stats_main']}")

        # Optimización de períodos para el meta-modelo
        n_periods_meta = 1 # trial.suggest_int('n_periods_meta', 1, 3, log=True)
        meta_periods = []
        for i in range(n_periods_meta):
            period_meta = trial.suggest_int(f'period_meta_{i}', 2, 6, log=True)
            meta_periods.append(period_meta)
        meta_periods = sorted(list(set(meta_periods)))
        hp['periods_meta'] = meta_periods
        # Selección de estadísticas para el meta-modelo
        meta_stat_choices = [
            "std", "skew", "kurt", "zscore", "range", "mad", "entropy", 
            "slope", "momentum", "fractal", "hurst", "autocorr", "max_dd", 
            "sharpe", "fisher", "chande", "var", "approx_entropy", 
            "eff_ratio", "corr_skew", "jump_vol", "vol_skew", "hurst"
        ]
        n_meta_stats = trial.suggest_int('n_meta_stats', 1, 3, log=True)
        selected_meta_stats = []
        for i in range(n_meta_stats):
            stat = trial.suggest_categorical(f'meta_stat_{i}', meta_stat_choices)
            selected_meta_stats.append(stat)
        selected_meta_stats = list(set(selected_meta_stats))
        hp["stats_meta"] = selected_meta_stats
        #print(f"Meta features seleccionadas: {hp['stats_meta']} | Periodo: {hp['periods_meta']}")
        return hp

    def save_best_trial(
            self,
            trial: optuna.Trial,
            study: optuna.study.Study,
            *,
            metrics: dict,
            models: list | None = None,
    ):
        """Registra métricas y conserva solo los modelos del mejor score."""
        # ---- guardar métricas en el trial ---------------------------------
        for k, v in metrics.items():
            trial.set_user_attr(k, v)

        if study is None:
            return

        best = study.user_attrs.get("best_score", -math.inf)
        if metrics["score"] > best and models is not None:
            #      ─── nuevo récord global ────────────────────────────
            # liberar modelos anteriores
            old = study.user_attrs.get("best_models")
            if old:
                old.clear()
                gc.collect()

            study.set_user_attr("best_score", metrics["score"])
            study.set_user_attr("best_models", models)
            study.set_user_attr("best_metrics", metrics)
            study.set_user_attr("best_trial_number", trial.number)
        else:
            # score no mejora -> descartar modelos recibidos
            if models is not None:
                del models[:]
                gc.collect()
    
    def fit_final_models(self, main_data: pd.DataFrame,
                        meta_data: pd.DataFrame,
                        ds_train: pd.DataFrame,
                        ds_test: pd.DataFrame,
                        hp: Dict[str, Any],
                        trial: optuna.Trial) -> tuple[dict, object, object]:
        """Ajusta los modelos finales.
        
        Args:
            main_data: Datos principales
            meta_data: Datos meta
            ds_train: Dataset de entrenamiento
            ds_test: Dataset de prueba
            hp: Diccionario de hiperparámetros
            trial: Trial actual de Optuna
            
        Returns:
            Dict[str, Any]: Hiperparámetros actualizados
        """
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
                test_size=0.2,
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
                test_size=0.2,
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
                early_stopping_rounds=hp['cat_main_early_stopping'],
                eval_metric='Accuracy',
                store_all_simple_ctr=False,
                verbose=False,
                thread_count=self.n_jobs,
                task_type='CPU'
            )
            xgb_main_params = dict(
                n_estimators=hp['xgb_main_estimators'],
                max_depth=hp['xgb_main_max_depth'],
                learning_rate=hp['xgb_main_learning_rate'],
                reg_lambda=hp['xgb_main_reg_lambda'],
                early_stopping_rounds=hp['xgb_main_early_stopping'],
                eval_metric='logloss',
                verbosity=0,
                n_jobs=self.n_jobs,
                tree_method= "gpu_hist",
                device_type="cuda"
            )
            base_main_models = [
                ('catboost', CatWithEval(
                    **cat_main_params,
                    eval_set=[(X_val_main, y_val_main)],
                    callbacks=[CatBoostPruningCallback(trial, "Accuracy")]
                )),
                ('xgboost', XGBWithEval(
                    **xgb_main_params, 
                    eval_set=[(X_val_main, y_val_main)],
                    callbacks=[XGBoostPruningCallback(trial, "validation_0-logloss")]
                )),
            ]
            model_main = VotingClassifier(
                    estimators=base_main_models,
                    voting='soft',
                    flatten_transform=False,
                    n_jobs=1
                )
            # print("training main model...")
            # start_time = time.time()
            model_main.fit(X_train_main, y_train_main)
            # print(f"main model trained in {time.time() - start_time:.2f} seconds")

            # Meta-modelo
            cat_meta_params = dict(
                iterations=hp['cat_meta_iterations'],
                depth=hp['cat_meta_depth'],
                learning_rate=hp['cat_meta_learning_rate'],
                l2_leaf_reg=hp['cat_meta_l2_leaf_reg'],
                early_stopping_rounds=hp['cat_meta_early_stopping'],
                eval_metric='F1',
                store_all_simple_ctr=False,
                verbose=False,
                thread_count=self.n_jobs,
                task_type='CPU'
            )
            xgb_meta_params = dict(
                n_estimators=hp['xgb_meta_estimators'],
                max_depth=hp['xgb_meta_max_depth'],
                learning_rate=hp['xgb_meta_learning_rate'],
                reg_lambda=hp['xgb_meta_reg_lambda'],
                early_stopping_rounds=hp['xgb_meta_early_stopping'],
                eval_metric='logloss',
                verbosity=0,
                verbose_eval=False,
                n_jobs=self.n_jobs,
                tree_method= "gpu_hist",
                device_type="cuda"
            )
            base_meta_models = [
                ('catboost', CatWithEval(
                    **cat_meta_params,
                    eval_set=[(X_val_meta, y_val_meta)],
                    callbacks=[CatBoostPruningCallback(trial, "F1")]
                )),
                ('xgboost', XGBWithEval(
                    **xgb_meta_params, 
                    eval_set=[(X_val_meta, y_val_meta)],
                    callbacks=[XGBoostPruningCallback(trial, "validation_0-logloss")]
                )),
            ]

            model_meta = VotingClassifier(
                    estimators=base_meta_models,
                    voting='soft',
                    flatten_transform=False,
                    n_jobs=1
                )
            # print("training meta model...")
            # start_time = time.time()
            model_meta.fit(X_train_meta, y_train_meta)
            # print(f"meta model trained in {time.time() - start_time:.2f} seconds")

            # ── evaluación ───────────────────────────────────────────────
            test_len = len(ds_test)
            train_indices = np.random.choice(len(ds_train), size=test_len, replace=False)
            ds_train_eval = ds_train.iloc[train_indices].copy()
            
            # print("evaluating in-sample...")
            # start_time = time.time()
            r2_ins = robust_oos_score_one_direction(
                dataset=ds_train_eval,
                model_main=model_main,
                model_meta=model_meta,
                direction=hp['direction'],
                n_sim=100,
                agg="q05"
            )
            # print(f"in-sample score calculated in {time.time() - start_time:.2f} seconds")
            # print("evaluating out-of-sample...")
            # start_time = time.time()
            r2_oos = robust_oos_score_one_direction(
                dataset=ds_test,
                model_main=model_main,
                model_meta=model_meta,
                direction=hp['direction'],
                n_sim=100,
                agg="q05"
            )
            # print(f"out-of-sample score calculated in {time.time() - start_time:.2f} seconds\n")

            metrics = dict(
                r2_ins=r2_ins,
                r2_oos=r2_oos,
                score=self.calc_score(r2_ins, r2_oos),
                stats_main=hp['stats_main'],
                stats_meta=hp['stats_meta'],
                periods_main=hp['periods_main'],
                periods_meta=hp['periods_meta'],
            )
            return metrics, model_main, model_meta
        finally:
            _ONNX_CACHE.clear()
            gc.collect()