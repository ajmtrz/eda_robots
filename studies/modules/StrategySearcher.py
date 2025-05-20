import gc
import copy
import math
import time
import warnings
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
from optuna.integration import LightGBMPruningCallback
from sklearn import set_config
from sklearn.model_selection import train_test_split
from functools import lru_cache, partial
from weakref import WeakValueDictionary

from modules.labeling_lib import (
    get_prices, get_features, get_labels_one_direction,
    sliding_window_clustering, clustering_simple
)
from modules.tester_lib import robust_oos_score_one_direction, _ONNX_CACHE
from modules.export_lib import (
    export_model_to_ONNX, XGBWithEval, CatWithEval, LGBMWithEval
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
        subtype_clustering: str = 'simple',
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
        self.subtype_clustering = subtype_clustering
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
        
        # Configuración de warnings, sklearn y optuna
        warnings.filterwarnings("ignore")
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
            'causal': self.search_causal,
            'filter': self.search_filter
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
    
    def search_clusters(self, trial: optuna.Trial, study: optuna.study.Study) -> float:
        """Implementa la búsqueda de estrategias usando clustering.
        
        Args:
            trial: Trial actual de Optuna
            study: Estudio de Optuna
            
        Returns:
            float: Mejor puntuación encontrada
        """
        best_score = -math.inf
        hp = self.common_hyper_params(trial)

        # Parámetros a optimizar
        hp['markup'] = trial.suggest_float("markup", 0.1, 1.0, step=0.1)
        hp['label_max'] = trial.suggest_int('label_max', 2, 6, step=1, log=True)
        hp['atr_period'] = trial.suggest_int('atr_period', 5, 50, step=5)
        hp['n_clusters'] = trial.suggest_int('n_clusters', 5, 50, step=5)
        if self.subtype_clustering == 'advanced':
            hp['k'] = trial.suggest_int('k', 3, 10, step=1)
        
        # Obtener datos de entrenamiento y prueba
        ds_train, ds_test = self.get_train_test_data(hp)
        
        # Clustering
        # print("clustering...")
        # start_time = time.time()
        if self.subtype_clustering == 'advanced':
            ds_train = sliding_window_clustering(
                ds_train,
                n_clusters=hp['n_clusters'],
                step=hp.get('step', None),
                atr_period=hp['atr_period'],
                k=hp['k']
            )
        else:
            ds_train = clustering_simple(
                ds_train,
                min_cluster_size=hp['n_clusters']
            )
        # print(f"clustering done in {time.time() - start_time:.2f} seconds")
        # Evaluar clusters ordenados por tamaño
        cluster_sizes = ds_train['labels_meta'].value_counts()

        # Filtrar el cluster 0 (inválido) si existe
        if 0 in cluster_sizes.index:
            cluster_sizes = cluster_sizes.drop(0)
        
        # Evaluar cada cluster
        for clust in cluster_sizes.index:
            # Main data
            main_cols = [c for c in ds_train.columns if '_feature' in c and '_meta_feature' not in c]
            meta_cols = [c for c in ds_train.columns if '_meta_feature' in c]

            ohlc_cols = ["open", "high", "low", "close"]          # las que existan
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

        # Devolver la mejor puntuación encontrada
        return best_score if best_score != -math.inf else -1.0

    def search_causal(self, trial: optuna.Trial, study: optuna.study.Study) -> float:
        """Implementa la búsqueda de estrategias usando causalidad.
        
        Args:
            trial: Trial actual de Optuna
            study: Estudio de Optuna
            
        Returns:
            float: Mejor puntuación encontrada
        """
        raise NotImplementedError("Búsqueda causal no implementada aún")

    def search_filter(self, trial: optuna.Trial, study: optuna.study.Study) -> float:
        """Implementa la búsqueda de estrategias usando filtros.
        
        Args:
            trial: Trial actual de Optuna
            study: Estudio de Optuna
            
        Returns:
            float: Mejor puntuación encontrada
        """
        raise NotImplementedError("Búsqueda por filtros no implementada aún")

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
    
    def sample_lgbm_params(self, trial: optuna.Trial, prefix: str) -> Dict[str, Any]:
        """Muestra parámetros para LightGBM.
        
        Args:
            trial: Trial actual de Optuna
            prefix: Prefijo para los parámetros
            
        Returns:
            Dict[str, Any]: Parámetros de LightGBM
        """
        n_estimators = trial.suggest_int(f"{prefix}_estimators", 100, 300, step=50)
        max_depth    = trial.suggest_int(f"{prefix}_max_depth", 3, 6)
        lr           = trial.suggest_float(f"{prefix}_learning_rate", 0.15, 0.3, log=True)
        reg_lambda   = trial.suggest_float(f"{prefix}_reg_lambda", 1.0, 5.0, log=True)
        es_rounds    = trial.suggest_int(f"{prefix}_early_stopping", 30, 60, step=10)
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
        # Optimización de hiperparámetros
        hp.update(self.sample_cat_params(trial, "cat_main"))
        hp.update(self.sample_cat_params(trial, "cat_meta"))
        hp.update(self.sample_xgb_params(trial, "xgb_main"))
        hp.update(self.sample_xgb_params(trial, "xgb_meta"))
        hp.update(self.sample_lgbm_params(trial, "lgbm_main"))
        hp.update(self.sample_lgbm_params(trial, "lgbm_meta"))

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
            def check_constant_features(X):
                return np.any(np.var(X, axis=0) < 1e-10)
            # ---------- 1) main model_main ----------
            # Get feature columns and rename them to follow f%d pattern
            main_feature_cols = main_data.columns[main_data.columns.str.contains('_feature') & ~main_data.columns.str.contains('_meta_feature')]
            X_main = main_data[main_feature_cols]
            X_main.columns = [f'f{i}' for i in range(len(main_feature_cols))]
            if check_constant_features(X_main.to_numpy()):
                return None
            y_main = main_data['labels_main'].astype('int16')
            # Check for inf values in main features
            inf_cols_main = X_main.columns[X_main.isin([np.inf, -np.inf]).any()].tolist()
            if inf_cols_main:
                print("Main features with inf values:", inf_cols_main)
            # Check for NaN values in main features
            nan_cols_main = X_main.columns[X_main.isna().any()].tolist()
            if nan_cols_main:
                print("Main features with NaN values:", nan_cols_main)
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
            if check_constant_features(X_meta.to_numpy()):
                return None
            y_meta = meta_data['labels_meta'].astype('int16')
            # Check for inf values in meta features
            inf_cols_meta = X_meta.columns[X_meta.isin([np.inf, -np.inf]).any()].tolist()
            if inf_cols_meta:
                print("Meta features with inf values:", inf_cols_meta)
            # Check for NaN values in meta features
            nan_cols_meta = X_meta.columns[X_meta.isna().any()].tolist()
            if nan_cols_meta:
                print("Meta features with NaN values:", nan_cols_meta)
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
            lgbm_main_params = dict(
                n_estimators=hp['lgbm_main_estimators'],
                max_depth=hp['lgbm_main_max_depth'],
                learning_rate=hp['lgbm_main_learning_rate'],
                reg_lambda=hp['lgbm_main_reg_lambda'],
                early_stopping_round=hp['lgbm_main_early_stopping'],
                metric='auc',
                tree_learner="serial",
                device="cpu",
                verbosity=-1,
                n_jobs=self.n_jobs
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
                ('lightgbm', LGBMWithEval(
                    **lgbm_main_params,
                    eval_set=[(X_val_main, y_val_main)],
                    callbacks=[LightGBMPruningCallback(trial, "auc")]
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
            lgbm_meta_params = dict(
                n_estimators=hp['lgbm_meta_estimators'],
                max_depth=hp['lgbm_meta_max_depth'],
                learning_rate=hp['lgbm_meta_learning_rate'],
                reg_lambda=hp['lgbm_meta_reg_lambda'],
                early_stopping_round=hp['lgbm_meta_early_stopping'],
                metric='auc',
                tree_learner="serial",
                device="cpu",
                verbosity=-1,
                n_jobs=self.n_jobs
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
                ('lightgbm', LGBMWithEval(
                    **lgbm_meta_params,
                    eval_set=[(X_val_meta, y_val_meta)],
                    callbacks=[LightGBMPruningCallback(trial, "binary_logloss")]
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