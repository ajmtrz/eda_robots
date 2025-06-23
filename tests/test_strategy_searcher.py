import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import optuna

sys.path.append(str(Path(__file__).resolve().parents[1] / "studies"))

from modules.StrategySearcher import StrategySearcher

class DummySearcher(StrategySearcher):
    def __init__(self):
        # Avoid heavy init
        self.symbol = "TEST"
        self.timeframe = "M1"
        self.direction = "buy"
        self.train_start = datetime(2020,1,1)
        self.train_end = datetime(2020,1,2)
        self.test_start = datetime(2020,1,3)
        self.test_end = datetime(2020,1,4)
        self.models_export_path = ""
        self.include_export_path = ""
        self.history_path = ""
        self.search_type = "lgmm"
        self.search_subtype = "simple"
        self.label_method = "atr"
        self.pruner_type = "hyperband"
        self.n_trials = 1
        self.n_models = 1
        self.n_jobs = 1
        self.tag = ""
        rng = pd.date_range("2020-01-01", periods=4)
        self.base_df = pd.DataFrame({
            "open":   [1,2,3,4],
            "high":   [1.1,2.1,3.1,4.1],
            "low":    [0.9,1.9,2.9,3.9],
            "close":  [1,2,3,4],
            "volume": [1,1,1,1],
        }, index=rng)

    def get_train_test_data(self, hp):
        data = pd.DataFrame({
            "meta_feature1": [0.1, 0.2, 0.3, 0.4],
            "meta_feature2": [0.0, 0.1, 0.0, 0.1],
            "open":  [1,2,3,4],
            "high":  [1.1,2.1,3.1,4.1],
            "low":   [0.9,1.9,2.9,3.9],
            "close": [1,2,3,4],
            "volume": [1,1,1,1],
        }, index=self.base_df.index)
        return data, data

    def _evaluate_clusters(self, ds_train, ds_test, hp):
        return (0.1, 0.2), (None, None)

    def suggest_all_params(self, trial):
        return {
            "n_components": trial.suggest_int("n_components", 2, 3),
            "covariance_type": trial.suggest_categorical("covariance_type", ["full"]),
            "max_iter": trial.suggest_int("max_iter", 10, 20)
        }


class DummySearcherConstant(StrategySearcher):
    def __init__(self):
        self.symbol = "TEST"
        self.timeframe = "M1"
        self.direction = "buy"
        self.train_start = datetime(2020,1,1)
        self.train_end = datetime(2020,1,2)
        self.test_start = datetime(2020,1,3)
        self.test_end = datetime(2020,1,4)
        self.models_export_path = ""
        self.include_export_path = ""
        self.history_path = ""
        self.search_type = "lgmm"
        self.search_subtype = "simple"
        self.label_method = "atr"
        self.pruner_type = "hyperband"
        self.n_trials = 1
        self.n_models = 1
        self.n_jobs = 1
        self.tag = ""
        rng = pd.date_range("2020-01-01", periods=4)
        self.base_df = pd.DataFrame({
            "open":   [1,2,3,4],
            "high":   [1.1,2.1,3.1,4.1],
            "low":    [0.9,1.9,2.9,3.9],
            "close":  [1,2,3,4],
            "volume": [1,1,1,1],
        }, index=rng)

    def get_train_test_data(self, hp):
        data = pd.DataFrame({
            "constant_feature": [1.0, 1.0, 1.0, 1.0],
            "good_feature": [0.0, 1.0, 2.0, 3.0],
            "open":  [1,2,3,4],
            "high":  [1.1,2.1,3.1,4.1],
            "low":   [0.9,1.9,2.9,3.9],
            "close": [1,2,3,4],
            "volume": [1,1,1,1]
        }, index=self.base_df.index)

        feature_cols = ["constant_feature", "good_feature"]
        problematic = self.check_constant_features(data, feature_cols)
        if problematic:
            data.drop(columns=problematic, inplace=True)
            feature_cols = [c for c in feature_cols if c not in problematic]
            if not feature_cols:
                return None, None

        train = data.iloc[:2]
        test = data.iloc[2:]
        return train, test

    def _evaluate_clusters(self, ds_train, ds_test, hp):
        return (0.1, 0.2), (None, None)

    def suggest_all_params(self, trial):
        return {}


def test_search_lgmm_returns_floats():
    searcher = DummySearcher()
    trial = optuna.trial.FixedTrial({"n_components":2,"covariance_type":"full","max_iter":10})
    res = searcher.search_lgmm(trial)
    assert isinstance(res[0], float)
    assert isinstance(res[1], float)


def test_get_train_test_data_drops_constant_features():
    searcher = DummySearcherConstant()
    train, test = searcher.get_train_test_data({})
    assert train is not None and test is not None
    assert "constant_feature" not in train.columns


def test_get_train_test_data_contains_ohlcv():
    searcher = DummySearcher()
    train, test = searcher.get_train_test_data({})
    for df in (train, test):
        assert {"open", "high", "low", "close", "volume"}.issubset(df.columns)


def test_apply_labeling_preserves_ohlcv():
    searcher = DummySearcher()
    df, _ = searcher.get_train_test_data({})
    searcher.label_method = "filter_one"
    labeled = searcher._apply_labeling(df.copy(), {"rolling": 5, "polyorder": 2, "quantiles": [0.45, 0.55]})
    assert {"open", "high", "low", "close", "volume"}.issubset(labeled.columns)

