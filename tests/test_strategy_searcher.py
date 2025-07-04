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
    labeled = searcher.apply_labeling(
        df.copy(), {"rolling": 5, "polyorder": 2, "quantiles": [0.45, 0.55]}
    )
    assert {"open", "high", "low", "close", "volume"}.issubset(labeled.columns)

    
class DummySearcherPrune(StrategySearcher):
    def __init__(self):
        self.symbol = "TEST"
        self.timeframe = "M1"
        self.direction = "buy"
        self.train_start = datetime(2020, 1, 1)
        self.train_end = datetime(2020, 4, 30)
        self.test_start = datetime(2020, 5, 1)
        self.test_end = datetime(2020, 6, 30)
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
        self.base_df = pd.DataFrame({"close": range(200)}, index=pd.date_range("2020-01-01", periods=200))


def test_get_train_test_data_prunes_hp():
    searcher = DummySearcherPrune()
    hp = {
        "periods_main": (10, 20),
        "periods_meta": (5, 6),
        "stats_main": ("std", "skew"),
        "stats_meta": ("std",),
    }

    from modules import labeling_lib
    import modules.StrategySearcher as ss

    def fake_get_features(data, _hp):
        idx = data.index
        return pd.DataFrame({
            "10_std_feature": [1.0] * len(idx),
            "20_skew_feature": range(len(idx)),
            "5_std_meta_feature": [0.0] * len(idx),
            "6_std_meta_feature": range(len(idx)),
            "open": range(len(idx)),
            "high": range(len(idx)),
            "low": range(len(idx)),
            "close": range(len(idx)),
            "volume": range(len(idx)),
        }, index=idx)

    original = labeling_lib.get_features
    labeling_lib.get_features = fake_get_features
    ss.get_features = fake_get_features
    try:
        train, test = searcher.get_train_test_data(hp)
    finally:
        labeling_lib.get_features = original
        ss.get_features = original

    assert train is not None and test is not None
    assert hp["periods_main"] == (20,)
    assert hp["stats_main"] == ("skew",)
    assert hp["periods_meta"] == (6,)
    assert hp["stats_meta"] == ("std",)


class DummySearcherCausal(StrategySearcher):
    def __init__(self):
        self.symbol = "TEST"
        self.timeframe = "M1"
        self.direction = "buy"
        self.train_start = datetime(2020, 1, 1)
        self.train_end = datetime(2020, 1, 2)
        self.test_start = datetime(2020, 1, 3)
        self.test_end = datetime(2020, 1, 4)
        self.models_export_path = ""
        self.include_export_path = ""
        self.history_path = ""
        self.search_type = "causal"
        self.search_subtype = "simple"
        self.label_method = "atr"
        self.pruner_type = "hyperband"
        self.n_trials = 1
        self.n_models = 1
        self.n_jobs = 1
        self.tag = ""
        self.base_df = pd.DataFrame({"close": [1, 2, 3, 4]},
                                    index=pd.date_range("2020-01-01", periods=4))


def test_suggest_all_params_causal_includes_meta_learners():
    searcher = DummySearcherCausal()
    study = optuna.create_study(direction="maximize")
    trial = study.ask()
    params = searcher.suggest_all_params(trial)
    assert "n_meta_learners" in params

