import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "studies"))

from modules.labeling_lib import (
    get_labels_filter_one_direction,
    get_labels_trend_one_direction,
)


def _sample_df(n=50):
    values = np.linspace(1, 10, n)
    return pd.DataFrame({
        "open": values,
        "high": values + 0.1,
        "low": values - 0.1,
        "close": values,
        "volume": np.ones(n),
    })


def test_filter_one_direction_both():
    df = _sample_df()
    res = get_labels_filter_one_direction(df, rolling=5, polyorder=2, direction="both")
    assert set(res["labels_main"].unique()) <= {0.0, 1.0}
    assert {"open", "high", "low", "close", "volume"}.issubset(res.columns)


def test_filter_one_direction_buy():
    df = _sample_df()
    res = get_labels_filter_one_direction(df, rolling=5, polyorder=2, direction="buy")
    assert set(res["labels_main"].unique()) <= {0.0, 1.0}
    assert {"open", "high", "low", "close", "volume"}.issubset(res.columns)


def test_trend_one_direction_both():
    df = _sample_df()
    res = get_labels_trend_one_direction(
        df,
        rolling=5,
        polyorder=2,
        threshold=0.01,
        vol_window=5,
        direction="both",
    )
    assert set(res["labels_main"].unique()) <= {0.0, 1.0}
    assert {"open", "high", "low", "close", "volume"}.issubset(res.columns)


def test_trend_one_direction_sell():
    df = _sample_df()
    res = get_labels_trend_one_direction(
        df,
        rolling=5,
        polyorder=2,
        threshold=0.01,
        vol_window=5,
        direction="sell",
    )
    assert set(res["labels_main"].unique()) <= {0.0, 1.0}
    assert {"open", "high", "low", "close", "volume"}.issubset(res.columns)
    
    
def test_trend_one_direction_buy():
    df = _sample_df()
    res = get_labels_trend_one_direction(
        df,
        rolling=5,
        polyorder=2,
        threshold=0.01,
        vol_window=5,
        direction="buy",
    )
    assert set(res["labels_main"].unique()) <= {0.0, 1.0}
    assert {"open", "high", "low", "close", "volume"}.issubset(res.columns)
