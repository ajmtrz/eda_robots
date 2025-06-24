import sys
from pathlib import Path
import numpy as np
import pandas as pd
import random

sys.path.append(str(Path(__file__).resolve().parents[1] / "studies"))

from modules.labeling_lib import get_labels_trend_with_profit


def _sample_ohlc_df(n=50, bidirectional=False):
    if bidirectional:
        close = np.concatenate([
            np.linspace(1, 5, n // 2),
            np.linspace(5, 2, n - n // 2),
        ])
    else:
        close = np.linspace(1, 10, n)
    return pd.DataFrame({
        "close": close,
        "high": close + 0.5,
        "low": close - 0.5,
    })


def _label(df):
    random.seed(0)
    return get_labels_trend_with_profit(
        df,
        rolling=5,
        polyorder=2,
        threshold=0.01,
        vol_window=5,
        markup=0.0,
        min_l=1,
        max_l=5,
        atr_period=5,
    )


def test_trend_profit_single_direction():
    df = _sample_ohlc_df()
    res = _label(df)
    assert not res.empty
    assert set(res["labels"].unique()) <= {0.0}


def test_trend_profit_both_directions():
    df = _sample_ohlc_df(bidirectional=True)
    res = _label(df)
    assert not res.empty
    labels = set(res["labels"].unique())
    assert labels <= {0.0, 1.0, 2.0}
    assert 0.0 in labels and 1.0 in labels
