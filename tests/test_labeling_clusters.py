import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "studies"))

from modules.labeling_lib import get_labels_clusters


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
    np.random.seed(0)
    return get_labels_clusters(df, markup=0.1, num_clusters=5, atr_period=5)


def test_clusters_single_direction():
    df = _sample_ohlc_df()
    res = _label(df)
    assert not res.empty
    assert set(res["labels"].unique()) <= {0.0}


def test_clusters_both_directions():
    df = _sample_ohlc_df(bidirectional=True)
    res = _label(df)
    assert not res.empty
    labels = set(res["labels"].unique())
    assert labels <= {0.0, 1.0}
    assert 0.0 in labels and 1.0 in labels
