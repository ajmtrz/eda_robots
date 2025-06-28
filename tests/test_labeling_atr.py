import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "studies"))

from modules.labeling_lib import get_labels_one_direction


def _sample_ohlc_df(n=50):
    close = np.linspace(1, 10, n)
    return pd.DataFrame({
        "close": close,
        "high": close + 0.5,
        "low": close - 0.5,
    })


def test_labels_one_direction_buy():
    df = _sample_ohlc_df()
    res = get_labels_one_direction(
        df,
        markup=0.5,
        min_val=1,
        max_val=5,
        direction="buy",
        atr_period=5,
    )
    assert not res.empty
    assert set(res["labels_main"].unique()) <= {0.0, 1.0}


def test_labels_one_direction_both():
    df = _sample_ohlc_df()
    res = get_labels_one_direction(
        df,
        markup=0.5,
        min_val=1,
        max_val=5,
        direction="both",
        atr_period=5,
    )
    assert not res.empty
    labels = set(res["labels_main"].unique())
    assert labels <= {0.0, 1.0, 2.0}
    assert 0.0 in labels or 1.0 in labels

