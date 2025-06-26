import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "studies"))

from modules.labeling_lib import get_labels_multi_window


def _sample_ohlc_df(n=10):
    close = np.linspace(1, 10, n)
    return pd.DataFrame({
        "close": close,
        "high": close + 0.5,
        "low": close - 0.5,
    })


def test_multi_window_small_df():
    df = _sample_ohlc_df()
    res = get_labels_multi_window(df)
    assert len(res) == len(res["labels"])
