import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "studies"))

from modules.labeling_lib import get_labels_trend


def _sample_df(n=50):
    return pd.DataFrame({"close": np.linspace(1, 10, n)})


def test_get_labels_trend():
    df = _sample_df()
    res = get_labels_trend(
        df,
        rolling=5,
        polyorder=2,
        threshold=0.01,
        vol_window=5,
    )
    assert not res.empty
    assert set(res["labels"].unique()) <= {0.0, 1.0, 2.0}
