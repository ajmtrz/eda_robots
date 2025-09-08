import numpy as np

from studies.modules.labeling_lib import calculate_labels_random


def test_j_eq_i_single_up_is_valid_first_touch():
    n = 5
    close = np.zeros(n, dtype=np.float64)
    high = np.zeros(n, dtype=np.float64)
    low = np.zeros(n, dtype=np.float64)
    atr = np.ones(n, dtype=np.float64)
    label_markup = 0.5
    label_min_val = 1
    label_max_val = 2
    direction_int = 2  # both

    # i = 0: only up target hit on the same bar
    high[0] = 1.0   # >= 0.5
    low[0] = 0.0    # > -0.5 (so down not hit)

    labels = calculate_labels_random(close, high, low, atr, label_markup, label_min_val, label_max_val, direction_int)
    # Result length = n - label_max_val = 3; we assert for i = 0
    assert labels[0] == 0.0


def test_j_eq_i_both_up_and_down_is_ambiguous():
    n = 5
    close = np.zeros(n, dtype=np.float64)
    high = np.zeros(n, dtype=np.float64)
    low = np.zeros(n, dtype=np.float64)
    atr = np.ones(n, dtype=np.float64)
    label_markup = 0.5
    label_min_val = 1
    label_max_val = 2
    direction_int = 2

    # i = 0: both up and down targets hit on same bar
    high[0] = 1.0   # >= 0.5
    low[0] = -1.0   # <= -0.5

    labels = calculate_labels_random(close, high, low, atr, label_markup, label_min_val, label_max_val, direction_int)
    assert labels[0] == 2.0


def test_future_bar_both_up_and_down_is_ambiguous():
    n = 5
    close = np.zeros(n, dtype=np.float64)
    high = np.zeros(n, dtype=np.float64)
    low = np.zeros(n, dtype=np.float64)
    atr = np.ones(n, dtype=np.float64)
    label_markup = 0.5
    label_min_val = 1
    label_max_val = 2
    direction_int = 2

    # i = 0: no touch at j==i, but both hit at j = 1
    high[1] = 1.0
    low[1] = -1.0

    labels = calculate_labels_random(close, high, low, atr, label_markup, label_min_val, label_max_val, direction_int)
    assert labels[0] == 2.0


def test_earliest_touch_resolution_different_bars():
    n = 6
    close = np.zeros(n, dtype=np.float64)
    high = np.zeros(n, dtype=np.float64)
    low = np.zeros(n, dtype=np.float64)
    atr = np.ones(n, dtype=np.float64)
    label_markup = 0.5
    label_min_val = 1
    label_max_val = 3
    direction_int = 2

    # i = 0: up hit first at j=1, down hit later at j=2
    high[1] = 1.0
    low[2] = -1.0

    labels = calculate_labels_random(close, high, low, atr, label_markup, label_min_val, label_max_val, direction_int)
    assert labels[0] == 0.0

