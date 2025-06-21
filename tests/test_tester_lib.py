"""Pytest cases for tester_lib.

Run the tests with:
    pytest
"""

import sys
from pathlib import Path
import numpy as np

# Allow importing modules from the repository's studies directory
sys.path.append(str(Path(__file__).resolve().parents[1] / "studies"))

from modules import tester_lib


def test_signed_r2_positive():
    equity = np.arange(10, dtype=np.float64)
    result = tester_lib._signed_r2(equity)
    assert result > 0


def test_evaluate_report_score_range():
    report = np.arange(10, dtype=np.float64)
    score = tester_lib.evaluate_report(report)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_process_data_bidirectional_closes_open_position():
    close = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    labels = np.array([0.6, 0.6, 0.6, 0.6], dtype=np.float64)
    meta = np.array([0.6, 0.4, 0.4, 0.4], dtype=np.float64)

    rpt, chart = tester_lib.process_data(close, labels, meta)
    assert rpt.size == 2
    assert chart.size == 2
    assert np.isclose(abs(rpt[-1]), 3.0)

