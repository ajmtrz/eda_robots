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

