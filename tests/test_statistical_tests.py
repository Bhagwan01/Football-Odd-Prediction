"""
Unit Tests for Statistical Analysis Module
==========================================
Verifies mathematical test routines (Kruskal-Wallis, Wilcoxon, Chi-Square, Z-test).
"""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.statistical_tests import (
    evaluate_bookmaker_margin_differences,
    evaluate_odds_drift_significance,
    evaluate_league_outcome_independence,
    evaluate_home_advantage_significance,
)


def test_home_advantage_z_test():
    """Verify one-sample proportion test produces valid stats and confidence interval."""
    mock_matches = pd.DataFrame({"outcome": ["H"] * 60 + ["D"] * 20 + ["A"] * 20})
    res = evaluate_home_advantage_significance(mock_matches)

    assert res["home_wins"] == 60
    assert pytest.approx(res["home_win_rate"], 0.01) == 0.60
    assert res["z_statistic"] > 0
    assert res["is_significant"] is True
    assert res["ci_95"][0] < 0.60 < res["ci_95"][1]


def test_bookmaker_margin_differences():
    """Verify Kruskal-Wallis detects differences between synthetic margins."""
    mock_closing = pd.DataFrame({
        "bookmaker": ["BookA"] * 50 + ["BookB"] * 50,
        "close_overround": [0.03] * 50 + [0.08] * 50  # 3% vs 8%
    })
    res = evaluate_bookmaker_margin_differences(mock_closing)
    assert res["is_significant"] is True
    assert "BookA" in res["bookmaker_summaries"]
    assert "BookB" in res["bookmaker_summaries"]


def test_odds_drift_computation():
    """Verify odds drift Wilcoxon calculation on paired series."""
    np.random.seed(42)
    mock_closing = pd.DataFrame({
        "open_1x2_home": np.linspace(2.0, 2.5, 30),
        "close_1x2_home": np.linspace(2.1, 2.7, 30),
        "open_1x2_draw": np.linspace(3.1, 3.5, 30),
        "close_1x2_draw": np.linspace(3.0, 3.4, 30),
        "open_1x2_away": np.linspace(3.2, 3.8, 30),
        "close_1x2_away": np.linspace(3.0, 3.5, 30)
    })
    res = evaluate_odds_drift_significance(mock_closing)
    assert "home" in res
    assert "draw" in res
    assert "away" in res
    assert res["home"]["mean_drift"] > 0
    assert res["away"]["mean_drift"] < 0
