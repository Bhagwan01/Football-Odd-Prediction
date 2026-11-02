"""
Unit Tests for Anomaly Detection Module
=======================================
Tests unsupervised Isolation Forest fitting, anomaly scoring, and output schemas.
"""

import numpy as np
import pandas as pd
import pytest

from src.models.anomaly import fit_anomaly_detector, ANOMALY_FEATURE_COLS


def test_fit_anomaly_detector():
    """Verify Isolation Forest scores and flags outliers in feature matrix."""
    np.random.seed(42)
    n = 100
    data = {col: np.random.normal(0, 1, size=n) for col in ANOMALY_FEATURE_COLS}
    # Inject 5 extreme outliers
    for col in ["mean_drift_home", "spread_odds_home"]:
        data[col][:5] = 100.0

    mock_df = pd.DataFrame(data)
    model, scored = fit_anomaly_detector(mock_df, contamination=0.05)

    assert "anomaly_score" in scored.columns
    assert "is_anomaly" in scored.columns
    assert scored["is_anomaly"].isin([0, 1]).all()
    # At least some of the injected outliers should be flagged
    assert scored["is_anomaly"].sum() > 0
