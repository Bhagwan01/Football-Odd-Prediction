"""
Unit Tests for Machine Learning & Metrics
=========================================
Tests multi-class Brier score, log loss calculation, prediction evaluation, and data splits.
"""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import calculate_multiclass_brier_score, evaluate_predictions
from src.models.trainer import prepare_data_splits, FEATURE_COLUMNS


def test_multiclass_brier_score():
    """Verify Brier score gives 0.0 for perfect forecasts and >0 for imperfect ones."""
    y_true = np.array([0, 1, 2])
    # Perfect probabilities
    y_prob_perfect = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])
    score_perfect = calculate_multiclass_brier_score(y_true, y_prob_perfect)
    assert pytest.approx(score_perfect, 1e-6) == 0.0

    # Imperfect probabilities
    y_prob_imperfect = np.array([
        [0.5, 0.3, 0.2],
        [0.2, 0.6, 0.2],
        [0.1, 0.2, 0.7]
    ])
    score_imperfect = calculate_multiclass_brier_score(y_true, y_prob_imperfect)
    assert score_imperfect > 0.0


def test_evaluate_predictions_structure():
    """Verify evaluate_predictions dictionary schema and ranges."""
    y_true = np.array([0, 1, 0, 2])
    y_prob = np.array([
        [0.7, 0.2, 0.1],
        [0.1, 0.8, 0.1],
        [0.6, 0.2, 0.2],
        [0.2, 0.3, 0.5]
    ])
    metrics = evaluate_predictions(y_true, y_prob, model_name="TestModel")
    assert metrics["model_name"] == "TestModel"
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert metrics["log_loss"] > 0
    assert metrics["brier_score"] > 0
    assert len(metrics["confusion_matrix"]) == 3


def test_prepare_data_splits():
    """Verify chronological split divides data by cutoff year."""
    data = {col: np.random.rand(20) for col in FEATURE_COLUMNS}
    data["target"] = np.random.choice([0, 1, 2], size=20)
    data["kickoff_year"] = [2022] * 10 + [2024] * 10

    mock_df = pd.DataFrame(data)
    train_df, test_df, cols = prepare_data_splits(mock_df, split_year=2024)

    assert len(train_df) == 10
    assert len(test_df) == 10
    assert (train_df["kickoff_year"] < 2024).all()
    assert (test_df["kickoff_year"] >= 2024).all()
