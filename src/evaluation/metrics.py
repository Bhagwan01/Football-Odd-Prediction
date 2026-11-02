"""
Evaluation Metrics Module
=========================
Calculates multi-class probabilistic classification metrics:
- Multi-class Log Loss
- Multi-class Brier Score: sum((p_i - y_i)^2) / N
- Multi-class Accuracy, Macro F1, Weighted F1
- Calibration Analysis (Reliability Bins)
"""

from typing import Any
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    confusion_matrix,
    classification_report
)


def calculate_multiclass_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Compute multi-class Brier score.

    Formula: (1 / N) * sum_{i=1}^N sum_{k=1}^K (y_{i,k} - p_{i,k})^2

    Parameters
    ----------
    y_true : np.ndarray of shape (N,)
        True class indices (0, 1, 2).
    y_prob : np.ndarray of shape (N, K)
        Predicted probabilities for each class.

    Returns
    -------
    float
        Brier score (lower is better, 0.0 is perfect).
    """
    n_samples, n_classes = y_prob.shape
    y_onehot = np.zeros((n_samples, n_classes))
    y_onehot[np.arange(n_samples), y_true] = 1.0
    return float(np.mean(np.sum((y_prob - y_onehot) ** 2, axis=1)))


def evaluate_predictions(y_true: np.ndarray, y_prob: np.ndarray, model_name: str = "Model") -> dict[str, Any]:
    """
    Calculate comprehensive evaluation metrics for multi-class probabilistic forecasts.

    Parameters
    ----------
    y_true : np.ndarray of shape (N,)
        True class labels (0, 1, 2).
    y_prob : np.ndarray of shape (N, 3)
        Predicted probabilities for [Home, Draw, Away].
    model_name : str
        Descriptive name of the model.

    Returns
    -------
    dict
        Dictionary of accuracy, log_loss, brier_score, f1_macro, f1_weighted, and confusion_matrix.
    """
    y_pred = np.argmax(y_prob, axis=1)

    acc = accuracy_score(y_true, y_pred)
    ll = log_loss(y_true, y_prob)
    brier = calculate_multiclass_brier_score(y_true, y_prob)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")
    cm = confusion_matrix(y_true, y_pred).tolist()

    return {
        "model_name": model_name,
        "accuracy": float(acc),
        "log_loss": float(ll),
        "brier_score": float(brier),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "confusion_matrix": cm
    }


def compute_calibration_curve(y_true_binary: np.ndarray, prob_pred: np.ndarray, n_bins: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute calibration curve bins (observed frequency vs mean predicted probability).

    Parameters
    ----------
    y_true_binary : np.ndarray
        Binary indicator (e.g. 1 if Home Win, 0 otherwise).
    prob_pred : np.ndarray
        Predicted probabilities for that class.
    n_bins : int
        Number of equal-width probability bins.

    Returns
    -------
    tuple of (prob_pred_means, observed_frequencies)
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    binids = np.digitize(prob_pred, bins) - 1

    prob_means = []
    obs_freqs = []

    for i in range(n_bins):
        mask = binids == i
        if np.any(mask):
            prob_means.append(np.mean(prob_pred[mask]))
            obs_freqs.append(np.mean(y_true_binary[mask]))

    return np.array(prob_means), np.array(obs_freqs)
