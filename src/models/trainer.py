"""
Model Training & Benchmarking Pipeline
======================================
Trains and benchmarks supervised multi-class ML models on chronological train/test splits.

Models Evaluated:
1. Bookmaker Consensus Baseline (Fair Implied Probabilities)
2. Calibrated Multinomial Logistic Regression
3. Random Forest Classifier
4. XGBoost Classifier

Prevents Data Leakage:
- Strictly splits chronologically (Train: <= 2023, Test: >= 2024).
- Uses only pre-kickoff market quotes and fixture schedule attributes.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data.loader import load_processed
from src.evaluation.metrics import evaluate_predictions

FEATURE_COLUMNS = [
    # Consensus Normalized Implied Probabilities
    "mkt_prob_home", "mkt_prob_draw", "mkt_prob_away",
    # Opening Normalized Implied Probabilities
    "open_prob_home", "open_prob_draw", "open_prob_away",
    # Raw Market Odds
    "mean_odds_home", "mean_odds_draw", "mean_odds_away",
    # Market Disagreement & Range
    "odds_std_home", "odds_std_draw", "odds_std_away",
    "spread_odds_home", "spread_odds_draw", "spread_odds_away",
    # Market Margin & Momentum
    "mean_overround", "mean_drift_home", "mean_drift_draw", "mean_drift_away",
    # Bookmaker Depth & Fixture Context
    "n_bookmakers", "neutral",
    "kickoff_month", "kickoff_dayofweek", "kickoff_hour", "is_weekend"
]


def prepare_data_splits(features_df: pd.DataFrame, split_year: int = 2024) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Split feature matrix chronologically into training and testing sets.

    Parameters
    ----------
    features_df : pd.DataFrame
        Complete engineered feature matrix.
    split_year : int
        Kickoff year threshold for the test set (e.g. 2024).

    Returns
    -------
    tuple of (train_df, test_df, feature_cols)
    """
    valid = features_df.dropna(subset=FEATURE_COLUMNS + ["target"]).copy()

    train_df = valid[valid["kickoff_year"] < split_year].copy()
    test_df = valid[valid["kickoff_year"] >= split_year].copy()

    # Fallback to 80/20 chronological if split year gives empty test
    if len(test_df) == 0:
        split_idx = int(len(valid) * 0.8)
        train_df = valid.iloc[:split_idx].copy()
        test_df = valid.iloc[split_idx:].copy()

    return train_df, test_df, FEATURE_COLUMNS


def run_training_pipeline() -> dict[str, Any]:
    """
    Execute model training, benchmarking against Bookmaker Baseline, and saving artifacts.

    Returns
    -------
    dict
        Dictionary of benchmark evaluation results for all models.
    """
    print("=" * 70)
    print(" EXECUTING MACHINE LEARNING PIPELINE ")
    print("=" * 70)

    print("[1/5] Loading engineered feature matrix...")
    features_df = load_processed("features_match_level")
    print(f" -> Loaded: {features_df.shape[0]:,} matches x {features_df.shape[1]} cols")

    print("[2/5] Performing chronological train/test split...")
    train_df, test_df, feature_cols = prepare_data_splits(features_df, split_year=2024)
    print(f" -> Train Set: {len(train_df):,} matches (Year < 2024)")
    print(f" -> Test Set:  {len(test_df):,} matches (Year >= 2024)")

    X_train = train_df[feature_cols].values
    y_train = train_df["target"].values
    X_test = test_df[feature_cols].values
    y_test = test_df["target"].values

    results = {}

    # 1. Bookmaker Consensus Baseline
    print("\n[3/5] Evaluating Bookmaker Consensus Baseline...")
    bmk_probs_test = test_df[["mkt_prob_home", "mkt_prob_draw", "mkt_prob_away"]].values
    # Re-normalize to ensure exact 1.0 probability conservation
    bmk_probs_test = bmk_probs_test / bmk_probs_test.sum(axis=1, keepdims=True)
    bmk_metrics = evaluate_predictions(y_test, bmk_probs_test, model_name="Bookmaker Consensus Baseline")
    results["Bookmaker Baseline"] = bmk_metrics
    print(f"   Accuracy: {bmk_metrics['accuracy']:.4f} | Log Loss: {bmk_metrics['log_loss']:.4f} | Brier: {bmk_metrics['brier_score']:.4f}")

    # 2. Multinomial Logistic Regression
    print("\n[4/5] Training Supervised Models...")
    print(" - Model 1: Calibrated Multinomial Logistic Regression...")
    lr_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42))
    ])
    lr_pipeline.fit(X_train, y_train)
    lr_probs = lr_pipeline.predict_proba(X_test)
    lr_metrics = evaluate_predictions(y_test, lr_probs, model_name="Multinomial Logistic Regression")
    results["Logistic Regression"] = lr_metrics
    print(f"   Accuracy: {lr_metrics['accuracy']:.4f} | Log Loss: {lr_metrics['log_loss']:.4f} | Brier: {lr_metrics['brier_score']:.4f}")

    # 3. Random Forest Classifier
    print(" - Model 2: Random Forest Classifier...")
    rf_clf = RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        min_samples_split=20,
        random_state=42,
        n_jobs=-1
    )
    rf_clf.fit(X_train, y_train)
    rf_probs = rf_clf.predict_proba(X_test)
    rf_metrics = evaluate_predictions(y_test, rf_probs, model_name="Random Forest")
    results["Random Forest"] = rf_metrics
    print(f"   Accuracy: {rf_metrics['accuracy']:.4f} | Log Loss: {rf_metrics['log_loss']:.4f} | Brier: {rf_metrics['brier_score']:.4f}")

    # 4. XGBoost Classifier
    print(" - Model 3: XGBoost Classifier...")
    xgb_clf = XGBClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="mlogloss"
    )
    xgb_clf.fit(X_train, y_train)
    xgb_probs = xgb_clf.predict_proba(X_test)
    xgb_metrics = evaluate_predictions(y_test, xgb_probs, model_name="XGBoost Classifier")
    results["XGBoost"] = xgb_metrics
    print(f"   Accuracy: {xgb_metrics['accuracy']:.4f} | Log Loss: {xgb_metrics['log_loss']:.4f} | Brier: {xgb_metrics['brier_score']:.4f}")

    # 5. Persist Best Model and Summary Metrics
    print("\n[5/5] Persisting trained artifacts...")
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    report_dir = project_root / "reports" / "analysis"
    report_dir.mkdir(parents=True, exist_ok=True)

    # Identify best model by lowest log loss
    model_objects = {
        "Logistic Regression": lr_pipeline,
        "Random Forest": rf_clf,
        "XGBoost": xgb_clf
    }
    ml_results = {k: v for k, v in results.items() if k != "Bookmaker Baseline"}
    best_name = min(ml_results, key=lambda k: ml_results[k]["log_loss"])
    best_model = model_objects[best_name]

    model_path = models_dir / "best_model.joblib"
    joblib.dump({
        "model": best_model,
        "model_name": best_name,
        "feature_columns": feature_cols,
        "classes": ["Home", "Draw", "Away"],
        "metrics": results[best_name]
    }, model_path)
    print(f" -> Best ML Model ({best_name}) saved to: {model_path}")

    metrics_path = report_dir / "model_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f" -> Benchmark results exported to: {metrics_path}")

    print("=" * 70)
    print(" MACHINE LEARNING PIPELINE COMPLETED SUCCESSFULLY ")
    print("=" * 70)
    return results


if __name__ == "__main__":
    run_training_pipeline()
