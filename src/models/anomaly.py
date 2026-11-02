"""
Anomaly Detection Module
========================
Applies unsupervised learning (Isolation Forest) to detect unusual or erratic
betting market behavior:
- Abrupt or massive odds drift percentages
- Extreme bookmaker pricing divergences (high spread / variance)
- Irregular bookmaker overrounds
"""

import json
import os
import sys
from pathlib import Path
from typing import Any
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

os.environ["MPLCONFIGDIR"] = str(project_root / ".matplotlib_cache")

from src.data.loader import load_processed, save_processed

ANOMALY_FEATURE_COLS = [
    # Drift dynamics
    "mean_drift_home", "mean_drift_draw", "mean_drift_away",
    # Bookmaker disagreement
    "odds_std_home", "odds_std_draw", "odds_std_away",
    # Best vs Worst Spreads
    "spread_odds_home", "spread_odds_draw", "spread_odds_away",
    # Margins & Bookmaker Depth
    "mean_overround", "n_bookmakers"
]


def fit_anomaly_detector(features_df: pd.DataFrame, contamination: float = 0.02) -> tuple[IsolationForest, pd.DataFrame]:
    """
    Fit an Isolation Forest model to detect unusual market behavior.

    Parameters
    ----------
    features_df : pd.DataFrame
        Engineered feature matrix.
    contamination : float
        Expected proportion of outliers in the data (default: 2%).

    Returns
    -------
    tuple of (IsolationForest, pd.DataFrame with 'anomaly_score' and 'is_anomaly')
    """
    valid = features_df.dropna(subset=ANOMALY_FEATURE_COLS).copy()
    X = valid[ANOMALY_FEATURE_COLS].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    iso = IsolationForest(
        n_estimators=150,
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )
    preds = iso.fit_predict(X_scaled)  # 1: Normal, -1: Anomaly
    scores = iso.decision_function(X_scaled)  # Lower is more abnormal

    valid["anomaly_score"] = scores
    valid["is_anomaly"] = (preds == -1).astype(int)

    return iso, valid


def run_anomaly_pipeline(contamination: float = 0.02) -> pd.DataFrame:
    """
    Execute full anomaly detection pipeline, save artifacts, and generate report figures.

    Parameters
    ----------
    contamination : float
        Proportion of anomalies to identify.

    Returns
    -------
    pd.DataFrame
        DataFrame of matches with anomaly scores.
    """
    print("=" * 70)
    print(" EXECUTING UNSUPERVISED ANOMALY DETECTION PIPELINE ")
    print("=" * 70)

    print("[1/4] Loading engineered feature matrix...")
    features_df = load_processed("features_match_level")

    print(f"[2/4] Fitting Isolation Forest on {len(features_df):,} matches (contamination={contamination*100:.1f}%)...")
    model, scored_df = fit_anomaly_detector(features_df, contamination=contamination)

    n_anomalies = scored_df["is_anomaly"].sum()
    print(f" -> Flagged {n_anomalies:,} anomalous fixtures ({n_anomalies / len(scored_df) * 100:.2f}%)")

    # Analyze Outcome distribution among Anomalies vs Normal Matches
    normal_outcomes = scored_df[scored_df["is_anomaly"] == 0]["outcome"].value_counts(normalize=True)
    anomaly_outcomes = scored_df[scored_df["is_anomaly"] == 1]["outcome"].value_counts(normalize=True)

    print("\nOutcome Comparison:")
    print("Normal Matches:   ", normal_outcomes.round(4).to_dict())
    print("Anomalous Matches:", anomaly_outcomes.round(4).to_dict())

    # 3. Save Model & Scored Dataset
    print("\n[3/4] Persisting model and anomalous match records...")
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, models_dir / "isolation_forest.joblib")

    # Save flagged data to processed
    save_processed(scored_df, "features_with_anomalies")

    # 4. Generate Visualizations
    print("[4/4] Generating anomaly visualization in reports/figures/...")
    figures_dir = project_root / "reports" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Anomaly score distribution
    axes[0].hist(scored_df[scored_df["is_anomaly"] == 0]["anomaly_score"], bins=50, color="#2980b9", alpha=0.7, label="Normal Market")
    axes[0].hist(scored_df[scored_df["is_anomaly"] == 1]["anomaly_score"], bins=30, color="#e74c3c", alpha=0.85, label="Anomalous Market")
    axes[0].set_title("Isolation Forest Decision Score Distribution", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Anomaly Score (Lower = More Abnormal)")
    axes[0].set_ylabel("Count")
    axes[0].legend()

    # Outcome comparison bar chart
    comp_df = pd.DataFrame({"Normal Market": normal_outcomes, "Anomalous Market": anomaly_outcomes}) * 100
    comp_df.plot.bar(ax=axes[1], color=["#3498db", "#e74c3c"], edgecolor="black", alpha=0.85)
    axes[1].set_title("Match Outcome Distribution: Normal vs Anomalous", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Proportion (%)")
    axes[1].set_xticklabels(["Home Win", "Draw", "Away Win"], rotation=0)
    axes[1].legend()

    plt.tight_layout()
    fig_path = figures_dir / "anomaly_distribution.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f" -> Saved: {fig_path}")

    print("=" * 70)
    print(" ANOMALY DETECTION COMPLETED SUCCESSFULLY ")
    print("=" * 70)
    return scored_df


if __name__ == "__main__":
    run_anomaly_pipeline()
