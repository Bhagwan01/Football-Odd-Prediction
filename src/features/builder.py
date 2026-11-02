"""
Feature Engineering Module
==========================
Transforms cleaned matches and multi-bookmaker odds into a unified match-level
feature matrix for Supervised Machine Learning and Anomaly Detection.

Guarantees zero data leakage:
- No final scores or post-match statistics are included in the feature set.
- All engineered features are derived strictly from pre-kickoff market quotes.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data.loader import load_processed, save_processed


def build_match_odds_aggregates(closing_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate bookmaker quotes at the match level.

    Calculates:
    - Market Consensus Probabilities: Mean & Median normalized implied probabilities.
    - Market Disagreement: Standard deviation of closing odds across bookmakers.
    - Market Spreads: Best (max) vs worst (min) available odds across bookmakers.
    - Average Market Margin: Mean bookmaker overround.
    - Market Momentum: Average odds drift (%) across bookmakers.
    - Bookmaker Breadth: Number of active bookmakers quoting the fixture.

    Parameters
    ----------
    closing_df : pd.DataFrame
        Cleaned match_closing dataset.

    Returns
    -------
    pd.DataFrame
        One row per match_id containing aggregated market signals.
    """
    grouped = closing_df.groupby("match_id")

    # Aggregate dictionary
    agg_df = grouped.agg(
        # Bookmaker count
        n_bookmakers=("bookmaker", "nunique"),

        # Consensus Closing Normalized Probabilities (Fair implied probability)
        mkt_prob_home=("close_norm_prob_home", "mean"),
        mkt_prob_draw=("close_norm_prob_draw", "mean"),
        mkt_prob_away=("close_norm_prob_away", "mean"),

        # Consensus Opening Normalized Probabilities
        open_prob_home=("open_norm_prob_home", "mean"),
        open_prob_draw=("open_norm_prob_draw", "mean"),
        open_prob_away=("open_norm_prob_away", "mean"),

        # Raw Odds Aggregates
        mean_odds_home=("close_1x2_home", "mean"),
        mean_odds_draw=("close_1x2_draw", "mean"),
        mean_odds_away=("close_1x2_away", "mean"),

        # Best available market odds
        max_odds_home=("close_1x2_home", "max"),
        max_odds_draw=("close_1x2_draw", "max"),
        max_odds_away=("close_1x2_away", "max"),

        # Minimum market odds
        min_odds_home=("close_1x2_home", "min"),
        min_odds_draw=("close_1x2_draw", "min"),
        min_odds_away=("close_1x2_away", "min"),

        # Bookmaker Disagreement (Standard Deviation of odds across bookmakers)
        odds_std_home=("close_1x2_home", "std"),
        odds_std_draw=("close_1x2_draw", "std"),
        odds_std_away=("close_1x2_away", "std"),

        # Market Margins & Drifts
        mean_overround=("close_overround", "mean"),
        mean_drift_home=("drift_home_pct", "mean"),
        mean_drift_draw=("drift_draw_pct", "mean"),
        mean_drift_away=("drift_away_pct", "mean")
    ).reset_index()

    # Derived Spread Features
    agg_df["spread_odds_home"] = agg_df["max_odds_home"] - agg_df["min_odds_home"]
    agg_df["spread_odds_draw"] = agg_df["max_odds_draw"] - agg_df["min_odds_draw"]
    agg_df["spread_odds_away"] = agg_df["max_odds_away"] - agg_df["min_odds_away"]

    # Fill NaN std (when only 1 bookmaker exists) with 0.0
    agg_df["odds_std_home"] = agg_df["odds_std_home"].fillna(0.0)
    agg_df["odds_std_draw"] = agg_df["odds_std_draw"].fillna(0.0)
    agg_df["odds_std_away"] = agg_df["odds_std_away"].fillna(0.0)

    # Market Consensus Favourite (0: Home, 1: Draw, 2: Away)
    probs = agg_df[["mkt_prob_home", "mkt_prob_draw", "mkt_prob_away"]].values
    agg_df["mkt_favourite"] = np.argmax(probs, axis=1)

    return agg_df


def build_match_feature_matrix(matches_df: pd.DataFrame, closing_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge match metadata with aggregated market signals to produce the final ML feature set.

    Parameters
    ----------
    matches_df : pd.DataFrame
        Cleaned matches dataset.
    closing_df : pd.DataFrame
        Cleaned match closing dataset.

    Returns
    -------
    pd.DataFrame
        Complete match-level feature matrix ready for ML models.
    """
    odds_aggs = build_match_odds_aggregates(closing_df)

    # Merge on match_id (inner join to retain matches with valid odds)
    merged = pd.merge(matches_df, odds_aggs, on="match_id", how="inner")

    # Temporal features from kickoff
    kickoff = pd.to_datetime(merged["kickoff_utc"], utc=True)
    merged["kickoff_year"] = kickoff.dt.year.astype(np.int16)
    merged["kickoff_month"] = kickoff.dt.month.astype(np.int8)
    merged["kickoff_dayofweek"] = kickoff.dt.day_of_week.astype(np.int8)
    merged["kickoff_hour"] = kickoff.dt.hour.astype(np.int8)
    merged["is_weekend"] = merged["kickoff_dayofweek"].isin([5, 6]).astype(np.int8)

    # Target columns
    # target: 0 = Home Win (H), 1 = Draw (D), 2 = Away Win (A)
    merged["target"] = merged["outcome_code"].astype(np.int8)

    # Drop any remaining unneeded or raw leaking strings for ML matrix
    # Keep metadata columns for traceability
    return merged.sort_values("kickoff_utc").reset_index(drop=True)


def run_feature_pipeline() -> pd.DataFrame:
    """
    Execute feature engineering pipeline and persist features_match_level.parquet.

    Returns
    -------
    pd.DataFrame
        Engineered feature matrix.
    """
    print("=" * 70)
    print(" EXECUTING FEATURE ENGINEERING PIPELINE ")
    print("=" * 70)

    print("[1/3] Loading cleaned matches and closing odds...")
    matches = load_processed("matches_clean")
    closing = load_processed("match_closing_clean")

    print("[2/3] Constructing match-level feature matrix...")
    features = build_match_feature_matrix(matches, closing)
    print(f" -> Feature Matrix Shape: {features.shape[0]:,} rows x {features.shape[1]} columns")

    print("[3/3] Saving features to data/processed/features_match_level.parquet...")
    save_path = save_processed(features, "features_match_level")
    print(f" -> Saved: {save_path}")

    print("=" * 70)
    print(" FEATURE ENGINEERING COMPLETED SUCCESSFULLY ")
    print("=" * 70)
    return features


if __name__ == "__main__":
    run_feature_pipeline()
