"""
Data Cleaning Module
====================
Provides robust, reproducible cleaning functions for football match metadata
and multi-bookmaker odds datasets.

Preserves raw inputs by returning new DataFrame instances.
Saves cleaned datasets in Apache Parquet format inside `data/processed/`.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data.loader import load_matches, load_match_closing, save_processed


def clean_matches(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the master matches dataset.

    Transformations:
    - Verifies match_id uniqueness and drops null identifiers.
    - Standardizes kickoff_utc to timezone-aware UTC datetime.
    - Validates scores and removes any rows with missing or negative full-time scores.
    - Derives match outcome ('H': Home Win, 'D': Draw, 'A': Away Win).
    - Derives outcome_code (0: H, 1: D, 2: A).
    - Derives total_goals (ft_home + ft_away) and goal_diff (ft_home - ft_away).
    - Derives ht_total_goals (ht_home + ht_away) and ht_goal_diff.
    - Formats categorical attributes (league_name, season, comp_key).

    Parameters
    ----------
    df : pd.DataFrame
        Raw matches DataFrame.

    Returns
    -------
    pd.DataFrame
        Cleaned matches DataFrame.
    """
    clean = df.copy()

    # Drop any null match_id and ensure integer type
    clean = clean.dropna(subset=["match_id"])
    clean["match_id"] = clean["match_id"].astype(np.int64)

    # Standardize kickoff datetime
    if not pd.api.types.is_datetime64_any_dtype(clean["kickoff_utc"]):
        clean["kickoff_utc"] = pd.to_datetime(clean["kickoff_utc"], utc=True, errors="coerce")

    # Drop matches without valid kickoff or full-time scores
    clean = clean.dropna(subset=["kickoff_utc", "ft_home", "ft_away"])

    # Ensure non-negative goal integers
    clean = clean[(clean["ft_home"] >= 0) & (clean["ft_away"] >= 0)].copy()
    clean["ft_home"] = clean["ft_home"].astype(np.int16)
    clean["ft_away"] = clean["ft_away"].astype(np.int16)

    # Clean half-time goals where present
    if "ht_home" in clean.columns and "ht_away" in clean.columns:
        # If HT score exceeds FT score, impute with FT score or flag
        clean["ht_home"] = clean["ht_home"].clip(lower=0, upper=clean["ft_home"])
        clean["ht_away"] = clean["ht_away"].clip(lower=0, upper=clean["ft_away"])
        clean["ht_total_goals"] = clean["ht_home"] + clean["ht_away"]
        clean["ht_goal_diff"] = clean["ht_home"] - clean["ht_away"]

    # Match outcome derivations
    conditions = [
        clean["ft_home"] > clean["ft_away"],
        clean["ft_home"] == clean["ft_away"],
        clean["ft_home"] < clean["ft_away"]
    ]
    clean["outcome"] = np.select(conditions, ["H", "D", "A"], default="Unknown")
    clean["outcome_code"] = np.select(conditions, [0, 1, 2], default=-1).astype(np.int8)

    clean["total_goals"] = (clean["ft_home"] + clean["ft_away"]).astype(np.int16)
    clean["goal_diff"] = (clean["ft_home"] - clean["ft_away"]).astype(np.int16)

    # Clean string attributes
    string_cols = ["league_name", "home_team", "away_team", "comp_key", "season", "stage"]
    for col in string_cols:
        if col in clean.columns:
            clean[col] = clean[col].astype(str).str.strip()

    # Neutral venue binary indicator
    if "neutral" in clean.columns:
        clean["neutral"] = clean["neutral"].fillna(0).astype(np.int8)

    # Sort chronologically
    clean = clean.sort_values("kickoff_utc").reset_index(drop=True)
    return clean


def clean_match_closing(df: pd.DataFrame, valid_match_ids: set[int] | None = None) -> pd.DataFrame:
    """
    Clean the match_closing dataset and calculate mathematical odds metrics.

    Transformations:
    - Enforces referential integrity against valid_match_ids.
    - Audits decimal odds: replaces invalid values (<= 1.0) with NaN.
    - Computes raw implied probabilities: P_raw = 1 / odds.
    - Computes bookmaker overround (margin): overround = (P_h + P_d + P_a) - 1.
    - Computes normalized (margin-removed) fair implied probabilities:
        P_fair = P_raw / (1 + overround)
    - Computes odds movement (drift percentage) between opening and closing odds:
        drift% = (close - open) / open * 100
    - Identifies market favourites at open and close.

    Parameters
    ----------
    df : pd.DataFrame
        Raw match_closing DataFrame.
    valid_match_ids : set of int, optional
        Set of valid match_id values from cleaned matches table.

    Returns
    -------
    pd.DataFrame
        Cleaned match_closing DataFrame with implied probabilities and margins.
    """
    clean = df.copy()

    # Referential integrity filter
    if valid_match_ids is not None:
        clean = clean[clean["match_id"].isin(valid_match_ids)].copy()

    clean["match_id"] = clean["match_id"].astype(np.int64)
    clean["bookmaker"] = clean["bookmaker"].astype(str).str.strip()

    # Odds sanity: replace any odds <= 1.0 with NaN
    odds_cols = [
        "open_1x2_home", "open_1x2_draw", "open_1x2_away",
        "close_1x2_home", "close_1x2_draw", "close_1x2_away",
        "open_ah_home", "open_ah_away", "close_ah_home", "close_ah_away",
        "open_ou_over", "open_ou_under", "close_ou_over", "close_ou_under"
    ]
    for col in odds_cols:
        if col in clean.columns:
            clean[col] = pd.to_numeric(clean[col], errors="coerce")
            clean.loc[clean[col] <= 1.0, col] = np.nan

    # 1. Opening 1X2 Implied Probabilities & Overround
    clean["open_ip_home"] = 1.0 / clean["open_1x2_home"]
    clean["open_ip_draw"] = 1.0 / clean["open_1x2_draw"]
    clean["open_ip_away"] = 1.0 / clean["open_1x2_away"]

    clean["open_overround"] = (clean["open_ip_home"] + clean["open_ip_draw"] + clean["open_ip_away"]) - 1.0
    open_sum = clean["open_ip_home"] + clean["open_ip_draw"] + clean["open_ip_away"]

    clean["open_norm_prob_home"] = clean["open_ip_home"] / open_sum
    clean["open_norm_prob_draw"] = clean["open_ip_draw"] / open_sum
    clean["open_norm_prob_away"] = clean["open_ip_away"] / open_sum

    # 2. Closing 1X2 Implied Probabilities & Overround
    clean["close_ip_home"] = 1.0 / clean["close_1x2_home"]
    clean["close_ip_draw"] = 1.0 / clean["close_1x2_draw"]
    clean["close_ip_away"] = 1.0 / clean["close_1x2_away"]

    clean["close_overround"] = (clean["close_ip_home"] + clean["close_ip_draw"] + clean["close_ip_away"]) - 1.0
    close_sum = clean["close_ip_home"] + clean["close_ip_draw"] + clean["close_ip_away"]

    clean["close_norm_prob_home"] = clean["close_ip_home"] / close_sum
    clean["close_norm_prob_draw"] = clean["close_ip_draw"] / close_sum
    clean["close_norm_prob_away"] = clean["close_ip_away"] / close_sum

    # 3. Odds Drift (Movement percentage from open to close)
    clean["drift_home_pct"] = ((clean["close_1x2_home"] - clean["open_1x2_home"]) / clean["open_1x2_home"]) * 100.0
    clean["drift_draw_pct"] = ((clean["close_1x2_draw"] - clean["open_1x2_draw"]) / clean["open_1x2_draw"]) * 100.0
    clean["drift_away_pct"] = ((clean["close_1x2_away"] - clean["open_1x2_away"]) / clean["open_1x2_away"]) * 100.0

    # 4. Market Favourites
    def determine_favourite(h, d, a):
        if pd.isna(h) or pd.isna(d) or pd.isna(a):
            return "Unknown"
        min_val = min(h, d, a)
        if h == min_val:
            return "H"
        elif a == min_val:
            return "A"
        return "D"

    clean["close_favourite"] = [
        determine_favourite(h, d, a)
        for h, d, a in zip(clean["close_1x2_home"], clean["close_1x2_draw"], clean["close_1x2_away"])
    ]

    return clean.reset_index(drop=True)


def run_cleaning_pipeline() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Execute full end-to-end data cleaning pipeline and save processed Parquet files.

    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame)
        Cleaned matches and cleaned match_closing DataFrames.
    """
    print("=" * 70)
    print(" EXECUTING DATA CLEANING PIPELINE ")
    print("=" * 70)

    print("[1/4] Loading raw data...")
    raw_matches = load_matches()
    raw_closing = load_match_closing()

    print("[2/4] Cleaning matches dataset...")
    matches_clean = clean_matches(raw_matches)
    print(f" -> Matches clean: {matches_clean.shape[0]:,} rows x {matches_clean.shape[1]} cols")

    print("[3/4] Cleaning match_closing dataset & computing implied probabilities...")
    valid_ids = set(matches_clean["match_id"])
    closing_clean = clean_match_closing(raw_closing, valid_match_ids=valid_ids)
    print(f" -> Closing clean: {closing_clean.shape[0]:,} rows x {closing_clean.shape[1]} cols")

    print("[4/4] Saving processed datasets to data/processed/...")
    p1 = save_processed(matches_clean, "matches_clean")
    p2 = save_processed(closing_clean, "match_closing_clean")
    print(f" -> Saved: {p1}")
    print(f" -> Saved: {p2}")

    print("=" * 70)
    print(" DATA CLEANING PIPELINE COMPLETED SUCCESSFULLY ")
    print("=" * 70)
    return matches_clean, closing_clean


if __name__ == "__main__":
    run_cleaning_pipeline()
