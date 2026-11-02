"""
Unit Tests for Feature Engineering Module
=========================================
Tests feature aggregations, market spread math, consensus probabilities, and matrix construction.
"""

import numpy as np
import pandas as pd
import pytest

from src.features.builder import build_match_odds_aggregates, build_match_feature_matrix


def test_build_match_odds_aggregates():
    """Verify aggregation across multiple bookmakers for a single match."""
    mock_closing = pd.DataFrame({
        "match_id": [1, 1],
        "bookmaker": ["BookA", "BookB"],
        "close_1x2_home": [2.00, 2.20],
        "close_1x2_draw": [3.40, 3.20],
        "close_1x2_away": [3.60, 3.40],
        "close_norm_prob_home": [0.47, 0.43],
        "close_norm_prob_draw": [0.28, 0.29],
        "close_norm_prob_away": [0.25, 0.28],
        "open_norm_prob_home": [0.45, 0.44],
        "open_norm_prob_draw": [0.30, 0.28],
        "open_norm_prob_away": [0.25, 0.28],
        "close_overround": [0.05, 0.04],
        "drift_home_pct": [2.0, -1.0],
        "drift_draw_pct": [0.0, 1.0],
        "drift_away_pct": [-1.0, 0.0],
    })

    aggs = build_match_odds_aggregates(mock_closing)
    assert len(aggs) == 1
    assert aggs["n_bookmakers"].iloc[0] == 2
    # Mean home probability = (0.47 + 0.43) / 2 = 0.45
    assert pytest.approx(aggs["mkt_prob_home"].iloc[0], 0.001) == 0.45
    # Max home odds = 2.20, min = 2.00 -> spread = 0.20
    assert pytest.approx(aggs["spread_odds_home"].iloc[0], 0.001) == 0.20
    # Favourite is Home (index 0)
    assert aggs["mkt_favourite"].iloc[0] == 0


def test_build_match_feature_matrix():
    """Verify feature matrix correctly combines match attributes and market aggregates."""
    mock_matches = pd.DataFrame({
        "match_id": [1],
        "kickoff_utc": ["2023-10-14T15:00:00Z"],
        "league_name": ["Premier League"],
        "home_team": ["Arsenal"],
        "away_team": ["Chelsea"],
        "ft_home": [2],
        "ft_away": [1],
        "outcome": ["H"],
        "outcome_code": [0]
    })

    mock_closing = pd.DataFrame({
        "match_id": [1],
        "bookmaker": ["BookA"],
        "close_1x2_home": [1.90],
        "close_1x2_draw": [3.60],
        "close_1x2_away": [4.00],
        "close_norm_prob_home": [0.50],
        "close_norm_prob_draw": [0.26],
        "close_norm_prob_away": [0.24],
        "open_norm_prob_home": [0.48],
        "open_norm_prob_draw": [0.27],
        "open_norm_prob_away": [0.25],
        "close_overround": [0.04],
        "drift_home_pct": [-2.0],
        "drift_draw_pct": [1.0],
        "drift_away_pct": [2.0],
    })

    matrix = build_match_feature_matrix(mock_matches, mock_closing)
    assert len(matrix) == 1
    assert matrix["target"].iloc[0] == 0
    assert matrix["kickoff_year"].iloc[0] == 2023
    assert matrix["kickoff_month"].iloc[0] == 10
    # 2023-10-14 was Saturday -> is_weekend = 1
    assert matrix["is_weekend"].iloc[0] == 1
