"""
Unit Tests for Data Cleaner Module
==================================
Tests data cleaning, outcome derivations, odds sanity checks, and implied probability math.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.cleaner import clean_matches, clean_match_closing


def test_clean_matches_outcomes():
    """Verify that match outcome logic maps scores to H, D, A accurately."""
    mock_df = pd.DataFrame({
        "match_id": [1, 2, 3],
        "kickoff_utc": ["2023-08-12T15:00:00Z", "2023-08-12T17:30:00Z", "2023-08-13T14:00:00Z"],
        "league_name": ["Premier League", "Premier League", "La Liga"],
        "home_team": ["Arsenal", "Chelsea", "Real Madrid"],
        "away_team": ["Forest", "Liverpool", "Athletic"],
        "ft_home": [2, 1, 0],
        "ft_away": [1, 1, 2],
        "ht_home": [1, 1, 0],
        "ht_away": [0, 1, 1],
    })

    cleaned = clean_matches(mock_df)
    assert list(cleaned["outcome"]) == ["H", "D", "A"]
    assert list(cleaned["outcome_code"]) == [0, 1, 2]
    assert list(cleaned["total_goals"]) == [3, 2, 2]
    assert list(cleaned["goal_diff"]) == [1, 0, -2]


def test_clean_match_closing_odds_math():
    """Verify implied probability and overround math on known values."""
    mock_df = pd.DataFrame({
        "match_id": [101],
        "bookmaker": ["TestBookie"],
        "open_1x2_home": [2.0],
        "open_1x2_draw": [4.0],
        "open_1x2_away": [4.0],
        "close_1x2_home": [1.90],
        "close_1x2_draw": [3.80],
        "close_1x2_away": [4.50],
        "open_ah_home": [np.nan],
        "open_ah_away": [np.nan],
        "close_ah_home": [np.nan],
        "close_ah_away": [np.nan],
        "open_ou_over": [np.nan],
        "open_ou_under": [np.nan],
        "close_ou_over": [np.nan],
        "close_ou_under": [np.nan]
    })

    cleaned = clean_match_closing(mock_df)

    # Opening math:
    # 1/2.0 = 0.50, 1/4.0 = 0.25, 1/4.0 = 0.25 -> sum = 1.00 (fair market, 0% overround)
    assert pytest.approx(cleaned["open_ip_home"].iloc[0], 0.0001) == 0.50
    assert pytest.approx(cleaned["open_ip_draw"].iloc[0], 0.0001) == 0.25
    assert pytest.approx(cleaned["open_ip_away"].iloc[0], 0.0001) == 0.25
    assert pytest.approx(cleaned["open_overround"].iloc[0], 0.0001) == 0.00
    assert pytest.approx(cleaned["open_norm_prob_home"].iloc[0], 0.0001) == 0.50

    # Normalized probabilities must strictly sum to 1.0
    norm_sum = (
        cleaned["close_norm_prob_home"].iloc[0] +
        cleaned["close_norm_prob_draw"].iloc[0] +
        cleaned["close_norm_prob_away"].iloc[0]
    )
    assert pytest.approx(norm_sum, 1e-6) == 1.0

    # Odds drift test:
    # Home went from 2.0 to 1.90 -> (1.90 - 2.0)/2.0 * 100 = -5%
    assert pytest.approx(cleaned["drift_home_pct"].iloc[0], 0.01) == -5.0
    # Favourite test: home has lowest odds (1.90) -> 'H'
    assert cleaned["close_favourite"].iloc[0] == "H"


def test_invalid_odds_replaced_with_nan():
    """Verify that decimal odds <= 1.0 are replaced with NaN."""
    mock_df = pd.DataFrame({
        "match_id": [201],
        "bookmaker": ["BadBookie"],
        "open_1x2_home": [0.95],   # Invalid
        "open_1x2_draw": [1.00],   # Invalid
        "open_1x2_away": [2.50],   # Valid
        "close_1x2_home": [1.80],
        "close_1x2_draw": [3.40],
        "close_1x2_away": [4.20],
    })

    cleaned = clean_match_closing(mock_df)
    assert pd.isna(cleaned["open_1x2_home"].iloc[0])
    assert pd.isna(cleaned["open_1x2_draw"].iloc[0])
    assert cleaned["open_1x2_away"].iloc[0] == 2.50
