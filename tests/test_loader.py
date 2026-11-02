"""
Unit Tests for Data Loader Module
=================================
Tests data loading, format handling, schema integrity, and filtering.
"""

import pytest
import pandas as pd
from pathlib import Path
from src.data.loader import load_matches, load_match_closing, load_odds_1x2


def test_load_matches_schema():
    """Verify matches dataframe loads with expected shape, columns, and types."""
    df = load_matches()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 63597
    assert "match_id" in df.columns
    assert "kickoff_utc" in df.columns
    assert "ft_home" in df.columns
    assert "ft_away" in df.columns
    assert df["match_id"].is_unique


def test_load_match_closing_schema():
    """Verify match closing dataset loads with expected dimensions and bookmakers."""
    df = load_match_closing()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 251326
    assert "bookmaker" in df.columns
    assert "close_1x2_home" in df.columns
    assert "close_1x2_draw" in df.columns
    assert "close_1x2_away" in df.columns
    # Check that the 4 major bookmakers exist
    bookmakers = set(df["bookmaker"].dropna().unique())
    assert {"Bet365", "Pinnacle", "SBOBET", "Singbet"}.issubset(bookmakers)


def test_load_odds_1x2_filtered():
    """Verify loading odds_1x2 by match_ids retrieves only requested records."""
    matches = load_matches()
    test_id = int(matches["match_id"].iloc[0])
    df_sample = load_odds_1x2(match_ids=[test_id])
    assert isinstance(df_sample, pd.DataFrame)
    assert len(df_sample) > 0
    assert (df_sample["match_id"] == test_id).all()
