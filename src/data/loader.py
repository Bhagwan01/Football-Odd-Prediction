"""
Data Loader Module
==================
Provides functions to load raw and processed football odds datasets.

All data loading goes through this module to ensure:
- Consistent file paths (relative to project root)
- Consistent data types (especially datetime parsing)
- Memory-efficient loading (Parquet for large files)
- Clear error messages when files are missing

Usage:
    from src.data.loader import load_matches, load_match_closing, load_odds_1x2
    matches = load_matches()
    closing = load_match_closing()
"""

import os
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _project_root() -> Path:
    """Return the project root directory (two levels up from this file)."""
    return Path(__file__).resolve().parent.parent.parent


def _raw_dir() -> Path:
    """Return the path to the raw data directory."""
    return _project_root() / "data" / "raw"


def _processed_dir() -> Path:
    """Return the path to the processed data directory."""
    return _project_root() / "data" / "processed"


# ---------------------------------------------------------------------------
# Raw data loaders
# ---------------------------------------------------------------------------

def load_matches(source: str = "parquet") -> pd.DataFrame:
    """
    Load the matches dataset.

    Parameters
    ----------
    source : str
        File format to load: 'parquet' (default, faster) or 'csv'.

    Returns
    -------
    pd.DataFrame
        DataFrame with 63,597 rows and 26 columns.
        Datetime columns are parsed automatically.

    Raises
    ------
    FileNotFoundError
        If the data file is not found in data/raw/.
    """
    raw = _raw_dir()
    if source == "parquet":
        path = raw / "matches.parquet"
    else:
        path = raw / "matches.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Matches data not found at {path}. "
            "Ensure raw data files are in data/raw/."
        )

    if source == "parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, parse_dates=["kickoff_utc"])

    # Ensure datetime parsing for parquet too
    if "kickoff_utc" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["kickoff_utc"]):
        df["kickoff_utc"] = pd.to_datetime(df["kickoff_utc"], utc=True)

    return df


def load_match_closing() -> pd.DataFrame:
    """
    Load the match closing odds dataset (opening and closing odds per bookmaker).

    Returns
    -------
    pd.DataFrame
        DataFrame with ~251,326 rows and 26 columns.

    Raises
    ------
    FileNotFoundError
        If the data file is not found.
    """
    path = _raw_dir() / "match_closing.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Match closing data not found at {path}. "
            "Ensure raw data files are in data/raw/."
        )

    df = pd.read_parquet(path)

    # Parse timestamp columns if they aren't already datetime
    for col in ["close_ah_ts", "close_ou_ts"]:
        if col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")

    return df


def load_odds_1x2(columns: list[str] | None = None,
                  sample_frac: float | None = None,
                  match_ids: list[int] | None = None) -> pd.DataFrame:
    """
    Load the 1X2 odds time-series dataset.

    This is a large dataset (~11.6M rows). Use parameters to limit
    the data loaded into memory.

    Parameters
    ----------
    columns : list of str, optional
        Subset of columns to load. Default loads all.
    sample_frac : float, optional
        Fraction of rows to randomly sample (0.0 to 1.0).
        Applied after loading, so still reads the full file.
    match_ids : list of int, optional
        Only load rows for these match IDs. More memory-efficient
        than loading all and filtering.

    Returns
    -------
    pd.DataFrame
        Odds time-series data.

    Raises
    ------
    FileNotFoundError
        If the data file is not found.
    """
    path = _raw_dir() / "odds_1x2.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Odds 1X2 data not found at {path}. "
            "Ensure raw data files are in data/raw/."
        )

    # Use PyArrow filters for match_id filtering (memory efficient)
    filters = None
    if match_ids is not None:
        filters = [("match_id", "in", match_ids)]

    df = pd.read_parquet(path, columns=columns, filters=filters)

    if sample_frac is not None:
        df = df.sample(frac=sample_frac, random_state=42)

    # Parse timestamp
    if "ts_utc" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["ts_utc"]):
        df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)

    return df


def load_odds_ah(columns: list[str] | None = None,
                 sample_frac: float | None = None,
                 match_ids: list[int] | None = None) -> pd.DataFrame:
    """
    Load the Asian Handicap odds time-series dataset.

    This is the largest dataset (~15.6M rows). Use parameters to limit
    the data loaded into memory.

    Parameters
    ----------
    columns : list of str, optional
        Subset of columns to load.
    sample_frac : float, optional
        Fraction of rows to randomly sample.
    match_ids : list of int, optional
        Only load rows for these match IDs.

    Returns
    -------
    pd.DataFrame
        Asian Handicap odds time-series data.
    """
    path = _raw_dir() / "odds_ah.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Odds AH data not found at {path}. "
            "Ensure raw data files are in data/raw/."
        )

    filters = None
    if match_ids is not None:
        filters = [("match_id", "in", match_ids)]

    df = pd.read_parquet(path, columns=columns, filters=filters)

    if sample_frac is not None:
        df = df.sample(frac=sample_frac, random_state=42)

    if "ts_utc" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["ts_utc"]):
        df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)

    return df


def load_odds_ou(columns: list[str] | None = None,
                 sample_frac: float | None = None,
                 match_ids: list[int] | None = None) -> pd.DataFrame:
    """
    Load the Over/Under odds time-series dataset.

    This is a large dataset (~9.8M rows). Use parameters to limit
    the data loaded into memory.

    Parameters
    ----------
    columns : list of str, optional
        Subset of columns to load.
    sample_frac : float, optional
        Fraction of rows to randomly sample.
    match_ids : list of int, optional
        Only load rows for these match IDs.

    Returns
    -------
    pd.DataFrame
        Over/Under odds time-series data.
    """
    path = _raw_dir() / "odds_ou.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Odds OU data not found at {path}. "
            "Ensure raw data files are in data/raw/."
        )

    filters = None
    if match_ids is not None:
        filters = [("match_id", "in", match_ids)]

    df = pd.read_parquet(path, columns=columns, filters=filters)

    if sample_frac is not None:
        df = df.sample(frac=sample_frac, random_state=42)

    if "ts_utc" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["ts_utc"]):
        df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)

    return df


# ---------------------------------------------------------------------------
# Processed data loaders
# ---------------------------------------------------------------------------

def load_processed(name: str) -> pd.DataFrame:
    """
    Load a processed dataset by name.

    Parameters
    ----------
    name : str
        Name of the processed file (without extension).
        Example: 'matches_clean', 'features_match_level'

    Returns
    -------
    pd.DataFrame
        The processed dataset.
    """
    path = _processed_dir() / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Processed data '{name}' not found at {path}. "
            "Run the data cleaning pipeline first."
        )
    return pd.read_parquet(path)


def save_processed(df: pd.DataFrame, name: str) -> Path:
    """
    Save a processed dataset.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to save.
    name : str
        Name for the file (without extension).

    Returns
    -------
    Path
        Path to the saved file.
    """
    path = _processed_dir() / f"{name}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path
