"""
Statistical Analysis & Hypothesis Testing Module
================================================
Conducts rigorous parametric and non-parametric statistical tests:
1. Bookmaker Margin Differences (Kruskal-Wallis H-test & Mann-Whitney U)
2. Opening vs Closing Odds Drift (Wilcoxon Signed-Rank Test)
3. League vs Match Outcome Independence (Chi-Square Test)
4. Home Advantage Significance (One-sample Z-test / Binomial Test)
"""

from typing import Any
import numpy as np
import pandas as pd
from scipy import stats


def evaluate_bookmaker_margin_differences(closing_df: pd.DataFrame) -> dict[str, Any]:
    """
    Kruskal-Wallis test evaluating whether bookmaker margins differ significantly.

    Parameters
    ----------
    closing_df : pd.DataFrame
        Cleaned match closing dataset.

    Returns
    -------
    dict
        Test statistics, p-value, and per-bookmaker descriptive stats.
    """
    groups = {}
    for bmk in closing_df["bookmaker"].unique():
        s = closing_df[closing_df["bookmaker"] == bmk]["close_overround"].dropna() * 100.0
        # Filter plausible range
        s = s[(s >= 0) & (s <= 30)]
        groups[bmk] = s.values

    # Kruskal-Wallis H-test (non-parametric ANOVA)
    h_stat, p_val = stats.kruskal(*groups.values())

    summary = {
        bmk: {
            "count": len(vals),
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "std": float(np.std(vals))
        }
        for bmk, vals in groups.items()
    }

    # Pairwise comparison between Pinnacle (sharpest) and Bet365 if present
    p_pin_b365 = None
    if "Pinnacle" in groups and "Bet365" in groups:
        _, p_val_u = stats.mannwhitneyu(groups["Pinnacle"], groups["Bet365"], alternative="less")
        p_pin_b365 = float(p_val_u)

    return {
        "test_name": "Kruskal-Wallis H-Test for Overround Margins",
        "h_statistic": float(h_stat),
        "p_value": float(p_val),
        "is_significant": bool(p_val < 0.05),
        "bookmaker_summaries": summary,
        "pinnacle_vs_bet365_p_value": p_pin_b365
    }


def evaluate_odds_drift_significance(closing_df: pd.DataFrame) -> dict[str, Any]:
    """
    Wilcoxon Signed-Rank test evaluating whether closing odds systematically
    drift higher or lower compared to opening odds.

    Parameters
    ----------
    closing_df : pd.DataFrame
        Cleaned match closing dataset.

    Returns
    -------
    dict
        Test results for Home, Draw, and Away odds drift.
    """
    results = {}
    cols = [
        ("open_1x2_home", "close_1x2_home", "home"),
        ("open_1x2_draw", "close_1x2_draw", "draw"),
        ("open_1x2_away", "close_1x2_away", "away")
    ]

    for open_col, close_col, market in cols:
        valid = closing_df.dropna(subset=[open_col, close_col])
        diff = valid[close_col] - valid[open_col]

        # Use 10,000 sample if dataset is massive to ensure fast computation
        if len(diff) > 20000:
            sample_diff = diff.sample(20000, random_state=42)
        else:
            sample_diff = diff

        w_stat, p_val = stats.wilcoxon(sample_diff, alternative="two-sided")
        results[market] = {
            "mean_drift": float(diff.mean()),
            "median_drift": float(diff.median()),
            "w_statistic": float(w_stat),
            "p_value": float(p_val),
            "is_significant": bool(p_val < 0.05)
        }

    return results


def evaluate_league_outcome_independence(matches_df: pd.DataFrame, min_matches: int = 100) -> dict[str, Any]:
    """
    Chi-Square Test of Independence evaluating whether match outcome proportions
    (H, D, A) vary significantly across competitions.

    Parameters
    ----------
    matches_df : pd.DataFrame
        Cleaned matches DataFrame.
    min_matches : int
        Minimum matches required for league inclusion.

    Returns
    -------
    dict
        Chi-square statistic, p-value, degrees of freedom, and contingency table.
    """
    counts = matches_df["league_name"].value_counts()
    valid_leagues = counts[counts >= min_matches].index

    sub = matches_df[matches_df["league_name"].isin(valid_leagues)]
    contingency = pd.crosstab(sub["league_name"], sub["outcome"])

    chi2, p_val, dof, expected = stats.chi2_contingency(contingency)

    return {
        "test_name": "Chi-Square Test of Independence (League vs Outcome)",
        "chi2_statistic": float(chi2),
        "p_value": float(p_val),
        "degrees_of_freedom": int(dof),
        "n_leagues_evaluated": len(valid_leagues),
        "is_significant": bool(p_val < 0.05)
    }


def evaluate_home_advantage_significance(matches_df: pd.DataFrame) -> dict[str, Any]:
    """
    One-sample proportion Z-test evaluating whether Home Win rate is
    significantly greater than the 1/3 (33.33%) uniform random expectation.

    Parameters
    ----------
    matches_df : pd.DataFrame
        Cleaned matches DataFrame.

    Returns
    -------
    dict
        Sample size, home win rate, z-statistic, p-value, and 95% confidence interval.
    """
    n = len(matches_df)
    k = (matches_df["outcome"] == "H").sum()
    p_hat = k / n
    p_null = 1.0 / 3.0

    se = np.sqrt(p_null * (1.0 - p_null) / n)
    z_stat = (p_hat - p_null) / se
    p_val = 1.0 - stats.norm.cdf(z_stat)

    # 95% Wald Confidence Interval
    ci_se = np.sqrt(p_hat * (1.0 - p_hat) / n)
    ci_lower = p_hat - 1.96 * ci_se
    ci_upper = p_hat + 1.96 * ci_se

    return {
        "total_matches": int(n),
        "home_wins": int(k),
        "home_win_rate": float(p_hat),
        "z_statistic": float(z_stat),
        "p_value": float(p_val),
        "ci_95": (float(ci_lower), float(ci_upper)),
        "is_significant": bool(p_val < 0.05)
    }
