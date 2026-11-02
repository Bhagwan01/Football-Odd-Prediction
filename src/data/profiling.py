"""
Data Profiling & Quality Audit Runner
=====================================
Executes automated profiling across matches, match_closing, and odds data:
- Missing value audit
- Referential integrity verification
- Anomaly / consistency checks (HT vs FT, negative odds, margins)
- Generation of Phase 2 figures in reports/figures/
"""

import os
import sys
from pathlib import Path

# Set MPLCONFIGDIR to a local directory within the project or temp to avoid Windows permissions error
project_root = Path(__file__).resolve().parent.parent.parent
os.environ["MPLCONFIGDIR"] = str(project_root / ".matplotlib_cache")

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.data.loader import load_matches, load_match_closing, load_odds_1x2, load_odds_ah, load_odds_ou

figures_dir = project_root / "reports" / "figures"
figures_dir.mkdir(parents=True, exist_ok=True)


def run_profiling():
    print("=" * 70)
    print(" PHASE 2: DATA UNDERSTANDING & QUALITY AUDIT ")
    print("=" * 70)

    # 1. Load Data
    print("\n[1/6] Loading datasets...")
    matches = load_matches()
    closing = load_match_closing()
    print(f"Matches loaded:       {matches.shape[0]:,} rows x {matches.shape[1]} cols")
    print(f"Match Closing loaded: {closing.shape[0]:,} rows x {closing.shape[1]} cols")

    # 2. Check Match ID & Referential Integrity
    print("\n[2/6] Auditing keys & referential integrity...")
    match_id_unique = matches["match_id"].is_unique
    print(f" - matches['match_id'] is globally unique: {match_id_unique}")
    
    closing_matches = set(closing["match_id"])
    master_matches = set(matches["match_id"])
    orphans = closing_matches - master_matches
    unmatched_in_closing = master_matches - closing_matches
    print(f" - Orphan match_ids in match_closing (not in matches): {len(orphans)}")
    print(f" - Matches without closing odds: {len(unmatched_in_closing)}")

    # 3. Score & Logic Audits
    print("\n[3/6] Auditing match score logic...")
    # Outcome derivation: H, D, A
    def get_outcome(row):
        if pd.isna(row["ft_home"]) or pd.isna(row["ft_away"]):
            return np.nan
        if row["ft_home"] > row["ft_away"]:
            return "H"
        elif row["ft_home"] == row["ft_away"]:
            return "D"
        return "A"

    matches["outcome"] = matches.apply(get_outcome, axis=1)
    outcome_dist = matches["outcome"].value_counts(dropna=False)
    print(" - Outcome distribution:\n", outcome_dist)

    invalid_scores = (matches["ft_home"] < 0) | (matches["ft_away"] < 0)
    print(f" - Negative full-time scores: {invalid_scores.sum()}")

    ht_inconsistent = (matches["ht_home"] > matches["ft_home"]) | (matches["ht_away"] > matches["ft_away"])
    print(f" - Half-time > Full-time violations: {ht_inconsistent.sum()}")

    # 4. Odds Sanity & Bookmaker Margins
    print("\n[4/6] Auditing decimal odds & overround...")
    odds_1x2_cols = ["open_1x2_home", "open_1x2_draw", "open_1x2_away",
                     "close_1x2_home", "close_1x2_draw", "close_1x2_away"]
    for col in odds_1x2_cols:
        invalid_odds = (closing[col] <= 1.0).sum()
        print(f" - {col:<18}: missing={closing[col].isna().sum():<6} invalid(<=1.0)={invalid_odds}")

    closing_valid = closing.dropna(subset=["close_1x2_home", "close_1x2_draw", "close_1x2_away"]).copy()
    closing_valid["overround"] = (
        (1 / closing_valid["close_1x2_home"]) +
        (1 / closing_valid["close_1x2_draw"]) +
        (1 / closing_valid["close_1x2_away"]) - 1.0
    ) * 100.0

    print("\nBookmaker Overround Statistics (%):")
    margin_summary = closing_valid.groupby("bookmaker")["overround"].agg(["count", "mean", "median", "std", "min", "max"])
    print(margin_summary.round(2))

    # 5. Missing Values Plot
    print("\n[5/6] Generating visualizations in reports/figures/...")
    
    # 5.1 Missing Values Chart
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    m_missing = matches.isna().sum()
    m_missing = m_missing[m_missing > 0].sort_values(ascending=True)
    if len(m_missing) > 0:
        m_missing.plot.barh(ax=axes[0], color="#e74c3c")
        axes[0].set_title("Missing Values — Matches Dataset", fontsize=12, fontweight="bold")
        axes[0].set_xlabel("Missing Count")
        for i, val in enumerate(m_missing):
            axes[0].text(val + 500, i, f"{val:,} ({val/len(matches)*100:.1f}%)", va="center", fontsize=8)
    
    c_missing = closing.isna().sum()
    c_missing = c_missing[c_missing > 0].sort_values(ascending=True)
    if len(c_missing) > 0:
        c_missing.plot.barh(ax=axes[1], color="#3498db")
        axes[1].set_title("Missing Values — Match Closing Dataset", fontsize=12, fontweight="bold")
        axes[1].set_xlabel("Missing Count")
        for i, val in enumerate(c_missing):
            axes[1].text(val + 1000, i, f"{val:,} ({val/len(closing)*100:.1f}%)", va="center", fontsize=8)
            
    plt.tight_layout()
    fig_missing = figures_dir / "missing_values.png"
    plt.savefig(fig_missing, dpi=150, bbox_inches="tight")
    plt.close()
    print(f" -> Saved: {fig_missing.name}")

    # 5.2 Initial Distributions
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = ["#2ecc71", "#f39c12", "#e74c3c"]
    valid_outcomes = matches["outcome"].dropna().value_counts()
    valid_outcomes.plot.bar(ax=axes[0], color=colors, edgecolor="black", alpha=0.85)
    axes[0].set_title("Match Outcomes Distribution (Full Time)", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Matches")
    axes[0].set_xticklabels(["Home Win (H)", "Draw (D)", "Away Win (A)"], rotation=0)
    for i, v in enumerate(valid_outcomes):
        axes[0].text(i, v + 400, f"{v:,}\n({v/len(matches)*100:.1f}%)", ha="center", fontsize=9)

    total_goals = (matches["ft_home"] + matches["ft_away"]).dropna()
    axes[1].hist(total_goals, bins=range(0, int(total_goals.max()) + 2), color="#3498db", edgecolor="black", align="left", alpha=0.85)
    axes[1].set_title("Total Goals per Match Distribution", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Goals")
    axes[1].set_ylabel("Frequency")
    axes[1].set_xlim(-0.5, 12)

    matches["year"] = pd.to_datetime(matches["kickoff_utc"], errors="coerce").dt.year
    year_counts = matches["year"].value_counts().sort_index()
    year_counts.plot.bar(ax=axes[2], color="#9b59b6", edgecolor="black", alpha=0.85)
    axes[2].set_title("Matches by Year", fontsize=12, fontweight="bold")
    axes[2].set_ylabel("Matches")
    axes[2].set_xticklabels(axes[2].get_xticklabels(), rotation=45)

    plt.tight_layout()
    fig_dist = figures_dir / "initial_distributions.png"
    plt.savefig(fig_dist, dpi=150, bbox_inches="tight")
    plt.close()
    print(f" -> Saved: {fig_dist.name}")

    # 5.3 Closing Odds Distribution
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    odds_specs = [
        ("close_1x2_home", "Closing Home Win Odds", "#16a085"),
        ("close_1x2_draw", "Closing Draw Odds", "#e67e22"),
        ("close_1x2_away", "Closing Away Win Odds", "#c0392b")
    ]
    for idx, (col, label, colr) in enumerate(odds_specs):
        s = closing[col].dropna()
        q99 = s.quantile(0.99)
        s_clipped = s[s <= q99]
        axes[idx].hist(s_clipped, bins=60, color=colr, edgecolor="white", alpha=0.85)
        axes[idx].axvline(s.median(), color="black", linestyle="--", linewidth=1.5, label=f"Median: {s.median():.2f}")
        axes[idx].set_title(label, fontsize=12, fontweight="bold")
        axes[idx].set_xlabel("Decimal Odds")
        axes[idx].set_ylabel("Count")
        axes[idx].legend()

    plt.tight_layout()
    fig_odds = figures_dir / "closing_odds_distribution.png"
    plt.savefig(fig_odds, dpi=150, bbox_inches="tight")
    plt.close()
    print(f" -> Saved: {fig_odds.name}")

    # 5.4 Overround Distribution by Bookmaker
    fig, ax = plt.subplots(figsize=(10, 5))
    for bmk in sorted(closing_valid["bookmaker"].unique()):
        sub = closing_valid[closing_valid["bookmaker"] == bmk]["overround"]
        sub_filtered = sub[(sub >= 0) & (sub <= 20)]
        ax.hist(sub_filtered, bins=50, alpha=0.45, label=f"{bmk} (median {sub.median():.1f}%)")

    ax.set_title("Bookmaker Margin (Overround %) Distribution", fontsize=13, fontweight="bold")
    ax.set_xlabel("Overround Margin (%)")
    ax.set_ylabel("Count")
    ax.axvline(0, color="gray", linestyle=":", label="Zero Margin (Fair Market)")
    ax.legend()
    plt.tight_layout()
    fig_overround = figures_dir / "overround_distribution.png"
    plt.savefig(fig_overround, dpi=150, bbox_inches="tight")
    plt.close()
    print(f" -> Saved: {fig_overround.name}")

    # 5.5 Odds Movement Example for a Match
    sample_match_id = matches["match_id"].iloc[0]
    m_info = matches[matches["match_id"] == sample_match_id].iloc[0]
    sample_1x2 = load_odds_1x2(match_ids=[sample_match_id])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    titles = ["Home Odds Movement", "Draw Odds Movement", "Away Odds Movement"]
    cols = ["home", "draw", "away"]

    for bmk in sample_1x2["bookmaker"].unique():
        b_df = sample_1x2[sample_1x2["bookmaker"] == bmk].sort_values("ts_utc")
        for i, c in enumerate(cols):
            axes[i].plot(b_df["ts_utc"], b_df[c], marker="o", markersize=3, label=bmk, alpha=0.85)

    for i in range(3):
        axes[i].set_title(titles[i], fontsize=12, fontweight="bold")
        axes[i].set_xlabel("Timestamp (UTC)")
        axes[i].set_ylabel("Decimal Odds")
        axes[i].tick_params(axis="x", rotation=30)
        axes[i].legend()

    plt.suptitle(f"Match ID {sample_match_id}: {m_info['home_team']} vs {m_info['away_team']} (FT: {int(m_info['ft_home'])}-{int(m_info['ft_away'])})",
                 fontsize=14, fontweight="bold", y=1.03)
    plt.tight_layout()
    fig_mov = figures_dir / "sample_odds_movement.png"
    plt.savefig(fig_mov, dpi=150, bbox_inches="tight")
    plt.close()
    print(f" -> Saved: {fig_mov.name}")

    print("\n[6/6] Profiling completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    run_profiling()
