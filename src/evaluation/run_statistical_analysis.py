"""
Phase 4 — Statistical Analysis & EDA Execution Runner
=====================================================
Executes statistical tests and generates Phase 4 figures using real cleaned data:
1. Bookmaker margin differences (Kruskal-Wallis & Mann-Whitney U)
2. Opening vs closing odds drift (Wilcoxon Signed-Rank)
3. League independence (Chi-Square)
4. Home advantage significance (Z-test)
Saves figures to reports/figures/ and results to reports/analysis/statistical_summary.json.
"""

import json
import os
import sys
from pathlib import Path
import pandas as pd

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

os.environ["MPLCONFIGDIR"] = str(project_root / ".matplotlib_cache")

from src.data.loader import load_processed
from src.evaluation.statistical_tests import (
    evaluate_bookmaker_margin_differences,
    evaluate_odds_drift_significance,
    evaluate_league_outcome_independence,
    evaluate_home_advantage_significance,
)
from src.visualization.plots import (
    plot_bookmaker_margins,
    plot_odds_drift_distribution,
    plot_league_home_advantage,
)


def run_analysis():
    print("=" * 70)
    print(" PHASE 4: STATISTICAL ANALYSIS & EDA EXECUTION ")
    print("=" * 70)

    # 1. Load Cleaned Data
    print("[1/4] Loading cleaned datasets...")
    matches = load_processed("matches_clean")
    closing = load_processed("match_closing_clean")
    print(f" -> Matches: {matches.shape[0]:,} rows")
    print(f" -> Closing: {closing.shape[0]:,} rows")

    # 2. Run Statistical Tests
    print("\n[2/4] Executing hypothesis tests...")

    # Test 1: Bookmaker Margins
    margin_res = evaluate_bookmaker_margin_differences(closing)
    print(f"\n1. Bookmaker Margin H-test: H = {margin_res['h_statistic']:.2f}, p-value = {margin_res['p_value']:.4e}")
    print(f"   Pinnacle vs Bet365 Mann-Whitney p-value: {margin_res['pinnacle_vs_bet365_p_value']:.4e}")
    for bmk, stats_dict in margin_res["bookmaker_summaries"].items():
        print(f"   - {bmk:<10}: Median={stats_dict['median']:.2f}%, Mean={stats_dict['mean']:.2f}% (std={stats_dict['std']:.2f})")

    # Test 2: Odds Drift
    drift_res = evaluate_odds_drift_significance(closing)
    print("\n2. Odds Drift Wilcoxon Tests:")
    for mkt, d in drift_res.items():
        print(f"   - {mkt:<6}: Mean drift = {d['mean_drift']:+.4f}%, Median = {d['median_drift']:+.4f}%, p = {d['p_value']:.4e}")

    # Test 3: League Independence
    league_res = evaluate_league_outcome_independence(matches, min_matches=200)
    print(f"\n3. League Independence Chi-Square: chi2 = {league_res['chi2_statistic']:.2f}, dof = {league_res['degrees_of_freedom']}, p = {league_res['p_value']:.4e}")
    print(f"   Evaluated across {league_res['n_leagues_evaluated']} major competitions.")

    # Test 4: Home Advantage
    home_res = evaluate_home_advantage_significance(matches)
    print(f"\n4. Home Advantage Proportion Z-Test: Home Win Rate = {home_res['home_win_rate']*100:.2f}%")
    print(f"   Z-stat = {home_res['z_statistic']:.2f}, p-value = {home_res['p_value']:.4e}")
    print(f"   95% Confidence Interval: [{home_res['ci_95'][0]*100:.2f}%, {home_res['ci_95'][1]*100:.2f}%]")

    # 3. Generate Visualizations
    print("\n[3/4] Generating publication-grade charts...")
    fig_dir = project_root / "reports" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    f1 = plot_bookmaker_margins(closing, save_path=fig_dir / "bookmaker_margins.png")
    print(" -> Saved: reports/figures/bookmaker_margins.png")

    f2 = plot_odds_drift_distribution(closing, save_path=fig_dir / "odds_drift_distribution.png")
    print(" -> Saved: reports/figures/odds_drift_distribution.png")

    f3 = plot_league_home_advantage(matches, min_matches=200, save_path=fig_dir / "league_home_advantage.png")
    print(" -> Saved: reports/figures/league_home_advantage.png")

    # 4. Save JSON Summary
    print("\n[4/4] Saving analysis report...")
    report_dir = project_root / "reports" / "analysis"
    report_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "bookmaker_margin_test": margin_res,
        "odds_drift_test": drift_res,
        "league_independence_test": league_res,
        "home_advantage_test": home_res
    }
    with open(report_dir / "statistical_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f" -> Saved: reports/analysis/statistical_summary.json")
    print("=" * 70)


if __name__ == "__main__":
    run_analysis()
