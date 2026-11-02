"""
Visualization Module
====================
Provides reusable, styled plotting routines for football odds intelligence,
market margin analysis, drift distributions, ML model benchmarks, and confusion matrices.
"""

import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

project_root = Path(__file__).resolve().parent.parent.parent
cache_dir = project_root / ".matplotlib_cache"
cache_dir.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(cache_dir)

# Set global aesthetic style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8


def plot_bookmaker_margins(closing_df: pd.DataFrame, save_path: Path | None = None) -> plt.Figure:
    """
    Plot box plot and KDE distribution comparing overround margins across bookmakers.

    Parameters
    ----------
    closing_df : pd.DataFrame
        Cleaned match_closing DataFrame with 'bookmaker' and 'close_overround'.
    save_path : Path, optional
        Destination image path.

    Returns
    -------
    plt.Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    valid = closing_df.dropna(subset=["close_overround", "bookmaker"]).copy()
    valid["margin_pct"] = valid["close_overround"] * 100.0
    valid = valid[(valid["margin_pct"] >= 0) & (valid["margin_pct"] <= 25)]

    # 1. Boxplot
    sns.boxplot(
        data=valid,
        x="bookmaker",
        y="margin_pct",
        hue="bookmaker",
        legend=False,
        ax=axes[0],
        palette="Set2"
    )
    axes[0].set_title("Closing Margin (%) by Bookmaker", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Bookmaker", fontsize=11)
    axes[0].set_ylabel("Overround Margin (%)", fontsize=11)

    # 2. KDE Density
    sns.kdeplot(
        data=valid,
        x="margin_pct",
        hue="bookmaker",
        common_norm=False,
        fill=True,
        alpha=0.25,
        ax=axes[1],
        palette="Set2"
    )
    axes[1].set_title("Margin Density Distribution", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Overround Margin (%)", fontsize=11)
    axes[1].set_ylabel("Density", fontsize=11)

    plt.suptitle("Bookmaker Pricing Strategy & Margin Comparison", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_odds_drift_distribution(closing_df: pd.DataFrame, save_path: Path | None = None) -> plt.Figure:
    """
    Plot distribution of odds drift percentages for Home, Draw, and Away.

    Parameters
    ----------
    closing_df : pd.DataFrame
        Cleaned match_closing DataFrame.
    save_path : Path, optional
        Destination image path.

    Returns
    -------
    plt.Figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    drift_cols = [
        ("drift_home_pct", "Home Odds Drift (%)", "#27ae60"),
        ("drift_draw_pct", "Draw Odds Drift (%)", "#f39c12"),
        ("drift_away_pct", "Away Odds Drift (%)", "#c0392b")
    ]

    for idx, (col, title, color) in enumerate(drift_cols):
        s = closing_df[col].dropna()
        s_clipped = s[(s >= -50) & (s <= 50)]
        axes[idx].hist(s_clipped, bins=60, color=color, edgecolor="white", alpha=0.85)
        axes[idx].axvline(0, color="black", linestyle="--", linewidth=1.5, label="No Movement (0%)")
        axes[idx].axvline(s.median(), color="blue", linestyle=":", linewidth=1.5, label=f"Median: {s.median():.2f}%")
        axes[idx].set_title(title, fontsize=12, fontweight="bold")
        axes[idx].set_xlabel("Movement from Open to Close (%)", fontsize=10)
        axes[idx].set_ylabel("Count", fontsize=10)
        axes[idx].legend(fontsize=9)

    plt.suptitle("1X2 Market Odds Drift (Opening vs Closing Odds)", fontsize=15, fontweight="bold", y=1.03)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_league_home_advantage(matches_df: pd.DataFrame, min_matches: int = 200, save_path: Path | None = None) -> plt.Figure:
    """
    Plot Top Leagues ranked by Home Win Rate.

    Parameters
    ----------
    matches_df : pd.DataFrame
        Cleaned matches DataFrame with 'league_name' and 'outcome'.
    min_matches : int
        Minimum matches required for league inclusion.
    save_path : Path, optional
        Destination image path.

    Returns
    -------
    plt.Figure
    """
    league_counts = matches_df["league_name"].value_counts()
    top_leagues = league_counts[league_counts >= min_matches].index

    sub = matches_df[matches_df["league_name"].isin(top_leagues)]
    league_stats = sub.groupby("league_name")["outcome"].apply(
        lambda s: (s == "H").mean() * 100.0
    ).sort_values(ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(12, 8))
    league_stats.plot.barh(ax=ax, color="#2980b9", edgecolor="white", alpha=0.85)
    overall_mean = (matches_df["outcome"] == "H").mean() * 100.0
    ax.axvline(overall_mean, color="red", linestyle="--", linewidth=1.5, label=f"Global Mean ({overall_mean:.1f}%)")
    ax.set_title(f"Top 20 Competitions by Home Win Percentage (min {min_matches} matches)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Home Win Rate (%)", fontsize=11)
    ax.set_ylabel("Competition", fontsize=11)
    ax.legend(fontsize=10)
    ax.invert_yaxis()

    for idx, val in enumerate(league_stats):
        ax.text(val + 0.4, idx, f"{val:.1f}%", va="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_model_comparison(metrics_dict: dict, save_path: Path | None = None) -> plt.Figure:
    """
    Plot comparative bar charts of Accuracy, Log Loss, and Brier Score across models.

    Parameters
    ----------
    metrics_dict : dict
        Dictionary of model names to evaluation metric dictionaries.
    save_path : Path, optional
        Destination image path.

    Returns
    -------
    plt.Figure
    """
    models = list(metrics_dict.keys())
    accs = [metrics_dict[m]["accuracy"] * 100 for m in models]
    losses = [metrics_dict[m]["log_loss"] for m in models]
    briers = [metrics_dict[m]["brier_score"] for m in models]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Accuracy
    axes[0].bar(models, accs, color="#2ecc71", edgecolor="black", alpha=0.85)
    axes[0].set_title("Test Accuracy (%) — Higher is Better", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_ylim(50, 55)
    axes[0].tick_params(axis="x", rotation=25)
    for i, v in enumerate(accs):
        axes[0].text(i, v + 0.1, f"{v:.2f}%", ha="center", fontsize=9, fontweight="bold")

    # Log Loss
    axes[1].bar(models, losses, color="#e74c3c", edgecolor="black", alpha=0.85)
    axes[1].set_title("Multi-Class Log Loss — Lower is Better", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Log Loss (Cross-Entropy)")
    axes[1].set_ylim(0.95, 0.99)
    axes[1].tick_params(axis="x", rotation=25)
    for i, v in enumerate(losses):
        axes[1].text(i, v + 0.0005, f"{v:.4f}", ha="center", fontsize=9, fontweight="bold")

    # Brier Score
    axes[2].bar(models, briers, color="#3498db", edgecolor="black", alpha=0.85)
    axes[2].set_title("Brier Score — Lower is Better", fontsize=12, fontweight="bold")
    axes[2].set_ylabel("Brier Score")
    axes[2].set_ylim(0.57, 0.59)
    axes[2].tick_params(axis="x", rotation=25)
    for i, v in enumerate(briers):
        axes[2].text(i, v + 0.0003, f"{v:.4f}", ha="center", fontsize=9, fontweight="bold")

    plt.suptitle("Supervised ML Model Benchmark vs Bookmaker Consensus (2024+ Out-of-Time Test)",
                 fontsize=14, fontweight="bold", y=1.03)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_confusion_matrices(metrics_dict: dict, save_path: Path | None = None) -> plt.Figure:
    """
    Plot confusion matrix heatmaps comparing models on Home, Draw, Away outcomes.

    Parameters
    ----------
    metrics_dict : dict
        Dictionary of model names to evaluation metrics.
    save_path : Path, optional
        Destination image path.

    Returns
    -------
    plt.Figure
    """
    class_names = ["Home", "Draw", "Away"]
    models = list(metrics_dict.keys())
    n = len(models)

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    if n == 1:
        axes = [axes]

    for idx, model_name in enumerate(models):
        cm = np.array(metrics_dict[model_name]["confusion_matrix"])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=axes[idx],
            cbar=False
        )
        axes[idx].set_title(model_name, fontsize=11, fontweight="bold")
        axes[idx].set_xlabel("Predicted Label")
        axes[idx].set_ylabel("True Label")

    plt.suptitle("Confusion Matrices (Test Set: 2024–2026 Matches)", fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
