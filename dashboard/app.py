"""
ODDSPULSE — Football Odds Intelligence & Match Analytics Platform
==================================================================
Interactive Streamlit application exploring bookmaker margins, odds drift,
supervised ML predictions, and market anomaly detection across 63,597 matches
and ~37 million odds ticks.

Modular architecture powered by:
- dashboard.theme: Design system, dark palette (#080D16, #C5F23B, #27B8D1)
- dashboard.views: Dedicated views for Overview, Match Explorer, Odds Analytics,
  Bookmaker Analysis, Predictive Analytics, Anomaly Radar, Model Performance, and About.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Tuple

import joblib
import pandas as pd
import streamlit as st

# Configure project path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data.loader import load_processed
from dashboard.theme import (
    COLOR_ACCENT_PRIMARY,
    COLOR_BORDER,
    COLOR_TEXT_MAIN,
    COLOR_TEXT_MUTED,
    inject_custom_css,
)
from dashboard.views.overview import render_overview
from dashboard.views.match_explorer import render_match_explorer
from dashboard.views.odds_analytics import render_odds_analytics
from dashboard.views.bookmaker_analysis import render_bookmaker_analysis
from dashboard.views.predictive_analytics import render_predictive_analytics
from dashboard.views.anomaly_radar import render_anomaly_radar
from dashboard.views.model_performance import render_model_performance
from dashboard.views.about import render_about

# -----------------------------------------------------------------------------
# STREAMLIT PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="ODDSPULSE | Football Odds Intelligence",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply global dark sports analytics design system
inject_custom_css()


# -----------------------------------------------------------------------------
# DATA LOADING WITH PERSISTENT CACHING
# -----------------------------------------------------------------------------
@st.cache_data
def load_dashboard_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, dict]:
    """Load and cache processed datasets, evaluation metrics, and trained model bundle."""
    matches = load_processed("matches_clean")
    closing = load_processed("match_closing_clean")
    features = load_processed("features_with_anomalies")

    # Load metrics
    metrics_path = project_root / "reports" / "analysis" / "model_metrics.json"
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
    else:
        metrics = {}

    # Load best model artifact
    model_path = project_root / "models" / "best_model.joblib"
    if model_path.exists():
        model_bundle = joblib.load(model_path)
    else:
        model_bundle = None

    return matches, closing, features, metrics, model_bundle


matches_df, closing_df, features_df, metrics_data, model_bundle = load_dashboard_data()


# -----------------------------------------------------------------------------
# SIDEBAR NAVIGATION & PLATFORM BRANDING
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 12px; padding: 10px 0 16px 0;">
        <div style="background: linear-gradient(135deg, #C5F23B 0%, #27B8D1 100%); width: 38px; height: 38px; border-radius: 8px; display: flex; align-items: center; justify-content: center; font-size: 1.3rem;">
            ⚽
        </div>
        <div>
            <div style="font-size: 1.25rem; font-weight: 800; color: #F4F7FA; letter-spacing: -0.5px;">ODDSPULSE</div>
            <div style="font-size: 0.7rem; font-weight: 600; color: #C5F23B; letter-spacing: 0.1em; text-transform: uppercase;">Football Intelligence</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.caption("ANALYTICS SUITE")
    nav_choice = st.radio(
        "Navigation",
        [
            "📊 Overview",
            "🔍 Match Explorer",
            "📈 Odds Analytics",
            "🏦 Bookmaker Analysis",
            "🤖 Predictive Analytics",
            "🚨 Anomaly Radar",
            "📋 Model Performance",
            "ℹ️ About Project"
        ],
        label_visibility="collapsed"
    )

    st.markdown("<hr style='margin: 18px 0; border-color: #253449;'>", unsafe_allow_html=True)

    # Sidebar Platform Metadata Badge
    st.markdown("""
    <div style="background-color: #172333; border: 1px solid #253449; border-radius: 10px; padding: 14px; font-size: 0.78rem;">
        <div style="color: #C5F23B; font-weight: 700; margin-bottom: 4px;">DATASET AUDIT</div>
        <div style="color: #F4F7FA;"><b>63,597</b> Matches (2018–2026)</div>
        <div style="color: #94A3B8;">~37M Tick Observations</div>
        <div style="color: #94A3B8;">4 Global Bookmakers</div>
        <div style="margin-top: 10px; padding-top: 8px; border-top: 1px solid #253449; color: #94A3B8;">
            <b>Developer:</b> Mohd Afzal<br>
            <b>Domain:</b> Quant Sports Analytics
        </div>
    </div>
    """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# VIEW ROUTING
# -----------------------------------------------------------------------------
if nav_choice == "📊 Overview":
    render_overview(matches_df, closing_df, features_df, metrics_data, model_bundle)

elif nav_choice == "🔍 Match Explorer":
    render_match_explorer(matches_df, closing_df, features_df)

elif nav_choice == "📈 Odds Analytics":
    render_odds_analytics(matches_df, closing_df)

elif nav_choice == "🏦 Bookmaker Analysis":
    render_bookmaker_analysis(closing_df, matches_df)

elif nav_choice == "🤖 Predictive Analytics":
    render_predictive_analytics(matches_df, features_df, model_bundle)

elif nav_choice == "🚨 Anomaly Radar":
    render_anomaly_radar(features_df)

elif nav_choice == "📋 Model Performance":
    render_model_performance(metrics_data, model_bundle)

elif nav_choice == "ℹ️ About Project":
    render_about()
