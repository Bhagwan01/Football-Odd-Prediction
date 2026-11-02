"""
ODDSPULSE — Football Odds Intelligence & Match Analytics Platform
==================================================================
A quantitative sports analytics interface analyzing historical odds movements,
bookmaker pricing behavior, statistical market efficiency, supervised ML modeling,
and unsupervised anomaly detection across 63,597 matches and ~37 million odds ticks.

Design System:
- Main Background: #080D16
- Secondary Background: #101927
- Card Background: #172333
- Border Color: #253449
- Primary Accent: #C5F23B (Lime green)
- Secondary Accent: #27B8D1 (Cyan)
- Text Color: #F4F7FA / #94A3B8
"""

import json
import os
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Configure project path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data.loader import load_processed
from src.models.trainer import FEATURE_COLUMNS

# =============================================================================
# STREAMLIT PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="ODDSPULSE | Football Odds Intelligence",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# PREMIUM DARK THEME CSS
# =============================================================================
st.markdown("""
<style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* App background */
    .stApp {
        background-color: #080D16;
        color: #F4F7FA;
    }
    
    /* Sidebar customization */
    section[data-testid="stSidebar"] {
        background-color: #101927;
        border-right: 1px solid #253449;
    }
    section[data-testid="stSidebar"] div.stRadio > label {
        display: none;
    }
    section[data-testid="stSidebar"] div.stRadio > div[role="radiogroup"] > label {
        background-color: #172333;
        border: 1px solid #253449;
        border-radius: 8px;
        padding: 8px 12px;
        margin-bottom: 6px;
        color: #94A3B8;
        font-weight: 500;
        transition: all 0.2s ease-in-out;
    }
    section[data-testid="stSidebar"] div.stRadio > div[role="radiogroup"] > label:hover {
        border-color: #27B8D1;
        color: #F4F7FA;
    }
    section[data-testid="stSidebar"] div.stRadio > div[role="radiogroup"] > label[data-checked="true"] {
        background: linear-gradient(90deg, rgba(197, 242, 59, 0.12) 0%, rgba(23, 35, 51, 0.8) 100%);
        border-left: 4px solid #C5F23B !important;
        border-color: #253449;
        color: #C5F23B !important;
        font-weight: 700;
    }
    
    /* Card Styles */
    .stat-card {
        background: #172333;
        border: 1px solid #253449;
        border-radius: 12px;
        padding: 18px 20px;
        position: relative;
        overflow: hidden;
        transition: transform 0.2s ease, border-color 0.2s ease;
        margin-bottom: 12px;
    }
    .stat-card:hover {
        border-color: #27B8D1;
        transform: translateY(-2px);
    }
    .stat-card::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #C5F23B, #27B8D1);
    }
    .stat-label {
        color: #94A3B8;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 6px;
    }
    .stat-value {
        color: #F4F7FA;
        font-size: 1.85rem;
        font-weight: 800;
        font-family: 'JetBrains Mono', monospace;
        line-height: 1.1;
    }
    .stat-subtext {
        font-size: 0.8rem;
        color: #27B8D1;
        margin-top: 6px;
        font-weight: 500;
    }
    
    /* Match card */
    .match-hero {
        background: linear-gradient(180deg, #172333 0%, #101927 100%);
        border: 1px solid #253449;
        border-radius: 14px;
        padding: 24px;
        margin-bottom: 16px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.3);
    }
    .badge-comp {
        background-color: rgba(39, 184, 209, 0.15);
        color: #27B8D1;
        border: 1px solid rgba(39, 184, 209, 0.35);
        border-radius: 6px;
        padding: 3px 8px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        display: inline-block;
    }
    .badge-outcome-h {
        background-color: rgba(54, 201, 143, 0.15);
        color: #36C98F;
        border: 1px solid #36C98F;
        border-radius: 6px;
        padding: 3px 10px;
        font-size: 0.8rem;
        font-weight: 700;
    }
    .badge-outcome-d {
        background-color: rgba(245, 185, 66, 0.15);
        color: #F5B942;
        border: 1px solid #F5B942;
        border-radius: 6px;
        padding: 3px 10px;
        font-size: 0.8rem;
        font-weight: 700;
    }
    .badge-outcome-a {
        background-color: rgba(240, 100, 100, 0.15);
        color: #F06464;
        border: 1px solid #F06464;
        border-radius: 6px;
        padding: 3px 10px;
        font-size: 0.8rem;
        font-weight: 700;
    }
    .team-name {
        font-size: 1.4rem;
        font-weight: 700;
        color: #F4F7FA;
    }
    .score-display {
        font-size: 2.4rem;
        font-weight: 900;
        color: #C5F23B;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: 4px;
        text-align: center;
    }
    
    /* Odds box */
    .odds-pill {
        background: #101927;
        border: 1px solid #253449;
        border-radius: 8px;
        padding: 10px 14px;
        text-align: center;
    }
    .odds-pill-title {
        color: #94A3B8;
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
    }
    .odds-pill-val {
        color: #C5F23B;
        font-size: 1.3rem;
        font-weight: 800;
        font-family: 'JetBrains Mono', monospace;
    }
    .odds-pill-sub {
        color: #94A3B8;
        font-size: 0.75rem;
    }

    /* Custom containers */
    div[data-testid="stMetric"] {
        background-color: #172333;
        border: 1px solid #253449;
        border-radius: 10px;
        padding: 12px 16px;
    }
    div[data-testid="stMetricValue"] {
        color: #C5F23B;
        font-family: 'JetBrains Mono', monospace;
    }
    
    /* Headers */
    h1, h2, h3, h4 {
        color: #F4F7FA !important;
        font-weight: 700 !important;
    }
    hr {
        border-color: #253449 !important;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# DATA LOADING & PERSISTENCE
# =============================================================================
@st.cache_data
def load_dashboard_data():
    """Load and cache processed datasets and analytical artifacts."""
    matches = load_processed("matches_clean")
    closing = load_processed("match_closing_clean")
    features = load_processed("features_with_anomalies")

    metrics_path = project_root / "reports" / "analysis" / "model_metrics.json"
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
    else:
        metrics = {}

    stats_path = project_root / "reports" / "analysis" / "statistical_summary.json"
    if stats_path.exists():
        with open(stats_path, "r", encoding="utf-8") as f:
            stats_summary = json.load(f)
    else:
        stats_summary = {}

    model_path = project_root / "models" / "best_model.joblib"
    if model_path.exists():
        model_bundle = joblib.load(model_path)
    else:
        model_bundle = None

    return matches, closing, features, metrics, stats_summary, model_bundle


matches_df, closing_df, features_df, metrics_data, stats_data, model_bundle = load_dashboard_data()


# =============================================================================
# PLOTLY THEME HELPER
# =============================================================================
def apply_dark_theme(fig: go.Figure, title: str = "", height: int = 380) -> go.Figure:
    """Apply consistent premium dark styling to Plotly figures."""
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>" if title else "",
            font=dict(size=14, color="#F4F7FA", family="Inter")
        ),
        paper_bgcolor="#172333",
        plot_bgcolor="#101927",
        font=dict(color="#94A3B8", family="Inter", size=11),
        height=height,
        margin=dict(l=40, r=30, t=50, b=40),
        xaxis=dict(
            gridcolor="#253449",
            zerolinecolor="#253449",
            tickfont=dict(color="#94A3B8")
        ),
        yaxis=dict(
            gridcolor="#253449",
            zerolinecolor="#253449",
            tickfont=dict(color="#94A3B8")
        ),
        legend=dict(
            bgcolor="rgba(16, 25, 39, 0.8)",
            bordercolor="#253449",
            borderwidth=1,
            font=dict(color="#F4F7FA")
        )
    )
    return fig


# =============================================================================
# SIDEBAR NAVIGATION
# =============================================================================
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
        "Navigation Menu",
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

    # Project metadata pill in sidebar
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


# =============================================================================
# 1. OVERVIEW DASHBOARD
# =============================================================================
if nav_choice == "📊 Overview":
    st.markdown("""
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
        <div>
            <h1 style="margin: 0; font-size: 2.0rem; letter-spacing: -0.5px;">Executive Analytics Overview</h1>
            <p style="color: #94A3B8; margin: 4px 0 0 0; font-size: 0.95rem;">
                Quantitative intelligence and market efficiency benchmark across 63,597 historical fixtures
            </p>
        </div>
        <div style="background: rgba(197, 242, 59, 0.1); border: 1px solid #C5F23B; border-radius: 8px; padding: 6px 14px; color: #C5F23B; font-size: 0.8rem; font-weight: 700;">
            ● 100% DATASET INTEGRITY
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 4 Top KPI Cards
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">TOTAL MATCHES</div>
            <div class="stat-value">{len(matches_df):,}</div>
            <div class="stat-subtext">33 Competitions (2018–2026)</div>
        </div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-label">ODDS OBSERVATIONS</div>
            <div class="stat-value">~37.0M</div>
            <div class="stat-subtext">1X2, Asian Handicap, Over/Under</div>
        </div>
        """, unsafe_allow_html=True)
    with k3:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-label">SHARPEST BOOKMAKER</div>
            <div class="stat-value">Pinnacle</div>
            <div class="stat-subtext">3.34% Median Overround Margin</div>
        </div>
        """, unsafe_allow_html=True)
    with k4:
        best_acc = metrics_data.get("Logistic Regression", {}).get("accuracy", 0.5285) * 100
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">BEST ML ACCURACY</div>
            <div class="stat-value">{best_acc:.2f}%</div>
            <div class="stat-subtext">Calibrated Out-of-Time Model</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # Main Analytical Section: 2 Columns (Match Spotlight vs Model Intelligence)
    col_left, col_right = st.columns([1.1, 0.9])

    with col_left:
        st.markdown("### 🎯 Fixture & Market Spotlight")
        # Match selector inside overview
        recent_leagues = sorted(matches_df["league_name"].dropna().unique().tolist())
        sel_l = st.selectbox("Filter League", ["All Leagues"] + recent_leagues, key="ov_league")
        
        ov_filtered = matches_df if sel_l == "All Leagues" else matches_df[matches_df["league_name"] == sel_l]
        sample_match_id = st.selectbox(
            "Select Fixture",
            ov_filtered["match_id"].head(50),
            format_func=lambda mid: f"{ov_filtered[ov_filtered['match_id']==mid]['home_team'].values[0]} vs {ov_filtered[ov_filtered['match_id']==mid]['away_team'].values[0]} ({ov_filtered[ov_filtered['match_id']==mid]['season'].values[0]})"
        )
        
        m_row = matches_df[matches_df["match_id"] == sample_match_id].iloc[0]
        c_sub = closing_df[closing_df["match_id"] == sample_match_id]
        
        outcome_badge = (
            '<span class="badge-outcome-h">HOME WIN</span>' if m_row['outcome'] == 'H'
            else ('<span class="badge-outcome-d">DRAW</span>' if m_row['outcome'] == 'D'
            else '<span class="badge-outcome-a">AWAY WIN</span>')
        )
        
        st.markdown(f"""
        <div class="match-hero">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px;">
                <span class="badge-comp">{m_row['league_name']} • {m_row['season']}</span>
                <div>{outcome_badge}</div>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center; padding: 10px 0;">
                <div style="flex: 1; text-align: left;">
                    <div style="color: #94A3B8; font-size: 0.75rem; text-transform: uppercase;">HOME</div>
                    <div class="team-name">{m_row['home_team']}</div>
                </div>
                <div style="padding: 0 20px;">
                    <div class="score-display">{int(m_row['ft_home'])} - {int(m_row['ft_away'])}</div>
                    <div style="color: #94A3B8; font-size: 0.75rem; text-align: center;">Full Time Score</div>
                </div>
                <div style="flex: 1; text-align: right;">
                    <div style="color: #94A3B8; font-size: 0.75rem; text-transform: uppercase;">AWAY</div>
                    <div class="team-name">{m_row['away_team']}</div>
                </div>
            </div>
            <div style="color: #94A3B8; font-size: 0.8rem; margin-top: 14px; border-top: 1px solid #253449; padding-top: 10px;">
                Kickoff UTC: <b>{m_row['kickoff_utc']}</b> | Total Goals: <b>{m_row['total_goals']}</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Available Bookmaker Quotes for this spotlight match
        if len(c_sub) > 0:
            st.markdown("<div style='font-size: 0.85rem; font-weight: 700; color: #F4F7FA; margin-bottom: 6px;'>CLOSING 1X2 QUOTES & MARGINS</div>", unsafe_allow_html=True)
            bk_cols = st.columns(len(c_sub.head(4)))
            for idx, (_, b_row) in enumerate(c_sub.head(4).iterrows()):
                with bk_cols[idx]:
                    st.markdown(f"""
                    <div class="odds-pill">
                        <div class="odds-pill-title">{b_row['bookmaker']}</div>
                        <div style="display: flex; justify-content: space-around; margin: 6px 0;">
                            <div><div class="odds-pill-sub">H</div><div class="odds-pill-val">{b_row['close_1x2_home']:.2f}</div></div>
                            <div><div class="odds-pill-sub">D</div><div class="odds-pill-val">{b_row['close_1x2_draw']:.2f}</div></div>
                            <div><div class="odds-pill-sub">A</div><div class="odds-pill-val">{b_row['close_1x2_away']:.2f}</div></div>
                        </div>
                        <div style="font-size: 0.72rem; color: #27B8D1;">Margin: {b_row['close_overround']*100:.2f}%</div>
                    </div>
                    """, unsafe_allow_html=True)

    with col_right:
        st.markdown("### 🤖 Predictive Intelligence")
        # Compute consensus and model forecast for selected fixture if available in features
        feat_match = features_df[features_df["match_id"] == sample_match_id]
        
        if len(feat_match) > 0 and model_bundle is not None:
            f_row = feat_match.iloc[0]
            feature_cols = model_bundle["feature_columns"]
            x_vec = f_row[feature_cols].values.reshape(1, -1)
            model_probs = model_bundle["model"].predict_proba(x_vec)[0]
            mkt_probs = [f_row["mkt_prob_home"], f_row["mkt_prob_draw"], f_row["mkt_prob_away"]]
            classes = ["Home Win (H)", "Draw (D)", "Away Win (A)"]

            prob_fig = go.Figure()
            prob_fig.add_trace(go.Bar(
                name="Calibrated ML Model",
                x=classes,
                y=[p * 100 for p in model_probs],
                marker_color="#C5F23B",
                text=[f"{p*100:.1f}%" for p in model_probs],
                textposition="auto"
            ))
            prob_fig.add_trace(go.Bar(
                name="Bookmaker Consensus",
                x=classes,
                y=[p * 100 for p in mkt_probs],
                marker_color="#27B8D1",
                text=[f"{p*100:.1f}%" for p in mkt_probs],
                textposition="auto"
            ))
            prob_fig.update_layout(
                barmode="group",
                title="Model Forecast vs Market Fair Consensus (%)",
                yaxis_range=[0, 100],
                height=260
            )
            apply_dark_theme(prob_fig, height=270)
            st.plotly_chart(prob_fig, use_container_width=True)

            pred_idx = np.argmax(model_probs)
            pred_class = ["Home Win", "Draw", "Away Win"][pred_idx]
            conf = model_probs[pred_idx] * 100
            
            st.markdown(f"""
            <div style="background-color: #101927; border: 1px solid #253449; border-radius: 10px; padding: 12px 16px; margin-top: 8px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <div style="color: #94A3B8; font-size: 0.75rem; text-transform: uppercase;">MODEL PREDICTION</div>
                        <div style="color: #C5F23B; font-size: 1.25rem; font-weight: 800;">{pred_class}</div>
                    </div>
                    <div style="text-align: right;">
                        <div style="color: #94A3B8; font-size: 0.75rem; text-transform: uppercase;">CONFIDENCE</div>
                        <div style="color: #27B8D1; font-size: 1.25rem; font-weight: 800; font-family: 'JetBrains Mono', monospace;">{conf:.1f}%</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Feature vector for this fixture is being aggregated. Select another match from the dropdown.")

    st.markdown("<hr style='margin: 28px 0; border-color: #253449;'>", unsafe_allow_html=True)

    # Lower Analytics Cards
    st.markdown("### 📊 Market Dynamics & Empirical Discoveries")
    c1, c2, c3 = st.columns(3)

    with c1:
        # Global Outcome Pie Chart
        outcome_counts = matches_df["outcome"].value_counts()
        fig_donut = px.pie(
            values=outcome_counts.values,
            names=["Home Win (H)", "Away Win (A)", "Draw (D)"],
            title="Outcome Proportions (63,597 Matches)",
            color=["Home Win (H)", "Away Win (A)", "Draw (D)"],
            color_discrete_map={
                "Home Win (H)": "#36C98F",
                "Draw (D)": "#F5B942",
                "Away Win (A)": "#F06464"
            },
            hole=0.55
        )
        apply_dark_theme(fig_donut, title="Outcome Proportions (63,597 Matches)", height=320)
        st.plotly_chart(fig_donut, use_container_width=True)

    with c2:
        # Match Coverage by Season/Year
        matches_df["year"] = pd.to_datetime(matches_df["kickoff_utc"]).dt.year
        yr_counts = matches_df["year"].value_counts().sort_index().reset_index()
        yr_counts.columns = ["Year", "Matches"]
        fig_bar = px.bar(
            yr_counts,
            x="Year",
            y="Matches",
            title="Match Coverage by Kickoff Year",
            color_discrete_sequence=["#27B8D1"]
        )
        apply_dark_theme(fig_bar, title="Match Coverage by Kickoff Year", height=320)
        st.plotly_chart(fig_bar, use_container_width=True)

    with c3:
        # Bookmaker Median Margins Comparison Bar
        bmk_margins = pd.DataFrame({
            "Bookmaker": ["Pinnacle", "Singbet", "Bet365", "SBOBET"],
            "Median Margin (%)": [3.34, 5.45, 5.95, 6.23],
            "Type": ["Sharp Market Maker", "Asian Handicap Specialist", "Recreational Bookmaker", "Asian Bookmaker"]
        })
        fig_margins = px.bar(
            bmk_margins,
            x="Median Margin (%)",
            y="Bookmaker",
            orientation="h",
            color="Median Margin (%)",
            color_continuous_scale=["#C5F23B", "#27B8D1", "#F5B942", "#F06464"],
            title="Bookmaker Median Commission (%)"
        )
        fig_margins.update_layout(coloraxis_showscale=False)
        apply_dark_theme(fig_margins, title="Bookmaker Median Commission (%)", height=320)
        st.plotly_chart(fig_margins, use_container_width=True)


# =============================================================================
# 2. MATCH EXPLORER
# =============================================================================
elif nav_choice == "🔍 Match Explorer":
    st.markdown("""
    <div>
        <h1 style="margin: 0; font-size: 2.0rem;">Match & Multi-Bookmaker Explorer</h1>
        <p style="color: #94A3B8; margin: 4px 0 16px 0;">
            Filter, search, and inspect historical football fixtures, scorelines, closing odds, and margin structures.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Search and Filter Controls
    f_col1, f_col2, f_col3, f_col4 = st.columns(4)
    with f_col1:
        leagues = ["All Leagues"] + sorted(matches_df["league_name"].dropna().unique().tolist())
        sel_league = st.selectbox("Competition", leagues)
    with f_col2:
        seasons = ["All Seasons"] + sorted(matches_df["season"].dropna().unique().tolist(), reverse=True)
        sel_season = st.selectbox("Season", seasons)
    with f_col3:
        outcomes = ["All Outcomes", "Home Win (H)", "Draw (D)", "Away Win (A)"]
        sel_outcome = st.selectbox("Match Outcome", outcomes)
    with f_col4:
        team_search = st.text_input("Search Team Name", placeholder="e.g. Arsenal, Real Madrid...")

    # Apply filters
    filtered = matches_df.copy()
    if sel_league != "All Leagues":
        filtered = filtered[filtered["league_name"] == sel_league]
    if sel_season != "All Seasons":
        filtered = filtered[filtered["season"] == sel_season]
    if sel_outcome != "All Outcomes":
        code_map = {"Home Win (H)": "H", "Draw (D)": "D", "Away Win (A)": "A"}
        filtered = filtered[filtered["outcome"] == code_map[sel_outcome]]
    if team_search.strip():
        term = team_search.strip().lower()
        filtered = filtered[
            filtered["home_team"].str.lower().str.contains(term, na=False) |
            filtered["away_team"].str.lower().str.contains(term, na=False)
        ]

    st.markdown(f"<div style='color: #27B8D1; font-weight: 600; margin-bottom: 8px;'>Filtered Records: {len(filtered):,} Fixtures</div>", unsafe_allow_html=True)

    # Fixture Browser Table
    preview_cols = [
        "match_id", "kickoff_utc", "league_name", "season", "home_team", "away_team",
        "ft_home", "ft_away", "outcome", "total_goals"
    ]
    st.dataframe(
        filtered[preview_cols].head(150),
        use_container_width=True,
        height=320,
        column_config={
            "match_id": st.column_config.NumberColumn("Match ID", format="%d"),
            "kickoff_utc": st.column_config.DatetimeColumn("Kickoff UTC", format="YYYY-MM-DD HH:mm"),
            "ft_home": st.column_config.NumberColumn("H Goals"),
            "ft_away": st.column_config.NumberColumn("A Goals")
        }
    )

    # Detailed Single-Match Breakdown Card
    if len(filtered) > 0:
        st.markdown("<hr style='margin: 24px 0; border-color: #253449;'>", unsafe_allow_html=True)
        st.markdown("### 🔍 Single Match Deep-Dive Inspector")

        match_pick = st.selectbox(
            "Select Fixture to Inspect in Detail",
            filtered["match_id"].head(50),
            format_func=lambda mid: f"ID {mid}: {filtered[filtered['match_id']==mid]['home_team'].values[0]} vs {filtered[filtered['match_id']==mid]['away_team'].values[0]} ({filtered[filtered['match_id']==mid]['kickoff_utc'].values[0]})"
        )
        
        m_row = matches_df[matches_df["match_id"] == match_pick].iloc[0]
        c_rows = closing_df[closing_df["match_id"] == match_pick]

        # Hero Presentation Card
        out_badge = (
            '<span class="badge-outcome-h">HOME WIN</span>' if m_row['outcome'] == 'H'
            else ('<span class="badge-outcome-d">DRAW</span>' if m_row['outcome'] == 'D'
            else '<span class="badge-outcome-a">AWAY WIN</span>')
        )

        st.markdown(f"""
        <div class="match-hero">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <span class="badge-comp">{m_row['league_name']} • Season {m_row['season']}</span>
                <div>{out_badge}</div>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div style="flex: 1; text-align: left;">
                    <div style="color: #94A3B8; font-size: 0.75rem;">HOME TEAM</div>
                    <div class="team-name">{m_row['home_team']}</div>
                </div>
                <div style="padding: 0 30px;">
                    <div class="score-display">{int(m_row['ft_home'])} - {int(m_row['ft_away'])}</div>
                    <div style="color: #94A3B8; font-size: 0.75rem; text-align: center;">Full Time</div>
                </div>
                <div style="flex: 1; text-align: right;">
                    <div style="color: #94A3B8; font-size: 0.75rem;">AWAY TEAM</div>
                    <div class="team-name">{m_row['away_team']}</div>
                </div>
            </div>
            <div style="color: #94A3B8; font-size: 0.8rem; margin-top: 14px; border-top: 1px solid #253449; padding-top: 8px;">
                Match ID: <code>{m_row['match_id']}</code> | Kickoff: <b>{m_row['kickoff_utc']}</b> | Goal Difference: <b>{m_row['goal_diff']}</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Multi-bookmaker comparison table and chart
        if len(c_rows) > 0:
            c_left, c_right = st.columns([1.2, 0.8])
            with c_left:
                st.markdown("#### Bookmaker Quote Breakdown")
                odds_table = c_rows[[
                    "bookmaker", "open_1x2_home", "open_1x2_draw", "open_1x2_away",
                    "close_1x2_home", "close_1x2_draw", "close_1x2_away",
                    "close_overround", "close_norm_prob_home", "close_norm_prob_draw", "close_norm_prob_away"
                ]].copy()
                odds_table["close_overround"] = (odds_table["close_overround"] * 100).round(2)
                odds_table["close_norm_prob_home"] = (odds_table["close_norm_prob_home"] * 100).round(1)
                odds_table["close_norm_prob_draw"] = (odds_table["close_norm_prob_draw"] * 100).round(1)
                odds_table["close_norm_prob_away"] = (odds_table["close_norm_prob_away"] * 100).round(1)
                
                odds_table.columns = [
                    "Bookmaker", "Open H", "Open D", "Open A",
                    "Close H", "Close D", "Close A",
                    "Margin %", "Fair P(H) %", "Fair P(D) %", "Fair P(A) %"
                ]
                st.dataframe(odds_table, use_container_width=True)

            with c_right:
                st.markdown("#### Closing Decimal Odds Comparison")
                fig_comp = go.Figure()
                fig_comp.add_trace(go.Bar(
                    name="Home Win Odds",
                    x=c_rows["bookmaker"],
                    y=c_rows["close_1x2_home"],
                    marker_color="#C5F23B"
                ))
                fig_comp.add_trace(go.Bar(
                    name="Draw Odds",
                    x=c_rows["bookmaker"],
                    y=c_rows["close_1x2_draw"],
                    marker_color="#F5B942"
                ))
                fig_comp.add_trace(go.Bar(
                    name="Away Win Odds",
                    x=c_rows["bookmaker"],
                    y=c_rows["close_1x2_away"],
                    marker_color="#27B8D1"
                ))
                fig_comp.update_layout(barmode="group", height=280)
                apply_dark_theme(fig_comp, title="1X2 Quotes by Bookmaker", height=280)
                st.plotly_chart(fig_comp, use_container_width=True)


# =============================================================================
# 3. ODDS ANALYTICS
# =============================================================================
elif nav_choice == "📈 Odds Analytics":
    st.markdown("""
    <div>
        <h1 style="margin: 0; font-size: 2.0rem;">Odds Dynamics & Distribution Analytics</h1>
        <p style="color: #94A3B8; margin: 4px 0 16px 0;">
            Comprehensive statistical analysis of decimal odds distributions, implied probabilities, and line drift.
        </p>
    </div>
    """, unsafe_allow_html=True)

    row1_c1, row1_c2 = st.columns(2)
    with row1_c1:
        # Closing Decimal Odds Boxplot
        closing_melt = closing_df[["close_1x2_home", "close_1x2_draw", "close_1x2_away"]].dropna().copy()
        closing_melt = closing_melt[(closing_melt["close_1x2_home"] <= 15) & (closing_melt["close_1x2_away"] <= 15)]
        
        fig_odds_dist = go.Figure()
        fig_odds_dist.add_trace(go.Box(y=closing_melt["close_1x2_home"], name="Home Win", marker_color="#36C98F"))
        fig_odds_dist.add_trace(go.Box(y=closing_melt["close_1x2_draw"], name="Draw", marker_color="#F5B942"))
        fig_odds_dist.add_trace(go.Box(y=closing_melt["close_1x2_away"], name="Away Win", marker_color="#27B8D1"))
        fig_odds_dist.update_layout(title="Closing Decimal Odds Distribution (Clipped at 15.0)")
        apply_dark_theme(fig_odds_dist, title="Closing Decimal Odds Distribution", height=340)
        st.plotly_chart(fig_odds_dist, use_container_width=True)

    with row1_c2:
        # Fair Implied Probability Distributions
        prob_melt = closing_df[["close_norm_prob_home", "close_norm_prob_draw", "close_norm_prob_away"]].dropna().copy() * 100
        fig_prob_dist = go.Figure()
        fig_prob_dist.add_trace(go.Violin(y=prob_melt["close_norm_prob_home"], name="P(Home)", line_color="#36C98F", box_visible=True, meanline_visible=True))
        fig_prob_dist.add_trace(go.Violin(y=prob_melt["close_norm_prob_draw"], name="P(Draw)", line_color="#F5B942", box_visible=True, meanline_visible=True))
        fig_prob_dist.add_trace(go.Violin(y=prob_melt["close_norm_prob_away"], name="P(Away)", line_color="#27B8D1", box_visible=True, meanline_visible=True))
        fig_prob_dist.update_layout(title="Fair Margin-Removed Probability (%)")
        apply_dark_theme(fig_prob_dist, title="Fair Normalized Implied Probability (%)", height=340)
        st.plotly_chart(fig_prob_dist, use_container_width=True)

    st.markdown("<hr style='margin: 20px 0; border-color: #253449;'>", unsafe_allow_html=True)

    # Odds Drift Section
    st.markdown("### 📊 Market Momentum & Odds Drift Distribution")
    st.markdown("""
    Odds drift captures percentage line movement from opening quotes to closing:
    $$\\Delta \\% = \\frac{Odds_{close} - Odds_{open}}{Odds_{open}} \\times 100$$
    """)

    drift_clean = closing_df.dropna(subset=["drift_home_pct", "drift_away_pct"]).copy()
    drift_clean = drift_clean[(drift_clean["drift_home_pct"] >= -35) & (drift_clean["drift_home_pct"] <= 35)]

    d_col1, d_col2 = st.columns(2)
    with d_col1:
        fig_drift = px.histogram(
            drift_clean,
            x="drift_home_pct",
            nbins=70,
            title="Home Odds Drift (%) Distribution",
            color_discrete_sequence=["#C5F23B"],
            marginal="box"
        )
        fig_drift.add_vline(x=0, line_dash="dash", line_color="#F4F7FA", annotation_text="0% Baseline")
        apply_dark_theme(fig_drift, title="Home Odds Drift (%) [Open to Close]", height=340)
        st.plotly_chart(fig_drift, use_container_width=True)

    with d_col2:
        # League Home Advantage Ranking
        min_m = 250
        l_counts = matches_df["league_name"].value_counts()
        eligible_l = l_counts[l_counts >= min_m].index
        l_stats = matches_df[matches_df["league_name"].isin(eligible_l)].groupby("league_name")["outcome"].apply(
            lambda s: (s == "H").mean() * 100.0
        ).sort_values(ascending=False).head(15).reset_index()
        l_stats.columns = ["Competition", "Home Win Rate (%)"]

        fig_l = px.bar(
            l_stats,
            x="Home Win Rate (%)",
            y="Competition",
            orientation="h",
            title=f"Top 15 Competitions by Home Advantage (min {min_m} matches)",
            color="Home Win Rate (%)",
            color_continuous_scale=["#101927", "#27B8D1", "#C5F23B"]
        )
        fig_l.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
        apply_dark_theme(fig_l, title="Top Competitions by Home Win %", height=340)
        st.plotly_chart(fig_l, use_container_width=True)


# =============================================================================
# 4. BOOKMAKER ANALYSIS
# =============================================================================
elif nav_choice == "🏦 Bookmaker Analysis":
    st.markdown("""
    <div>
        <h1 style="margin: 0; font-size: 2.0rem;">Bookmaker Margin Audit & Sharp Market Analysis</h1>
        <p style="color: #94A3B8; margin: 4px 0 16px 0;">
            Empirical validation of bookmaker pricing strategies, overround commission structures, and market efficiency.
        </p>
    </div>
    """, unsafe_allow_html=True)

    valid_c = closing_df.dropna(subset=["close_overround", "bookmaker"]).copy()
    valid_c["margin_pct"] = (valid_c["close_overround"] * 100.0).round(2)
    valid_c = valid_c[(valid_c["margin_pct"] >= 0) & (valid_c["margin_pct"] <= 25)]

    # Statistical Summary Table
    margin_table = valid_c.groupby("bookmaker")["margin_pct"].agg(
        ["count", "mean", "median", "std", "min", "max"]
    ).round(2)

    col1, col2 = st.columns([1, 1.2])
    with col1:
        st.markdown("### 📋 Empirical Margin Summary")
        st.dataframe(margin_table, use_container_width=True)

        st.markdown("""
        <div style="background-color: #172333; border: 1px solid #253449; border-left: 4px solid #C5F23B; border-radius: 8px; padding: 14px; margin-top: 14px;">
            <div style="color: #C5F23B; font-weight: 700; font-size: 0.85rem; margin-bottom: 4px;">KEY DOMAIN FINDINGS</div>
            <ul style="margin: 0; padding-left: 18px; font-size: 0.82rem; color: #94A3B8;">
                <li><b>Pinnacle</b> operates as the sharpest global market maker with a <b>3.34% median overround</b>.</li>
                <li><b>Bet365</b> charges recreational bettors a <b>5.95% median margin</b>.</li>
                <li><b>SBOBET</b> maintains the highest average margin at <b>6.23%</b>.</li>
                <li>Kruskal-Wallis H-test confirms differences are statistically significant ($H = 107,374.91, p \\approx 0.0$).</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("### 📦 Margin Distribution by Bookmaker")
        fig_box = px.box(
            valid_c,
            x="bookmaker",
            y="margin_pct",
            color="bookmaker",
            color_discrete_map={
                "Pinnacle": "#C5F23B",
                "Singbet": "#27B8D1",
                "Bet365": "#F5B942",
                "SBOBET": "#F06464"
            },
            title="Overround Margin Distribution (%)",
            points=False
        )
        apply_dark_theme(fig_box, title="Overround Commission Distribution by Bookmaker (%)", height=360)
        st.plotly_chart(fig_box, use_container_width=True)

    st.markdown("<hr style='margin: 24px 0; border-color: #253449;'>", unsafe_allow_html=True)

    # Statistical hypothesis test badge callouts
    st.markdown("### 🧪 Hypothesis Testing Results")
    h1, h2, h3 = st.columns(3)
    with h1:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-label">KRUSKAL-WALLIS H-TEST</div>
            <div class="stat-value">107,374.9</div>
            <div class="stat-subtext">p-value: 0.0 (Reject Null of Equal Margins)</div>
        </div>
        """, unsafe_allow_html=True)
    with h2:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-label">PINNACLE VS BET365 TEST</div>
            <div class="stat-value">p &lt; 10⁻³⁰⁰</div>
            <div class="stat-subtext">Mann-Whitney U: Pinnacle is significantly sharper</div>
        </div>
        """, unsafe_allow_html=True)
    with h3:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-label">GLOBAL HOME ADVANTAGE</div>
            <div class="stat-value">43.81%</div>
            <div class="stat-subtext">Z = 56.06, 95% CI: [43.43%, 44.20%]</div>
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# 5. PREDICTIVE ANALYTICS
# =============================================================================
elif nav_choice == "🤖 Predictive Analytics":
    st.markdown("""
    <div>
        <h1 style="margin: 0; font-size: 2.0rem;">Supervised Machine Learning & Live Probability Engine</h1>
        <p style="color: #94A3B8; margin: 4px 0 16px 0;">
            Calibrated multinomial classification benchmarked strictly on 20,747 out-of-time test fixtures (2024–2026).
        </p>
    </div>
    """, unsafe_allow_html=True)

    p_left, p_right = st.columns([1.1, 0.9])

    with p_left:
        st.markdown("### 🎮 Interactive Decimal Odds Calculator")
        st.markdown("Enter closing decimal odds below to extract raw implied probabilities and margin-removed fair odds:")

        in_c1, in_c2, in_c3 = st.columns(3)
        with in_c1:
            in_home = st.number_input("Home Win Odds (1)", min_value=1.01, max_value=50.0, value=2.20, step=0.05)
        with in_c2:
            in_draw = st.number_input("Draw Odds (X)", min_value=1.01, max_value=50.0, value=3.40, step=0.05)
        with in_c3:
            in_away = st.number_input("Away Win Odds (2)", min_value=1.01, max_value=50.0, value=3.25, step=0.05)

        raw_h = 1.0 / in_home
        raw_d = 1.0 / in_draw
        raw_a = 1.0 / in_away
        raw_sum = raw_h + raw_d + raw_a
        fair_h = raw_h / raw_sum
        fair_d = raw_d / raw_sum
        fair_a = raw_a / raw_sum
        live_margin = (raw_sum - 1.0) * 100

        # Bar chart of fair probabilities
        calc_fig = go.Figure(data=[
            go.Bar(
                name="Fair Probability",
                x=["Home Win", "Draw", "Away Win"],
                y=[fair_h * 100, fair_d * 100, fair_a * 100],
                marker_color=["#36C98F", "#F5B942", "#27B8D1"],
                text=[f"{fair_h*100:.1f}%", f"{fair_d*100:.1f}%", f"{fair_a*100:.1f}%"],
                textposition="auto"
            )
        ])
        calc_fig.update_layout(yaxis_range=[0, 100])
        apply_dark_theme(calc_fig, title=f"Calculated Fair Probabilities (Overround Commission: {live_margin:.2f}%)", height=280)
        st.plotly_chart(calc_fig, use_container_width=True)

    with p_right:
        st.markdown("### 📋 Calibrated Fair Odds Comparison")
        st.markdown(f"""
        <div style="background-color: #172333; border: 1px solid #253449; border-radius: 10px; padding: 18px;">
            <div style="color: #94A3B8; font-size: 0.75rem; text-transform: uppercase;">BOOKMAKER COMMISSION OVERROUND</div>
            <div style="color: {'#36C98F' if live_margin < 4.0 else '#F5B942'}; font-size: 1.8rem; font-weight: 800; font-family: 'JetBrains Mono', monospace;">
                {live_margin:.2f}%
            </div>
            <hr style="border-color: #253449; margin: 12px 0;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                <span style="color: #94A3B8;">Fair Home Odds:</span>
                <span style="color: #C5F23B; font-weight: 700; font-family: 'JetBrains Mono';">{1/fair_h:.2f} (was {in_home:.2f})</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                <span style="color: #94A3B8;">Fair Draw Odds:</span>
                <span style="color: #C5F23B; font-weight: 700; font-family: 'JetBrains Mono';">{1/fair_d:.2f} (was {in_draw:.2f})</span>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span style="color: #94A3B8;">Fair Away Odds:</span>
                <span style="color: #C5F23B; font-weight: 700; font-family: 'JetBrains Mono';">{1/fair_a:.2f} (was {in_away:.2f})</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="background-color: #101927; border: 1px solid #253449; border-radius: 8px; padding: 12px; margin-top: 14px; font-size: 0.8rem; color: #94A3B8;">
            <b>Leakage Prevention Note:</b>
            All supervised model inferences rely exclusively on pre-kickoff market features, temporal markers, and bookmaker consensus metrics. Zero post-match scores or statistics are accessible to predictors.
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# 6. ANOMALY RADAR
# =============================================================================
elif nav_choice == "🚨 Anomaly Radar":
    st.markdown("""
    <div>
        <h1 style="margin: 0; font-size: 2.0rem;">Market Anomaly Radar (Isolation Forest)</h1>
        <p style="color: #94A3B8; margin: 4px 0 16px 0;">
            Unsupervised machine learning identifying fixtures with erratic odds drift, bookmaker disagreement, and market polarization.
        </p>
    </div>
    """, unsafe_allow_html=True)

    anom_only = features_df[features_df["is_anomaly"] == 1].copy()
    normal_only = features_df[features_df["is_anomaly"] == 0].copy()

    # KPI Banner
    ak1, ak2, ak3 = st.columns(3)
    with ak1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">FLAGGED ANOMALIES</div>
            <div class="stat-value">{len(anom_only):,}</div>
            <div class="stat-subtext">2.00% Contamination Rate (63,115 Fixtures)</div>
        </div>
        """, unsafe_allow_html=True)
    with ak2:
        draw_anom = (anom_only['outcome'] == 'D').mean() * 100
        draw_norm = (normal_only['outcome'] == 'D').mean() * 100
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">DRAW RATE COLLAPSE</div>
            <div class="stat-value">{draw_anom:.1f}%</div>
            <div class="stat-subtext">-72.6% vs Normal Market ({draw_norm:.1f}%)</div>
        </div>
        """, unsafe_allow_html=True)
    with ak3:
        home_anom = (anom_only['outcome'] == 'H').mean() * 100
        home_norm = (normal_only['outcome'] == 'H').mean() * 100
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">HOME WIN SPIKE</div>
            <div class="stat-value">{home_anom:.1f}%</div>
            <div class="stat-subtext">+36.2% vs Normal Market ({home_norm:.1f}%)</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr style='margin: 20px 0; border-color: #253449;'>", unsafe_allow_html=True)

    # Anomaly visual comparison
    a_col1, a_col2 = st.columns(2)
    with a_col1:
        # Decision score distribution
        fig_score = go.Figure()
        fig_score.add_trace(go.Histogram(
            x=normal_only["anomaly_score"],
            nbins=45,
            name="Normal Market",
            marker_color="#27B8D1",
            opacity=0.7
        ))
        fig_score.add_trace(go.Histogram(
            x=anom_only["anomaly_score"],
            nbins=30,
            name="Anomalous Fixtures",
            marker_color="#F06464",
            opacity=0.85
        ))
        fig_score.update_layout(barmode="overlay")
        apply_dark_theme(fig_score, title="Isolation Forest Decision Score Distribution", height=320)
        st.plotly_chart(fig_score, use_container_width=True)

    with a_col2:
        # Outcome Comparison
        comp_df = pd.DataFrame({
            "Outcome": ["Home Win", "Draw", "Away Win"],
            "Normal Market (%)": [home_norm, draw_norm, 100 - home_norm - draw_norm],
            "Anomalous Market (%)": [home_anom, draw_anom, 100 - home_anom - draw_anom]
        })
        fig_out = go.Figure()
        fig_out.add_trace(go.Bar(
            name="Normal Market",
            x=comp_df["Outcome"],
            y=comp_df["Normal Market (%)"],
            marker_color="#27B8D1"
        ))
        fig_out.add_trace(go.Bar(
            name="Anomalous Market",
            x=comp_df["Outcome"],
            y=comp_df["Anomalous Market (%)"],
            marker_color="#F06464"
        ))
        fig_out.update_layout(barmode="group")
        apply_dark_theme(fig_out, title="Match Outcome Comparison: Normal vs Anomalous", height=320)
        st.plotly_chart(fig_out, use_container_width=True)

    st.markdown("### 📋 Anomalous Fixtures Registry")
    anom_cols = [
        "match_id", "kickoff_utc", "league_name", "home_team", "away_team",
        "ft_home", "ft_away", "outcome", "mean_drift_home", "spread_odds_home",
        "odds_std_home", "anomaly_score"
    ]
    st.dataframe(
        anom_only[anom_cols].sort_values("anomaly_score").head(100),
        use_container_width=True,
        height=380
    )


# =============================================================================
# 7. MODEL PERFORMANCE
# =============================================================================
elif nav_choice == "📋 Model Performance":
    st.markdown("""
    <div>
        <h1 style="margin: 0; font-size: 2.0rem;">Model Benchmark & Calibration Audit</h1>
        <p style="color: #94A3B8; margin: 4px 0 16px 0;">
            Multi-class probabilistic evaluation across 20,747 out-of-time test fixtures (Kickoff 2024 to 2026).
        </p>
    </div>
    """, unsafe_allow_html=True)

    # 4 Model Benchmark Cards
    m_cols = st.columns(len(metrics_data))
    for idx, (m_name, m_dict) in enumerate(metrics_data.items()):
        with m_cols[idx]:
            is_best = m_name == "Logistic Regression"
            border_c = "#C5F23B" if is_best else "#253449"
            bg_c = "rgba(197, 242, 59, 0.05)" if is_best else "#172333"
            
            st.markdown(f"""
            <div style="background-color: {bg_c}; border: 1px solid {border_c}; border-radius: 12px; padding: 16px;">
                <div style="font-weight: 700; color: {'#C5F23B' if is_best else '#F4F7FA'}; font-size: 1.0rem;">
                    {m_name} {'⭐ BEST' if is_best else ''}
                </div>
                <hr style="margin: 8px 0; border-color: #253449;">
                <div style="font-size: 0.85rem; margin-bottom: 4px;">Accuracy: <b style="color: #F4F7FA;">{m_dict['accuracy']*100:.2f}%</b></div>
                <div style="font-size: 0.85rem; margin-bottom: 4px;">Log Loss: <b style="color: #27B8D1;">{m_dict['log_loss']:.4f}</b></div>
                <div style="font-size: 0.85rem; margin-bottom: 4px;">Brier Score: <b style="color: #27B8D1;">{m_dict['brier_score']:.4f}</b></div>
                <div style="font-size: 0.85rem;">Macro F1: <b style="color: #94A3B8;">{m_dict['f1_macro']:.4f}</b></div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<hr style='margin: 24px 0; border-color: #253449;'>", unsafe_allow_html=True)

    m_left, m_right = st.columns(2)
    with m_left:
        # Comparison Bar Chart
        models_list = list(metrics_data.keys())
        acc_vals = [metrics_data[m]["accuracy"] * 100 for m in models_list]
        loss_vals = [metrics_data[m]["log_loss"] for m in models_list]

        fig_bench = go.Figure()
        fig_bench.add_trace(go.Bar(
            name="Accuracy (%)",
            x=models_list,
            y=acc_vals,
            marker_color="#C5F23B"
        ))
        fig_bench.update_layout(yaxis_range=[50, 54])
        apply_dark_theme(fig_bench, title="Test Set Accuracy (%) — 20,747 Matches", height=320)
        st.plotly_chart(fig_bench, use_container_width=True)

    with m_right:
        # Confusion Matrix Heatmap
        sel_m = st.selectbox("Select Model for Confusion Matrix", models_list)
        cm = np.array(metrics_data[sel_m]["confusion_matrix"])
        
        fig_cm = px.imshow(
            cm,
            labels=dict(x="Predicted", y="True", color="Matches"),
            x=["Home Win", "Draw", "Away Win"],
            y=["Home Win", "Draw", "Away Win"],
            text_auto=True,
            color_continuous_scale=["#101927", "#172333", "#27B8D1", "#C5F23B"]
        )
        apply_dark_theme(fig_cm, title=f"Confusion Matrix: {sel_m}", height=320)
        st.plotly_chart(fig_cm, use_container_width=True)


# =============================================================================
# 8. ABOUT PROJECT
# =============================================================================
elif nav_choice == "ℹ️ About Project":
    st.markdown("""
    <div>
        <h1 style="margin: 0; font-size: 2.0rem;">About ODDSPULSE & Technical Architecture</h1>
        <p style="color: #94A3B8; margin: 4px 0 16px 0;">
            End-to-End Quantitative Sports Analytics & Machine Learning Pipeline.
        </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns([1.2, 0.8])
    with c1:
        st.markdown("""
        ### 📖 Executive Overview
        **ODDSPULSE** is a high-performance quantitative sports analytics platform developed to study
        market efficiency, bookmaker pricing power, odds momentum, and predictive machine learning models
        across **63,597 matches** and **~37 million odds ticks** from 2018 to 2026.

        ### 📐 Mathematical Formulations
        
        1. **Raw Implied Probability:**
           $$P_{raw, i} = \\frac{1}{Odds_i}$$

        2. **Bookmaker Overround (Margin):**
           $$\\Omega = \\left( \\sum_{i \\in \\{H, D, A\\}} \\frac{1}{Odds_i} \\right) - 1$$

        3. **Margin-Removed Fair Probability:**
           $$P_{fair, i} = \\frac{P_{raw, i}}{1 + \\Omega}$$

        4. **Odds Drift Percentage:**
           $$\\Delta \\% = \\frac{Odds_{close} - Odds_{open}}{Odds_{open}} \\times 100$$

        5. **Multi-Class Brier Score:**
           $$\\text{Brier} = \\frac{1}{N} \\sum_{i=1}^N \\sum_{k=1}^K (y_{i,k} - p_{i,k})^2$$
        """)

    with c2:
        st.markdown("""
        ### 👨‍💻 Project Metadata
        - **Developer:** Mohd Afzal
        - **Academic Track:** B.Tech Electronics & Communication Engineering (3rd Year)
        - **Target Domain:** Quantitative Sports Analytics, Machine Learning
        - **Repository:** [Bhagwan01/Football-Odd-Prediction](https://github.com/Bhagwan01/Football-Odd-Prediction)
        - **Standards:** Python 3.12 / 3.13, PEP 8, 100% Test Passing (15 unit tests)

        ### 🛠️ Tech Stack
        - **Core:** Python, Pandas, NumPy, PyArrow
        - **Machine Learning:** Scikit-Learn, XGBoost, Isolation Forest
        - **Hypothesis Testing:** SciPy (Kruskal-Wallis, Wilcoxon, Chi-Square, Z-Test)
        - **Visualization:** Streamlit, Plotly Express & Graph Objects
        """)
