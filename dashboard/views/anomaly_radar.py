"""
Market Anomaly Radar View
=========================
Unsupervised machine learning outlier detection (Isolation Forest):
- 1,262 anomalous market fixtures flagged (2.0% contamination)
- Discovery: 72.6% collapse in Draw probabilities in anomalous fixtures
- Inspection and filtering of anomalous fixtures
- Clear analytical framing (unusual market behavior ≠ wrongdoing)
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.theme import (
    COLOR_ACCENT_PRIMARY,
    COLOR_ACCENT_SECONDARY,
    COLOR_BG_CARD,
    COLOR_BORDER,
    COLOR_DANGER,
    COLOR_SUCCESS,
    COLOR_TEXT_MAIN,
    COLOR_TEXT_MUTED,
    COLOR_WARNING,
    format_plotly_figure,
    render_kpi_card,
)


def render_anomaly_radar(features_df: pd.DataFrame):
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <div style="color:#C5F23B; font-size:0.8rem; font-weight:700; letter-spacing:0.1em; text-transform:uppercase;">
            Unsupervised Outlier Detection
        </div>
        <h1 style="margin:0; font-size: 2.1rem; font-weight: 800;">Market Anomaly Radar (Isolation Forest)</h1>
        <div style="color:#94A3B8; font-size: 0.95rem; margin-top: 4px;">
            Identifying fixtures exhibiting erratic odds drift, extreme bookmaker disagreement, and abnormal market spreads.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.info("💡 **Methodology & Ethics Note:** Anomaly detection identifies statistical outliers in price discovery (e.g. abrupt line shortening, wide bid-ask spreads, or major syndicate disagreement). An anomaly is a mathematical pattern, not proof of wrongdoing or external interference.")

    anom_df = features_df[features_df["is_anomaly"] == 1].copy()
    norm_df = features_df[features_df["is_anomaly"] == 0].copy()

    # 1. Top Anomaly KPIs
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_kpi_card(
            label="Anomalous Fixtures",
            value=f"{len(anom_df):,}",
            subtext="Isolation Forest (contamination=0.02)",
            badge="Top 2.0% Outliers",
            badge_type="danger"
        )
    with k2:
        draw_anom = (anom_df["outcome"] == "D").mean() * 100
        render_kpi_card(
            label="Anomaly Draw Rate",
            value=f"{draw_anom:.1f}%",
            subtext="vs 25.71% in Normal Matches",
            badge="-72.6% Collapse",
            badge_type="danger"
        )
    with k3:
        home_anom = (anom_df["outcome"] == "H").mean() * 100
        render_kpi_card(
            label="Anomaly Home Win Rate",
            value=f"{home_anom:.1f}%",
            subtext="vs 43.50% in Normal Matches",
            badge="+36.2% Surge",
            badge_type="lime"
        )
    with k4:
        avg_drift = anom_df["mean_drift_home"].abs().mean()
        render_kpi_card(
            label="Avg Home Odds Drift",
            value=f"±{avg_drift:.1f}%",
            subtext="vs ±2.4% in Normal Markets",
            badge="Extreme Volatility",
            badge_type="warning"
        )

    st.markdown("<hr style='margin: 25px 0;'>", unsafe_allow_html=True)

    # 2. Key Discovery Chart: Draw Probability Collapse & Score Distributions
    c1, c2 = st.columns([1, 1], gap="medium")

    with c1:
        st.markdown("<div style='font-size:1.1rem; font-weight:700; margin-bottom:12px;'>Outcome Distribution: Normal vs Anomalous Markets</div>", unsafe_allow_html=True)
        norm_dist = norm_df["outcome"].value_counts(normalize=True) * 100
        anom_dist = anom_df["outcome"].value_counts(normalize=True) * 100

        fig_comp = go.Figure()
        fig_comp.add_trace(go.Bar(
            name="Normal Market",
            x=["Home Win", "Draw", "Away Win"],
            y=[norm_dist.get("H", 0), norm_dist.get("D", 0), norm_dist.get("A", 0)],
            marker_color=COLOR_ACCENT_SECONDARY
        ))
        fig_comp.add_trace(go.Bar(
            name="Anomalous Market",
            x=["Home Win", "Draw", "Away Win"],
            y=[anom_dist.get("H", 0), anom_dist.get("D", 0), anom_dist.get("A", 0)],
            marker_color=COLOR_DANGER
        ))
        fig_comp.update_layout(barmode="group", yaxis_title="Outcome Frequency (%)", height=320)
        st.plotly_chart(format_plotly_figure(fig_comp, title="Draw Rate Collapse in Erratic Matches"), use_container_width=True)

    with c2:
        st.markdown("<div style='font-size:1.1rem; font-weight:700; margin-bottom:12px;'>Isolation Forest Anomaly Score Distribution</div>", unsafe_allow_html=True)
        fig_score = go.Figure()
        fig_score.add_trace(go.Histogram(
            x=norm_df["anomaly_score"],
            name="Normal Market",
            marker_color=COLOR_ACCENT_SECONDARY,
            opacity=0.6,
            nbinsx=40
        ))
        fig_score.add_trace(go.Histogram(
            x=anom_df["anomaly_score"],
            name="Flagged Anomalies",
            marker_color=COLOR_DANGER,
            opacity=0.8,
            nbinsx=20
        ))
        fig_score.update_layout(barmode="overlay", xaxis_title="Decision Score (Lower = More Abnormal)", yaxis_title="Count", height=320)
        st.plotly_chart(format_plotly_figure(fig_score, title="Continuous Anomaly Score Spectrum"), use_container_width=True)

    st.markdown("<hr style='margin: 25px 0;'>", unsafe_allow_html=True)

    # 3. Interactive Anomalous Match Registry
    st.markdown("### 📋 Anomalous Fixtures Registry & Deep Dive")

    af_c1, af_c2, af_c3 = st.columns(3)
    with af_c1:
        anom_leagues = ["All Leagues"] + sorted(anom_df["league_name"].dropna().unique().tolist())
        sel_anom_league = st.selectbox("Filter Anomaly by League", anom_leagues)
    with af_c2:
        min_drift = st.slider("Min Absolute Home Odds Drift (%)", 0.0, 50.0, 10.0, step=2.5)
    with af_c3:
        anom_outcomes = ["All Outcomes", "H", "D", "A"]
        sel_anom_out = st.selectbox("Filter by Outcome", anom_outcomes)

    filtered_anom = anom_df.copy()
    if sel_anom_league != "All Leagues":
        filtered_anom = filtered_anom[filtered_anom["league_name"] == sel_anom_league]
    filtered_anom = filtered_anom[filtered_anom["mean_drift_home"].abs() >= min_drift]
    if sel_anom_out != "All Outcomes":
        filtered_anom = filtered_anom[filtered_anom["outcome"] == sel_anom_out]

    st.markdown(f"**Found {len(filtered_anom):,} fixtures with extreme market behavior.**")

    anom_table_cols = [
        "match_id", "kickoff_utc", "league_name", "home_team", "away_team",
        "ft_home", "ft_away", "outcome", "mean_drift_home", "spread_odds_home",
        "odds_std_home", "anomaly_score"
    ]
    st.dataframe(filtered_anom[anom_table_cols].sort_values("anomaly_score").head(50), use_container_width=True, height=350)
