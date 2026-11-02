"""
Predictive Analytics View
=========================
Supervised ML probability estimation & interactive simulation:
- Offline match inference on actual historical fixtures
- Comparison against the Bookmaker Consensus Baseline
- Live interactive decimal odds simulator
- Clear labeling as offline statistical modeling
"""

import numpy as np
import pandas as pd
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
)
from src.models.trainer import FEATURE_COLUMNS


def render_predictive_analytics(matches_df: pd.DataFrame, features_df: pd.DataFrame, model_bundle: dict):
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <div style="color:#C5F23B; font-size:0.8rem; font-weight:700; letter-spacing:0.1em; text-transform:uppercase;">
            Supervised Intelligence
        </div>
        <h1 style="margin:0; font-size: 2.1rem; font-weight: 800;">Predictive Analytics & Probability Engine</h1>
        <div style="color:#94A3B8; font-size: 0.95rem; margin-top: 4px;">
            Calibrated multinomial probabilistic forecasts evaluated on 20,747 out-of-time test fixtures.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.warning("⚠️ **Academic & Research Disclaimer:** Predictions generated here represent historical offline machine learning estimations trained to evaluate market efficiency. They do not constitute live financial or gambling advice.")

    model = model_bundle["model"]
    model_name = model_bundle["model_name"]
    feat_cols = model_bundle["feature_columns"]

    tab_pred, tab_sim = st.tabs(["🔮 Historical Fixture Model Inference", "🎮 Interactive Odds Probability Simulator"])

    # -------------------------------------------------------------------------
    # TAB 1: Historical Fixture Inference
    # -------------------------------------------------------------------------
    with tab_pred:
        st.markdown("<div style='font-size:1.1rem; font-weight:700; margin-bottom:12px;'>Select Match from Out-of-Time Test Set (2024–2026)</div>", unsafe_allow_html=True)

        test_fixtures = features_df[features_df["kickoff_year"] >= 2024]
        if len(test_fixtures) == 0:
            test_fixtures = features_df

        fixture_options = test_fixtures.head(50)
        fix_dict = {
            r["match_id"]: f"{r['home_team']} vs {r['away_team']} ({str(r['kickoff_utc'])[:10]}) • Actual: {r['outcome']}"
            for _, r in fixture_options.iterrows()
        }
        pick_fix = st.selectbox("Choose Fixture", list(fix_dict.keys()), format_func=lambda x: fix_dict[x])

        f_row = features_df[features_df["match_id"] == pick_fix].iloc[0]

        # Extract features and predict
        X_vec = f_row[feat_cols].values.reshape(1, -1)
        y_prob = model.predict_proba(X_vec)[0]

        p_h_ml, p_d_ml, p_a_ml = y_prob[0] * 100, y_prob[1] * 100, y_prob[2] * 100
        p_h_bk = f_row["mkt_prob_home"] * 100
        p_d_bk = f_row["mkt_prob_draw"] * 100
        p_a_bk = f_row["mkt_prob_away"] * 100

        pred_idx = np.argmax(y_prob)
        class_names = ["Home Win (H)", "Draw (D)", "Away Win (A)"]
        predicted_outcome = class_names[pred_idx]
        actual_outcome = f_row["outcome"]
        actual_map = {"H": "Home Win (H)", "D": "Draw (D)", "A": "Away Win (A)"}

        is_correct = (pred_idx == 0 and actual_outcome == "H") or (pred_idx == 1 and actual_outcome == "D") or (pred_idx == 2 and actual_outcome == "A")

        # Top Results Cards
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""
            <div class="dash-card">
                <div style="font-size:0.75rem; color:{COLOR_TEXT_MUTED}; font-weight:700; text-transform:uppercase;">MODEL PREDICTION</div>
                <div style="font-size:1.5rem; font-weight:800; color:{COLOR_ACCENT_PRIMARY};">{predicted_outcome}</div>
                <div style="font-size:0.8rem; color:{COLOR_TEXT_MUTED}; margin-top:4px;">Confidence: <b>{np.max(y_prob)*100:.1f}%</b></div>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
            <div class="dash-card">
                <div style="font-size:0.75rem; color:{COLOR_TEXT_MUTED}; font-weight:700; text-transform:uppercase;">ACTUAL MATCH OUTCOME</div>
                <div style="font-size:1.5rem; font-weight:800; color:{COLOR_TEXT_MAIN};">{actual_map.get(actual_outcome, actual_outcome)}</div>
                <div style="font-size:0.8rem; color:{COLOR_TEXT_MUTED}; margin-top:4px;">Full Time Score: <b>{int(f_row['ft_home'])} - {int(f_row['ft_away'])}</b></div>
            </div>
            """, unsafe_allow_html=True)

        with c3:
            st.markdown(f"""
            <div class="dash-card" style="border-left: 4px solid {COLOR_SUCCESS if is_correct else COLOR_DANGER};">
                <div style="font-size:0.75rem; color:{COLOR_TEXT_MUTED}; font-weight:700; text-transform:uppercase;">FORECAST ACCURACY</div>
                <div style="font-size:1.5rem; font-weight:800; color:{COLOR_SUCCESS if is_correct else COLOR_DANGER};">
                    {'✓ HIT' if is_correct else '✗ MISS'}
                </div>
                <div style="font-size:0.8rem; color:{COLOR_TEXT_MUTED}; margin-top:4px;">Evaluated via Out-of-Time Test Set</div>
            </div>
            """, unsafe_allow_html=True)

        # Comparative Bar Chart
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            name=f"{model_name}",
            x=["Home Win", "Draw", "Away Win"],
            y=[p_h_ml, p_d_ml, p_a_ml],
            marker_color=COLOR_ACCENT_PRIMARY,
            text=[f"{p_h_ml:.1f}%", f"{p_d_ml:.1f}%", f"{p_a_ml:.1f}%"],
            textposition="auto"
        ))
        fig_bar.add_trace(go.Bar(
            name="Bookmaker Consensus",
            x=["Home Win", "Draw", "Away Win"],
            y=[p_h_bk, p_d_bk, p_a_bk],
            marker_color=COLOR_ACCENT_SECONDARY,
            text=[f"{p_h_bk:.1f}%", f"{p_d_bk:.1f}%", f"{p_a_bk:.1f}%"],
            textposition="auto"
        ))
        fig_bar.update_layout(
            barmode="group",
            yaxis=dict(title="Probability (%)", range=[0, 100]),
            height=320
        )
        st.plotly_chart(format_plotly_figure(fig_bar, title="Predicted Probability Comparison: ML Model vs Market Consensus"), use_container_width=True)

    # -------------------------------------------------------------------------
    # TAB 2: Interactive Odds Simulator
    # -------------------------------------------------------------------------
    with tab_sim:
        st.markdown("<div style='font-size:1.1rem; font-weight:700; margin-bottom:12px;'>Custom Odds Probability Simulator</div>", unsafe_allow_html=True)
        st.markdown("Enter custom decimal odds to calculate implied fair probabilities, bookmaker overround margin, and estimated probability distribution:")

        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            user_h = st.number_input("Home Win Odds (1)", min_value=1.01, max_value=40.0, value=2.20, step=0.05, key="sim_h")
        with sc2:
            user_d = st.number_input("Draw Odds (X)", min_value=1.01, max_value=40.0, value=3.30, step=0.05, key="sim_d")
        with sc3:
            user_a = st.number_input("Away Win Odds (2)", min_value=1.01, max_value=40.0, value=3.40, step=0.05, key="sim_a")

        # Math
        raw_h = 1.0 / user_h
        raw_d = 1.0 / user_d
        raw_a = 1.0 / user_a
        raw_sum = raw_h + raw_d + raw_a
        fair_h = (raw_h / raw_sum) * 100
        fair_d = (raw_d / raw_sum) * 100
        fair_a = (raw_a / raw_sum) * 100
        sim_margin = (raw_sum - 1.0) * 100

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.metric("Fair P(Home Win)", f"{fair_h:.1f}%")
        with k2:
            st.metric("Fair P(Draw)", f"{fair_d:.1f}%")
        with k3:
            st.metric("Fair P(Away Win)", f"{fair_a:.1f}%")
        with k4:
            st.metric("House Overround Margin", f"{sim_margin:.2f}%")

        fig_gauge = go.Figure(data=[
            go.Bar(
                x=["Home Win", "Draw", "Away Win"],
                y=[fair_h, fair_d, fair_a],
                marker_color=[COLOR_ACCENT_PRIMARY, COLOR_ACCENT_SECONDARY, COLOR_WARNING],
                text=[f"{fair_h:.1f}%", f"{fair_d:.1f}%", f"{fair_a:.1f}%"],
                textposition="auto"
            )
        ])
        fig_gauge.update_layout(yaxis=dict(title="Fair Probability (%)", range=[0, 100]), height=280)
        st.plotly_chart(format_plotly_figure(fig_gauge, title="Simulated Fair Probability Distribution"), use_container_width=True)
