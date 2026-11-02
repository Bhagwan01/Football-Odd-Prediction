"""
Odds Analytics View
===================
Deep quantitative analysis of betting market structures:
- Decimal odds and implied probability distributions
- Opening vs closing odds paired drift dynamics
- Interactive tick-level odds movement over time for selected matches
- Overround distribution comparisons
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
)
from src.data.loader import load_odds_1x2


def render_odds_analytics(matches_df: pd.DataFrame, closing_df: pd.DataFrame):
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <div style="color:#C5F23B; font-size:0.8rem; font-weight:700; letter-spacing:0.1em; text-transform:uppercase;">
            Market Microstructure
        </div>
        <h1 style="margin:0; font-size: 2.1rem; font-weight: 800;">Odds Analytics & Dynamics</h1>
        <div style="color:#94A3B8; font-size: 0.95rem; margin-top: 4px;">
            Distributional properties, overround compression, drift velocity, and tick-level movements.
        </div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Odds & Probability Distributions",
        "⚡ Opening vs Closing Drift",
        "⏱️ Tick-Level Odds Movements (Time-Series)",
        "📐 Overround Margin Anatomy"
    ])

    # -------------------------------------------------------------------------
    # TAB 1: Distributions
    # -------------------------------------------------------------------------
    with tab1:
        st.markdown("<div style='font-size:1.1rem; font-weight:700; margin-bottom:14px;'>Decimal Odds & Fair Probability Spectra</div>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        with col1:
            odds_type = st.selectbox("Select Outcome Market", ["Home Win Odds", "Draw Odds", "Away Win Odds"])
            col_map = {"Home Win Odds": "close_1x2_home", "Draw Odds": "close_1x2_draw", "Away Win Odds": "close_1x2_away"}
            target_col = col_map[odds_type]

            s = closing_df[target_col].dropna()
            q99 = s.quantile(0.99)
            s_clipped = s[s <= q99]

            fig_odds = px.histogram(
                s_clipped,
                nbins=60,
                color_discrete_sequence=[COLOR_ACCENT_PRIMARY],
                labels={"value": "Decimal Odds"}
            )
            fig_odds.add_vline(x=float(s.median()), line_dash="dash", line_color="#F4F7FA", annotation_text=f"Median: {s.median():.2f}")
            fig_odds.update_layout(showlegend=False, xaxis_title="Closing Decimal Odds", yaxis_title="Frequency")
            st.plotly_chart(format_plotly_figure(fig_odds, title=f"Distribution of Closing {odds_type}"), use_container_width=True)

        with col2:
            prob_col_map = {"Home Win Odds": "close_norm_prob_home", "Draw Odds": "close_norm_prob_draw", "Away Win Odds": "close_norm_prob_away"}
            target_p = prob_col_map[odds_type]
            sp = closing_df[target_p].dropna() * 100

            fig_prob = px.histogram(
                sp,
                nbins=60,
                color_discrete_sequence=[COLOR_ACCENT_SECONDARY],
                labels={"value": "Fair Implied Probability (%)"}
            )
            fig_prob.add_vline(x=float(sp.median()), line_dash="dash", line_color=COLOR_ACCENT_PRIMARY, annotation_text=f"Median: {sp.median():.1f}%")
            fig_prob.update_layout(showlegend=False, xaxis_title="Fair Implied Probability (%)", yaxis_title="Frequency")
            st.plotly_chart(format_plotly_figure(fig_prob, title=f"Fair Normalized Probability: {odds_type}"), use_container_width=True)

    # -------------------------------------------------------------------------
    # TAB 2: Opening vs Closing Drift
    # -------------------------------------------------------------------------
    with tab2:
        st.markdown("<div style='font-size:1.1rem; font-weight:700; margin-bottom:14px;'>Odds Drift Dynamics (Open vs Close)</div>", unsafe_allow_html=True)
        st.markdown("""
        Odds drift measures the percentage price shift between opening quotes and kickoff closing lines:
        $$\\Delta \\% = \\frac{Odds_{close} - Odds_{open}}{Odds_{open}} \\times 100$$
        """)

        d_col1, d_col2 = st.columns([1, 1])
        with d_col1:
            drift_clean = closing_df.dropna(subset=["drift_home_pct", "drift_away_pct", "bookmaker"]).copy()
            drift_clean = drift_clean[(drift_clean["drift_home_pct"] >= -35) & (drift_clean["drift_home_pct"] <= 35)]

            fig_drift_box = px.box(
                drift_clean,
                x="bookmaker",
                y="drift_home_pct",
                color="bookmaker",
                color_discrete_sequence=[COLOR_ACCENT_PRIMARY, COLOR_ACCENT_SECONDARY, COLOR_WARNING, COLOR_DANGER],
                points=False
            )
            fig_drift_box.add_hline(y=0, line_dash="dash", line_color="#F4F7FA")
            fig_drift_box.update_layout(showlegend=False, xaxis_title="Bookmaker", yaxis_title="Home Odds Drift (%)")
            st.plotly_chart(format_plotly_figure(fig_drift_box, title="Home Odds Drift Dispersion by Bookmaker"), use_container_width=True)

        with d_col2:
            fig_scatter = px.scatter(
                drift_clean.sample(min(3000, len(drift_clean)), random_state=42),
                x="open_1x2_home",
                y="close_1x2_home",
                color="bookmaker",
                opacity=0.6,
                color_discrete_sequence=[COLOR_ACCENT_PRIMARY, COLOR_ACCENT_SECONDARY, COLOR_WARNING, COLOR_DANGER],
                range_x=[1.0, 8.0],
                range_y=[1.0, 8.0]
            )
            fig_scatter.add_shape(type="line", x0=1.0, y0=1.0, x1=8.0, y1=8.0, line=dict(color="#F4F7FA", dash="dash"))
            fig_scatter.update_layout(xaxis_title="Opening Home Odds", yaxis_title="Closing Home Odds")
            st.plotly_chart(format_plotly_figure(fig_scatter, title="Open vs Close Price Parity (Diagonal = Zero Drift)"), use_container_width=True)

    # -------------------------------------------------------------------------
    # TAB 3: Tick-Level Time Series
    # -------------------------------------------------------------------------
    with tab3:
        st.markdown("<div style='font-size:1.1rem; font-weight:700; margin-bottom:14px;'>Tick-Level Odds Movements Over Time</div>", unsafe_allow_html=True)
        st.markdown("Inspect minute-by-minute time-series odds updates streamed directly from `data/raw/odds_1x2.parquet` via PyArrow filters:")

        # Match selector for tick data
        sample_fixtures = matches_df.head(40)
        fixture_dict = {
            r["match_id"]: f"{r['home_team']} vs {r['away_team']} ({str(r['kickoff_utc'])[:10]})"
            for _, r in sample_fixtures.iterrows()
        }
        pick_tick_match = st.selectbox(
            "Select Fixture for Time-Series Analysis",
            options=list(fixture_dict.keys()),
            format_func=lambda x: fixture_dict[x],
            key="tick_match_picker"
        )

        with st.spinner("Streaming tick-level price series via PyArrow..."):
            try:
                ticks_df = load_odds_1x2(match_ids=[pick_tick_match])
                if len(ticks_df) > 0:
                    ticks_df["ts_utc"] = pd.to_datetime(ticks_df["ts_utc"])
                    ticks_df = ticks_df.sort_values("ts_utc")

                    c_ts1, c_ts2, c_ts3 = st.columns(3)
                    with c_ts1:
                        fig_th = px.line(
                            ticks_df,
                            x="ts_utc",
                            y="home",
                            color="bookmaker",
                            markers=True,
                            color_discrete_sequence=[COLOR_ACCENT_PRIMARY, COLOR_ACCENT_SECONDARY, COLOR_WARNING, COLOR_DANGER]
                        )
                        fig_th.update_layout(xaxis_title="Timestamp (UTC)", yaxis_title="Home Odds")
                        st.plotly_chart(format_plotly_figure(fig_th, title="Home Win (1) Odds Movement"), use_container_width=True)

                    with c_ts2:
                        fig_td = px.line(
                            ticks_df,
                            x="ts_utc",
                            y="draw",
                            color="bookmaker",
                            markers=True,
                            color_discrete_sequence=[COLOR_ACCENT_PRIMARY, COLOR_ACCENT_SECONDARY, COLOR_WARNING, COLOR_DANGER]
                        )
                        fig_td.update_layout(xaxis_title="Timestamp (UTC)", yaxis_title="Draw Odds")
                        st.plotly_chart(format_plotly_figure(fig_td, title="Draw (X) Odds Movement"), use_container_width=True)

                    with c_ts3:
                        fig_ta = px.line(
                            ticks_df,
                            x="ts_utc",
                            y="away",
                            color="bookmaker",
                            markers=True,
                            color_discrete_sequence=[COLOR_ACCENT_PRIMARY, COLOR_ACCENT_SECONDARY, COLOR_WARNING, COLOR_DANGER]
                        )
                        fig_ta.update_layout(xaxis_title="Timestamp (UTC)", yaxis_title="Away Odds")
                        st.plotly_chart(format_plotly_figure(fig_ta, title="Away Win (2) Odds Movement"), use_container_width=True)

                    st.caption(f"Streamed {len(ticks_df):,} distinct price ticks across {ticks_df['bookmaker'].nunique()} active market makers for Match #{pick_tick_match}.")
                else:
                    st.info("No time-series ticks recorded for this match.")
            except Exception as e:
                st.error(f"Error loading time-series: {e}")

    # -------------------------------------------------------------------------
    # TAB 4: Overround Anatomy
    # -------------------------------------------------------------------------
    with tab4:
        st.markdown("<div style='font-size:1.1rem; font-weight:700; margin-bottom:14px;'>Market Margin (Overround) Structure</div>", unsafe_allow_html=True)
        valid_ovr = closing_df.dropna(subset=["close_overround", "bookmaker"]).copy()
        valid_ovr["ovr_pct"] = valid_ovr["close_overround"] * 100
        valid_ovr = valid_ovr[(valid_ovr["ovr_pct"] >= 0) & (valid_ovr["ovr_pct"] <= 20)]

        fig_kde = px.histogram(
            valid_ovr,
            x="ovr_pct",
            color="bookmaker",
            barmode="overlay",
            nbins=60,
            opacity=0.55,
            color_discrete_sequence=[COLOR_ACCENT_PRIMARY, COLOR_ACCENT_SECONDARY, COLOR_WARNING, COLOR_DANGER]
        )
        fig_kde.add_vline(x=0, line_dash="dash", line_color="#F4F7FA", annotation_text="0% (Zero Margin Market)")
        fig_kde.update_layout(xaxis_title="Overround Margin (%)", yaxis_title="Frequency")
        st.plotly_chart(format_plotly_figure(fig_kde, title="Overround Distribution Across 251,326 Bookmaker Quotes"), use_container_width=True)
