"""
Match Explorer View
===================
Deep-dive inspection tool for historical matches:
- Multi-dimensional filters (League, Season, Outcome, Team Search)
- Match card presentation
- Side-by-side bookmaker comparisons across 1X2, Asian Handicap, and Over/Under
- Implied fair probabilities and margin comparisons
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.theme import (
    COLOR_ACCENT_PRIMARY,
    COLOR_ACCENT_SECONDARY,
    COLOR_BG_CARD,
    COLOR_BG_MAIN,
    COLOR_BG_SECONDARY,
    COLOR_BORDER,
    COLOR_DANGER,
    COLOR_SUCCESS,
    COLOR_TEXT_MAIN,
    COLOR_TEXT_MUTED,
    COLOR_WARNING,
    format_plotly_figure,
)


def render_match_explorer(matches_df: pd.DataFrame, closing_df: pd.DataFrame, features_df: pd.DataFrame):
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <div style="color:#C5F23B; font-size:0.8rem; font-weight:700; letter-spacing:0.1em; text-transform:uppercase;">
            Historical Fixture Registry
        </div>
        <h1 style="margin:0; font-size: 2.1rem; font-weight: 800;">Match & Odds Explorer</h1>
        <div style="color:#94A3B8; font-size: 0.95rem; margin-top: 4px;">
            Filter 63,597 matches and inspect multi-bookmaker opening, closing, and Asian handicap lines.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 1. Search and Filter Bar
    with st.container():
        st.markdown(f"<div class='dash-card'>", unsafe_allow_html=True)
        fc1, fc2, fc3, fc4 = st.columns([1.2, 1, 1, 1.2])

        with fc1:
            team_query = st.text_input("🔍 Search Team Name", "", placeholder="e.g. Arsenal, Real Madrid...")
        with fc2:
            leagues = ["All Leagues"] + sorted(matches_df["league_name"].dropna().unique().tolist())
            sel_league = st.selectbox("Competition", leagues)
        with fc3:
            seasons = ["All Seasons"] + sorted(matches_df["season"].dropna().unique().tolist(), reverse=True)
            sel_season = st.selectbox("Season", seasons)
        with fc4:
            outcomes = ["All Outcomes", "Home Win (H)", "Draw (D)", "Away Win (A)"]
            sel_outcome = st.selectbox("Match Outcome", outcomes)
        st.markdown("</div>", unsafe_allow_html=True)

    # Apply filters
    filtered = matches_df.copy()
    if team_query.strip():
        q = team_query.strip().lower()
        filtered = filtered[
            filtered["home_team"].str.lower().str.contains(q) |
            filtered["away_team"].str.lower().str.contains(q)
        ]
    if sel_league != "All Leagues":
        filtered = filtered[filtered["league_name"] == sel_league]
    if sel_season != "All Seasons":
        filtered = filtered[filtered["season"] == sel_season]
    if sel_outcome != "All Outcomes":
        code = sel_outcome[sel_outcome.find("(") + 1]
        filtered = filtered[filtered["outcome"] == code]

    st.markdown(f"**Found {len(filtered):,} matching fixtures.** Displaying top matches below:")

    # Table preview
    display_cols = [
        "match_id", "kickoff_utc", "league_name", "season", "home_team", "away_team",
        "ft_home", "ft_away", "outcome", "total_goals"
    ]
    st.dataframe(filtered[display_cols].head(50), use_container_width=True, height=260)

    if len(filtered) == 0:
        st.warning("No matches found matching the specified filters.")
        return

    st.markdown("<hr style='margin: 25px 0;'>", unsafe_allow_html=True)

    # 2. Detailed Single Match Inspector
    st.markdown("### 🎯 Single Match Inspection & Bookmaker Deep Dive")

    match_pick = st.selectbox(
        "Select a Match to Inspect",
        options=filtered["match_id"].head(50).tolist(),
        format_func=lambda x: f"Match ID #{x} — {filtered[filtered['match_id']==x]['home_team'].iloc[0]} vs {filtered[filtered['match_id']==x]['away_team'].iloc[0]} (FT: {int(filtered[filtered['match_id']==x]['ft_home'].iloc[0])}-{int(filtered[filtered['match_id']==x]['ft_away'].iloc[0])})"
    )

    m_row = matches_df[matches_df["match_id"] == match_pick].iloc[0]
    c_rows = closing_df[closing_df["match_id"] == match_pick]

    outcome_colors = {"H": COLOR_SUCCESS, "D": COLOR_WARNING, "A": COLOR_DANGER}
    outcome_labels = {"H": "Home Win", "D": "Draw", "A": "Away Win"}
    out_c = outcome_colors.get(m_row["outcome"], COLOR_TEXT_MUTED)

    # Match Presentation Card
    st.markdown(f"""
    <div class="match-hero">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:16px;">
            <span style="font-size:0.85rem; font-weight:700; color:{COLOR_ACCENT_PRIMARY}; text-transform:uppercase;">
                {m_row['league_name']} • Season {m_row['season']}
            </span>
            <span style="font-size:0.8rem; color:{COLOR_TEXT_MUTED};">
                Kickoff: {str(m_row['kickoff_utc'])[:16].replace('T', ' ')} UTC • Neutral: {'Yes' if m_row.get('neutral', 0) == 1 else 'No'}
            </span>
        </div>
        <div style="display:grid; grid-template-columns: 1fr auto 1fr; align-items:center; text-align:center; gap: 20px;">
            <div style="text-align: right;">
                <div style="font-size: 1.6rem; font-weight: 800; color:#F4F7FA;">{m_row['home_team']}</div>
                <div style="font-size:0.85rem; color:{COLOR_TEXT_MUTED};">HT Goals: {m_row.get('ht_home', 0)}</div>
            </div>
            <div>
                <div class="score-badge">{int(m_row['ft_home'])} - {int(m_row['ft_away'])}</div>
                <div style="margin-top:8px; font-weight:700; color:{out_c}; font-size:0.85rem;">
                    {outcome_labels.get(m_row['outcome'], 'Unknown')} (FT)
                </div>
            </div>
            <div style="text-align: left;">
                <div style="font-size: 1.6rem; font-weight: 800; color:#F4F7FA;">{m_row['away_team']}</div>
                <div style="font-size:0.85rem; color:{COLOR_TEXT_MUTED};">HT Goals: {m_row.get('ht_away', 0)}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Multi-Bookmaker Odds Breakdown Tabs
    tab_1x2, tab_ah_ou, tab_probs = st.tabs(["📊 1X2 Market Odds Comparison", "🎯 Asian Handicap & Over/Under", "📈 Implied Probability Breakdown"])

    with tab_1x2:
        if len(c_rows) > 0:
            grid_cols = st.columns(len(c_rows))
            for i, (_, row) in enumerate(c_rows.iterrows()):
                is_pinnacle = row["bookmaker"] == "Pinnacle"
                ovr = (row["close_overround"] * 100) if pd.notna(row["close_overround"]) else 0
                drift_h = row.get("drift_home_pct", 0)
                with grid_cols[i]:
                    st.markdown(f"""
                    <div class="dash-card {'highlight' if is_pinnacle else ''}" style="{'border-color:' + COLOR_ACCENT_PRIMARY if is_pinnacle else ''}">
                        <div style="font-size:1rem; font-weight:800; color:{COLOR_ACCENT_PRIMARY if is_pinnacle else '#F4F7FA'}; margin-bottom:10px;">
                            {row['bookmaker']} {'<span style="font-size:0.65rem; background:#C5F23B; color:#080D16; padding:2px 6px; border-radius:10px; font-weight:800;">SHARP</span>' if is_pinnacle else ''}
                        </div>
                        <div style="display:grid; grid-template-columns: 1fr 1fr 1fr; gap:6px; text-align:center; margin-bottom:10px;">
                            <div style="background:{COLOR_BG_SECONDARY}; padding:6px; border-radius:6px;">
                                <div style="font-size:0.68rem; color:#94A3B8;">OPEN 1</div>
                                <div style="font-size:0.95rem; font-weight:700;">{row['open_1x2_home']:.2f}</div>
                            </div>
                            <div style="background:{COLOR_BG_SECONDARY}; padding:6px; border-radius:6px;">
                                <div style="font-size:0.68rem; color:#94A3B8;">OPEN X</div>
                                <div style="font-size:0.95rem; font-weight:700;">{row['open_1x2_draw']:.2f}</div>
                            </div>
                            <div style="background:{COLOR_BG_SECONDARY}; padding:6px; border-radius:6px;">
                                <div style="font-size:0.68rem; color:#94A3B8;">OPEN 2</div>
                                <div style="font-size:0.95rem; font-weight:700;">{row['open_1x2_away']:.2f}</div>
                            </div>
                        </div>
                        <div style="display:grid; grid-template-columns: 1fr 1fr 1fr; gap:6px; text-align:center; margin-bottom:10px;">
                            <div style="background:rgba(39, 184, 209, 0.1); border:1px solid rgba(39, 184, 209, 0.3); padding:6px; border-radius:6px;">
                                <div style="font-size:0.68rem; color:#27B8D1;">CLOSE 1</div>
                                <div style="font-size:1.05rem; font-weight:800; color:#F4F7FA;">{row['close_1x2_home']:.2f}</div>
                            </div>
                            <div style="background:rgba(39, 184, 209, 0.1); border:1px solid rgba(39, 184, 209, 0.3); padding:6px; border-radius:6px;">
                                <div style="font-size:0.68rem; color:#27B8D1;">CLOSE X</div>
                                <div style="font-size:1.05rem; font-weight:800; color:#F4F7FA;">{row['close_1x2_draw']:.2f}</div>
                            </div>
                            <div style="background:rgba(39, 184, 209, 0.1); border:1px solid rgba(39, 184, 209, 0.3); padding:6px; border-radius:6px;">
                                <div style="font-size:0.68rem; color:#27B8D1;">CLOSE 2</div>
                                <div style="font-size:1.05rem; font-weight:800; color:#F4F7FA;">{row['close_1x2_away']:.2f}</div>
                            </div>
                        </div>
                        <div style="display:flex; justify-content:space-between; font-size:0.75rem; color:#94A3B8;">
                            <span>Margin: <b>{ovr:.2f}%</b></span>
                            <span>Home Drift: <b style="color:{COLOR_SUCCESS if drift_h > 0 else COLOR_DANGER};">{drift_h:+.1f}%</b></span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("No closing odds quotes available for this fixture.")

    with tab_ah_ou:
        st.markdown("<div style='font-size:0.95rem; font-weight:700; margin-bottom:12px;'>Asian Handicap & Over/Under Quotes</div>", unsafe_allow_html=True)
        ah_cols = [
            "bookmaker", "open_ah_line", "close_ah_line", "close_ah_home", "close_ah_away", "ah_home_result",
            "open_ou_line", "close_ou_line", "close_ou_over", "close_ou_under", "ou_over_result"
        ]
        avail_ah = c_rows[ah_cols].dropna(subset=["close_ah_line", "close_ou_line"], how="all")
        if len(avail_ah) > 0:
            st.dataframe(avail_ah, use_container_width=True)
            st.caption("Note: Pinnacle typically operates on pure 1X2 market volume in this dataset.")
        else:
            st.info("Asian Handicap / Goal Line data not provided for this specific match.")

    with tab_probs:
        st.markdown("<div style='font-size:0.95rem; font-weight:700; margin-bottom:12px;'>Normalized Fair Probabilities (Bookmaker Margin Removed)</div>", unsafe_allow_html=True)
        if len(c_rows) > 0:
            prob_df = c_rows[["bookmaker", "close_norm_prob_home", "close_norm_prob_draw", "close_norm_prob_away"]].copy()
            prob_df["close_norm_prob_home"] = (prob_df["close_norm_prob_home"] * 100).round(2)
            prob_df["close_norm_prob_draw"] = (prob_df["close_norm_prob_draw"] * 100).round(2)
            prob_df["close_norm_prob_away"] = (prob_df["close_norm_prob_away"] * 100).round(2)
            prob_df.columns = ["Bookmaker", "Fair P(Home) %", "Fair P(Draw) %", "Fair P(Away) %"]

            fig_p = px.bar(
                prob_df,
                x="Bookmaker",
                y=["Fair P(Home) %", "Fair P(Draw) %", "Fair P(Away) %"],
                barmode="group",
                color_discrete_sequence=[COLOR_ACCENT_PRIMARY, COLOR_WARNING, COLOR_ACCENT_SECONDARY]
            )
            st.plotly_chart(format_plotly_figure(fig_p, height=300), use_container_width=True)
            st.dataframe(prob_df, use_container_width=True)
