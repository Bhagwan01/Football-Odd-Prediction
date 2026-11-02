"""
About View
==========
Project provenance, architectural overview, data integrity standards,
and academic portfolio credentials.
"""

import streamlit as st
from dashboard.theme import (
    COLOR_ACCENT_PRIMARY,
    COLOR_ACCENT_SECONDARY,
    COLOR_BG_CARD,
    COLOR_BG_SECONDARY,
    COLOR_BORDER,
    COLOR_TEXT_MAIN,
    COLOR_TEXT_MUTED,
)


def render_about():
    st.markdown("""
    <div style="margin-bottom: 24px;">
        <div style="color:#C5F23B; font-size:0.8rem; font-weight:700; letter-spacing:0.1em; text-transform:uppercase;">
            System Documentation & Provenance
        </div>
        <h1 style="margin:0; font-size: 2.1rem; font-weight: 800;">About ODDSPULSE Intelligence Platform</h1>
        <div style="color:#94A3B8; font-size: 0.95rem; margin-top: 4px;">
            An end-to-end data science & quantitative machine learning portfolio project.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1.1, 0.9], gap="large")

    with col1:
        st.markdown(f"""
        <div class="dash-card">
            <h3 style="color:{COLOR_ACCENT_PRIMARY}; margin-top:0;">Project Overview</h3>
            <p style="color:{COLOR_TEXT_MAIN}; line-height:1.7;">
                <b>Football Odds Intelligence & Match Analytics (ODDSPULSE)</b> is an applied quantitative research platform developed to analyze betting market dynamics, multi-bookmaker pricing efficiency, and match outcome probability estimation.
            </p>
            <p style="color:{COLOR_TEXT_MUTED}; line-height:1.7;">
                Unlike standard classroom tutorials that rely on toy datasets or disconnected scripts, this project operates on <b>63,597 genuine matches</b> and <b>~37 million tick-level odds updates</b> across Pinnacle, Bet365, SBOBET, and Singbet from 2018 through 2026.
            </p>
            <hr style="margin:16px 0;">
            <h4 style="color:#F4F7FA;">Core Research Questions Answered</h4>
            <ol style="color:{COLOR_TEXT_MUTED}; line-height:1.8; padding-left:20px;">
                <li><b>Bookmaker Pricing Hierarchies:</b> Formally confirmed Pinnacle as the sharpest benchmark bookmaker (3.34% median overround) vs recreational bookmakers (5.95%–6.23%) using Kruskal-Wallis & Mann-Whitney U tests (p &lt; 10<sup>-300</sup>).</li>
                <li><b>Home Advantage Quantification:</b> One-sample Z-test confirms global Home Win rate of <b>43.81%</b> (Z = 56.06, p &lt; 10<sup>-100</sup>).</li>
                <li><b>Leakage-Free Supervised Modeling:</b> Evaluated on an out-of-time test set (20,747 matches) where Calibrated Logistic Regression matches/edges the bookmaker consensus baseline (52.85% accuracy, 0.9705 Log Loss).</li>
                <li><b>Unsupervised Anomaly Discovery:</b> Flagged 1,262 fixtures with extreme market disagreement; revealed a <b>72.6% collapse in Draw probability</b> in erratic fixtures.</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="dash-card">
            <h3 style="color:{COLOR_ACCENT_SECONDARY}; margin-top:0;">Author & Engineering Profile</h3>
            <div style="color:{COLOR_TEXT_MAIN}; line-height:1.8;">
                • <b>Lead Developer:</b> Mohd Afzal<br>
                • <b>Academic Track:</b> B.Tech Electronics & Communication Engineering (3rd Year)<br>
                • <b>Focus Areas:</b> Data Science, Machine Learning Engineering, Quantitative Analytics<br>
                • <b>GitHub Repository:</b> <a href="https://github.com/Bhagwan01/Football-Odd-Prediction" target="_blank" style="color:{COLOR_ACCENT_PRIMARY}; text-decoration:none;">Bhagwan01/Football-Odd-Prediction</a>
            </div>
            <hr style="margin:16px 0;">
            <h4 style="color:#F4F7FA;">Engineering Standards Enforced</h4>
            <div style="color:{COLOR_TEXT_MUTED}; font-size:0.85rem; line-height:1.7;">
                ✓ <b>Python 3.14 / 3.12</b> compatibility<br>
                ✓ <b>Zero Data Leakage:</b> Strict chronological train/test split<br>
                ✓ <b>Zero Synthetic Data:</b> 100% real historical datasets<br>
                ✓ <b>Test-Driven Development:</b> 15 automated unit tests with 100% pass rate<br>
                ✓ <b>Optimized Memory:</b> Apache Parquet columnar storage reducing 2.5 GB to 289 MB
            </div>
        </div>

        <div class="dash-card" style="border-left: 4px solid {COLOR_ACCENT_PRIMARY};">
            <h4 style="color:#F4F7FA; margin-top:0;">Academic & Portfolio Disclaimer</h4>
            <p style="font-size:0.82rem; color:{COLOR_TEXT_MUTED}; line-height:1.6; margin-bottom:0;">
                This project was created purely for academic research, data science portfolio demonstration, and educational purposes. It does not provide real-time gambling or commercial betting advice.
            </p>
        </div>
        """, unsafe_allow_html=True)
