"""
Design System & Styling Module
==============================
Implements the premium sports analytics theme:
- Deep navy / dark canvas (#080D16, #101927)
- Analytical card containers (#172333) with crisp borders (#253449)
- High-contrast accents: Lime Green (#C5F23B) and Cyan (#27B8D1)
- Standardized Plotly figure layouts
"""

import plotly.graph_objects as go
import streamlit as st

# Color Palette Constants
COLOR_BG_MAIN = "#080D16"
COLOR_BG_SECONDARY = "#101927"
COLOR_BG_CARD = "#172333"
COLOR_BORDER = "#253449"
COLOR_ACCENT_PRIMARY = "#C5F23B"   # Lime green
COLOR_ACCENT_SECONDARY = "#27B8D1" # Cyan
COLOR_TEXT_MAIN = "#F4F7FA"
COLOR_TEXT_MUTED = "#94A3B8"
COLOR_SUCCESS = "#36C98F"
COLOR_WARNING = "#F5B942"
COLOR_DANGER = "#F06464"


def inject_custom_css():
    """Inject global CSS rules to style Streamlit into a sports analytics workstation."""
    css = f"""
    <style>
        /* Import Inter Font */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');

        /* Global Canvas */
        html, body, [data-testid="stAppViewContainer"], .main {{
            background-color: {COLOR_BG_MAIN} !important;
            color: {COLOR_TEXT_MAIN} !important;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        }}

        [data-testid="stHeader"] {{
            background-color: rgba(8, 13, 22, 0.9) !important;
            border-bottom: 1px solid {COLOR_BORDER};
        }}

        /* Sidebar Styling */
        [data-testid="stSidebar"] {{
            background-color: {COLOR_BG_SECONDARY} !important;
            border-right: 1px solid {COLOR_BORDER} !important;
        }}

        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{
            color: {COLOR_TEXT_MUTED};
        }}

        /* Radio Navigation Styling */
        [data-testid="stSidebar"] .stRadio > div {{
            background: transparent;
            gap: 6px;
        }}

        [data-testid="stSidebar"] .stRadio label {{
            background: {COLOR_BG_CARD} !important;
            border: 1px solid {COLOR_BORDER} !important;
            border-radius: 8px !important;
            padding: 10px 14px !important;
            color: {COLOR_TEXT_MAIN} !important;
            font-size: 0.9rem !important;
            font-weight: 500 !important;
            transition: all 0.2s ease-in-out !important;
            cursor: pointer;
            margin-bottom: 4px;
        }}

        [data-testid="stSidebar"] .stRadio label:hover {{
            border-color: {COLOR_ACCENT_PRIMARY} !important;
            color: {COLOR_ACCENT_PRIMARY} !important;
            background: rgba(197, 242, 59, 0.05) !important;
        }}

        [data-testid="stSidebar"] .stRadio [data-checked="true"] {{
            background: rgba(197, 242, 59, 0.12) !important;
            border-color: {COLOR_ACCENT_PRIMARY} !important;
            color: {COLOR_ACCENT_PRIMARY} !important;
            font-weight: 700 !important;
        }}

        /* Standard Cards */
        .dash-card {{
            background-color: {COLOR_BG_CARD};
            border: 1px solid {COLOR_BORDER};
            border-radius: 12px;
            padding: 18px 20px;
            margin-bottom: 16px;
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.35);
        }}

        .dash-card:hover {{
            border-color: rgba(39, 184, 209, 0.4);
        }}

        /* KPI Card */
        .kpi-container {{
            background-color: {COLOR_BG_CARD};
            border: 1px solid {COLOR_BORDER};
            border-radius: 12px;
            padding: 16px 20px;
            margin-bottom: 12px;
            position: relative;
            overflow: hidden;
            border-top: 3px solid {COLOR_ACCENT_PRIMARY};
        }}

        .kpi-label {{
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: {COLOR_TEXT_MUTED};
            font-weight: 600;
            margin-bottom: 6px;
        }}

        .kpi-value {{
            font-size: 1.85rem;
            font-weight: 800;
            color: {COLOR_TEXT_MAIN};
            font-family: 'JetBrains Mono', monospace;
            line-height: 1.2;
        }}

        .kpi-badge {{
            display: inline-block;
            font-size: 0.75rem;
            font-weight: 600;
            padding: 2px 8px;
            border-radius: 20px;
            margin-top: 8px;
        }}

        /* Match Display Card */
        .match-hero {{
            background: linear-gradient(135deg, {COLOR_BG_CARD} 0%, {COLOR_BG_SECONDARY} 100%);
            border: 1px solid {COLOR_BORDER};
            border-radius: 14px;
            padding: 24px;
            margin-bottom: 20px;
        }}

        .team-pill {{
            font-size: 1.25rem;
            font-weight: 700;
            color: {COLOR_TEXT_MAIN};
        }}

        .score-badge {{
            background: {COLOR_BG_MAIN};
            border: 1px solid {COLOR_ACCENT_PRIMARY};
            color: {COLOR_ACCENT_PRIMARY};
            padding: 6px 18px;
            border-radius: 30px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.5rem;
            font-weight: 800;
            letter-spacing: 2px;
            display: inline-block;
        }}

        .odds-chip {{
            background: {COLOR_BG_SECONDARY};
            border: 1px solid {COLOR_BORDER};
            border-radius: 8px;
            padding: 10px 14px;
            text-align: center;
        }}

        .odds-chip.highlight {{
            border-color: {COLOR_ACCENT_PRIMARY};
            background: rgba(197, 242, 59, 0.08);
        }}

        .odds-chip .label {{
            font-size: 0.75rem;
            color: {COLOR_TEXT_MUTED};
            text-transform: uppercase;
            font-weight: 600;
        }}

        .odds-chip .value {{
            font-size: 1.25rem;
            font-weight: 800;
            color: {COLOR_TEXT_MAIN};
            font-family: 'JetBrains Mono', monospace;
        }}

        .odds-chip.highlight .value {{
            color: {COLOR_ACCENT_PRIMARY};
        }}

        .odds-chip .sub {{
            font-size: 0.75rem;
            color: {COLOR_ACCENT_SECONDARY};
            font-weight: 600;
        }}

        /* Inputs & Select Boxes */
        .stSelectbox > div > div, .stTextInput > div > div > input, .stNumberInput > div > div > input {{
            background-color: {COLOR_BG_SECONDARY} !important;
            border: 1px solid {COLOR_BORDER} !important;
            color: {COLOR_TEXT_MAIN} !important;
            border-radius: 8px !important;
        }}

        .stSelectbox > div > div:hover, .stTextInput > div > div > input:focus {{
            border-color: {COLOR_ACCENT_SECONDARY} !important;
        }}

        /* DataFrame styling */
        [data-testid="stDataFrame"] {{
            border: 1px solid {COLOR_BORDER} !important;
            border-radius: 10px !important;
            overflow: hidden !important;
        }}

        /* Buttons */
        .stButton > button {{
            background: linear-gradient(90deg, {COLOR_ACCENT_PRIMARY} 0%, #AEE620 100%) !important;
            color: #080D16 !important;
            font-weight: 700 !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 8px 20px !important;
            transition: all 0.2s ease !important;
        }}

        .stButton > button:hover {{
            transform: translateY(-1px) !important;
            box-shadow: 0 4px 14px rgba(197, 242, 59, 0.35) !important;
        }}

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {{
            background-color: {COLOR_BG_SECONDARY} !important;
            border-radius: 10px !important;
            padding: 4px !important;
            border: 1px solid {COLOR_BORDER} !important;
        }}

        .stTabs [data-baseweb="tab"] {{
            color: {COLOR_TEXT_MUTED} !important;
            font-weight: 600 !important;
            border-radius: 8px !important;
            padding: 8px 16px !important;
        }}

        .stTabs [aria-selected="true"] {{
            background-color: {COLOR_BG_CARD} !important;
            color: {COLOR_ACCENT_PRIMARY} !important;
        }}

        /* Section Headings */
        h1, h2, h3, h4 {{
            color: {COLOR_TEXT_MAIN} !important;
            font-weight: 700 !important;
            letter-spacing: -0.02em !important;
        }}

        /* Divider */
        hr {{
            border: none !important;
            border-top: 1px solid {COLOR_BORDER} !important;
            margin: 20px 0 !important;
        }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def format_plotly_figure(fig: go.Figure, title: str | None = None, height: int = 380) -> go.Figure:
    """Apply uniform dark analytics styling to any Plotly figure."""
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>" if title else "",
            font=dict(color=COLOR_TEXT_MAIN, size=14, family="Inter")
        ),
        paper_bgcolor=COLOR_BG_CARD,
        plot_bgcolor=COLOR_BG_CARD,
        font=dict(color=COLOR_TEXT_MUTED, family="Inter", size=11),
        height=height,
        margin=dict(l=40, r=30, t=50 if title else 20, b=40),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(0,0,0,0)",
            font=dict(color=COLOR_TEXT_MAIN, size=10),
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis=dict(
            gridcolor=COLOR_BORDER,
            zerolinecolor=COLOR_BORDER,
            linecolor=COLOR_BORDER,
            tickfont=dict(color=COLOR_TEXT_MUTED)
        ),
        yaxis=dict(
            gridcolor=COLOR_BORDER,
            zerolinecolor=COLOR_BORDER,
            linecolor=COLOR_BORDER,
            tickfont=dict(color=COLOR_TEXT_MUTED)
        ),
        hoverlabel=dict(
            bgcolor=COLOR_BG_MAIN,
            bordercolor=COLOR_ACCENT_PRIMARY,
            font=dict(color=COLOR_TEXT_MAIN, family="Inter")
        )
    )
    return fig


def render_kpi_card(label: str, value: str, subtext: str = "", badge: str = "", badge_type: str = "success"):
    """Render a KPI tile with lime-accent border."""
    badge_colors = {
        "success": ("rgba(54, 201, 143, 0.15)", COLOR_SUCCESS),
        "warning": ("rgba(245, 185, 66, 0.15)", COLOR_WARNING),
        "danger": ("rgba(240, 100, 100, 0.15)", COLOR_DANGER),
        "cyan": ("rgba(39, 184, 209, 0.15)", COLOR_ACCENT_SECONDARY),
        "lime": ("rgba(197, 242, 59, 0.15)", COLOR_ACCENT_PRIMARY)
    }
    bg_c, text_c = badge_colors.get(badge_type, badge_colors["lime"])

    badge_html = f'<div class="kpi-badge" style="background:{bg_c}; color:{text_c};">{badge}</div>' if badge else ""
    sub_html = f'<div style="color:{COLOR_TEXT_MUTED}; font-size:0.75rem; margin-top:4px;">{subtext}</div>' if subtext else ""

    html = f"""
    <div class="kpi-container">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {sub_html}
        {badge_html}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
