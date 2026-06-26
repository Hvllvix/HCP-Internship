"""
Light premium theme — palette: #000000, #14213d, #fca311, #e5e5e5, #ffffff
Inspired by app_style.py structure with lighter surfaces.
"""
import streamlit as st

PALETTE = {
    "black": "#000000",
    "navy": "#14213d",
    "amber": "#fca311",
    "gray": "#e5e5e5",
    "white": "#ffffff",
    "navy_light": "#1e3a5f",
    "navy_muted": "#4a5568",
    "amber_soft": "#fff4e0",
    "bg": "#f7f8fa",
    "danger": "#b42318",
    "success": "#0d7a4a",
}

CSS = f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    :root {{
        --black: {PALETTE['black']};
        --navy: {PALETTE['navy']};
        --amber: {PALETTE['amber']};
        --gray: {PALETTE['gray']};
        --white: {PALETTE['white']};
        --bg: {PALETTE['bg']};
        --text-main: {PALETTE['navy']};
        --text-dim: {PALETTE['navy_muted']};
        --card-bg: {PALETTE['white']};
        --card-border: {PALETTE['gray']};
        --accent-soft: {PALETTE['amber_soft']};
    }}

    .stApp {{
        background: var(--bg);
        color: var(--text-main);
        font-family: 'Inter', sans-serif;
    }}
    .block-container {{
        padding-top: 1.25rem !important;
        max-width: 1440px;
    }}

    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, var(--navy) 0%, #0f1a2e 100%) !important;
        border-right: 1px solid rgba(255,255,255,0.08);
        min-width: 22rem !important;
        width: 22rem !important;
    }}
    [data-testid="stSidebar"][aria-expanded="true"] ~ div .main .block-container {{
        margin-left: 0;
    }}
    section[data-testid="stSidebar"] > div {{
        width: 22rem !important;
        min-width: 22rem !important;
    }}
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown li,
    section[data-testid="stSidebar"] .stMarkdown span,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stCaption,
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {{
        color: rgba(255,255,255,0.92) !important;
    }}
    section[data-testid="stSidebar"] .stRadio label,
    section[data-testid="stSidebar"] .stRadio label span,
    section[data-testid="stSidebar"] .stRadio label p,
    section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {{
        color: #ffffff !important;
        font-size: 1.08rem !important;
        font-weight: 600 !important;
    }}
    section[data-testid="stSidebar"] .sidebar-nav-title {{
        color: #ffffff !important;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        margin: 0.5rem 0 0.35rem 0;
    }}
    section[data-testid="stSidebar"] .sidebar-desc {{
        color: rgba(255,255,255,0.78) !important;
        font-size: 0.82rem;
        line-height: 1.5;
        margin-bottom: 0.75rem;
    }}
    section[data-testid="stSidebar"] hr {{
        border-color: rgba(255,255,255,0.12) !important;
    }}

    div[data-testid="stVerticalBlockBorderWrapper"] > div[data-testid="stVerticalBlockBorderDiv"] {{
        background-color: {PALETTE['white']};
        padding: 1.25rem 1.35rem;
        border-radius: 14px;
        border: 1px solid {PALETTE['gray']};
        box-shadow: 0 8px 24px rgba(20, 33, 61, 0.06);
    }}

    .plot-desc {{
        font-size: 0.88rem;
        line-height: 1.55;
        color: var(--text-dim);
        margin: 0 0 0.75rem 0;
    }}
    .prose-block {{
        font-size: 0.92rem;
        line-height: 1.65;
        color: var(--text-main);
        margin: 0 0 1rem 0;
    }}
    .prose-block strong {{ color: var(--navy); }}
    .repo-tree {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        line-height: 1.55;
        background: #f0f2f6;
        border: 1px solid var(--gray);
        border-radius: 10px;
        padding: 1rem 1.15rem;
        white-space: pre;
        overflow-x: auto;
        color: var(--navy);
    }}
    div[data-testid="column"] > div[data-testid="stVerticalBlockBorderWrapper"] {{
        height: 100%;
    }}
    div[data-testid="column"] > div[data-testid="stVerticalBlockBorderWrapper"] > div[data-testid="stVerticalBlockBorderDiv"] {{
        min-height: 420px;
        display: flex;
        flex-direction: column;
    }}
    .sidebar-tagline {{
        color: rgba(255,255,255,0.72) !important;
        font-size: 0.8rem;
        line-height: 1.5;
        margin: 0.25rem 0 1rem 0;
    }}
    .section-heading {{
        font-size: 1.05rem;
        font-weight: 700;
        color: var(--navy);
        margin: 0 0 0.35rem 0;
    }}

    .main-title {{
        font-size: clamp(2rem, 4.5vw, 3.5rem);
        font-weight: 800;
        background: linear-gradient(135deg, var(--navy), var(--amber));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
        letter-spacing: -1.5px;
        line-height: 1.05;
    }}
    .hero-subtitle {{
        color: var(--text-dim);
        font-size: 0.85rem;
        letter-spacing: 2.5px;
        text-transform: uppercase;
        margin-bottom: 1.75rem;
        font-weight: 500;
    }}

    .premium-card {{
        background: var(--card-bg);
        padding: 1.35rem 1.5rem;
        border-radius: 14px;
        border: 1px solid var(--card-border);
        box-shadow: 0 8px 24px rgba(20, 33, 61, 0.06);
        transition: border-color 0.2s ease, transform 0.2s ease, box-shadow 0.2s ease;
        margin-bottom: 1rem;
    }}
    .premium-card:hover {{
        border-color: rgba(252, 163, 17, 0.45);
        transform: translateY(-1px);
        box-shadow: 0 12px 28px rgba(20, 33, 61, 0.09);
    }}
    .card-eyebrow {{
        font-size: 0.62rem;
        font-weight: 700;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        color: var(--amber);
        margin-bottom: 0.35rem;
    }}
    .card-heading {{
        font-size: 0.95rem;
        font-weight: 700;
        color: var(--navy);
        margin: 0 0 0.75rem 0;
    }}
    .metric-value {{
        font-family: 'JetBrains Mono', monospace;
        font-size: clamp(1.6rem, 2.5vw, 2.2rem);
        font-weight: 700;
        color: var(--navy);
        line-height: 1.1;
    }}
    .metric-value.accent {{ color: var(--amber); }}
    .metric-label {{
        font-size: 0.62rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--text-dim);
        margin-top: 0.25rem;
    }}

    .kpi-strip {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 1px;
        background: var(--gray);
        border: 1px solid var(--gray);
        border-radius: 14px;
        overflow: hidden;
        margin: 0.5rem 0 1.75rem 0;
    }}
    .kpi-cell {{
        background: var(--white);
        padding: 1.1rem 1rem;
        text-align: center;
    }}

    .bench-text {{
        font-size: 1rem;
        line-height: 1.65;
        color: var(--text-main);
        margin-bottom: 1rem;
    }}
    .highlight-accent {{
        color: var(--amber);
        font-weight: 600;
    }}

    .metric-container {{ margin-bottom: 0.65rem; }}
    .metric-container small {{
        color: var(--text-dim);
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        font-size: 0.62rem;
    }}
    .progress-bg {{
        background: var(--gray);
        border-radius: 6px;
        height: 8px;
        width: 100%;
        overflow: hidden;
        margin-top: 0.35rem;
    }}
    .progress-fill {{ height: 100%; border-radius: 6px; }}

    .sidebar-brand {{
        font-size: 1.15rem;
        font-weight: 800;
        color: var(--white);
        padding: 0.35rem 0 0.15rem 0;
        letter-spacing: -0.02em;
    }}
    .sidebar-brand span {{ color: var(--amber); }}

    .footer {{
        text-align: center;
        padding: 1.75rem 1rem 0.5rem;
        color: var(--text-dim);
        font-size: 0.78rem;
        border-top: 1px solid var(--gray);
        margin-top: 3rem;
    }}

    .result-banner {{
        background: linear-gradient(135deg, var(--white), var(--accent-soft));
        border: 1px solid rgba(252,163,17,0.35);
        border-radius: 14px;
        padding: 1.25rem 1.5rem;
    }}
</style>
"""


def inject():
    st.markdown(CSS, unsafe_allow_html=True)


def plotly_layout(**overrides):
    base = dict(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=PALETTE["navy"], size=11),
        margin=dict(l=12, r=12, t=36, b=12),
        colorway=[PALETTE["navy"], PALETTE["amber"], PALETTE["navy_light"], PALETTE["gray"]],
    )
    base.update(overrides)
    return base
