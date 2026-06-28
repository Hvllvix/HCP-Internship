"""Vercel-inspired light theme: zinc borders, stretch grids, depth shadows."""
import streamlit as st

PALETTE = {
    "black": "#09090b",
    "navy": "#18181b",
    "amber": "#ea580c",
    "amber_soft": "#fff7ed",
    "zinc50": "#fafafa",
    "zinc100": "#f4f4f5",
    "zinc200": "#e4e4e7",
    "zinc400": "#a1a1aa",
    "zinc500": "#71717a",
    "zinc700": "#3f3f46",
    "white": "#ffffff",
    "danger": "#dc2626",
    "success": "#16a34a",
    "bg": "#fafafa",
    "gray": "#e4e4e7",
    "navy_light": "#27272a",
    "navy_muted": "#71717a",
}

SHADOW = "0 4px 6px -1px rgba(0, 0, 0, 0.1)"

CSS = f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    :root {{
        --black: {PALETTE['black']};
        --navy: {PALETTE['navy']};
        --amber: {PALETTE['amber']};
        --zinc50: {PALETTE['zinc50']};
        --zinc100: {PALETTE['zinc100']};
        --zinc200: {PALETTE['zinc200']};
        --zinc400: {PALETTE['zinc400']};
        --zinc500: {PALETTE['zinc500']};
        --zinc700: {PALETTE['zinc700']};
        --white: {PALETTE['white']};
        --bg: {PALETTE['bg']};
        --text-main: {PALETTE['navy']};
        --text-dim: {PALETTE['zinc500']};
        --card-bg: {PALETTE['white']};
        --card-border: {PALETTE['zinc200']};
        --accent-soft: {PALETTE['amber_soft']};
        --shadow: {SHADOW};
    }}

    .stApp {{
        background: var(--bg);
        color: var(--text-main);
        font-family: 'Inter', sans-serif;
    }}
    .block-container {{
        padding-top: 1.5rem !important;
        max-width: 1280px;
    }}

    div[data-testid="stVerticalBlockBorderWrapper"] > div[data-testid="stVerticalBlockBorderDiv"] {{
        background: var(--white);
        padding: 1.25rem 1.35rem;
        border-radius: 12px;
        border: 1px solid var(--zinc200);
        box-shadow: var(--shadow);
        height: 100%;
    }}

    .stretch-row {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(0, 1fr));
        gap: 0.75rem;
        align-items: stretch;
        width: 100%;
        margin-bottom: 0.75rem;
    }}
    .stretch-row + div[data-testid="stHorizontalBlock"] {{
        align-items: stretch !important;
    }}
    div[data-testid="stHorizontalBlock"] {{
        align-items: stretch !important;
    }}
    div[data-testid="column"] {{
        display: flex !important;
        flex-direction: column !important;
        align-self: stretch !important;
    }}
    div[data-testid="column"] > div {{
        flex: 1 1 auto !important;
        display: flex !important;
        flex-direction: column !important;
        height: 100% !important;
    }}
    div[data-testid="column"] .stretch-card {{
        flex: 1 1 auto;
        display: flex;
        flex-direction: column;
        height: 100%;
    }}
    div[data-testid="column"] div[data-testid="stVerticalBlockBorderWrapper"] {{
        flex: 1 1 auto;
        height: 100%;
        display: flex;
        flex-direction: column;
    }}

    section[data-testid="stSidebar"] {{
        background: var(--white) !important;
        border-right: 1px solid var(--zinc200) !important;
        min-width: 17rem !important;
        width: 17rem !important;
        box-shadow: var(--shadow);
    }}
    section[data-testid="stSidebar"] > div {{
        width: 17rem !important;
        min-width: 17rem !important;
    }}
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] label {{
        color: var(--zinc700) !important;
    }}
    section[data-testid="stSidebar"] .stRadio label,
    section[data-testid="stSidebar"] .stRadio label span,
    section[data-testid="stSidebar"] .stRadio label p {{
        color: var(--navy) !important;
        font-size: 0.92rem !important;
        font-weight: 600 !important;
    }}
    section[data-testid="stSidebar"] .sidebar-nav-title {{
        color: var(--zinc500) !important;
        font-size: 0.68rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin: 0.75rem 0 0.35rem 0;
    }}
    section[data-testid="stSidebar"] .sidebar-desc {{
        color: var(--zinc500) !important;
        font-size: 0.8rem;
        line-height: 1.55;
        margin-bottom: 0.65rem;
    }}
    section[data-testid="stSidebar"] hr {{
        border-color: var(--zinc200) !important;
        margin: 0.85rem 0 !important;
    }}

    .main-title {{
        font-size: clamp(1.75rem, 3.5vw, 2.75rem);
        font-weight: 700;
        color: var(--navy);
        margin-bottom: 0.15rem;
        letter-spacing: -0.03em;
        line-height: 1.1;
    }}
    .hero-subtitle {{
        color: var(--zinc500);
        font-size: 0.8rem;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        margin-bottom: 1.5rem;
        font-weight: 500;
    }}
    .section-heading {{
        font-size: 0.95rem;
        font-weight: 600;
        color: var(--navy);
        margin: 0 0 0.35rem 0;
    }}
    .plot-desc {{
        font-size: 0.84rem;
        line-height: 1.55;
        color: var(--zinc500);
        margin: 0 0 0.75rem 0;
    }}
    .prose-block {{
        font-size: 0.88rem;
        line-height: 1.65;
        color: var(--zinc700);
        margin: 0 0 0.85rem 0;
    }}
    .card-eyebrow {{
        font-size: 0.62rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: var(--zinc500);
        margin: 1.25rem 0 0.5rem 0;
    }}

    .metric-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
        gap: 0.75rem;
        margin: 0.75rem 0 1.25rem 0;
        align-items: stretch;
    }}
    .metric-card {{
        background: var(--white);
        border: 1px solid var(--zinc200);
        border-radius: 10px;
        padding: 1rem 1.1rem;
        box-shadow: var(--shadow);
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }}
    .metric-card .metric-label {{
        font-size: 0.65rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--zinc500);
        margin-bottom: 0.35rem;
    }}
    .metric-card .metric-value {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.65rem;
        font-weight: 600;
        color: var(--navy);
        line-height: 1.1;
    }}
    .metric-card .metric-value.accent {{ color: var(--amber); }}
    .metric-card .metric-hint {{
        font-size: 0.72rem;
        color: var(--zinc400);
        margin-top: 0.35rem;
    }}

    .kpi-strip {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
        gap: 0.75rem;
        margin: 0.5rem 0 1.5rem 0;
        align-items: stretch;
    }}
    .kpi-cell {{
        background: var(--white);
        border: 1px solid var(--zinc200);
        border-radius: 10px;
        padding: 0.95rem 0.85rem;
        text-align: center;
        box-shadow: var(--shadow);
        height: 100%;
    }}
    .metric-value {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.5rem;
        font-weight: 600;
        color: var(--navy);
        line-height: 1.1;
    }}
    .metric-value.accent {{ color: var(--amber); }}
    .metric-label {{
        font-size: 0.62rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--zinc500);
        margin-top: 0.25rem;
    }}

    .tree-section div[data-testid="stCode"] {{
        border: 1px solid var(--zinc200);
        border-radius: 10px;
        box-shadow: var(--shadow);
    }}
    .tree-section div[data-testid="stCode"] pre,
    .tree-section div[data-testid="stCode"] code {{
        white-space: pre !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.72rem !important;
        line-height: 1.6 !important;
        background: var(--zinc50) !important;
        color: var(--zinc700) !important;
        tab-size: 4;
        overflow-x: auto !important;
    }}

    .dep-card {{
        background: var(--white);
        border: 1px solid var(--zinc200);
        border-radius: 10px;
        padding: 0.85rem 0.95rem;
        box-shadow: var(--shadow);
        height: 100%;
        min-height: 88px;
    }}
    .dep-icon {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.58rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        color: var(--amber);
        margin-bottom: 0.35rem;
    }}
    .dep-name {{
        font-size: 0.82rem;
        font-weight: 600;
        color: var(--navy);
        margin-bottom: 0.25rem;
        word-break: break-word;
    }}
    .dep-meta {{
        font-size: 0.72rem;
        color: var(--zinc500);
        line-height: 1.45;
    }}

    .sidebar-brand {{
        font-size: 1rem;
        font-weight: 700;
        color: var(--navy);
        padding: 0.25rem 0 0.1rem 0;
        letter-spacing: -0.02em;
    }}
    .sidebar-brand span {{ color: var(--amber); }}
    .sidebar-tagline {{
        color: var(--zinc500) !important;
        font-size: 0.78rem;
        line-height: 1.5;
        margin: 0.2rem 0 0.85rem 0;
    }}

    .disclaimer-banner {{
        background: var(--accent-soft);
        border: 1px solid rgba(234, 88, 12, 0.25);
        border-radius: 10px;
        padding: 0.85rem 1rem;
        font-size: 0.84rem;
        color: var(--zinc700);
        margin: 0.75rem 0;
        box-shadow: var(--shadow);
    }}

    .footer {{
        text-align: center;
        padding: 1.5rem 1rem 0.5rem;
        color: var(--zinc500);
        font-size: 0.75rem;
        border-top: 1px solid var(--zinc200);
        margin-top: 2.5rem;
    }}
</style>
"""


def inject():
    st.markdown(CSS, unsafe_allow_html=True)


def metric_row(items):
    cells = []
    for item in items:
        label, value = item[0], item[1]
        hint = item[2] if len(item) > 2 else ""
        accent = " accent" if len(item) > 3 and item[3] else ""
        hint_html = f'<div class="metric-hint">{hint}</div>' if hint else ""
        cells.append(
            f'<div class="metric-card">'
            f'<div class="metric-label">{label}</div>'
            f'<div class="metric-value{accent}">{value}</div>'
            f'{hint_html}</div>'
        )
    return f'<div class="metric-grid">{"".join(cells)}</div>'


def plotly_layout(**overrides):
    base = dict(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=PALETTE["navy"], size=11),
        margin=dict(l=12, r=12, t=36, b=12),
        colorway=[PALETTE["navy"], PALETTE["amber"], PALETTE["navy_light"], PALETTE["zinc400"]],
    )
    base.update(overrides)
    return base
