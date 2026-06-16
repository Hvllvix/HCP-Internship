from .palette import PALETTE

def css(theme):
    p = PALETTE[theme]
    return f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {{
        --bg: {p['bg']};
        --surface: {p['surface']};
        --surface2: {p['surface2']};
        --input-bg: {p.get('input_bg', p['surface2'])};
        --border: {p['border']};
        --text: {p['text']};
        --muted: {p['muted']};
        --accent: {p['accent']};
        --accent2: {p['accent2']};
        --poverty: {p['poverty']};
        --vulnerable: {p['vulnerable']};
        --positive: {p['positive']};
    }}

    #MainMenu, footer, .stDeployButton, [data-testid="stToolbar"], [data-testid="stDecoration"] {{
        display: none !important;
        visibility: hidden !important;
    }}
    header[data-testid="stHeader"] {{
        background: transparent !important;
        pointer-events: none;
    }}
    header[data-testid="stHeader"] * {{
        display: none !important;
    }}

    .stApp {{ background: var(--bg); color: var(--text); font-family: 'DM Sans', sans-serif; }}
    [data-testid="stSidebar"] {{ background: var(--surface); border-right: 1px solid var(--border); }}
    [data-testid="stSidebar"] [data-testid="stSidebarContent"] {{
        display: flex;
        flex-direction: column;
    }}
    .sidebar-brand {{ margin-bottom: 0.25rem; }}
    .sidebar-brand h3 {{ margin: 0; color: var(--text) !important; }}
    .theme-switch-wrap {{
        margin-top: auto;
        padding-top: 1.25rem;
        border-top: 1px solid var(--border);
    }}
    [data-testid="stSidebarNav"] {{ padding-top: 0.5rem; }}

    h1, h2, h3, h4, h5, h6, p, li, span, label {{ color: var(--text); }}
    .muted {{ color: var(--muted) !important; }}
    .accent {{ color: var(--accent) !important; }}

    .hero-title {{
        font-size: clamp(2rem, 4vw, 3.2rem);
        font-weight: 700;
        margin: 0;
        background: linear-gradient(135deg, var(--accent), var(--accent2));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}
    .hero-sub {{
        color: var(--muted);
        font-size: 0.8rem;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        margin: 0.4rem 0 1.5rem 0;
    }}
    .kpi-value {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.75rem;
        font-weight: 700;
        color: var(--accent);
    }}
    .kpi-label {{ color: var(--muted); font-size: 0.82rem; margin-top: 0.2rem; }}
    .body-text {{ color: var(--text); line-height: 1.7; font-size: 0.95rem; margin: 0 0 0.75rem 0; }}
    .plot-desc {{ color: var(--muted); font-size: 0.88rem; line-height: 1.55; margin: 0 0 0.75rem 0; min-height: 2.6rem; }}

    div[data-testid="stVerticalBlockBorderWrapper"] {{
        background: var(--surface) !important;
        border-color: var(--border) !important;
        border-radius: 14px !important;
        padding: 1rem 1.15rem !important;
    }}

    .pipeline-step {{
        background: var(--surface2);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 0.65rem 0.9rem;
        text-align: center;
        font-size: 0.78rem;
        color: var(--muted);
    }}
    .pipeline-arrow {{ color: var(--accent); font-size: 1.1rem; }}
    .tree {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        color: var(--muted);
        background: var(--surface2);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 1rem 1.2rem;
        line-height: 1.65;
        white-space: pre;
        margin: 0;
    }}
    .result-card {{
        background: var(--surface);
        border: 1px solid var(--border);
        border-left: 4px solid var(--accent);
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        margin: 1rem 0;
    }}
    .arch-box {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.82rem;
        color: var(--text);
        background: var(--surface2);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 1rem 1.2rem;
        line-height: 1.7;
        white-space: pre-line;
        margin: 0;
    }}
    .progress-bg {{ background: var(--border); border-radius: 6px; height: 10px; overflow: hidden; }}
    .progress-fill {{ height: 10px; border-radius: 6px; }}
    .section-label {{
        font-size: 0.72rem;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: var(--muted);
        margin-bottom: 0.35rem;
    }}
    .app-footer {{
        text-align: center;
        padding: 2rem 1rem 4rem 1rem;
        color: var(--muted);
        font-size: 0.82rem;
        border-top: 1px solid var(--border);
        margin-top: 3rem;
    }}
    div[data-testid="stMetric"] {{
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 0.6rem 0.8rem;
    }}
    [data-testid="stPlotlyChart"] {{ border-radius: 12px; overflow: hidden; }}

    [data-testid="stSelectbox"] div[data-baseweb="select"] > div,
    [data-testid="stMultiSelect"] div[data-baseweb="select"] > div,
    [data-testid="stNumberInput"] input,
    [data-testid="stTextInput"] input,
    [data-testid="stTextArea"] textarea {{
        background-color: var(--input-bg) !important;
        color: var(--text) !important;
        border-color: var(--border) !important;
    }}
    [data-testid="stDataFrame"],
    [data-testid="stDataFrame"] > div,
    [data-testid="stDataFrame"] [data-testid="glideDataEditor"],
    [data-testid="stDataFrame"] canvas {{
        background-color: var(--input-bg) !important;
    }}
    [data-testid="stDataFrame"] [role="gridcell"] {{
        color: var(--text) !important;
        background-color: var(--input-bg) !important;
    }}
    [data-testid="stForm"] {{
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1rem;
    }}
    </style>
    """

def mpl_style(theme):
    p = PALETTE[theme]
    return {
        'figure.facecolor': p['plot_bg'],
        'axes.facecolor': p['plot_bg'],
        'axes.edgecolor': p['border'],
        'axes.labelcolor': p['text'],
        'text.color': p['text'],
        'xtick.color': p['muted'],
        'ytick.color': p['muted'],
        'grid.color': p['plot_grid'],
        'axes.spines.top': False,
        'axes.spines.right': False,
        'font.family': 'sans-serif',
    }

def plotly_template(theme):
    p = PALETTE[theme]
    return dict(
        paper_bgcolor = 'rgba(0,0,0,0)',
        plot_bgcolor = 'rgba(0,0,0,0)',
        font = dict(color = p['text'], family = 'DM Sans'),
        margin = dict(l = 24, r = 16, t = 36, b = 24),
        height = 340,
        xaxis = dict(gridcolor = p['plot_grid'], zerolinecolor = p['plot_grid']),
        yaxis = dict(gridcolor = p['plot_grid'], zerolinecolor = p['plot_grid']),
    )
