THEMES = {
    'dark': {
        'bg': '#1e1e1e',
        'surface': '#252526',
        'border': '#3c3c3c',
        'text': '#cccccc',
        'muted': '#858585',
        'accent': '#62C1FE',
        'poverty': '#F4583E',
        'vulnerable': '#8957E5',
        'plot_frame': 'transparent',
    },
    'light': {
        'bg': '#ffffff',
        'surface': '#f3f3f3',
        'border': '#e5e5e5',
        'text': '#333333',
        'muted': '#6e6e6e',
        'accent': '#006ab1',
        'poverty': '#c42b1c',
        'vulnerable': '#6b3fa0',
        'plot_frame': '#22272E',
    },
}

def css(theme):
    t = THEMES[theme]
    return f"""
    <style>
    .stApp {{ background-color: {t['bg']}; color: {t['text']}; }}
    [data-testid="stSidebar"] {{ background-color: {t['surface']}; border-right: 1px solid {t['border']}; }}
    [data-testid="stSidebar"] * {{ color: {t['text']} !important; }}
    .kpi-card {{
        background: {t['surface']};
        border: 1px solid {t['border']};
        border-radius: 8px;
        padding: 1rem 1.25rem;
        text-align: center;
    }}
    .kpi-value {{ font-size: 1.75rem; font-weight: 700; color: {t['accent']}; }}
    .kpi-label {{ font-size: 0.85rem; color: {t['muted']}; margin-top: 0.25rem; }}
    .pipeline {{
        background: {t['surface']};
        border: 1px solid {t['border']};
        border-radius: 8px;
        padding: 0.75rem 1rem;
        font-family: monospace;
        color: {t['muted']};
    }}
    .dataset-card {{
        background: {t['surface']};
        border: 1px solid {t['border']};
        border-radius: 8px;
        padding: 1rem;
        height: 100%;
    }}
    .dataset-card h4 {{ margin: 0 0 0.5rem 0; color: {t['accent']}; }}
    .dataset-card p {{ margin: 0; color: {t['muted']}; font-size: 0.9rem; }}
    .app-footer {{
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid {t['border']};
        color: {t['muted']};
        font-size: 0.85rem;
        text-align: center;
    }}
    .header-strip {{
        color: {t['muted']};
        font-size: 0.9rem;
        margin-bottom: 1.5rem;
    }}
    .plot-frame {{
        background: {t['plot_frame']};
        border-radius: 8px;
        padding: 0.5rem;
        display: inline-block;
        width: 100%;
    }}
    .arch-box {{
        background: {t['surface']};
        border: 1px solid {t['border']};
        border-radius: 8px;
        padding: 1rem 1.25rem;
        font-family: monospace;
        white-space: pre-line;
        line-height: 1.6;
        color: {t['text']};
    }}
    div[data-testid="stMetric"] {{
        background: {t['surface']};
        border: 1px solid {t['border']};
        border-radius: 8px;
        padding: 0.75rem;
    }}
    </style>
    """
