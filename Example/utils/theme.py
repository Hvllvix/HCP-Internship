"""
Swiss-inspired institutional design system.
Clean, light, grid-based. No animations.
"""
import streamlit as st


def inject_theme():
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


GLOBAL_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

    * { box-sizing: border-box; }

    :root {
        --paper: #F5F5F0;
        --surface: #FFFFFF;
        --ink: #1A1A1A;
        --ink-secondary: #4A4A4A;
        --ink-muted: #7A7A7A;
        --rule: #E0E0E0;
        --rule-light: #EBEBEB;

        --blue: #2563EB;
        --blue-light: #EFF6FF;
        --teal: #0D9488;
        --teal-light: #F0FDFA;
        --amber: #D97706;
        --rose: #BE123C;

        --shadow-sm: 0 1px 2px rgba(0,0,0,0.04);
        --shadow: 0 2px 8px rgba(0,0,0,0.06);
        
        --sidebar-width: 280px;
        --control-panel-width: 320px;
    }

    .stApp {
        background: var(--paper);
        color: var(--ink);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        font-size: 14px;
        line-height: 1.6;
    }
    .stApp > header { background: transparent !important; }
    .block-container { padding-top: 2rem !important; max-width: 1440px; }
    
    /* ==========================================================================
       ENHANCED SIDEBAR NAVIGATION
       ========================================================================== */
    .enhanced-sidebar {
        display: flex;
        flex-direction: column;
        gap: 24px;
        padding: 8px 0;
    }
    
    .sidebar-brand {
        font-size: 1.125rem;
        font-weight: 700;
        color: var(--ink);
        padding: 0 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .brand-icon {
        color: var(--blue);
        font-size: 1.5rem;
    }
    
    .sidebar-section {
        display: flex;
        flex-direction: column;
        gap: 12px;
    }
    
    .sidebar-section-title {
        font-size: 0.7rem;
        font-weight: 600;
        color: var(--ink-muted);
        letter-spacing: 0.1em;
        text-transform: uppercase;
        padding: 0 12px;
    }
    
    .sidebar-nav-link {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 10px 12px;
        border-radius: 6px;
        color: var(--ink-secondary);
        text-decoration: none;
        font-size: 0.875rem;
        font-weight: 500;
        transition: all 0.15s ease;
        cursor: pointer;
    }
    
    .sidebar-nav-link:hover {
        background: var(--paper);
        color: var(--ink);
    }
    
    .sidebar-nav-link.active {
        background: var(--blue-light);
        color: var(--blue);
        font-weight: 600;
    }
    
    .nav-icon {
        font-size: 1rem;
        width: 20px;
        text-align: center;
    }
    
    .dataset-status {
        display: flex;
        flex-direction: column;
        gap: 8px;
        padding: 0 12px;
    }
    
    .status-indicator {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 0.8125rem;
        color: var(--ink-secondary);
    }
    
    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: var(--ink-muted);
        opacity: 0.4;
    }
    
    .status-dot.active {
        background: var(--teal);
        opacity: 1;
        box-shadow: 0 0 0 2px var(--teal-light);
    }
    
    .log-container {
        display: flex;
        flex-direction: column;
        gap: 6px;
        padding: 0 12px;
        max-height: 200px;
        overflow-y: auto;
    }
    
    .log-entry {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 0.75rem;
        padding: 6px 8px;
        border-radius: 4px;
        background: var(--paper);
    }
    
    .log-entry.success {
        background: var(--teal-light);
        color: var(--teal);
    }
    
    .log-entry.info {
        background: var(--blue-light);
        color: var(--blue);
    }
    
    .log-time {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.6875rem;
        font-weight: 500;
        opacity: 0.7;
    }
    
    .log-msg {
        flex: 1;
    }

    /* Typography */
    h1, h2, h3, h4, h5 {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        color: var(--ink);
        letter-spacing: -0.02em;
        line-height: 1.2;
    }
    h1 { font-size: 2.5rem; margin: 0 0 1rem 0; }
    h2 { font-size: 1.75rem; margin: 2rem 0 0.75rem 0; padding-bottom: 0.5rem; border-bottom: 2px solid var(--ink); }
    h3 { font-size: 1.25rem; margin: 1.5rem 0 0.5rem 0; }
    h4 { font-size: 1.05rem; margin: 1rem 0 0.4rem 0; }

    p, li {
        color: var(--ink-secondary);
        margin: 0 0 0.75rem 0;
    }

    /* Ghost container */
    div[data-testid="stVerticalBlock"] > div:empty,
    .element-container:empty,
    div[data-testid="column"]:empty {
        display: none !important;
    }

    /* Header */
    .page-header {
        background: var(--surface);
        border-bottom: 1px solid var(--rule);
        padding: 1.5rem 0;
        margin-bottom: 2rem;
    }
    .page-header h1 {
        margin: 0;
        font-size: 2.25rem;
        font-weight: 700;
    }
    .page-header .subtitle {
        color: var(--ink-muted);
        font-size: 0.9rem;
        margin-top: 0.25rem;
    }

    /* KPI strip */
    .kpi-strip {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1px;
        background: var(--rule);
        border: 1px solid var(--rule);
        border-radius: 6px;
        overflow: hidden;
        margin: 1.5rem 0;
    }
    .kpi-cell {
        background: var(--surface);
        padding: 1.25rem;
        text-align: center;
    }
    .kpi-label {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: var(--ink-muted);
        font-weight: 600;
        margin-bottom: 0.4rem;
    }
    .kpi-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--blue);
        line-height: 1;
    }
    .kpi-note {
        font-size: 0.75rem;
        color: var(--ink-muted);
        margin-top: 0.3rem;
    }

    /* Section labels */
    .section-label {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        color: var(--teal);
        font-weight: 600;
        margin: 2rem 0 0.5rem 0;
    }
    .section-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: var(--ink);
        margin: 0 0 1rem 0;
    }
    .section-desc {
        font-size: 0.95rem;
        color: var(--ink-secondary);
        line-height: 1.7;
        margin-bottom: 1.25rem;
        max-width: 900px;
    }

    /* Cards */
    .card {
        background: var(--surface);
        border: 1px solid var(--rule-light);
        border-radius: 4px;
        padding: 1.25rem;
        box-shadow: var(--shadow-sm);
    }
    .card-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: var(--ink);
        margin-bottom: 0.75rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid var(--rule-light);
    }
    .card-body { margin-top: 0.5rem; }

    /* Plot card */
    .plot-card {
        background: var(--surface);
        border: 1px solid var(--rule-light);
        border-radius: 4px;
        padding: 1rem;
        box-shadow: var(--shadow-sm);
    }
    .plot-card-title {
        font-size: 0.9rem;
        font-weight: 600;
        color: var(--ink);
        margin-bottom: 0.25rem;
    }
    .plot-card-desc {
        font-size: 0.8rem;
        color: var(--ink-muted);
        line-height: 1.5;
        margin-bottom: 0.75rem;
    }

    /* Metric strip */
    .metric-strip {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 1px;
        background: var(--rule);
        border: 1px solid var(--rule);
        border-radius: 4px;
        overflow: hidden;
        margin: 1rem 0;
    }
    .metric-cell {
        background: var(--surface);
        padding: 1rem;
        text-align: center;
    }
    .metric-label {
        font-size: 0.65rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--ink-muted);
        font-weight: 600;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--blue);
        margin-top: 0.2rem;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: var(--surface);
        border-bottom: 2px solid var(--rule);
        padding: 0;
        gap: 0;
        margin-bottom: 1.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 0;
        font-weight: 600;
        font-size: 0.85rem;
        color: var(--ink-muted);
        padding: 0.75rem 1.5rem;
        border-bottom: 3px solid transparent;
        margin-bottom: -2px;
        transition: none;
    }
    .stTabs [aria-selected="true"] {
        background: transparent !important;
        color: var(--blue) !important;
        border-bottom-color: var(--blue) !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: var(--paper);
        color: var(--ink);
    }

    /* Buttons */
    .stButton button {
        border-radius: 4px !important;
        font-weight: 600 !important;
        font-family: 'Inter', sans-serif !important;
        letter-spacing: 0.01em !important;
        border: 1px solid var(--rule) !important;
        transition: none !important;
    }
    .stButton button[kind="primary"] {
        background: var(--blue) !important;
        border-color: var(--blue) !important;
        color: white !important;
    }
    .stButton button:not([kind="primary"]):hover {
        background: var(--paper) !important;
    }

    /* Form controls */
    .stSelectbox > div > div,
    .stTextInput > div > div,
    .stNumberInput > div > div {
        border-radius: 4px !important;
        border-color: var(--rule) !important;
        background: var(--surface) !important;
    }
    .stSlider > div > div > div {
        background: var(--rule) !important;
    }

    /* Map container */
    .map-container {
        background: var(--surface);
        border: 1px solid var(--rule-light);
        border-radius: 4px;
        padding: 1rem;
        box-shadow: var(--shadow-sm);
    }
    
    .map-container-enhanced {
        background: var(--surface);
        border: 1px solid var(--rule-light);
        border-radius: 6px;
        padding: 1.25rem;
        box-shadow: var(--shadow-sm);
    }

    /* Data table */
    .dataframe {
        font-size: 0.8rem !important;
    }
    .dataframe th {
        font-weight: 600 !important;
        color: var(--ink) !important;
        background: var(--paper) !important;
        border-bottom: 2px solid var(--rule) !important;
    }
    .dataframe td {
        border-bottom: 1px solid var(--rule-light) !important;
    }

    /* Footer */
    .footer {
        margin-top: 3rem;
        padding-top: 1.5rem;
        border-top: 1px solid var(--rule);
        text-align: center;
        color: var(--ink-muted);
        font-size: 0.8rem;
    }

    /* Equal height columns */
    div[data-testid="column"] > div { height: 100%; }
    .stPlotlyChart { height: 100%; }

    /* Mono for code */
    code, pre {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.85em;
    }
    
    /* ==========================================================================
       ENHANCED COMPONENTS
       ========================================================================== */
    .enhanced-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        margin-bottom: 2rem;
        padding-bottom: 1.5rem;
        border-bottom: 2px solid var(--rule);
    }
    
    .header-left {
        flex: 1;
    }
    
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: var(--ink);
        margin: 0 0 0.5rem 0;
        letter-spacing: -0.03em;
    }
    
    .subtitle {
        font-size: 0.95rem;
        color: var(--ink-muted);
        margin: 0;
        line-height: 1.5;
    }
    
    .header-right {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    .workflow-indicator {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 8px 16px;
        background: var(--surface);
        border: 1px solid var(--rule-light);
        border-radius: 6px;
        box-shadow: var(--shadow-sm);
    }
    
    .workflow-label {
        font-size: 0.75rem;
        font-weight: 600;
        color: var(--ink-muted);
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .workflow-value {
        font-size: 0.875rem;
        font-weight: 600;
        color: var(--blue);
    }
    
    .kpi-strip-enhanced {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 1px;
        background: var(--rule);
        border: 1px solid var(--rule);
        border-radius: 8px;
        overflow: hidden;
        margin: 1.5rem 0 2.5rem 0;
    }
    
    .kpi-cell-enhanced {
        background: var(--surface);
        padding: 1.5rem;
        text-align: center;
        transition: background 0.15s ease;
    }
    
    .kpi-cell-enhanced:hover {
        background: var(--paper);
    }
    
    .section-header {
        margin: 2.5rem 0 1.25rem 0;
    }
    
    .section-desc-enhanced {
        font-size: 0.95rem;
        color: var(--ink-secondary);
        line-height: 1.7;
        margin-bottom: 1.5rem;
        max-width: 900px;
    }
    
    .card-enhanced {
        background: var(--surface);
        border: 1px solid var(--rule-light);
        border-radius: 6px;
        box-shadow: var(--shadow-sm);
        overflow: hidden;
    }
    
    .card-enhanced.highlight {
        border-color: var(--amber);
        background: #FFFBEB;
    }
    
    .card-header {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 1rem 1.25rem;
        border-bottom: 1px solid var(--rule-light);
        background: var(--paper);
    }
    
    .card-icon {
        font-size: 1.25rem;
    }
    
    .card-title {
        font-size: 0.9rem;
        font-weight: 600;
        color: var(--ink);
    }
    
    .card-body-enhanced {
        padding: 1.25rem;
    }
    
    .info-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.5rem 0;
        border-bottom: 1px solid var(--rule-light);
    }
    
    .info-row:last-child {
        border-bottom: none;
    }
    
    .info-label {
        font-size: 0.8125rem;
        color: var(--ink-muted);
        font-weight: 500;
    }
    
    .info-value {
        font-size: 0.875rem;
        color: var(--ink);
        font-weight: 600;
        text-align: right;
    }
    
    .plot-wrapper {
        background: var(--surface);
        border: 1px solid var(--rule-light);
        border-radius: 6px;
        box-shadow: var(--shadow-sm);
        overflow: hidden;
        margin-bottom: 1.5rem;
    }
    
    .plot-header {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 0.875rem 1.25rem;
        border-bottom: 1px solid var(--rule-light);
        background: var(--paper);
    }
    
    .plot-badge {
        font-size: 0.65rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: white;
        background: var(--blue);
        padding: 3px 8px;
        border-radius: 4px;
    }
    
    .plot-title {
        font-size: 0.9rem;
        font-weight: 600;
        color: var(--ink);
        margin: 0;
    }
    
    .plot-content {
        padding: 1rem;
    }
    
    .preview-card {
        background: var(--surface);
        border: 1px solid var(--rule-light);
        border-radius: 6px;
        box-shadow: var(--shadow-sm);
        overflow: hidden;
    }
    
    .preview-header {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 0.875rem 1.25rem;
        border-bottom: 1px solid var(--rule-light);
        background: var(--paper);
    }
    
    .preview-badge {
        font-size: 0.65rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: white;
        background: var(--teal);
        padding: 3px 8px;
        border-radius: 4px;
    }
    
    .preview-header h4 {
        font-size: 0.9rem;
        font-weight: 600;
        color: var(--ink);
        margin: 0;
    }
    
    .preview-body {
        padding: 1rem;
    }
    
    /* ==========================================================================
       RIGHT CONTROL PANEL
       ========================================================================== */
    .control-panel-enhanced {
        background: var(--surface);
        border: 1px solid var(--rule-light);
        border-radius: 6px;
        box-shadow: var(--shadow-sm);
        overflow: hidden;
        margin-bottom: 1.5rem;
    }
    
    .panel-header {
        padding: 1rem 1.25rem;
        border-bottom: 1px solid var(--rule-light);
        background: var(--paper);
    }
    
    .panel-header h3 {
        font-size: 0.875rem;
        font-weight: 600;
        color: var(--ink);
        margin: 0;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .region-info-card {
        padding: 1.25rem;
        border-bottom: 1px solid var(--rule-light);
        background: linear-gradient(135deg, var(--blue-light) 0%, var(--surface) 100%);
    }
    
    .region-name {
        font-size: 1.125rem;
        font-weight: 700;
        color: var(--ink);
        margin-bottom: 0.25rem;
    }
    
    .region-code {
        font-size: 0.75rem;
        color: var(--ink-muted);
        font-family: 'IBM Plex Mono', monospace;
    }
    
    .panel-section {
        padding: 1.25rem;
        border-bottom: 1px solid var(--rule-light);
    }
    
    .panel-section:last-child {
        border-bottom: none;
    }
    
    .panel-section h4 {
        font-size: 0.75rem;
        font-weight: 600;
        color: var(--ink-muted);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin: 0 0 1rem 0;
    }
    
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 10px;
    }
    
    .metric-box {
        background: var(--paper);
        border: 1px solid var(--rule-light);
        border-radius: 6px;
        padding: 0.875rem;
        text-align: center;
    }
    
    .metric-box .metric-value {
        font-size: 1.25rem;
        font-weight: 700;
        color: var(--blue);
        line-height: 1;
        margin-bottom: 0.25rem;
    }
    
    .metric-box .metric-label {
        font-size: 0.6875rem;
        color: var(--ink-muted);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
    }
    
    .pipeline-status {
        display: flex;
        flex-direction: column;
        gap: 8px;
    }
    
    .pipeline-step {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 8px 10px;
        border-radius: 4px;
        font-size: 0.8125rem;
        color: var(--ink-muted);
        background: var(--paper);
    }
    
    .pipeline-step.complete {
        color: var(--teal);
        background: var(--teal-light);
    }
    
    .pipeline-step.active {
        color: var(--blue);
        background: var(--blue-light);
        font-weight: 600;
    }
    
    .step-indicator {
        font-size: 0.875rem;
        font-weight: 700;
    }
    
    .lineage-tree {
        display: flex;
        flex-direction: column;
        gap: 4px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.75rem;
        color: var(--ink-secondary);
        line-height: 1.6;
    }
    
    .tree-item {
        padding: 2px 0 2px 12px;
        border-left: 1px solid var(--rule);
    }
    
    /* ==========================================================================
       FOOTER
       ========================================================================== */
    .footer-enhanced {
        margin-top: 3rem;
        padding-top: 1.5rem;
        border-top: 1px solid var(--rule);
        text-align: center;
    }
    
    .footer-content {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 12px;
        flex-wrap: wrap;
        font-size: 0.8125rem;
        color: var(--ink-muted);
    }
    
    .footer-content span:not(:last-child)::after {
        content: "|";
        margin-left: 12px;
        opacity: 0.5;
    }
</style>
"""
