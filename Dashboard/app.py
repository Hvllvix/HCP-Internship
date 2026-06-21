"""
HCP Poverty Dashboard - Morocco
Comprehensive multi-page interactive dashboard for socioeconomic analysis and poverty prediction.
Built with Streamlit, Plotly, and modular Python utilities.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from utils.data_loader import (
    load_clean_encdm, load_clean_rgph, load_raw_encdm,
    load_models, load_scalers_encdm, load_imputers,
    load_morocco_geojson,
    get_encdm_feature_stats, get_rgph_feature_stats,
    ENCDM_LABELS, RGPH_LABELS, get_column_label, get_column_labels,
    resolve_colname
)
from utils.plots import get_all_plots
from utils.mapping import get_all_maps, compute_region_data, ENCDM_CODE_TO_GEOJSON as MAP_CODES
from utils.network import get_all_networks
from utils.sandbox import run_prediction, build_shap_waterfall, render_sandbox_form

st.set_page_config(page_title="HCP Poverty Dashboard - Morocco", page_icon="", layout="wide", initial_sidebar_state="collapsed")

# ============================================================================
# OVERHAULED CSS - High-contrast, modern SaaS aesthetic
# ============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:wght@400;700&display=swap');
    
    * { box-sizing: border-box; }
    
    :root {
        --bg: #F8F9FA;
        --white: #FFFFFF;
        --text: #212529;
        --text-light: #6C757D;
        --text-muted: #ADB5BD;
        --primary: #1A5F7A;
        --primary-light: #2C7A7B;
        --primary-dark: #134B63;
        --secondary: #2C7A7B;
        --accent1: #805A3B;
        --accent2: #9B4D4D;
        --border: #DEE2E6;
        --border-light: #E9ECEF;
        --shadow: 0 4px 6px rgba(0,0,0,0.07);
        --shadow-hover: 0 8px 15px rgba(0,0,0,0.1);
        --radius: 10px;
    }
    
    .stApp { background-color: var(--bg); color: var(--text); }
    .stApp > header { background-color: transparent !important; }
    .stApp > .main { background-color: var(--bg); }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Playfair Display', serif;
        color: var(--text);
        margin-top: 0;
    }
    
    p, li, div, span, label, .stMarkdown {
        font-family: 'Inter', sans-serif;
        color: var(--text);
    }
    
    /* Main Title */
    .main-title {
        font-family: 'Playfair Display', serif;
        font-size: 2.8rem;
        font-weight: 700;
        color: var(--primary);
        margin-bottom: 0.2rem;
        letter-spacing: -0.5px;
        line-height: 1.15;
    }
    
    .main-subtitle {
        font-family: 'Inter', sans-serif;
        color: var(--text-light);
        font-size: 0.85rem;
        font-weight: 400;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        margin-bottom: 1rem;
        border-bottom: 1px solid var(--border);
        padding-bottom: 0.6rem;
    }
    
    .section-title {
        font-family: 'Playfair Display', serif;
        font-size: 1.4rem;
        font-weight: 700;
        color: var(--primary);
        margin-top: 1.2rem;
        margin-bottom: 0.6rem;
        padding-bottom: 0.3rem;
        border-bottom: 2px solid var(--primary-light);
    }
    
    .subsection-title {
        font-family: 'Playfair Display', serif;
        font-size: 1.15rem;
        font-weight: 700;
        color: var(--secondary);
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    
    /* Cards with depth and equal-height support */
    .card {
        background: var(--white);
        padding: 1.5rem;
        border-radius: var(--radius);
        border: 1px solid var(--border);
        box-shadow: var(--shadow);
        margin-bottom: 1.2rem;
        transition: all 0.3s ease-in-out;
        height: auto;
        min-height: 100%;
        display: flex;
        flex-direction: column;
    }
    .card:hover {
        box-shadow: var(--shadow-hover);
        border-color: var(--primary-light);
        transform: translateY(-1px);
    }
    
    .card h3 {
        color: var(--primary);
        margin-top: 0;
        margin-bottom: 0.6rem;
        border-bottom: 1px solid var(--border-light);
        padding-bottom: 0.4rem;
        font-size: 1.1rem;
    }
    
    .card > *:last-child {
        margin-top: auto;
    }
    
    /* Card grid containers - force equal height via flex */
    .card-grid {
        display: flex;
        flex-wrap: wrap;
        gap: 1.2rem;
    }
    .card-grid > * {
        flex: 1 1 0;
        min-width: 250px;
        display: flex;
        flex-direction: column;
    }
    .card-grid > * > .card {
        flex: 1;
    }
    
    /* Metric boxes */
    .metric-box {
        background: var(--white);
        border-radius: var(--radius);
        padding: 1.1rem;
        border-left: 4px solid var(--primary);
        box-shadow: var(--shadow);
        text-align: center;
        margin-bottom: 0.8rem;
    }
    .metric-box .mlabel {
        font-size: 0.7rem;
        color: var(--text-light);
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }
    .metric-box .mvalue {
        font-family: 'Playfair Display', serif;
        font-size: 1.6rem;
        font-weight: 700;
        color: var(--primary);
        margin: 0.2rem 0;
    }
    .metric-box .mdelta {
        font-size: 0.75rem;
        color: var(--text-muted);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: var(--white);
        border-radius: var(--radius);
        padding: 0.35rem;
        border: 1px solid var(--border);
        gap: 0.2rem;
        margin-bottom: 1.5rem;
        box-shadow: var(--shadow);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.45rem 1rem;
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        font-size: 0.8rem;
        color: var(--text-light);
        background: transparent;
        border: none;
        transition: all 0.3s ease;
    }
    .stTabs [aria-selected="true"] {
        background: var(--primary) !important;
        color: white !important;
        box-shadow: 0 2px 8px rgba(26, 95, 122, 0.3);
    }
    
    /* Buttons */
    .stButton button {
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-family: 'Inter', sans-serif !important;
        transition: all 0.3s ease !important;
        border: 1px solid var(--primary) !important;
        color: var(--primary) !important;
        background: white !important;
        box-shadow: var(--shadow);
    }
    .stButton button:hover {
        background: var(--primary) !important;
        color: white !important;
        box-shadow: var(--shadow-hover);
    }
    .stButton button[kind="primary"] {
        background: var(--primary) !important;
        color: white !important;
        border: none !important;
    }
    .stButton button[kind="primary"]:hover {
        background: var(--primary-dark) !important;
    }
    
    /* Dataframes */
    .stDataFrame {
        border-radius: var(--radius);
        border: 1px solid var(--border);
        box-shadow: var(--shadow);
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background: var(--white) !important;
        border-radius: var(--radius) !important;
        border: 1px solid var(--border) !important;
        font-weight: 600 !important;
        font-family: 'Inter', sans-serif !important;
        box-shadow: var(--shadow);
    }
    
    /* Info boxes */
    .info-callout {
        background: #E8F0F5;
        border-left: 4px solid var(--primary);
        padding: 0.9rem 1.1rem;
        border-radius: 0 var(--radius) var(--radius) 0;
        margin: 0.8rem 0;
        font-size: 0.88rem;
        line-height: 1.5;
    }
    .info-callout strong { color: var(--primary); }
    
    /* Analysis text */
    .analysis-text {
        font-size: 0.9rem;
        line-height: 1.65;
        color: var(--text);
        margin-bottom: 0.8rem;
    }
    
    /* Code block */
    .structure-block {
        background: var(--white);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 0.9rem 1.2rem;
        font-size: 0.78rem;
        line-height: 1.45;
        white-space: pre;
        overflow-x: auto;
        font-family: 'Inter', monospace;
    }
    
    .highlight-primary { color: var(--primary); font-weight: 600; }
    .highlight-secondary { color: var(--secondary); font-weight: 600; }
    
    /* Prediction badges */
    .pred-label {
        display: inline-block;
        padding: 0.25rem 0.9rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .pred-high { background: linear-gradient(135deg, #9B4D4D, #B87373); color: white; }
    .pred-mod { background: linear-gradient(135deg, #805A3B, #A8896E); color: white; }
    .pred-low { background: linear-gradient(135deg, #1A5F7A, #3B7A8B); color: white; }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 1.2rem;
        color: var(--text-light);
        font-size: 0.75rem;
        border-top: 1px solid var(--border);
        margin-top: 2.5rem;
    }
    
    /* Region detail panel */
    .region-panel {
        background: var(--white);
        border-radius: var(--radius);
        padding: 1.2rem;
        border: 1px solid var(--primary-light);
        box-shadow: var(--shadow);
        margin-top: 0;
    }
    .region-panel h3 {
        color: var(--primary);
        border-bottom: 2px solid var(--primary-light);
        padding-bottom: 0.3rem;
    }
    
    /* Fix empty div rectangle bug */
    .stPlotlyChart, .stDataFrame, div[data-testid="stVerticalBlock"] > div {
        overflow: visible !important;
    }
    .element-container {
        overflow: visible !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# HEADER
# ============================================================================
st.markdown('<h1 class="main-title">Poverty and Census Intelligence Dashboard</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="main-subtitle">HCP Morocco -- ENCDM x RGPH Interactive Analysis'
    ' -- Socioeconomic Diagnostics, Spatial Analytics and Predictive Modeling</p>',
    unsafe_allow_html=True
)

# ============================================================================
# CACHED RESOURCES
# ============================================================================
@st.cache_resource
def load_all_resources():
    return {
        "encdm": load_clean_encdm(),
        "rgph": load_clean_rgph(),
        "encdm_raw": load_raw_encdm()[0] if load_raw_encdm()[0] is not None else None,
        "encdm_stats": get_encdm_feature_stats(),
        "rgph_stats": get_rgph_feature_stats(),
        "models": load_models(),
        "scalers": load_scalers_encdm(),
        "imputers": load_imputers(),
        "geojson": load_morocco_geojson(),
    }

res = load_all_resources()

# ============================================================================
# TABS
# ============================================================================
t1_name = "Project Synopsis and Data Integrity"
t2_name = "Socioeconomic Diagnostics and Spatial Analytics"
t3_name = "Predictive Modeling and Scenario Simulator"
tab1, tab2, tab3 = st.tabs([t1_name, t2_name, t3_name])

# ============================================================================
# TAB 1
# ============================================================================
with tab1:
    st.markdown('<div class="section-title">Context and Methodology</div>', unsafe_allow_html=True)
    
    col_i1, col_i2 = st.columns([2, 1])
    with col_i1:
        st.markdown("""
        <div class="card">
        <h3>Project Overview</h3>
        <p class="analysis-text">
        This dashboard provides a comprehensive analytical framework for understanding socioeconomic
        stratification and poverty dynamics in Morocco. The analysis leverages two primary data sources
        from the Haut-Commissariat au Plan (HCP): the <strong class="highlight-primary">ENCDM</strong>
        (National Survey on Household Consumption and Expenditure, 2019-2020) and the
        <strong class="highlight-primary">RGPH</strong> (General Census of Population and Housing, 2014).
        </p>
        <p class="analysis-text">
        The ENCDM survey captures detailed household consumption across 15,970 households, with expenditure
        data across nine major categories, sampling weights for population-level inference, and poverty
        classification based on official national poverty lines. The RGPH census provides housing quality
        indicators for 730,099 households across all 12 Moroccan regions, including construction materials,
        utility access, and household amenities.
        </p>
        <p class="analysis-text">
        By combining these datasets, the analysis identifies key determinants of poverty and vulnerability,
        and builds predictive models (LightGBM + PyTorch Hypernetwork) that estimate poverty risk from
        household characteristics alone. The interactive sandbox allows users to input household data and
        receive real-time predictions with feature contribution explanations.
        </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_i2:
        es = res["encdm_stats"]
        rs = res["rgph_stats"]
        if es:
            pr = es.get("poverty_rate", 0)
            pr_color = "#9B4D4D" if pr > 10 else "#1A5F7A"
            st.markdown(f"""
            <div class="card">
            <h3>Dataset Summary</h3>
            <div class="metric-box"><div class="mlabel">ENCDM Households</div><div class="mvalue">{es.get('n_households', 0):,}</div><div class="mdelta">{es.get('n_features', 0)} features</div></div>
            <div class="metric-box"><div class="mlabel">Poverty Rate</div><div class="mvalue" style="color:{pr_color}">{pr:.1f}%</div><div class="mdelta">of surveyed households</div></div>
            <div class="metric-box"><div class="mlabel">RGPH Households</div><div class="mvalue">{rs.get('n_households', 0):,}</div><div class="mdelta">{rs.get('n_features', 0)} features</div></div>
            <div class="metric-box"><div class="mlabel">Regions</div><div class="mvalue">{es.get('n_regions', 0)}</div><div class="mdelta">Moroccan regions</div></div>
            </div>
            """, unsafe_allow_html=True)

    # Architecture
    st.markdown('<div class="section-title">Technical Architecture</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
    <h3>Repository Structure</h3>
    <div class="structure-block">
<strong>Internship-HCP/</strong>
  ├── <strong>Dashboard/</strong>
  │   ├── <strong>app.py</strong>              # Main application
  │   ├── <strong>utils/</strong>
  │   │   ├── data_loader.py  # Data I/O & caching
  │   │   ├── plots.py         # 20 interactive charts
  │   │   ├── mapping.py       # Morocco choropleth + region click
  │   │   ├── network.py       # Feature dependency graphs
  │   │   └── sandbox.py       # Prediction pipeline
  │   └── MoroccoGeoMap.geojson
  ├── <strong>Data/</strong>
  │   ├── Raw/                 # ENCDM.sav, RGPH.sav
  │   └── Processed/           # Cleaned parquet files
  ├── <strong>Models/</strong>
  │   ├── Classifier/          # LGBM + Hypernetwork
  │   ├── Scalers/             # StandardScaler objects
  │   └── Imputers/
  └── <strong>Assets/</strong>                     # Maps, Dependencies, Types
    </div>
    </div>
    """, unsafe_allow_html=True)

    # Data previews
    st.markdown('<div class="section-title">Dataset Previews</div>', unsafe_allow_html=True)
    cp1, cp2 = st.columns(2)
    with cp1:
        st.markdown('<div class="card"><h3>Clean ENCDM Data</h3>', unsafe_allow_html=True)
        encdm = res["encdm"]
        if encdm is not None:
            # Use column labels
            display_df = encdm.head(100).rename(columns=get_column_labels("encdm"))
            st.dataframe(display_df, use_container_width=True, height=220)
            st.caption(f"Showing 100 of {len(encdm):,} rows")
        st.markdown('</div>', unsafe_allow_html=True)
    with cp2:
        st.markdown('<div class="card"><h3>Clean RGPH Data</h3>', unsafe_allow_html=True)
        rgph = res["rgph"]
        if rgph is not None:
            display_df = rgph.head(100).rename(columns=get_column_labels("rgph"))
            st.dataframe(display_df, use_container_width=True, height=220)
            st.caption(f"Showing 100 of {len(rgph):,} rows")
        st.markdown('</div>', unsafe_allow_html=True)

    # Data integrity plots
    st.markdown('<div class="section-title">Data Integrity Analysis</div>', unsafe_allow_html=True)
    all_plots = get_all_plots()
    di_plots = [p for p in all_plots if p[0] == "Data Integrity"]
    
    # Display in 2-column grid with equal heights
    for i in range(0, len(di_plots), 2):
        cols = st.columns(2)
        for j in range(2):
            if i + j < len(di_plots):
                _, pf, desc = di_plots[i + j]
                with cols[j]:
                    fig = pf()
                    if fig is not None:
                        st.markdown('<div class="card">', unsafe_allow_html=True)
                        st.plotly_chart(fig, use_container_width=True)
                        st.caption(desc)
                        st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# TAB 2
# ============================================================================
with tab2:
    st.markdown('<div class="section-title">Exploratory Data Analysis and Spatial Patterns</div>', unsafe_allow_html=True)
    st.markdown('<p class="analysis-text">This section provides a comprehensive exploration of Moroccan socioeconomic conditions through interactive visualizations and spatial mapping. The interactive Morocco map enables region-specific analysis: click on any region to view detailed statistics.</p>', unsafe_allow_html=True)
    
    # --- Interactive Morocco Map + Region Detail Panel ---
    st.markdown('<div class="subsection-title">Interactive Morocco Regional Map</div>', unsafe_allow_html=True)
    
    # Session state for selected region
    if "selected_region" not in st.session_state:
        st.session_state.selected_region = None
    
    geo_maps = get_all_maps()
    map_name, map_func, map_desc = geo_maps[0]  # Morocco choropleth
    
    col_map, col_detail = st.columns([3, 2])
    
    with col_map:
        fig = map_func()
        if fig is not None:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.plotly_chart(fig, use_container_width=True)
            st.caption(map_desc)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Streamlit plotly events - use selectbox fallback for click interaction
            fig = st.plotly_chart(fig, use_container_width=True, key="morocco_map")
    
    with col_detail:
        # Region selector (simulates click interaction)
        region_names = sorted([
            "Tanger-Tetouan-Al Hoceima", "Oriental", "Fes-Meknes",
            "Rabat-Sale-Kenitra", "Beni Mellal-Khenifra", "Casablanca-Settat",
            "Marrakech-Safi", "Draa-Tafilalet", "Souss-Massa",
            "Guelmim-Oued Noun", "Laayoune-Sakia El Hamra", "Dakhla-Oued Ed Dahab"
        ])
        selected_region_name = st.selectbox(
            "Select Region for Detailed Analysis",
            options=["None (show national)"] + region_names,
            index=0,
            key="region_selector"
        )
        
        if selected_region_name and selected_region_name != "None (show national)":
            # Map selected region name to ENCDM code
            region_code = None
            for code, name in ENCDM_LABELS["Region_12"].items():
                if name == selected_region_name:
                    region_code = code
                    break
            
            if region_code:
                region_data = compute_region_data(region_key=region_code)
                if region_data:
                    st.markdown(f"""
                    <div class="region-panel">
                    <h3>{selected_region_name}</h3>
                    <div class="metric-box"><div class="mlabel">Poverty Rate</div>
                    <div class="mvalue" style="color:{'#9B4D4D' if region_data.get('poverty_rate', 0) > 15 else '#1A5F7A'}">{region_data.get('poverty_rate', 0):.1f}%</div></div>
                    <div class="metric-box"><div class="mlabel">Households</div><div class="mvalue">{region_data.get('household_count', 0):,}</div></div>
                    <div class="metric-box"><div class="mlabel">Urban Population</div><div class="mvalue">{region_data.get('urban_pct', 0):.0f}%</div></div>
                    <div class="metric-box"><div class="mlabel">Avg Age of Head</div><div class="mvalue">{region_data.get('avg_age', 0):.0f}</div></div>
                    <div class="metric-box"><div class="mlabel">Female Headed</div><div class="mvalue">{region_data.get('female_headed_pct', 0):.1f}%</div></div>
                    <p style="font-size:0.8rem; margin-top:0.5rem; color:var(--text-light);">Education: No schooling {region_data.get('education_no_schooling', 0):.0f}% | Higher {region_data.get('education_higher', 0):.0f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="card" style="min-height: 300px; display: flex; align-items: center; justify-content: center; text-align: center;">
            <p style="color: var(--text-muted);">Select a region from the dropdown above to view detailed socioeconomic statistics, including poverty rate, household count, urban/rural composition, and demographic breakdowns.</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Other maps
    st.markdown('<div class="subsection-title">Regional Comparisons</div>', unsafe_allow_html=True)
    for i in range(1, len(geo_maps)):
        name, mf, desc = geo_maps[i]
        fig = mf()
        if fig is not None:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.plotly_chart(fig, use_container_width=True)
            st.caption(desc)
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Dependency networks
    st.markdown('<div class="subsection-title">Feature Dependency Networks</div>', unsafe_allow_html=True)
    nets = get_all_networks()
    col_n1, col_n2 = st.columns(2)
    for i, (name, nf, desc) in enumerate(nets):
        with [col_n1, col_n2][i % 2]:
            fig = nf()
            if fig is not None:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.plotly_chart(fig, use_container_width=True)
                st.caption(desc)
                st.markdown('</div>', unsafe_allow_html=True)
    
    # EDA plots
    st.markdown('<div class="subsection-title">Socioeconomic Diagnostics</div>', unsafe_allow_html=True)
    eda_plots = [p for p in all_plots if p[0] == "EDA"]
    for i in range(0, len(eda_plots), 2):
        cols = st.columns(2)
        for j in range(2):
            if i + j < len(eda_plots):
                _, pf, desc = eda_plots[i + j]
                with cols[j]:
                    fig = pf()
                    if fig is not None:
                        st.markdown('<div class="card">', unsafe_allow_html=True)
                        st.plotly_chart(fig, use_container_width=True)
                        st.caption(desc)
                        st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# TAB 3
# ============================================================================
with tab3:
    st.markdown('<div class="section-title">Model Performance and Predictive Sandbox</div>', unsafe_allow_html=True)
    st.markdown("""
    <p class="analysis-text">
    The predictive framework uses <strong class="highlight-primary">LightGBM</strong> gradient boosting for
    poverty ("Pauvre") and vulnerability ("Vulnerable") classification. A <strong class="highlight-primary">PyTorch Hypernetwork</strong>
    provides an alternative non-linear approach. Transfer learning variants extend predictions from the ENCDM survey
    to the RGPH census population. Below are evaluation metrics for all models.
    </p>
    """, unsafe_allow_html=True)
    
    # Model evaluation plots
    st.markdown('<div class="subsection-title">Model Evaluation</div>', unsafe_allow_html=True)
    model_plots = [p for p in all_plots if p[0] == "Model Eval"]
    for i in range(0, len(model_plots), 2):
        cols = st.columns(2)
        for j in range(2):
            if i + j < len(model_plots):
                _, pf, desc = model_plots[i + j]
                with cols[j]:
                    fig = pf()
                    if fig is not None:
                        st.markdown('<div class="card">', unsafe_allow_html=True)
                        st.plotly_chart(fig, use_container_width=True)
                        st.caption(desc)
                        st.markdown('</div>', unsafe_allow_html=True)
    
    # Sandbox
    st.markdown('<div class="subsection-title">Interactive Scenario Simulator</div>', unsafe_allow_html=True)
    st.markdown("""
    <p class="analysis-text">
    Input household characteristics below. The system maps categorical values to codes, applies
    <strong>StandardScaler</strong> normalization (Models/Scalers/ENCDM/), runs LightGBM inference,
    and returns probability estimates with a SHAP-inspired waterfall plot explaining the prediction.
    </p>
    <div class="info-callout">
    <strong>Pipeline:</strong> Raw human-readable inputs -> Categorical encoding -> StandardScaler -> Model inference -> Probability + Feature contributions
    </div>
    """, unsafe_allow_html=True)
    
    input_data = render_sandbox_form()
    
    cb1, cb2, cb3 = st.columns([1, 2, 1])
    with cb2:
        pred_btn = st.button("Run Prediction", type="primary", use_container_width=True)
    
    if pred_btn:
        with st.spinner("Running prediction pipeline..."):
            results = run_prediction(input_data)
        
        pp = results.get("pauvre_prob", 0)
        vp = results.get("vulnerable_prob", 0)
        shap_data = results.get("shap_values", [])
        
        pc = "Pauvre" if pp > 0.5 else "Non pauvre"
        vc = "Vulnerable" if vp > 0.5 else "Non vulnerable"
        
        if pp > 0.5: rl, rc = "HIGH", "pred-high"
        elif pp > 0.2: rl, rc = "MODERATE", "pred-mod"
        else: rl, rc = "LOW", "pred-low"
        
        st.markdown('<div class="subsection-title">Prediction Results</div>', unsafe_allow_html=True)
        
        cr1, cr2, cr3 = st.columns(3)
        with cr1:
            st.markdown(f"""
            <div class="card" style="text-align:center;">
                <h3 style="border:none; padding:0; text-align:center;">Poverty Risk</h3>
                <div style="font-size:2.8rem;font-weight:900;color:{'#9B4D4D' if pp > 0.5 else '#1A5F7A'};margin:0.3rem 0;">{pp:.1%}</div>
                <span class="pred-label {rc}">{rl} RISK</span>
                <p style="margin-top:0.5rem;">Classification: <strong>{pc}</strong></p>
            </div>
            """, unsafe_allow_html=True)
        
        with cr2:
            st.markdown(f"""
            <div class="card" style="text-align:center;">
                <h3 style="border:none; padding:0; text-align:center;">Vulnerability Risk</h3>
                <div style="font-size:2.8rem;font-weight:900;color:{'#805A3B' if vp > 0.5 else '#2C7A7B'};margin:0.3rem 0;">{vp:.1%}</div>
                <span class="pred-label {'pred-high' if vp > 0.5 else 'pred-low'}">{'HIGH' if vp > 0.5 else 'LOW'}</span>
                <p style="margin-top:0.5rem;">Classification: <strong>{vc}</strong></p>
            </div>
            """, unsafe_allow_html=True)
        
        with cr3:
            n_models = sum(1 for v in res["models"].values() if v is not None and not isinstance(v, str))
            st.markdown(f"""
            <div class="card" style="text-align:center;">
                <h3 style="border:none; padding:0; text-align:center;">Model Status</h3>
                <div style="font-size:1.1rem;font-weight:600;color:var(--primary);margin:0.5rem 0;">{n_models} / 5 models</div>
                <p style="font-size:0.85rem;">Pauvre LGBM: {"Active" if res["models"].get("pauvre_lgbm") else "N/A"}</p>
                <p style="font-size:0.85rem;">Vulnerable LGBM: {"Active" if res["models"].get("vulnerable_lgbm") else "N/A"}</p>
                <p style="font-size:0.85rem;">Transfer: {"Active" if res["models"].get("transfer_pauvre") else "N/A"}</p>
            </div>
            """, unsafe_allow_html=True)
        
        if shap_data:
            st.markdown('<div class="subsection-title">Feature Contribution Analysis</div>', unsafe_allow_html=True)
            wf = build_shap_waterfall(shap_data, pp)
            if wf is not None:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.plotly_chart(wf, use_container_width=True)
                st.caption("Waterfall plot showing how each feature contributes to the poverty prediction. Blue bars increase risk, red bars decrease it.")
                st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="subsection-title">Input Summary</div>', unsafe_allow_html=True)
        input_df = pd.DataFrame([input_data]).T.reset_index()
        input_df.columns = ["Feature", "Value"]
        st.dataframe(input_df, use_container_width=True, height=300)
    else:
        st.markdown('<div class="info-callout">Adjust the characteristics above and click <strong>Run Prediction</strong> to see poverty/vulnerability risk with feature contribution analysis.</div>', unsafe_allow_html=True)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("""
<div class="footer">
    <p><b>HCP Poverty Dashboard</b> - ENCDM x RGPH - Haut-Commissariat au Plan</p>
    <p>Developed as part of HCP Guelmim Internship. Data processed with official survey weights.</p>
    <p style="font-size:0.7rem;opacity:0.7;">Sources: ENCDM 2019-2020 | RGPH 2014 | Models: LightGBM + PyTorch Hypernetwork</p>
</div>
""", unsafe_allow_html=True)