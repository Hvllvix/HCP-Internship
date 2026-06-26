"""
Moroccan Socioeconomic Intelligence Platform — Enhanced Structural Dashboard.
Combines Streamlit interactivity with HTML/CSS layout patterns from example pages.
"""
import streamlit as st
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from utils.theme import inject_theme
from utils.data_loader import get_encdm_feature_stats, get_rgph_feature_stats, load_clean_encdm, load_clean_rgph
from utils.plots import get_all_plots
from utils.data_preview import render_paginated_preview
from utils.mapping import (
    build_morocco_choropleth, compute_region_data, get_region_narrative,
    build_region_income_profile, build_region_amenity_bars,
    build_region_education_profile, build_region_gender_profile,
    build_urban_rural_map, build_rgph_housing_map,
)
from utils.network import build_encdm_network, build_rgph_network
from utils.sandbox import render_sandbox_form, run_dual_prediction, build_shap_waterfall
from utils.hypernet import get_hypernet_engine
from utils.translations import REGION_NAMES, REGION_NAME_TO_CODE

st.set_page_config(
    page_title="Moroccan Socioeconomic Intelligence",
    page_icon=" ",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_theme()

# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================
if "selected_region_code" not in st.session_state:
    st.session_state.selected_region_code = 10
if "encdm_node" not in st.session_state:
    st.session_state.encdm_node = None
if "rgph_node" not in st.session_state:
    st.session_state.rgph_node = None
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Data Integrity"
if "sidebar_collapsed" not in st.session_state:
    st.session_state.sidebar_collapsed = False


# =============================================================================
# LOAD RESOURCES
# =============================================================================
@st.cache_resource
def load_all_resources():
    return {
        "encdm": load_clean_encdm(),
        "rgph": load_clean_rgph(),
        "encdm_stats": get_encdm_feature_stats(),
        "rgph_stats": get_rgph_feature_stats(),
    }


res = load_all_resources()
es, rs = res["encdm_stats"], res["rgph_stats"]
all_plots = get_all_plots()
di_plots = [p for p in all_plots if p[0] == "Data Integrity"]
eda_plots = [p for p in all_plots if p[0] == "EDA"]
model_plots = [p for p in all_plots if p[0] == "Model Eval"]

# =============================================================================
# CUSTOM SIDEBAR NAVIGATION
# =============================================================================
with st.sidebar:
    st.markdown("""
    <div class="enhanced-sidebar">
        <div class="sidebar-brand">
            <span class="brand-icon">◈</span>
            HCP Intelligence
        </div>
        
        <div class="sidebar-section">
            <div class="sidebar-section-title">WORKFLOW</div>
            <a href="#" class="sidebar-nav-link active" onclick="selectTab('Data Integrity')">
                <span class="nav-icon">⊙</span>
                <span>Data Integrity</span>
            </a>
            <a href="#" class="sidebar-nav-link" onclick="selectTab('Regional Analytics')">
                <span class="nav-icon">◉</span>
                <span>Regional Analytics</span>
            </a>
            <a href="#" class="sidebar-nav-link" onclick="selectTab('Predictive Engine')">
                <span class="nav-icon">◎</span>
                <span>Predictive Engine</span>
            </a>
        </div>
        
        <div class="sidebar-section">
            <div class="sidebar-section-title">DATASETS</div>
            <div class="dataset-status">
                <div class="status-indicator">
                    <span class="status-dot active"></span>
                    <span>ENCDM 2019-2020</span>
                </div>
                <div class="status-indicator">
                    <span class="status-dot active"></span>
                    <span>RGPH 2014</span>
                </div>
            </div>
        </div>
        
        <div class="sidebar-section">
            <div class="sidebar-section-title">SYSTEM LOGS</div>
            <div class="log-container">
                <div class="log-entry success">
                    <span class="log-time">12:31</span>
                    <span class="log-msg">Resources loaded</span>
                </div>
                <div class="log-entry success">
                    <span class="log-time">12:30</span>
                    <span class="log-msg">Schema validated</span>
                </div>
                <div class="log-entry info">
                    <span class="log-time">12:28</span>
                    <span class="log-msg">Cache refreshed</span>
                </div>
                <div class="log-entry info">
                    <span class="log-time">12:25</span>
                    <span class="log-msg">Models compiled</span>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# RIGHT CONTROL PANEL (Show only on Regional Analytics)
# =============================================================================
def render_right_panel(region_code, region_data):
    active_name = REGION_NAMES.get(float(region_code), "Guelmim-Oued Noun") if region_code else "Guelmim-Oued Noun"
    
    st.markdown(f"""
    <div class="control-panel-enhanced">
        <div class="panel-header">
            <h3>Active Region Profile</h3>
        </div>
        <div class="region-info-card">
            <div class="region-name">{active_name}</div>
            <div class="region-code">Code: {region_code if region_code else 'N/A'}</div>
        </div>
        
        <div class="panel-section">
            <h4>Key Metrics</h4>
            <div class="metric-grid">
                <div class="metric-box">
                    <div class="metric-value">{region_data.get('poverty_rate', 0):.1f}%</div>
                    <div class="metric-label">Poverty Rate</div>
                </div>
                <div class="metric-box">
                    <div class="metric-value">{region_data.get('household_count', 0):,}</div>
                    <div class="metric-label">Households</div>
                </div>
                <div class="metric-box">
                    <div class="metric-value">{region_data.get('urban_pct', 0):.0f}%</div>
                    <div class="metric-label">Urban</div>
                </div>
                <div class="metric-box">
                    <div class="metric-value">{region_data.get('avg_hh_size', 0):.1f}</div>
                    <div class="metric-label">Avg HH Size</div>
                </div>
            </div>
        </div>
        
        <div class="panel-section">
            <h4>Pipeline Health</h4>
            <div class="pipeline-status">
                <div class="pipeline-step complete">
                    <span class="step-indicator">✓</span>
                    <span>Data Ingestion</span>
                </div>
                <div class="pipeline-step complete">
                    <span class="step-indicator">✓</span>
                    <span>Preprocessing</span>
                </div>
                <div class="pipeline-step complete">
                    <span class="step-indicator">✓</span>
                    <span>Imputation</span>
                </div>
                <div class="pipeline-step active">
                    <span class="step-indicator">◉</span>
                    <span>Inference Ready</span>
                </div>
            </div>
        </div>
        
        <div class="panel-section">
            <h4>Data Lineage</h4>
            <div class="lineage-tree">
                <div class="tree-item">ENCDM_raw.sav</div>
                <div class="tree-item">→ clean_encdm.parquet</div>
                <div class="tree-item">RGPH_raw.sav</div>
                <div class="tree-item">→ clean_rgph.parquet</div>
                <div class="tree-item">→ merged_features</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Region selector
    st.markdown('<div class="panel-section"><h4>Region Selector</h4></div>', unsafe_allow_html=True)
    region_list = list(REGION_NAMES.values())
    current_idx = 0
    if region_code is not None:
        try:
            current_idx = region_list.index(REGION_NAMES.get(float(region_code), "Guelmim-Oued Noun"))
        except ValueError:
            current_idx = 0
    
    region_sel = st.selectbox(
        "Select Region",
        region_list,
        index=current_idx,
        key="region_sel_panel",
    )
    new_code = REGION_NAME_TO_CODE.get(region_sel, 10)
    if new_code != region_code:
        st.session_state.selected_region_code = new_code
        st.rerun()


# =============================================================================
# MAIN HEADER
# =============================================================================
st.markdown(f"""
<div class="enhanced-header">
    <div class="header-left">
        <h1 class="main-title">Moroccan Socioeconomic Intelligence</h1>
        <p class="subtitle">ENCDM 2019-2020 Household Survey | RGPH 2014 Population Census | HCP Analytical Platform</p>
    </div>
    <div class="header-right">
        <div class="workflow-indicator">
            <span class="workflow-label">Workflow Stage:</span>
            <span class="workflow-value" id="current-stage">Data Integrity</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# KPI STRIP - ENHANCED
# =============================================================================
kpis = [
    ("Total Sample", f"{es.get('n_households',0) + rs.get('n_households',0):,}", "Households"),
    ("Baseline Poverty", f"{es.get('poverty_rate',0):.1f}%", "ENCDM weighted"),
    ("Regions", "12", "Administrative units"),
    ("ENCDM Features", str(es.get('n_features',0)), "Variables"),
    ("RGPH Features", str(rs.get('n_features',0)), "Variables"),
]
cells = "".join([f"""
<div class="kpi-cell-enhanced">
    <div class="kpi-label">{m[0]}</div>
    <div class="kpi-value">{m[1]}</div>
    <div class="kpi-note">{m[2]}</div>
</div>
""" for m in kpis])
st.markdown(f'<div class="kpi-strip-enhanced">{cells}</div>', unsafe_allow_html=True)

# =============================================================================
# TAB NAVIGATION WITH ENHANCED STYLING
# =============================================================================
tab1, tab2, tab3 = st.tabs(["Data Integrity", "Regional Analytics", "Predictive Engine"])

# =============================================================================
# TAB 1 - DATA INTEGRITY
# =============================================================================
with tab1:
    st.markdown("""
    <div class="section-header">
        <div class="section-label">01 / Methodology</div>
        <div class="section-title">Survey Architecture and Data Lineage</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="section-desc-enhanced">
    This platform integrates two official HCP data sources: the National Survey on Household 
    Consumption and Expenditure (ENCDM 2019-2020) and the General Census of Population and Housing 
    (RGPH 2014). ENCDM provides high-granularity expenditure profiles and official poverty classifications 
    weighted by sampling coefficients. RGPH delivers near-universal coverage of structural housing-quality indicators.
    </div>
    """, unsafe_allow_html=True)
    
    col_survey, col_census = st.columns(2)
    with col_survey:
        st.markdown(f"""
        <div class="card-enhanced">
            <div class="card-header">
                <span class="card-icon">📊</span>
                <span class="card-title">ENCDM Survey Scope</span>
            </div>
            <div class="card-body-enhanced">
                <div class="info-row">
                    <span class="info-label">Sample:</span>
                    <span class="info-value">{es.get('n_households',0):,} households</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Features:</span>
                    <span class="info-value">{es.get('n_features',0)}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Poverty Rate:</span>
                    <span class="info-value">{es.get('poverty_rate',0):.1f}%</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Missingness:</span>
                    <span class="info-value">{es.get('missing_rate',0):.2f}%</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col_census:
        st.markdown(f"""
        <div class="card-enhanced">
            <div class="card-header">
                <span class="card-icon">🏘️</span>
                <span class="card-title">RGPH Census Scope</span>
            </div>
            <div class="card-body-enhanced">
                <div class="info-row">
                    <span class="info-label">Sample:</span>
                    <span class="info-value">{rs.get('n_households',0):,} households</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Features:</span>
                    <span class="info-value">{rs.get('n_features',0)}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Missingness:</span>
                    <span class="info-value">{rs.get('missing_rate',0):.2f}%</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="section-header">
        <div class="section-label">02 / Data Quality</div>
        <div class="section-title">Completeness and Structure</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="section-desc-enhanced">
    Missing value patterns, completeness rates, and distribution plots verify that cleaned parquet files 
    preserve the statistical properties of the raw .sav sources.
    </div>
    """, unsafe_allow_html=True)
    
    # First 4 data integrity plots in enhanced grid
    for i in range(0, min(4, len(di_plots))):
        p = di_plots[i]
        fig = p[1]() if callable(p[1]) else p[1]
        if fig is not None:
            st.markdown(f"""
            <div class="plot-wrapper">
                <div class="plot-header">
                    <span class="plot-badge">Data Integrity</span>
                    <h4 class="plot-title">{p[0]}: {p[2]}</h4>
                </div>
                <div class="plot-content">
            """, unsafe_allow_html=True)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div></div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="section-header">
        <div class="section-label">03 / Interactive Preview</div>
        <div class="section-title">Data Explorer</div>
    </div>
    """, unsafe_allow_html=True)
    
    p1, p2 = st.columns(2)
    with p1:
        st.markdown("""
        <div class="preview-card">
            <div class="preview-header">
                <span class="preview-badge">ENCDM</span>
                <h4>Clean ENCDM Dataset</h4>
            </div>
            <div class="preview-body">
        """, unsafe_allow_html=True)
        render_paginated_preview(res["encdm"], dataset="encdm", key_prefix="encdm_enh")
        st.markdown('</div></div>', unsafe_allow_html=True)
    with p2:
        st.markdown("""
        <div class="preview-card">
            <div class="preview-header">
                <span class="preview-badge">RGPH</span>
                <h4>Clean RGPH Dataset</h4>
            </div>
            <div class="preview-body">
        """, unsafe_allow_html=True)
        render_paginated_preview(res["rgph"], dataset="rgph", key_prefix="rgph_enh")
        st.markdown('</div></div>', unsafe_allow_html=True)
    
    # Remaining data integrity plots
    for p in di_plots[4:]:
        fig = p[1]() if callable(p[1]) else p[1]
        if fig is not None:
            st.markdown(f"""
            <div class="plot-wrapper">
                <div class="plot-header">
                    <span class="plot-badge">Data Integrity</span>
                    <h4 class="plot-title">{p[0]}: {p[2]}</h4>
                </div>
                <div class="plot-content">
            """, unsafe_allow_html=True)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div></div>', unsafe_allow_html=True)


# =============================================================================
# TAB 2 - REGIONAL ANALYTICS
# =============================================================================
with tab2:
    region_code = st.session_state.get("selected_region_code", 10)
    if region_code is None:
        region_code = 10
    
    region_data = compute_region_data(region_key=region_code) or {}
    active_name = REGION_NAMES.get(float(region_code), "Guelmim-Oued Noun")
    
    st.markdown("""
    <div class="section-header">
        <div class="section-label">01 / Spatial Analysis</div>
        <div class="section-title">Regional Choropleth and Localized Profiles</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="section-desc-enhanced">
    The map displays weighted poverty rates across all twelve administrative regions. 
    Default view: Guelmim-Oued Noun. Use the selector or the right panel to synchronize all downstream panels.
    </div>
    """, unsafe_allow_html=True)
    
    col_map, col_control = st.columns([3, 1])
    with col_map:
        st.markdown('<div class="map-container-enhanced">', unsafe_allow_html=True)
        fig_map = build_morocco_choropleth(selected_region_code=region_code)
        if fig_map is not None:
            st.plotly_chart(fig_map, use_container_width=True)
            st.caption("Static regional poverty choropleth.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_control:
        render_right_panel(region_code, region_data)
    
    st.markdown(f"""
    <div class="section-header">
        <div class="section-label">02 / Regional Profiles</div>
        <div class="section-title">Four Views — {active_name}</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="section-desc-enhanced">
    Top-left: spending density by poverty status. Top-right: amenity access rates from RGPH census. 
    Bottom-left: education vs poverty. Bottom-right: gender-disaggregated poverty rates.
    </div>
    """, unsafe_allow_html=True)
    
    r1c1, r1c2 = st.columns(2)
    with r1c1:
        fig = build_region_income_profile(region_code)
        if fig is not None:
            st.markdown('<div class="plot-wrapper"><div class="plot-header"><span class="plot-badge">Spending</span><h4>Spending Distribution</h4></div><div class="plot-content">', unsafe_allow_html=True)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Annual household expenditure (DAM) by poverty status.")
            st.markdown('</div></div>', unsafe_allow_html=True)
    with r1c2:
        fig = build_region_amenity_bars(region_code)
        if fig is not None:
            st.markdown('<div class="plot-wrapper"><div class="plot-header"><span class="plot-badge">Amenities</span><h4>Amenity Access</h4></div><div class="plot-content">', unsafe_allow_html=True)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Percentage of households with access to electricity, water, internet, mobile phones, and private vehicles.")
            st.markdown('</div></div>', unsafe_allow_html=True)
    
    r2c1, r2c2 = st.columns(2)
    with r2c1:
        fig = build_region_education_profile(region_code)
        if fig is not None:
            st.markdown('<div class="plot-wrapper"><div class="plot-header"><span class="plot-badge">Education</span><h4>Education vs Poverty</h4></div><div class="plot-content">', unsafe_allow_html=True)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Education level vs poverty status cross-tabulation.")
            st.markdown('</div></div>', unsafe_allow_html=True)
    with r2c2:
        fig = build_region_gender_profile(region_code)
        if fig is not None:
            st.markdown('<div class="plot-wrapper"><div class="plot-header"><span class="plot-badge">Gender</span><h4>Gender and Poverty</h4></div><div class="plot-content">', unsafe_allow_html=True)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Poverty rates disaggregated by gender of household head.")
            st.markdown('</div></div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="section-header">
        <div class="section-label">03 / Comparative Analytics</div>
        <div class="section-title">Cross-Regional and Housing Diagnostics</div>
    </div>
    """, unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1:
        fig = build_urban_rural_map()
        if fig is not None:
            st.markdown('<div class="plot-wrapper"><div class="plot-header"><span class="plot-badge">Urban/Rural</span><h4>Urban/Rural Composition</h4></div><div class="plot-content">', unsafe_allow_html=True)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div></div>', unsafe_allow_html=True)
    with c2:
        fig = build_rgph_housing_map()
        if fig is not None:
            st.markdown('<div class="plot-wrapper"><div class="plot-header"><span class="plot-badge">Housing</span><h4>Housing Quality Index</h4></div><div class="plot-content">', unsafe_allow_html=True)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div></div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="section-header">
        <div class="section-label">04 / Feature Dependencies</div>
        <div class="section-title">Imputation Schema Networks</div>
    </div>
    """, unsafe_allow_html=True)
    
    n1, n2 = st.columns(2)
    with n1:
        encdm_result = build_encdm_network(highlight_node=st.session_state.encdm_node)
        if encdm_result:
            fig_n, node_ids = encdm_result
            st.markdown('<div class="plot-wrapper"><div class="plot-header"><span class="plot-badge">ENCDM</span><h4>ENCDM Imputation Dependencies</h4></div><div class="plot-content">', unsafe_allow_html=True)
            st.plotly_chart(fig_n, use_container_width=True)
            choice = st.selectbox("Focus node", ["(overview)"] + node_ids, key="encdm_node_sel_enh")
            st.session_state.encdm_node = None if choice == "(overview)" else choice
            st.markdown('</div></div>', unsafe_allow_html=True)
    with n2:
        rgph_result = build_rgph_network(highlight_node=st.session_state.rgph_node)
        if rgph_result:
            fig_n, node_ids = rgph_result
            st.markdown('<div class="plot-wrapper"><div class="plot-header"><span class="plot-badge">RGPH</span><h4>RGPH Imputation Dependencies</h4></div><div class="plot-content">', unsafe_allow_html=True)
            st.plotly_chart(fig_n, use_container_width=True)
            choice = st.selectbox("Focus node", ["(overview)"] + node_ids, key="rgph_node_sel_enh")
            st.session_state.rgph_node = None if choice == "(overview)" else choice
            st.markdown('</div></div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="section-header">
        <div class="section-label">05 / Exploratory Analysis</div>
        <div class="section-title">Socioeconomic Distributions</div>
    </div>
    """, unsafe_allow_html=True)
    
    for i in range(0, len(eda_plots), 2):
        cols = st.columns(2)
        for j in range(2):
            if i + j < len(eda_plots):
                p = eda_plots[i + j]
                with cols[j]:
                    fig = p[1]() if callable(p[1]) else p[1]
                    if fig is not None:
                        st.markdown(f'''
                        <div class="plot-wrapper">
                            <div class="plot-header">
                                <span class="plot-badge">EDA</span>
                                <h4 class="plot-title">{p[2]}</h4>
                            </div>
                            <div class="plot-content">
                        ''', unsafe_allow_html=True)
                        st.plotly_chart(fig, use_container_width=True)
                        st.markdown('</div></div>', unsafe_allow_html=True)


# =============================================================================
# TAB 3 - PREDICTIVE ENGINE
# =============================================================================
with tab3:
    st.markdown("""
    <div class="section-header">
        <div class="section-label">01 / Model Benchmarks</div>
        <div class="section-title">Classifier Performance</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="section-desc-enhanced">
    LightGBM gradient-boosted baseline and PyTorch hypernetwork comparison with comprehensive evaluation metrics.
    </div>
    """, unsafe_allow_html=True)
    
    for i in range(0, min(5, len(model_plots)), 2):
        cols = st.columns(2)
        for j in range(2):
            if i + j < len(model_plots):
                p = model_plots[i + j]
                with cols[j]:
                    fig = p[1]() if callable(p[1]) else p[1]
                    if fig is not None:
                        st.markdown(f'''
                        <div class="plot-wrapper">
                            <div class="plot-header">
                                <span class="plot-badge">Model</span>
                                <h4 class="plot-title">{p[2]}</h4>
                            </div>
                            <div class="plot-content">
                        ''', unsafe_allow_html=True)
                        st.plotly_chart(fig, use_container_width=True)
                        st.markdown('</div></div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="section-header">
        <div class="section-label">02 / Scenario Simulation</div>
        <div class="section-title">Dual-Inference Sandbox</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="section-desc-enhanced">
    Enter household characteristics. Backend encodes, imputes, scales, and runs parallel inference 
    on LightGBM and the hypernetwork for comprehensive poverty prediction.
    </div>
    """, unsafe_allow_html=True)
    
    input_data = render_sandbox_form(default_region="Guelmim-Oued Noun")
    b1, b2 = st.columns([1, 1])
    with b1:
        run_btn = st.button("Run Dual Inference", type="primary", use_container_width=True)
    with b2:
        rural_btn = st.button("Simulate Rural Transfer", use_container_width=True)
    
    if run_btn or rural_btn:
        rural_mode = rural_btn and not run_btn
        with st.spinner("Encoding, imputing, scaling, and running dual-model inference..."):
            results = run_dual_prediction(input_data, rural_transfer=rural_mode)
        
        lgbm, hyper = results["lgbm"], results["hypernet"]
        pp, vp = lgbm["pauvre_prob"], lgbm["vulnerable_prob"]
        engine = get_hypernet_engine()
        
        st.markdown("""
        <div class="section-header">
            <div class="section-label">03 / Results</div>
            <div class="section-title">Side-by-Side Probability Comparison</div>
        </div>
        """, unsafe_allow_html=True)
        
        rc1, rc2, rc3, rc4 = st.columns(4)
        with rc1:
            st.metric("LGBM Poverty", f"{pp:.1%}", "Pauvre" if pp >= 0.5 else "Non pauvre")
        with rc2:
            st.metric("LGBM Vulnerability", f"{vp:.1%}", "Vulnerable" if vp >= 0.5 else "Non vulnerable")
        with rc3:
            hpp = hyper.get("pauvre_prob")
            st.metric("Hypernet Poverty", f"{hpp:.1%}" if hpp is not None else "N/A", hyper.get("pauvre_class","N/A"))
        with rc4:
            hvp = hyper.get("vulnerability_prob") or hyper.get("vulnerable_prob")
            hvc = hyper.get("vulnerability_class") or hyper.get("vulnerable_class","N/A")
            st.metric("Hypernet Vulnerability", f"{hvp:.1%}" if hvp is not None else "N/A", hvc)
        
        c_chart, c_shap = st.columns([3, 2])
        with c_chart:
            fig = engine.build_comparison_chart(lgbm, hyper, rural_transfer=rural_mode)
            if fig is not None:
                st.markdown('<div class="plot-wrapper"><div class="plot-header"><span class="plot-badge">Comparison</span><h4>LGBM vs Hypernet Comparison</h4></div><div class="plot-content">', unsafe_allow_html=True)
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div></div>', unsafe_allow_html=True)
        with c_shap:
            if results["shap_values"]:
                fig = build_shap_waterfall(results["shap_values"], pp)
                if fig is not None:
                    st.markdown('<div class="plot-wrapper"><div class="plot-header"><span class="plot-badge">Explainability</span><h4>Local Feature Importances</h4></div><div class="plot-content">', unsafe_allow_html=True)
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown('</div></div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="section-header">
            <div class="section-label">04 / Training Diagnostics</div>
            <div class="section-title">Neural Loss Curve Extractor</div>
        </div>
        """, unsafe_allow_html=True)
        
        fig = engine.build_training_loss_chart()
        if fig is not None:
            st.markdown('<div class="plot-wrapper"><div class="plot-header"><span class="plot-badge">Training</span><h4>Training and Validation Loss</h4></div><div class="plot-content">', unsafe_allow_html=True)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div></div>', unsafe_allow_html=True)
        
        if rural_mode:
            st.markdown("""
            <div class="card-enhanced highlight">
                <div class="card-header">
                    <span class="card-icon">⚠️</span>
                    <span class="card-title">Geographic Bias Stress-Test Active</span>
                </div>
                <div class="card-body-enhanced">
                    <p>Household characteristics held constant. Milieu and region context shifted to Rural. 
                    The hypernetwork regenerated target-network weights from the updated embedding, 
                    demonstrating context-sensitive inference.</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with st.expander("Pipeline diagnostics"):
            summary = pd.DataFrame([results["raw_row"]]).T.reset_index()
            summary.columns = ["Feature", "Encoded Value"]
            st.dataframe(summary, use_container_width=True)
    else:
        st.markdown("""
        <div class="card-enhanced">
            <div class="card-header">
                <span class="card-icon">ℹ️</span>
                <span class="card-title">Ready for Inference</span>
            </div>
            <div class="card-body-enhanced">
                <p>Configure household parameters in the form above, then run dual inference or simulate rural transfer 
                to view probability comparisons and model explanations.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Remaining model plots
    for p in model_plots[5:]:
        fig = p[1]() if callable(p[1]) else p[1]
        if fig is not None:
            st.markdown(f'''
            <div class="plot-wrapper">
                <div class="plot-header">
                    <span class="plot-badge">Model</span>
                    <h4 class="plot-title">{p[2]}</h4>
                </div>
                <div class="plot-content">
            ''', unsafe_allow_html=True)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div></div>', unsafe_allow_html=True)


# =============================================================================
# FOOTER
# =============================================================================
st.markdown("""
<div class="footer-enhanced">
    <div class="footer-content">
        <span>Moroccan Socioeconomic Intelligence Platform</span>
        <span>|</span>
        <span>HCP Internship</span>
        <span>|</span>
        <span>ENCDM 2019-2020 + RGPH 2014</span>
    </div>
</div>
""", unsafe_allow_html=True)