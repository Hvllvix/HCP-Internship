"""
Moroccan Socioeconomic Intelligence Platform — Clean rebuild.
No render_grid, no base_layout kwargs collisions, explicit None checks.
"""
import streamlit as st
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from Example.utils.theme import inject_theme
from Example.utils.data_loader import get_encdm_feature_stats, get_rgph_feature_stats
from Example.utils.plots import get_all_plots
from Example.utils.data_preview import render_paginated_preview
from Example.utils.mapping import (
    build_morocco_choropleth, compute_region_data, get_region_narrative,
    build_region_income_profile, build_region_amenity_bars,
    build_region_education_profile, build_region_gender_profile,
    build_urban_rural_map, build_rgph_housing_map,
)
from Example.utils.network import build_encdm_network, build_rgph_network
from Example.utils.sandbox import render_sandbox_form, run_dual_prediction, build_shap_waterfall
from Example.utils.hypernet import get_hypernet_engine

st.set_page_config(
    page_title="Moroccan Socioeconomic Intelligence",
    page_icon=" ",
    layout="wide",
    initial_sidebar_state="collapsed",
)

inject_theme()

if "selected_region_code" not in st.session_state:
    st.session_state.selected_region_code = 10
if "encdm_node" not in st.session_state:
    st.session_state.encdm_node = None
if "rgph_node" not in st.session_state:
    st.session_state.rgph_node = None


@st.cache_resource
def load_all_resources():
    from Example.utils.data_loader import load_clean_encdm, load_clean_rgph
    return {
        "encdm": load_clean_encdm(),
        "rgph": load_clean_rgph(),
        "encdm_stats": get_encdm_feature_stats(),
        "rgph_stats": get_rgph_feature_stats(),
    }


res = load_all_resources()
es, rs = res["encdm_stats"], res["rgph_stats"]

# Header
st.markdown(f"""
<div class="page-header">
    <h1>Moroccan Socioeconomic Intelligence</h1>
    <div class="subtitle">ENCDM 2019-2020 Household Survey &nbsp;|&nbsp; RGPH 2014 Population Census &nbsp;|&nbsp; HCP Analytical Platform</div>
</div>
""", unsafe_allow_html=True)

# KPI strip
kpis = [
    ("Total Sample", f"{es.get('n_households',0) + rs.get('n_households',0):,}", "Households"),
    ("Baseline Poverty", f"{es.get('poverty_rate',0):.1f}%", "ENCDM weighted"),
    ("Regions", "12", "Administrative units"),
    ("ENCDM Features", str(es.get('n_features',0)), "Variables"),
    ("RGPH Features", str(rs.get('n_features',0)), "Variables"),
]
cells = "".join([f"""
<div class="kpi-cell">
    <div class="kpi-label">{m[0]}</div>
    <div class="kpi-value">{m[1]}</div>
    <div class="kpi-note">{m[2]}</div>
</div>
""" for m in kpis])
st.markdown(f'<div class="kpi-strip">{cells}</div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Data Integrity", "Regional Analytics", "Predictive Engine"])

all_plots = get_all_plots()
di_plots = [p for p in all_plots if p[0] == "Data Integrity"]
eda_plots = [p for p in all_plots if p[0] == "EDA"]
model_plots = [p for p in all_plots if p[0] == "Model Eval"]

# ===========================================================================
# TAB 1 - DATA INTEGRITY
# ===========================================================================
with tab1:
    st.markdown('<div class="section-label">Methodology</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Survey Architecture and Data Lineage</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="section-desc">
    This platform integrates two official HCP data sources: the National Survey on Household 
    Consumption and Expenditure (ENCDM 2019-2020) and the General Census of Population and Housing 
    (RGPH 2014). ENCDM provides high-granularity expenditure profiles and official poverty classifications 
    weighted by sampling coefficients (coef_indiv, coef_ménage). RGPH delivers near-universal coverage of 
    structural housing-quality indicators. The fusion enables descriptive spatial analytics and predictive modeling.
    </div>
    """, unsafe_allow_html=True)

    col_survey, col_census = st.columns(2)
    with col_survey:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">ENCDM Survey Scope</div>
            <div class="card-body">
                <p><strong>Sample:</strong> {es.get('n_households',0):,} households<br>
                <strong>Features:</strong> {es.get('n_features',0)}<br>
                <strong>Poverty Rate:</strong> {es.get('poverty_rate',0):.1f}%<br>
                <strong>Missingness:</strong> {es.get('missing_rate',0):.2f}%</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col_census:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">RGPH Census Scope</div>
            <div class="card-body">
                <p><strong>Sample:</strong> {rs.get('n_households',0):,} households<br>
                <strong>Features:</strong> {rs.get('n_features',0)}<br>
                <strong>Missingness:</strong> {rs.get('missing_rate',0):.2f}%</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">Data Quality</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Completeness and Structure</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="section-desc">
    Missing value patterns, completeness rates, and distribution plots verify that cleaned parquet files 
    preserve the statistical properties of the raw .sav sources.
    </div>
    """, unsafe_allow_html=True)

    # First 4 data integrity plots in 2-column grid
    for i in range(0, min(4, len(di_plots))):
        p = di_plots[i]
        fig = p[1]() if callable(p[1]) else p[1]
        if fig is not None:
            st.markdown(f"""
            <div class="plot-card">
                <div class="plot-card-title">{p[0]}: {p[2]}</div>
            </div>
            """, unsafe_allow_html=True)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">Interactive Data Preview</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Filterable, paginated views of the cleaned parquet files.</div>', unsafe_allow_html=True)
    p1, p2 = st.columns(2)
    with p1:
        st.markdown('<div class="card"><div class="card-title">Clean ENCDM</div><div class="card-body">', unsafe_allow_html=True)
        render_paginated_preview(res["encdm"], dataset="encdm", key_prefix="encdm")
        st.markdown('</div></div>', unsafe_allow_html=True)
    with p2:
        st.markdown('<div class="card"><div class="card-title">Clean RGPH</div><div class="card-body">', unsafe_allow_html=True)
        render_paginated_preview(res["rgph"], dataset="rgph", key_prefix="rgph")
        st.markdown('</div></div>', unsafe_allow_html=True)

    # Remaining data integrity plots
    for p in di_plots[4:]:
        fig = p[1]() if callable(p[1]) else p[1]
        if fig is not None:
            st.markdown(f"""
            <div class="plot-card">
                <div class="plot-card-title">{p[0]}: {p[2]}</div>
            </div>
            """, unsafe_allow_html=True)
            st.plotly_chart(fig, use_container_width=True)

# ===========================================================================
# TAB 2 - REGIONAL ANALYTICS
# ===========================================================================
with tab2:
    st.markdown('<div class="section-label">Spatial Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Regional Choropleth and Localized Profiles</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="section-desc">
    The map displays weighted poverty rates across all twelve administrative regions. 
    Default view: Guelmim-Oued Noun. Use the selector to synchronize all downstream panels.
    </div>
    """, unsafe_allow_html=True)

    region_code = st.session_state.get("selected_region_code", 10)
    if region_code is None:
        region_code = 10

    from Example.utils.translations import REGION_NAMES, REGION_NAME_TO_CODE
    region_data = compute_region_data(region_key=region_code) or {}
    active_name = REGION_NAMES.get(float(region_code), "Guelmim-Oued Noun")

    col_map, col_region = st.columns([3, 2])
    with col_map:
        st.markdown('<div class="map-container">', unsafe_allow_html=True)
        fig_map = build_morocco_choropleth(selected_region_code=region_code)
        if fig_map is not None:
            st.plotly_chart(fig_map, use_container_width=True)
            st.caption("Static regional poverty choropleth.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_region:
        region_sel = st.selectbox(
            "Select Region",
            ["Guelmim-Oued Noun", "Tanger-Tetouan-Al Hoceima", "Oriental", "Fes-Meknes",
             "Rabat-Sale-Kenitra", "Beni Mellal-Khenifra", "Casablanca-Settat",
             "Marrakech-Safi", "Draa-Tafilalet", "Souss-Massa",
             "Laayoune-Sakia El Hamra", "Dakhla-Oued Ed Dahab"],
            index=0,
            key="region_sel_tab2",
        )
        new_code = REGION_NAME_TO_CODE.get(region_sel, 10)
        if new_code != region_code:
            st.session_state.selected_region_code = new_code
            st.rerun()

        pr = region_data.get("poverty_rate", 0)
        hh = region_data.get("household_count", 0)
        urban = region_data.get("urban_pct", 0)
        avg_age = region_data.get("avg_age", 0)
        avg_hh = region_data.get("avg_hh_size", 0)

        st.markdown(f"""
        <div class="card" style="margin-top:0.5rem;">
            <div class="card-title">{active_name}</div>
            <div class="card-body">
                <p style="font-size:0.9rem; line-height:1.6; color:#4A4A4A;">{get_region_narrative(region_code)}</p>
                <div class="metric-strip">
                    <div class="metric-cell">
                        <div class="metric-label">Poverty</div>
                        <div class="metric-value">{pr:.1f}%</div>
                    </div>
                    <div class="metric-cell">
                        <div class="metric-label">Households</div>
                        <div class="metric-value">{hh:,}</div>
                    </div>
                    <div class="metric-cell">
                        <div class="metric-label">Urban</div>
                        <div class="metric-value">{urban:.0f}%</div>
                    </div>
                    <div class="metric-cell">
                        <div class="metric-label">Avg Age</div>
                        <div class="metric-value">{avg_age:.0f}</div>
                    </div>
                    <div class="metric-cell">
                        <div class="metric-label">HH Size</div>
                        <div class="metric-value">{avg_hh:.1f}</div>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">Regional Profiles</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-title">Four Views — {active_name}</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="section-desc">
    Top-left: spending density by poverty status. Top-right: amenity access rates from RGPH census. 
    Bottom-left: education vs poverty. Bottom-right: gender-disaggregated poverty rates.
    </div>
    """, unsafe_allow_html=True)

    r1c1, r1c2 = st.columns(2)
    with r1c1:
        fig = build_region_income_profile(region_code)
        if fig is not None:
            st.markdown('<div class="plot-card"><div class="plot-card-title">Spending Distribution</div>', unsafe_allow_html=True)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Annual household expenditure (DAM) by poverty status.")
    with r1c2:
        fig = build_region_amenity_bars(region_code)
        if fig is not None:
            st.markdown('<div class="plot-card"><div class="plot-card-title">Amenity Access</div>', unsafe_allow_html=True)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Percentage of households with access to electricity, water, internet, mobile phones, and private vehicles.")

    r2c1, r2c2 = st.columns(2)
    with r2c1:
        fig = build_region_education_profile(region_code)
        if fig is not None:
            st.markdown('<div class="plot-card"><div class="plot-card-title">Education vs Poverty</div>', unsafe_allow_html=True)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Education level vs poverty status cross-tabulation.")
    with r2c2:
        fig = build_region_gender_profile(region_code)
        if fig is not None:
            st.markdown('<div class="plot-card"><div class="plot-card-title">Gender and Poverty</div>', unsafe_allow_html=True)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Poverty rates disaggregated by gender of household head.")

    st.markdown('<div class="section-label">Comparative Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Cross-Regional and Housing Diagnostics</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        fig = build_urban_rural_map()
        if fig is not None:
            st.markdown('<div class="plot-card"><div class="plot-card-title">Urban/Rural Composition</div>', unsafe_allow_html=True)
            st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = build_rgph_housing_map()
        if fig is not None:
            st.markdown('<div class="plot-card"><div class="plot-card-title">Housing Quality Index</div>', unsafe_allow_html=True)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-label">Feature Dependencies</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Imputation Schema Networks</div>', unsafe_allow_html=True)
    n1, n2 = st.columns(2)
    with n1:
        encdm_result = build_encdm_network(highlight_node=st.session_state.encdm_node)
        if encdm_result:
            fig_n, node_ids = encdm_result
            st.markdown('<div class="plot-card"><div class="plot-card-title">ENCDM Imputation Dependencies</div>', unsafe_allow_html=True)
            st.plotly_chart(fig_n, use_container_width=True)
            choice = st.selectbox("Focus node", ["(overview)"] + node_ids, key="encdm_node_sel")
            st.session_state.encdm_node = None if choice == "(overview)" else choice
    with n2:
        rgph_result = build_rgph_network(highlight_node=st.session_state.rgph_node)
        if rgph_result:
            fig_n, node_ids = rgph_result
            st.markdown('<div class="plot-card"><div class="plot-card-title">RGPH Imputation Dependencies</div>', unsafe_allow_html=True)
            st.plotly_chart(fig_n, use_container_width=True)
            choice = st.selectbox("Focus node", ["(overview)"] + node_ids, key="rgph_node_sel")
            st.session_state.rgph_node = None if choice == "(overview)" else choice

    st.markdown('<div class="section-label">Exploratory Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Socioeconomic Distributions</div>', unsafe_allow_html=True)
    for i in range(0, len(eda_plots), 2):
        cols = st.columns(2)
        for j in range(2):
            if i + j < len(eda_plots):
                p = eda_plots[i + j]
                with cols[j]:
                    fig = p[1]() if callable(p[1]) else p[1]
                    if fig is not None:
                        st.markdown(f"""
                        <div class="plot-card">
                            <div class="plot-card-title">{p[2]}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        st.plotly_chart(fig, use_container_width=True)

# ===========================================================================
# TAB 3 - PREDICTIVE ENGINE
# ===========================================================================
with tab3:
    st.markdown('<div class="section-label">Model Benchmarks</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Classifier Performance</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="section-desc">
    LightGBM gradient-boosted baseline and PyTorch hypernetwork comparison.
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
                        st.markdown(f'<div class="plot-card"><div class="plot-card-title">{p[2]}</div></div>', unsafe_allow_html=True)
                        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-label">Scenario Simulation</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Dual-Inference Sandbox</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="section-desc">
    Enter household characteristics. Backend encodes, imputes, scales, and runs parallel inference 
    on LightGBM and the hypernetwork.
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

        st.markdown('<div class="section-title">Side-by-Side Probability Comparison</div>', unsafe_allow_html=True)
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
                st.markdown('<div class="plot-card"><div class="plot-card-title">LGBM vs Hypernet Comparison</div>', unsafe_allow_html=True)
                st.plotly_chart(fig, use_container_width=True)
        with c_shap:
            if results["shap_values"]:
                fig = build_shap_waterfall(results["shap_values"], pp)
                if fig is not None:
                    st.markdown('<div class="plot-card"><div class="plot-card-title">Local Feature Importances</div>', unsafe_allow_html=True)
                    st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-title">Neural Loss Curve Extractor</div>', unsafe_allow_html=True)
        fig = engine.build_training_loss_chart()
        if fig is not None:
            st.markdown('<div class="plot-card"><div class="plot-card-title">Training and Validation Loss</div>', unsafe_allow_html=True)
            st.plotly_chart(fig, use_container_width=True)

        if rural_mode:
            st.markdown("""
            <div class="card"><div class="card-body">
            <h4 style="color:#2563EB; margin:0 0 0.4rem 0;">Geographic Bias Stress-Test Active</h4>
            <p style="font-size:0.9rem; color:#4A4A4A;">Household characteristics held constant. Milieu and region context shifted to Rural. The hypernetwork regenerated target-network weights from the updated embedding, demonstrating context-sensitive inference.</p>
            </div></div>
            """, unsafe_allow_html=True)

        with st.expander("Pipeline diagnostics"):
            summary = pd.DataFrame([results["raw_row"]]).T.reset_index()
            summary.columns = ["Feature", "Encoded Value"]
            st.dataframe(summary, use_container_width=True)
    else:
        st.markdown("""
        <div class="card"><div class="card-body">
        <p style="color:#4A4A4A;">Configure household parameters in the form above, then run dual inference or simulate rural transfer to view probability comparisons and model explanations.</p>
        </div></div>
        """, unsafe_allow_html=True)

    # Remaining model plots
    for p in model_plots[5:]:
        fig = p[1]() if callable(p[1]) else p[1]
        if fig is not None:
            st.markdown(f'<div class="plot-card"><div class="plot-card-title">{p[2]}</div></div>', unsafe_allow_html=True)
            st.plotly_chart(fig, use_container_width=True)

st.markdown("""
<div class="footer">
    Moroccan Socioeconomic Intelligence Platform &nbsp;|&nbsp; HCP Internship &nbsp;|&nbsp; ENCDM 2019-2020 + RGPH 2014
</div>
""", unsafe_allow_html=True)