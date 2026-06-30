"""
HCP Morocco Socioeconomic Intelligence Platform
ENCDM 2019-2020 · RGPH 2014 · LightGBM + Hypernetwork
"""
from datetime import datetime

import pandas as pd
import streamlit as st

from Utils.data_loader import GEOJSON_REGIONS, get_label, inverse_scale_encdm, translate, weighted_poverty_rate
from Utils.hypernet import get_hypernet_engine
from Utils.network import build_encdm_network, build_rgph_network
from Utils.plots import (
    CHART_H,
    CHART_H_TALL,
    compute_region_stats,
    fig_age_distribution,
    fig_choropleth,
    fig_contribution_waterfall,
    fig_dataset_dims,
    fig_dual_comparison,
    fig_feature_importance,
    fig_household_size_national,
    fig_hypernet_loss,
    fig_milieu_split,
    fig_national_education,
    fig_national_employment,
    fig_null_counts,
    fig_poverty_breakdown,
    fig_raw_missing_values,
    fig_region_amenities,
    fig_region_education,
    fig_region_gender,
    fig_region_household_size,
    fig_region_poverty,
    fig_rgph_infrastructure,
    fig_rgph_rooms,
    fig_roc_curves,
    fig_urban_rural_poverty,
)
from Utils.sandbox import InferenceError, run_dual_prediction
from Utils.theme import inject, metric_row
from Utils.utils import (
    MERMAID_INGEST,
    MERMAID_INFER,
    audit_nulls,
    boot,
    boot_raw,
    parse_map_click,
    plot_block,
    plot_row,
    print_audit,
    render_mermaid,
    safe_plot_row,
)

st.set_page_config(
    page_title="HCP Intelligence | Morocco",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject()

SIDEBAR_BLURBS = {
    "Overview": (
        "Audit dataset metadata, dimensional profiles, and raw missingness before imputation. "
        "Trace the Simpute lineage and inspect cleaned survey previews."
    ),
    "Regional Analytics": (
        "Click a region on the Morocco choropleth to unlock localized socioeconomic profiles. "
        "Demographics, amenities, employment, and housing diagnostics update instantly."
    ),
    "Predictive Engine": (
        "Simulate household scenarios through the dual LightGBM + Hypernetwork stack. "
        "Compare poverty and vulnerability probabilities with model diagnostics."
    ),
    "Smart Imputation": (
        "Upload CSV data and run adaptive per-column imputation using the Simpute library. "
        "Download the cleaned result with missing values resolved."
    ),
}

if "sel_reg" not in st.session_state:
    st.session_state.sel_reg = 2
if "encdm_node" not in st.session_state:
    st.session_state.encdm_node = None
if "rgph_node" not in st.session_state:
    st.session_state.rgph_node = None
if "inf" not in st.session_state:
    st.session_state.inf = None
if "rural_xfer" not in st.session_state:
    st.session_state.rural_xfer = False

nullreport = audit_nulls()
if not st.session_state.get("_audit_printed"):
    print_audit(nullreport)
    st.session_state._audit_printed = True

encdm, rgph, geojson, labels, codes, regions, deps, bundles, geoidmap = boot()
regstats = compute_region_stats(encdm, regions)
wpov = round(weighted_poverty_rate(encdm), 2)
raw_e = nullreport["encdm_raw"]
raw_r = nullreport["rgph_raw"]

with st.sidebar:
    st.markdown(
        '<div class="sidebar-brand">HCP <span>Intelligence</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="sidebar-tagline">Morocco poverty and vulnerability analytics '
        "fusing ENCDM survey microdata with RGPH census structure.</p>",
        unsafe_allow_html=True,
    )
    st.markdown('<p class="sidebar-nav-title">Navigation</p>', unsafe_allow_html=True)
    nav = st.radio("Navigation", list(SIDEBAR_BLURBS.keys()), label_visibility="collapsed")
    st.markdown(f'<p class="sidebar-desc">{SIDEBAR_BLURBS[nav]}</p>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<p class="sidebar-nav-title">Companion</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sidebar-desc"><strong>Simpute</strong> adaptive imputation library. '
        '<a href="https://pypi.org/project/simpute/" target="_blank">PyPI</a></p>',
        unsafe_allow_html=True,
    )
    st.markdown("---")
    engine = get_hypernet_engine(rgph)
    note = "LightGBM + Hypernet" if engine._ready else "LightGBM only"
    st.markdown(
        f'<p class="sidebar-desc">Models: {note}<br>{datetime.now():%d %b %Y, %H:%M}</p>',
        unsafe_allow_html=True,
    )


if nav == "Overview":
    import importlib
    render_overview = importlib.import_module("Pages.1_Overview").render
    raw_encdm, raw_rgph = boot_raw()
    render_overview(encdm, rgph, raw_encdm, raw_rgph, nullreport, wpov)

elif nav == "Regional Analytics":
    import importlib
    render_regional = importlib.import_module("Pages.2_Regional_Analysis").render
    render_regional(encdm, rgph, geojson, regstats, codes, labels, regions, geoidmap)

elif nav == "Predictive Engine":
    import importlib
    render_predictive = importlib.import_module("Pages.3_Prediction_Engine").render
    render_predictive(encdm, rgph, codes, bundles)

elif nav == "Smart Imputation":
    import importlib
    render_smart = importlib.import_module("Pages.4_Smart_Imputation").render
    render_smart()


st.markdown(
    """
<div class="footer">
    <p><b>MOROCCO SOCIOECONOMIC INTELLIGENCE PLATFORM</b></p>
    <p>HCP Data Fusion · ENCDM 2019-2020 · RGPH 2014 · LightGBM + Hypernetwork · Simpute</p>
</div>
""",
    unsafe_allow_html=True,
)