"""
HCP Morocco Socioeconomic Intelligence Platform
ENCDM 2019-2020 · RGPH 2014 · LightGBM + Hypernetwork
"""
from datetime import datetime

import pandas as pd
import streamlit as st

from data_loader import GEOJSON_REGIONS, get_label, inverse_scale_encdm, translate, weighted_poverty_rate
from hypernet import get_hypernet_engine
from network import build_encdm_network, build_rgph_network
from plots import (
    CHART_H,
    CHART_H_TALL,
    compute_region_stats,
    fig_age_distribution,
    fig_choropleth,
    fig_contribution_waterfall,
    fig_dataset_dims,
    fig_dual_comparison,
    fig_employment_mix,
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
    fig_vulnerability_breakdown,
)
from sandbox import InferenceError, run_dual_prediction
from theme import inject, metric_row
from utils import (
    MERMAID_INGEST,
    MERMAID_INFER,
    audit_nulls,
    boot,
    boot_raw,
    dep_cards,
    parse_map_click,
    pkg_cards,
    plot_block,
    plot_row,
    print_audit,
    render_mermaid,
    render_tree,
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


# ================================================================
# OVERVIEW
# ================================================================
if nav == "Overview":
    st.markdown('<h1 class="main-title">Overview</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Dataset Metadata and Pre-Processing Profile</p>', unsafe_allow_html=True)

    raw_encdm, raw_rgph = boot_raw()

    st.markdown(
        metric_row([
            ("Weighted Poverty", f"{wpov:.1f}%", "National household-weighted rate", True),
            ("ENCDM Raw Nulls", f"{raw_e['total_nulls']:,}", f"{raw_e['cols_with_nulls']} cols with gaps"),
            ("RGPH Raw Nulls", f"{raw_r['total_nulls']:,}", f"{raw_r['cols_with_nulls']} cols with gaps"),
            ("Clean Nulls", f"{nullreport['encdm_clean']['total_nulls'] + nullreport['rgph_clean']['total_nulls']:,}", "Both parquet files"),
        ]),
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        st.markdown('<p class="section-heading">Data Ingestion Pipeline</p>', unsafe_allow_html=True)
        st.markdown(
            '<p class="plot-desc">Verified null audit: ENCDM raw '
            f"{raw_e['total_nulls']:,} null cells across {raw_e['cols']} columns; "
            f"RGPH raw {raw_r['total_nulls']:,} null cells across {raw_r['cols']} columns. "
            "Clean parquet files retain zero nulls post-imputation.</p>",
            unsafe_allow_html=True,
        )
        render_mermaid(MERMAID_INGEST)

    with st.container(border=True):
        st.markdown('<p class="section-heading">Platform Scope</p>', unsafe_allow_html=True)
        st.markdown(
            '<p class="prose-block">Official <strong>Haut-Commissariat au Plan</strong> microdata: '
            "ENCDM household consumption (2019-2020) and RGPH census (2014). "
            "All rates respect inverse-scaled household weights (<code>coef_ménage</code>).</p>",
            unsafe_allow_html=True,
        )

    with st.container(border=True):
        st.markdown('<p class="section-heading">Simpute Companion Project</p>', unsafe_allow_html=True)
        st.markdown(
            '<p class="prose-block">Adaptive per-column imputation library generalizing the internship pipeline. '
            '<a href="https://pypi.org/project/simpute/">PyPI</a> · '
            '<a href="https://github.com/Hvllvix/Simpute">GitHub</a>.</p>',
            unsafe_allow_html=True,
        )

    with st.container(border=True):
        st.markdown('<p class="section-heading">Repository Structure</p>', unsafe_allow_html=True)
        render_tree()

    plot_row([
        (
            "Dataset Dimensions",
            "Row and column counts for raw and cleaned deliverables.",
            fig_dataset_dims(raw_encdm, raw_rgph, encdm, rgph),
        ),
        (
            "Null Cell Counts",
            "Verified aggregate missing vs filled cells in raw SPSS files.",
            fig_null_counts(raw_encdm, raw_rgph),
        ),
    ])

    plot_block(
        "Missing Values Profile (Raw)",
        "Features with zero missing count are omitted.",
        fig_raw_missing_values(raw_encdm, raw_rgph),
        h=CHART_H_TALL + 40,
    )

    with st.container(border=True):
        st.markdown('<p class="section-heading">Imputation Dependencies</p>', unsafe_allow_html=True)
        dep_cards(deps)

    with st.container(border=True):
        st.markdown('<p class="section-heading">Runtime Dependencies</p>', unsafe_allow_html=True)
        pkg_cards()

    st.markdown('<p class="card-eyebrow">Clean Data Preview</p>', unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["ENCDM Survey", "RGPH Census"])
    preview = [
        "Région_12", "Milieu", "Sexe_CM", "Age_CM", "Niveau_scolaire_agreg_CM",
        "Taille_ménage", "Pauvre", "Vulnérable",
    ]
    with tab1:
        with st.container(border=True):
            prev = inverse_scale_encdm(encdm[preview].head(3000).copy())
            for col in ["Région_12", "Milieu", "Sexe_CM", "Niveau_scolaire_agreg_CM"]:
                prev[col] = translate(prev[col], col, labels)
            prev["Pauvre"] = prev["Pauvre"].map({1: "Poor", 0: "Non-Poor"})
            prev["Vulnérable"] = prev["Vulnérable"].map({1: "Vulnerable", 0: "Not Vulnerable"})
            prev.columns = [get_label(c) for c in prev.columns]
            st.dataframe(prev, use_container_width=True, height=360)
    with tab2:
        with st.container(border=True):
            cols2 = ["REG", "MIL", "TAILLE", "PIECES", "ELEC", "NET", "EAU.MODE"]
            prev2 = rgph[cols2].head(3000).copy()
            prev2["MIL"] = prev2["MIL"].map({0: "Urban", 1: "Rural"})
            prev2["ELEC"] = prev2["ELEC"].map({0: "No", 1: "Yes"})
            prev2["NET"] = prev2["NET"].map({0: "No", 1: "Yes"})
            prev2.columns = [get_label(c) for c in prev2.columns]
            st.dataframe(prev2, use_container_width=True, height=360)


# ================================================================
# REGIONAL ANALYTICS
# ================================================================
elif nav == "Regional Analytics":
    st.markdown('<h1 class="main-title">Regional Analytics</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Spatial Socioeconomic Profiles Across Morocco</p>', unsafe_allow_html=True)

    try:
        sel = int(st.session_state.sel_reg)
        if sel not in regstats:
            sel = next(iter(regstats))
            st.session_state.sel_reg = sel
        selinfo = regstats[sel]
        selname = selinfo["name"]

        name_to_code = {regstats[c]["name"]: c for c in regstats}
        pick = st.selectbox(
            "Region selector",
            list(name_to_code.keys()),
            index=list(name_to_code.values()).index(sel),
            key="region_picker",
        )
        if name_to_code[pick] != sel:
            st.session_state.sel_reg = name_to_code[pick]
            st.rerun()

        with st.container(border=True):
            st.markdown('<p class="section-heading">Morocco Poverty Choropleth</p>', unsafe_allow_html=True)
            mapfig = fig_choropleth(geojson, regstats, GEOJSON_REGIONS, selected_code=sel)
            mapevt = st.plotly_chart(
                mapfig,
                use_container_width=True,
                height=CHART_H_TALL + 80,
                on_select="rerun",
                key="morocco_map",
                selection_mode="points",
            )
            clicked = parse_map_click(mapevt, geoidmap)
            if clicked is not None and clicked != sel:
                st.session_state.sel_reg = clicked
                st.rerun()

        st.markdown(f'<p class="plot-desc"><strong>Selected:</strong> {selname}</p>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="kpi-strip">'
            f'<div class="kpi-cell"><div class="metric-label">Poverty Rate</div>'
            f'<div class="metric-value accent">{selinfo["poverty_rate"]}%</div></div>'
            f'<div class="kpi-cell"><div class="metric-label">Households</div>'
            f'<div class="metric-value">{selinfo["households"]:,}</div></div>'
            f'<div class="kpi-cell"><div class="metric-label">Urban Share</div>'
            f'<div class="metric-value">{selinfo["urban_pct"]}%</div></div>'
            f'<div class="kpi-cell"><div class="metric-label">Avg Age</div>'
            f'<div class="metric-value">{selinfo["avg_age"]}</div></div>'
            f'<div class="kpi-cell"><div class="metric-label">Avg HH Size</div>'
            f'<div class="metric-value">{selinfo["avg_size"]}</div></div>'
            f"</div>",
            unsafe_allow_html=True,
        )

        st.markdown('<p class="card-eyebrow">National Structure</p>', unsafe_allow_html=True)
        safe_plot_row([
            ("Weighted Poverty Split", "National weighted poverty shares.", lambda: fig_poverty_breakdown(encdm)),
            ("Vulnerability Split", "National vulnerability shares.", lambda: fig_vulnerability_breakdown(encdm)),
        ])
        safe_plot_row([
            ("National Education", "Weighted education distribution.", lambda: fig_national_education(encdm, labels)),
            ("National Employment", "Top employment categories.", lambda: fig_national_employment(encdm, labels)),
        ], h=CHART_H_TALL)
        safe_plot_row([
            ("Household Size (National)", "Inverse-scaled ENCDM sizes.", lambda: fig_household_size_national(encdm)),
            ("RGPH Asset Access", "National infrastructure rates.", lambda: fig_rgph_infrastructure(rgph)),
        ])

        st.markdown(f'<p class="card-eyebrow">Regional Profile - {selname}</p>', unsafe_allow_html=True)
        safe_plot_row([
            ("Age Distribution", f"Ages in {selname}.", lambda: fig_age_distribution(encdm, sel)),
            ("Education Structure", f"Education in {selname}.", lambda: fig_region_education(encdm, sel, labels)),
            ("Amenity Access (RGPH)", f"Infrastructure in {selname}.", lambda: fig_region_amenities(rgph, sel)),
        ], ctx=f"Regional Profile {selname}")
        safe_plot_row([
            ("Urban / Rural Poverty", "Poverty by milieu.", lambda: fig_urban_rural_poverty(encdm, regions)),
            ("Poverty Rate by Region", "Regional ranking.", lambda: fig_region_poverty(regstats)),
        ], h=CHART_H_TALL, ctx=f"Regional Compare {selname}")
        safe_plot_row([
            ("Employment Mix", f"Employment in {selname}.", lambda: fig_employment_mix(encdm, sel, labels)),
            ("Urban-Rural Composition", f"Milieu split in {selname}.", lambda: fig_milieu_split(encdm, sel)),
            ("Household Size", f"HH sizes in {selname}.", lambda: fig_region_household_size(encdm, sel)),
        ], ctx=f"Regional Demographics {selname}")
        safe_plot_row([
            ("Gender Representation", f"Gender shares in {selname}.", lambda: fig_region_gender(encdm, sel)),
            ("Housing - Room Count", f"Rooms in {selname}.", lambda: fig_rgph_rooms(rgph, sel)),
        ], ctx=f"Regional Housing {selname}")

        st.markdown('<p class="card-eyebrow">Imputation Schema Networks</p>', unsafe_allow_html=True)
        n1, n2 = st.columns(2)
        with n1:
            with st.container(border=True):
                st.markdown('<p class="section-heading">ENCDM Dependencies</p>', unsafe_allow_html=True)
                encdm_net = build_encdm_network()
                encdm_ids = encdm_net[1] if encdm_net else []
                encdm_opts = ["(overview)"] + encdm_ids
                enc_idx = encdm_opts.index(st.session_state.encdm_node) if st.session_state.encdm_node in encdm_opts else 0
                choice = st.selectbox("Focus node (ENCDM)", encdm_opts, index=enc_idx, key="encdm_node_sel")
                st.session_state.encdm_node = None if choice == "(overview)" else choice
                focused = build_encdm_network(highlight_node=st.session_state.encdm_node)
                if focused:
                    st.plotly_chart(focused[0], use_container_width=True, height=CHART_H_TALL)
        with n2:
            with st.container(border=True):
                st.markdown('<p class="section-heading">RGPH Dependencies</p>', unsafe_allow_html=True)
                rgph_net = build_rgph_network()
                rgph_ids = rgph_net[1] if rgph_net else []
                rgph_opts = ["(overview)"] + rgph_ids
                rg_idx = rgph_opts.index(st.session_state.rgph_node) if st.session_state.rgph_node in rgph_opts else 0
                choice = st.selectbox("Focus node (RGPH)", rgph_opts, index=rg_idx, key="rgph_node_sel")
                st.session_state.rgph_node = None if choice == "(overview)" else choice
                focused = build_rgph_network(highlight_node=st.session_state.rgph_node)
                if focused:
                    st.plotly_chart(focused[0], use_container_width=True, height=CHART_H_TALL)

    except Exception as exc:
        st.error(f"Regional Analytics failed for region code {st.session_state.sel_reg}: {exc}")


# ================================================================
# PREDICTIVE ENGINE
# ================================================================
elif nav == "Predictive Engine":
    st.markdown('<h1 class="main-title">Predictive Engine</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Dual-Model Inference · LightGBM and Hypernetwork</p>', unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown('<p class="section-heading">Hypernetwork Inference Flow</p>', unsafe_allow_html=True)
        st.markdown(
            '<p class="plot-desc">LightGBM is census-agnostic. The Hypernetwork conditions on RGPH strata context '
            "and generates dynamic target-network weights per geography.</p>",
            unsafe_allow_html=True,
        )
        render_mermaid(MERMAID_INFER)

    plot_row([
        (
            "Feature Importance - Poverty",
            "Global LightGBM importances for Pauvre.",
            fig_feature_importance(bundles, "Pauvre"),
        ),
        (
            "Feature Importance - Vulnerability",
            "Global LightGBM importances for Vulnérable.",
            fig_feature_importance(bundles, "Vulnérable"),
        ),
    ], h=CHART_H_TALL)
    plot_row([
        ("ROC Curves - LightGBM", "Discrimination on a held sample.", fig_roc_curves(encdm, bundles)),
        ("Hypernetwork Training Loss", "Log-scaled loss from Hypernet.pt.", fig_hypernet_loss()),
    ])

    st.markdown('<p class="card-eyebrow">Scenario Simulation</p>', unsafe_allow_html=True)

    region_labels = list(codes["Région_12"].keys())
    milieu_labels = list(codes["Milieu"].keys())
    sexe_labels = list(codes["Sexe_CM"].keys())
    edu_labels = list(codes["Niveau_scolaire_agreg_CM"].keys())
    sit_labels = list(codes["Situation_profession_agreg_CM"].keys())
    taille_labels = list(codes["Taille_agregée"].keys())

    with st.container(border=True):
        st.markdown('<p class="section-heading">Household Parameters</p>', unsafe_allow_html=True)
        with st.form("infer_form"):
            r1, r2, r3 = st.columns(3)
            with r1:
                region = st.selectbox(
                    "Region", region_labels,
                    index=region_labels.index("Guelmim-Oued Noun") if "Guelmim-Oued Noun" in region_labels else 0,
                )
                milieu = st.selectbox("Area Type", milieu_labels)
                gender = st.selectbox("Gender", sexe_labels)
            with r2:
                education = st.selectbox("Education Level", edu_labels)
                employment = st.selectbox("Employment Status", sit_labels)
                size_cat = st.selectbox("Household Size Category", taille_labels)
            with r3:
                age = st.slider("Age", 18, 85, 35)
                hh_size = st.number_input("Household Size", 1, 15, 4)
                rural_xfer = st.checkbox("Simulate rural transfer (Hypernet counterfactual)")
            go_btn = st.form_submit_button("Run dual inference", use_container_width=True)

    if go_btn:
        inputs = {
            "Région_12": region,
            "Milieu": milieu,
            "Sexe_CM": gender,
            "Niveau_scolaire_agreg_CM": education,
            "Situation_profession_agreg_CM": employment,
            "Taille_agregée": size_cat,
            "Age_CM": age,
            "Taille_ménage": hh_size,
        }
        st.session_state.rural_xfer = rural_xfer
        try:
            with st.spinner("Running dual inference..."):
                st.session_state.inf = run_dual_prediction(inputs, codes, rgph, rural_transfer=rural_xfer)
        except InferenceError as exc:
            st.error(f"Inference failed: {exc}")
            st.session_state.inf = None

    if st.session_state.inf:
        res = st.session_state.inf
        lgbm, hyper = res["lgbm"], res["hypernet"]

        st.markdown(
            '<div class="disclaimer-banner"><strong>Policy research estimate.</strong> '
            "Probabilities are structural model outputs, not ground-truth individual classifications.</div>",
            unsafe_allow_html=True,
        )

        with st.container(border=True):
            st.markdown('<p class="section-heading">Dual Inference Results</p>', unsafe_allow_html=True)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("LGBM Poverty", f"{lgbm['Pauvre']['probability']:.1%}")
            c2.metric("LGBM Vulnerability", f"{lgbm['Vulnérable']['probability']:.1%}")
            hpp = hyper["Pauvre"]["probability"]
            hvp = hyper["Vulnérable"]["probability"]
            c3.metric("Hypernet Poverty", f"{hpp:.1%}" if hpp is not None else "N/A")
            c4.metric("Hypernet Vulnerability", f"{hvp:.1%}" if hvp is not None else "N/A")

        for flag in res.get("ood_flags", []):
            st.warning(flag)

        plot_row([
            (
                "Model Comparison (log-scaled)",
                "log10(probability) exposes variance in low absolute outputs.",
                fig_dual_comparison(lgbm, hyper, log_scale=True),
            ),
            (
                "Feature Contributions (LGBM)",
                "Top feature importances for this profile.",
                fig_contribution_waterfall(res["contributions"]),
            ),
        ])

        d1, d2 = st.columns(2)
        with d1:
            with st.expander("Feature contributions detail", expanded=False):
                st.dataframe(pd.DataFrame(res["contributions"]), use_container_width=True)
        with d2:
            with st.expander("Encoded feature vector", expanded=False):
                diag = pd.DataFrame({"Raw": res["feature_row"], "Scaled": res["scaled_row"]}).T
                st.dataframe(diag, use_container_width=True)

        if st.session_state.rural_xfer:
            st.info("Rural transfer: Hypernetwork RGPH context shifted to rural stratum.")
    elif not go_btn:
        st.info("Configure household characteristics and run dual inference.")


st.markdown(
    """
<div class="footer">
    <p><b>MOROCCO SOCIOECONOMIC INTELLIGENCE PLATFORM</b></p>
    <p>HCP Data Fusion · ENCDM 2019-2020 · RGPH 2014 · LightGBM + Hypernetwork · Simpute</p>
</div>
""",
    unsafe_allow_html=True,
)
