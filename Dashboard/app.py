"""
HCP Morocco Socioeconomic Intelligence Platform
ENCDM 2019–2020 · RGPH 2014 · LightGBM + Hypernetwork
"""
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st

from data_loader import (
    ENCDM_CONFIG,
    GEOJSON_REGIONS,
    build_code_maps,
    build_label_maps,
    build_region_name_map,
    get_label,
    inverse_scale_encdm,
    load_all_lgbm,
    load_deps_encdm,
    load_encdm,
    load_geojson,
    load_rgph,
    translate,
)
from hypernet import get_hypernet_engine
from network import build_encdm_network, build_rgph_network
from plots import (
    compute_region_stats,
    fig_choropleth,
    fig_contribution_waterfall,
    fig_dual_comparison,
    fig_feature_importance,
    fig_hypernet_loss,
    fig_missing_values,
    fig_poverty_breakdown,
    fig_region_education,
    fig_region_gender,
    fig_region_poverty,
    fig_rgph_housing_index,
    fig_roc_curves,
    fig_urban_rural_poverty,
)
from sandbox import run_dual_prediction
from theme import PALETTE, inject, plotly_layout

st.set_page_config(
    page_title="HCP Intelligence | Morocco",
    page_icon="🇲🇦",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject()

if "encdm_node" not in st.session_state:
    st.session_state.encdm_node = None
if "rgph_node" not in st.session_state:
    st.session_state.rgph_node = None


@st.cache_resource
def get_data():
    encdm = load_encdm()
    rgph = load_rgph()
    geojson = load_geojson()
    label_maps = build_label_maps()
    code_maps = build_code_maps()
    region_names = build_region_name_map()
    deps_encdm = load_deps_encdm()
    bundles = load_all_lgbm()
    get_hypernet_engine(rgph)
    return encdm, rgph, geojson, label_maps, code_maps, region_names, deps_encdm, bundles


encdm, rgph, geojson, label_maps, code_maps, region_names, deps_encdm, bundles = get_data()
region_stats = compute_region_stats(encdm, region_names)

w_total = encdm["coef_ménage"].sum()
w_pov = round((encdm["Pauvre"] * encdm["coef_ménage"]).sum() / w_total * 100, 2)

# --- SIDEBAR ---
with st.sidebar:
    st.markdown(
        '<div class="sidebar-brand">HCP <span>Intelligence</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")
    nav = st.radio(
        "Navigation",
        ["Data Integrity", "Regional Analytics", "Predictive Engine"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("Data Sources: ENCDM · RGPH")
    st.caption(f"Session · {datetime.now().strftime('%d %b %Y, %H:%M')}")
    hyper_engine = get_hypernet_engine(rgph)
    if hyper_engine._ready:
        st.caption("Models: LightGBM + Hypernet ✓")
    else:
        st.caption("Models: LightGBM only")


# ================================================================
# 1 · DATA INTEGRITY
# ================================================================
if nav == "Data Integrity":
    st.markdown('<h1 class="main-title">Data Integrity</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-subtitle">Dataset Overview & Quality Assessment</p>',
        unsafe_allow_html=True,
    )

    col_intro, col_k1, col_k2 = st.columns([2, 1, 1])
    with col_intro:
        st.markdown("""
        <div class="bench-text">
        This platform fuses official <span class="highlight-accent">Haut-Commissariat au Plan</span>
        microdata: the ENCDM household survey (2019–2020) and the RGPH census (2014).
        All statistics respect survey weights (<code>coef_ménage</code>, <code>coef_indiv</code>).
        </div>
        """, unsafe_allow_html=True)
    with col_k1:
        st.markdown(f"""
        <div class="premium-card">
            <div class="metric-value accent">{w_pov:.1f}%</div>
            <div class="metric-label">Weighted Poverty Rate</div>
        </div>
        """, unsafe_allow_html=True)
    with col_k2:
        st.markdown(f"""
        <div class="premium-card">
            <div class="metric-value">{len(encdm):,}</div>
            <div class="metric-label">ENCDM Households</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="kpi-strip">
        <div class="kpi-cell"><div class="metric-label">RGPH Observations</div>
            <div class="metric-value" style="font-size:1.4rem;">{len(rgph):,}</div></div>
        <div class="kpi-cell"><div class="metric-label">Regions</div>
            <div class="metric-value" style="font-size:1.4rem;">{encdm['Région_12'].nunique()}</div></div>
        <div class="kpi-cell"><div class="metric-label">ENCDM Features</div>
            <div class="metric-value" style="font-size:1.4rem;">{len(encdm.columns)}</div></div>
        <div class="kpi-cell"><div class="metric-label">RGPH Features</div>
            <div class="metric-value" style="font-size:1.4rem;">{len(rgph.columns)}</div></div>
    </div>
    """, unsafe_allow_html=True)

    left, right = st.columns(2)
    with left:
        st.markdown('<div class="card-eyebrow">Quality</div>', unsafe_allow_html=True)
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-heading">Missing Values — ENCDM</div>', unsafe_allow_html=True)
        fig_miss = fig_missing_values(encdm)
        if fig_miss:
            st.plotly_chart(fig_miss, use_container_width=True)
        else:
            st.success("No missing values detected in ENCDM.")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-heading">Imputation Dependencies</div>', unsafe_allow_html=True)
        dep_lines = []
        for target, deps in deps_encdm.items():
            if isinstance(deps, list) and deps:
                dep_lines.append(
                    f"<code>{target}</code> ← {', '.join(str(d) for d in deps[:4])}"
                )
        st.markdown("<br>".join(dep_lines[:6]) or "No dependency graph loaded.", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card-eyebrow">Distribution</div>', unsafe_allow_html=True)
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-heading">Weighted Poverty & Vulnerability</div>', unsafe_allow_html=True)
        st.plotly_chart(fig_poverty_breakdown(encdm), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        miss_pct_enc = round(encdm.isna().sum().sum() / (len(encdm) * len(encdm.columns)) * 100, 2)
        miss_pct_rgp = round(rgph.isna().sum().sum() / (len(rgph) * len(rgph.columns)) * 100, 2)
        sz = pd.DataFrame({
            "Dataset": ["ENCDM", "RGPH"],
            "Observations": [len(encdm), len(rgph)],
        })
        fig_sz = px.bar(
            sz, x="Dataset", y="Observations",
            color="Dataset",
            color_discrete_map={"ENCDM": PALETTE["navy"], "RGPH": PALETTE["amber"]},
            text=[f"{v:,}" for v in sz["Observations"]],
        )
        fig_sz.update_layout(**plotly_layout(height=240, showlegend=False))
        fig_sz.update_traces(textposition="outside")
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-heading">Dataset Scale</div>', unsafe_allow_html=True)
        st.plotly_chart(fig_sz, use_container_width=True)
        st.markdown(
            f"<p style='font-size:0.85rem;color:{PALETTE['navy_muted']};'>"
            f"Missing: ENCDM {miss_pct_enc}% · RGPH {miss_pct_rgp}%</p>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card-eyebrow">Preview</div>', unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["ENCDM Household Survey", "RGPH Census"])
    preview_cols = [
        "Région_12", "Milieu", "Sexe_CM", "Age_CM", "Niveau_scolaire_agreg_CM",
        "Taille_ménage", "Pauvre", "Vulnérable",
    ]
    with tab1:
        prev = inverse_scale_encdm(encdm[preview_cols].head(3000).copy())
        for col in ["Région_12", "Milieu", "Sexe_CM", "Niveau_scolaire_agreg_CM"]:
            prev[col] = translate(prev[col], col, label_maps)
        prev["Pauvre"] = prev["Pauvre"].map({1: "Poor", 0: "Non-Poor"})
        prev["Vulnérable"] = prev["Vulnérable"].map({1: "Vulnerable", 0: "Not Vulnerable"})
        prev.columns = [get_label(c) for c in prev.columns]
        st.dataframe(prev, use_container_width=True, height=360)

    with tab2:
        cols2 = ["REG", "MIL", "TAILLE", "PIECES", "ELEC", "NET", "EAU.MODE"]
        prev2 = rgph[cols2].head(3000).copy()
        prev2["MIL"] = prev2["MIL"].map({0: "Urban", 1: "Rural"})
        prev2["ELEC"] = prev2["ELEC"].map({0: "No", 1: "Yes"})
        prev2["NET"] = prev2["NET"].map({0: "No", 1: "Yes"})
        prev2.columns = [get_label(c) for c in prev2.columns]
        st.dataframe(prev2, use_container_width=True, height=360)


# ================================================================
# 2 · REGIONAL ANALYTICS
# ================================================================
elif nav == "Regional Analytics":
    st.markdown('<h1 class="main-title">Regional Analytics</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-subtitle">Spatial Socioeconomic Profiles Across Morocco</p>',
        unsafe_allow_html=True,
    )

    region_choices = sorted(region_stats.keys(), key=lambda c: region_stats[c]["name"])
    names_list = [region_stats[c]["name"] for c in region_choices]
    default_idx = names_list.index("Guelmim-Oued Noun") if "Guelmim-Oued Noun" in names_list else 0
    sel_name = st.selectbox("Focus region", names_list, index=default_idx)
    sel_code = next(c for c in region_stats if region_stats[c]["name"] == sel_name)
    sel = region_stats[sel_code]

    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    st.plotly_chart(
        fig_choropleth(geojson, region_stats, GEOJSON_REGIONS, selected_code=sel_code),
        use_container_width=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="kpi-strip">
        <div class="kpi-cell"><div class="metric-label">Poverty Rate</div>
            <div class="metric-value accent">{sel['poverty_rate']}%</div></div>
        <div class="kpi-cell"><div class="metric-label">Households</div>
            <div class="metric-value">{sel['households']:,}</div></div>
        <div class="kpi-cell"><div class="metric-label">Urban Share</div>
            <div class="metric-value">{sel['urban_pct']}%</div></div>
        <div class="kpi-cell"><div class="metric-label">Avg Age</div>
            <div class="metric-value">{sel['avg_age']}</div></div>
        <div class="kpi-cell"><div class="metric-label">Avg HH Size</div>
            <div class="metric-value">{sel['avg_size']}</div></div>
    </div>
    """, unsafe_allow_html=True)

    sub = encdm[encdm["Région_12"] == sel_code]
    sub_raw = inverse_scale_encdm(sub)
    rgph_sub = rgph[rgph["REG"] == sel_code]

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-heading">Age Distribution</div>', unsafe_allow_html=True)
        fig_age = px.histogram(
            sub_raw, x="Age_CM", nbins=28,
            color_discrete_sequence=[PALETTE["navy"]],
        )
        fig_age.update_layout(**plotly_layout(height=260))
        st.plotly_chart(fig_age, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-heading">Household Size</div>', unsafe_allow_html=True)
        fig_sz = px.histogram(
            sub_raw, x="Taille_ménage", nbins=14,
            color_discrete_sequence=[PALETTE["amber"]],
        )
        fig_sz.update_layout(**plotly_layout(height=260))
        st.plotly_chart(fig_sz, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    p1, p2 = st.columns(2)
    with p1:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-heading">Education vs Poverty</div>', unsafe_allow_html=True)
        fig_edu = fig_region_education(encdm, sel_code, label_maps)
        if fig_edu:
            st.plotly_chart(fig_edu, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with p2:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-heading">Gender & Poverty</div>', unsafe_allow_html=True)
        fig_gen = fig_region_gender(encdm, sel_code)
        if fig_gen:
            st.plotly_chart(fig_gen, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if len(rgph_sub) > 0:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-heading">Amenity Access (RGPH)</div>', unsafe_allow_html=True)
        has_water = rgph_sub["EAU.MODE"].isin([1, 2]).mean() * 100
        adm = pd.DataFrame({
            "Amenity": ["Electricity", "Improved Water", "Internet"],
            "Rate (%)": [
                rgph_sub["ELEC"].mean() * 100,
                has_water,
                rgph_sub["NET"].mean() * 100,
            ],
        })
        fig_am = px.bar(
            adm, x="Amenity", y="Rate (%)", color="Rate (%)",
            color_continuous_scale=[PALETTE["gray"], PALETTE["navy"]],
            text_auto=".1f", range_color=[0, 100],
        )
        fig_am.update_layout(**plotly_layout(height=260, coloraxis_showscale=False))
        st.plotly_chart(fig_am, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-heading">Urban / Rural Poverty</div>', unsafe_allow_html=True)
        fig_ur = fig_urban_rural_poverty(encdm, region_names)
        if fig_ur:
            st.plotly_chart(fig_ur, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-heading">Housing & Assets (RGPH)</div>', unsafe_allow_html=True)
        fig_house = fig_rgph_housing_index(rgph, sel_code)
        if fig_house:
            st.plotly_chart(fig_house, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-heading">Poverty Rate by Region (Weighted)</div>', unsafe_allow_html=True)
    st.plotly_chart(fig_region_poverty(region_stats), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card-eyebrow">Imputation Schema</div>', unsafe_allow_html=True)
    n1, n2 = st.columns(2)
    with n1:
        encdm_net = build_encdm_network(highlight_node=st.session_state.encdm_node)
        if encdm_net:
            fig_n, node_ids = encdm_net
            st.markdown('<div class="premium-card">', unsafe_allow_html=True)
            st.markdown('<div class="card-heading">ENCDM Dependencies</div>', unsafe_allow_html=True)
            st.plotly_chart(fig_n, use_container_width=True)
            choice = st.selectbox(
                "Focus node (ENCDM)", ["(overview)"] + node_ids, key="encdm_node_sel"
            )
            st.session_state.encdm_node = None if choice == "(overview)" else choice
            st.markdown("</div>", unsafe_allow_html=True)
    with n2:
        rgph_net = build_rgph_network(highlight_node=st.session_state.rgph_node)
        if rgph_net:
            fig_n, node_ids = rgph_net
            st.markdown('<div class="premium-card">', unsafe_allow_html=True)
            st.markdown('<div class="card-heading">RGPH Dependencies</div>', unsafe_allow_html=True)
            st.plotly_chart(fig_n, use_container_width=True)
            choice = st.selectbox(
                "Focus node (RGPH)", ["(overview)"] + node_ids, key="rgph_node_sel"
            )
            st.session_state.rgph_node = None if choice == "(overview)" else choice
            st.markdown("</div>", unsafe_allow_html=True)


# ================================================================
# 3 · PREDICTIVE ENGINE
# ================================================================
elif nav == "Predictive Engine":
    st.markdown('<h1 class="main-title">Predictive Engine</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-subtitle">Dual-Model Inference · LightGBM & Hypernetwork</p>',
        unsafe_allow_html=True,
    )

    st.markdown("""
    <div class="bench-text">
    Run live inference with persisted <span class="highlight-accent">LightGBM</span> classifiers
    and the geography-conditioned <span class="highlight-accent">Hypernetwork</span>.
    Feature importances and ROC curves are computed from your trained artifacts.
    </div>
    """, unsafe_allow_html=True)

    # --- Benchmarks ---
    st.markdown('<div class="card-eyebrow">Model Benchmarks</div>', unsafe_allow_html=True)
    b1, b2 = st.columns(2)
    with b1:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-heading">Feature Importance — Poverty</div>', unsafe_allow_html=True)
        fig_fi_p = fig_feature_importance(bundles, "Pauvre")
        if fig_fi_p:
            st.plotly_chart(fig_fi_p, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with b2:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-heading">Feature Importance — Vulnerability</div>', unsafe_allow_html=True)
        fig_fi_v = fig_feature_importance(bundles, "Vulnérable")
        if fig_fi_v:
            st.plotly_chart(fig_fi_v, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    b3, b4 = st.columns(2)
    with b3:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-heading">ROC Curves — LightGBM (Weighted Sample)</div>', unsafe_allow_html=True)
        st.plotly_chart(fig_roc_curves(encdm, bundles), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with b4:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-heading">Hypernetwork Training Loss</div>', unsafe_allow_html=True)
        fig_loss = fig_hypernet_loss()
        if fig_loss:
            st.plotly_chart(fig_loss, use_container_width=True)
        else:
            st.caption("Training history not available in checkpoint.")
        st.markdown("</div>", unsafe_allow_html=True)

    # --- Sandbox ---
    st.markdown('<div class="card-eyebrow">Scenario Simulation</div>', unsafe_allow_html=True)

    region_labels = list(code_maps["Région_12"].keys())
    milieu_labels = list(code_maps["Milieu"].keys())
    sexe_labels = list(code_maps["Sexe_CM"].keys())
    edu_labels = list(code_maps["Niveau_scolaire_agreg_CM"].keys())
    sit_labels = list(code_maps["Situation_profession_agreg_CM"].keys())
    taille_agg_labels = list(code_maps["Taille_agregée"].keys())

    with st.form("inference_form"):
        r1, r2, r3 = st.columns(3)
        with r1:
            region = st.selectbox("Region", region_labels, index=region_labels.index("Guelmim-Oued Noun") if "Guelmim-Oued Noun" in region_labels else 0)
            milieu = st.selectbox("Area Type", milieu_labels)
            gender = st.selectbox("Gender", sexe_labels)
        with r2:
            education = st.selectbox("Education Level", edu_labels)
            employment = st.selectbox("Employment Status", sit_labels)
            size_cat = st.selectbox("Household Size Category", taille_agg_labels)
        with r3:
            age = st.slider("Age", 18, 85, 35)
            hh_size = st.number_input("Household Size", 1, 15, 4)
            rural_transfer = st.checkbox("Simulate rural transfer (shift milieu → Rural)")

        submitted = st.form_submit_button("RUN DUAL INFERENCE", use_container_width=True)

    if submitted:
        form_inputs = {
            "Région_12": region,
            "Milieu": milieu,
            "Sexe_CM": gender,
            "Niveau_scolaire_agreg_CM": education,
            "Situation_profession_agreg_CM": employment,
            "Taille_agregée": size_cat,
            "Age_CM": age,
            "Taille_ménage": hh_size,
        }
        results = run_dual_prediction(form_inputs, code_maps, rgph, rural_transfer=rural_transfer)
        lgbm, hyper = results["lgbm"], results["hypernet"]

        def prob_card(label, res, accent=False):
            p = res["probability"]
            if p is None:
                return f"<div class='kpi-cell'><div class='metric-label'>{label}</div><div class='metric-value'>N/A</div></div>"
            color = PALETTE["danger"] if p >= 0.5 else PALETTE["navy"]
            if accent:
                color = PALETTE["amber"] if p >= 0.5 else PALETTE["navy"]
            status = "Positive" if res["label"] else "Negative"
            return (
                f"<div class='kpi-cell'><div class='metric-label'>{label}</div>"
                f"<div class='metric-value' style='color:{color};'>{p:.1%}</div>"
                f"<div style='font-size:0.72rem;color:{color};font-weight:600;'>{status}</div></div>"
            )

        st.markdown('<div class="result-banner">', unsafe_allow_html=True)
        st.markdown('<div class="card-heading">Dual Inference Results</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="kpi-strip" style="margin-bottom:0;">
            {prob_card("LGBM · Poverty", lgbm["Pauvre"])}
            {prob_card("LGBM · Vulnerability", lgbm["Vulnérable"], accent=True)}
            {prob_card("Hypernet · Poverty", hyper["Pauvre"])}
            {prob_card("Hypernet · Vulnerability", hyper["Vulnérable"], accent=True)}
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        d1, d2 = st.columns(2)
        with d1:
            st.markdown('<div class="premium-card">', unsafe_allow_html=True)
            st.markdown('<div class="card-heading">Model Comparison</div>', unsafe_allow_html=True)
            fig_cmp = fig_dual_comparison(lgbm, hyper)
            if fig_cmp:
                st.plotly_chart(fig_cmp, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with d2:
            st.markdown('<div class="premium-card">', unsafe_allow_html=True)
            st.markdown('<div class="card-heading">Feature Contributions (LGBM)</div>', unsafe_allow_html=True)
            fig_wf = fig_contribution_waterfall(results["contributions"])
            if fig_wf:
                st.plotly_chart(fig_wf, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        if rural_transfer:
            st.info(
                "Rural transfer shifts the Hypernetwork RGPH context to a rural stratum "
                "while holding household characteristics constant — useful for spatial counterfactuals."
            )
    else:
        st.info("Configure household characteristics above and run dual inference to see live model outputs.")


# --- FOOTER ---
st.markdown(f"""
<div class="footer">
    <p><b>MOROCCO SOCIOECONOMIC INTELLIGENCE PLATFORM</b></p>
    <p>HCP Data Fusion · ENCDM 2019–2020 · RGPH 2014 · LightGBM + Hypernetwork</p>
    <p style="font-size:0.65rem;opacity:0.75;">Models are probabilistic estimates for policy research — not individual classifications.</p>
</div>
""", unsafe_allow_html=True)
