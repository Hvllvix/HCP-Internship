"""
HCP Morocco Socioeconomic Intelligence Platform
ENCDM 2019–2020 · RGPH 2014 · LightGBM + Hypernetwork
"""
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st

from data_loader import (
    GEOJSON_REGIONS,
    build_code_maps,
    build_geo_id_map,
    build_label_maps,
    build_region_name_map,
    get_label,
    inverse_scale_encdm,
    load_all_lgbm,
    load_deps_encdm,
    load_encdm,
    load_geojson,
    load_raw_encdm,
    load_raw_rgph,
    load_rgph,
    translate,
    weighted_poverty_rate,
)
from hypernet import get_hypernet_engine
from network import build_encdm_network, build_rgph_network
from plots import (
    CHART_H,
    CHART_H_TALL,
    compute_region_stats,
    fig_age_distribution,
    fig_choropleth,
    fig_contribution_waterfall,
    fig_dual_comparison,
    fig_employment_mix,
    fig_feature_importance,
    fig_household_size_national,
    fig_hypernet_loss,
    fig_milieu_split,
    fig_national_education,
    fig_national_employment,
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
from sandbox import run_dual_prediction
from theme import PALETTE, inject, plotly_layout

st.set_page_config(
    page_title="HCP Intelligence | Morocco",
    page_icon="🇲🇦",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject()

PROJECT_TREE = """Internship-HCP/
├── Dashboard/          # Streamlit app (this UI)
│   ├── app.py
│   ├── data_loader.py  # Parquet, mappings, models, weights
│   ├── plots.py        # Plotly analytics
│   ├── sandbox.py      # Dual inference pipeline
│   ├── hypernet.py     # PyTorch hypernetwork engine
│   ├── network.py      # Imputation dependency graphs
│   └── theme.py
├── Data/
│   ├── Raw/            # ENCDM.sav · RGPH.sav
│   └── Processed/      # Clean parquet files
├── Models/
│   ├── Classifier/     # LightGBM + Hypernet.pt
│   ├── Scalers/
│   └── Imputers/       # ENCDM (KNN) · RGPH (LightGBM)
├── Assets/
│   ├── Maps/           # GeoJSON + codebooks
│   ├── Dependencies/   # Imputation schemas
│   └── Plots/          # Static report figures
├── Notebooks/          # Pre-processing · Analysis · Modeling
└── requirements.txt"""

SIDEBAR_BLURBS = {
    "Data Integrity": (
        "Audit raw HCP microdata, trace the Simpute imputation lineage, and verify "
        "that cleaned parquet files preserve survey structure before any modeling."
    ),
    "Regional Analytics": (
        "Click a region on the Morocco choropleth to unlock localized socioeconomic "
        "profiles — demographics, amenities, employment, and housing diagnostics."
    ),
    "Predictive Engine": (
        "Simulate household scenarios through the dual LightGBM + Hypernetwork stack "
        "and compare poverty/vulnerability probabilities with model diagnostics."
    ),
}

if "selected_region_code" not in st.session_state:
    st.session_state.selected_region_code = 2
if "encdm_node" not in st.session_state:
    st.session_state.encdm_node = None
if "rgph_node" not in st.session_state:
    st.session_state.rgph_node = None


def plot_block(title, description, fig, chart_height=CHART_H):
    with st.container(border=True):
        st.markdown(f'<p class="section-heading">{title}</p>', unsafe_allow_html=True)
        if description:
            st.markdown(f'<p class="plot-desc">{description}</p>', unsafe_allow_html=True)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True, height=chart_height)


def plot_row(items, chart_height=CHART_H):
    """Render equal-height side-by-side plot blocks."""
    cols = st.columns(len(items))
    for col, (title, desc, fig) in zip(cols, items):
        with col:
            plot_block(title, desc, fig, chart_height=chart_height)


def parse_map_selection(event, geo_id_map):
    if event is None or not getattr(event, "selection", None):
        return None
    sel = event.selection
    points = sel.get("points", []) if isinstance(sel, dict) else getattr(sel, "points", [])
    for pt in points:
        raw = pt.get("customdata") or pt.get("location")
        try:
            cid = int(raw[0] if isinstance(raw, (list, tuple)) else raw)
            if cid in geo_id_map:
                return int(geo_id_map[cid])
        except (TypeError, ValueError, IndexError):
            continue
    return None


@st.fragment
def region_map_fragment(geojson, region_stats, geo_id_map, sel_code):
    with st.container(border=True):
        st.markdown('<p class="section-heading">Morocco Poverty Choropleth</p>', unsafe_allow_html=True)
        st.markdown(
            '<p class="plot-desc">Click any administrative region to focus downstream panels. '
            "Shading reflects household-weighted poverty rate; the amber outline marks your selection.</p>",
            unsafe_allow_html=True,
        )
        map_fig = fig_choropleth(geojson, region_stats, GEOJSON_REGIONS, selected_code=sel_code)
        map_event = st.plotly_chart(
            map_fig,
            use_container_width=True,
            height=CHART_H_TALL + 80,
            on_select="rerun",
            key="morocco_map",
            selection_mode="points",
        )
        clicked = parse_map_selection(map_event, geo_id_map)
        if clicked is not None and clicked != int(sel_code):
            st.session_state.selected_region_code = clicked
            st.rerun(scope="fragment")


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
    geo_id_map = build_geo_id_map(geojson, GEOJSON_REGIONS)
    get_hypernet_engine(rgph)
    return encdm, rgph, geojson, label_maps, code_maps, region_names, deps_encdm, bundles, geo_id_map


@st.cache_data(show_spinner="Loading raw survey files…")
def get_raw_data():
    return load_raw_encdm(), load_raw_rgph()


encdm, rgph, geojson, label_maps, code_maps, region_names, deps_encdm, bundles, geo_id_map = get_data()
region_stats = compute_region_stats(encdm, region_names)
w_pov = round(weighted_poverty_rate(encdm), 2)

# --- SIDEBAR ---
with st.sidebar:
    st.markdown(
        '<div class="sidebar-brand">HCP <span>Intelligence</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="sidebar-tagline">Morocco poverty &amp; vulnerability analytics '
        "fusing ENCDM survey microdata with RGPH census structure.</p>",
        unsafe_allow_html=True,
    )
    st.markdown('<p class="sidebar-nav-title">Navigation</p>', unsafe_allow_html=True)
    nav = st.radio("Navigation", list(SIDEBAR_BLURBS.keys()), label_visibility="collapsed")
    st.markdown(f'<p class="sidebar-desc">{SIDEBAR_BLURBS[nav]}</p>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<p class="sidebar-nav-title">Companion Project</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sidebar-desc"><strong>Simpute</strong> (Smart Imputation) — adaptive per-column '
        'imputation library born from this preprocessing phase. '
        '<a href="https://pypi.org/project/simpute/" target="_blank" style="color:#fca311;">PyPI</a></p>',
        unsafe_allow_html=True,
    )
    st.markdown("---")
    hyper_engine = get_hypernet_engine(rgph)
    model_note = "LightGBM + Hypernet ✓" if hyper_engine._ready else "LightGBM only"
    st.markdown(
        f'<p class="sidebar-desc">Models: {model_note}<br>'
        f'{datetime.now().strftime("%d %b %Y, %H:%M")}</p>',
        unsafe_allow_html=True,
    )


# ================================================================
# 1 · DATA INTEGRITY
# ================================================================
if nav == "Data Integrity":
    st.markdown('<h1 class="main-title">Data Integrity</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Dataset Overview & Quality Assessment</p>', unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown('<p class="section-heading">Platform Overview</p>', unsafe_allow_html=True)
        st.markdown(
            '<p class="prose-block">This dashboard operationalizes official <strong>Haut-Commissariat au Plan</strong> '
            "microdata: the ENCDM household consumption survey (2019–2020) and the RGPH census (2014). "
            "The analytical stack moves from raw <code>.sav</code> ingestion through adaptive imputation, "
            "feature scaling, dual-model inference, and spatial visualization — with every rate respecting "
            "inverse-scaled household weights (<code>coef_ménage</code>).</p>",
            unsafe_allow_html=True,
        )
        k1, k2, k3 = st.columns(3)
        k1.metric("Weighted Poverty Rate", f"{w_pov:.1f}%")
        k2.metric("ENCDM Households", f"{len(encdm):,}")
        k3.metric("RGPH Households", f"{len(rgph):,}")

    with st.container(border=True):
        st.markdown('<p class="section-heading">Simpute — Smart Imputation (Companion Project)</p>', unsafe_allow_html=True)
        st.markdown(
            '<p class="prose-block">Before modeling, both datasets required extensive imputation — expenditure '
            "quintiles, profession codes, dwelling characteristics, and infrastructure flags arrived with "
            "heterogeneous missingness patterns. The internship pipeline originally coupled KNN imputers for "
            "ENCDM and LightGBM imputers for RGPH to explicit dependency graphs.</p>"
            '<p class="prose-block">That logic evolved into <strong>Simpute</strong> (<em>Sim</em> + <em>impute</em>): '
            "a standalone sklearn-compatible library that profiles each column (type, cardinality, missingness, "
            "distribution shape) and routes it to the best backend — CatBoost, LightGBM, KNN, or Bayesian Ridge. "
            "Sequential column-wise imputation lets earlier fills inform later ones. Published on "
            '<a href="https://pypi.org/project/simpute/">PyPI as simpute</a> · '
            '<a href="https://github.com/Hvllvix/Simpute">GitHub</a>.</p>',
            unsafe_allow_html=True,
        )

    with st.container(border=True):
        st.markdown('<p class="section-heading">Repository Structure</p>', unsafe_allow_html=True)
        st.markdown(
            '<p class="plot-desc">High-level layout of this internship repository (core paths only).</p>',
            unsafe_allow_html=True,
        )
        st.markdown(f'<pre class="repo-tree">{PROJECT_TREE}</pre>', unsafe_allow_html=True)

    raw_encdm, raw_rgph = get_raw_data()
    plot_block(
        "Raw Missingness — Before Imputation",
        "Null rates in the original HCP .sav deliverables. This is the baseline completeness picture "
        "that motivated Simpute and the persisted KNN/LightGBM imputers — not the cleaned parquet.",
        fig_raw_missing_values(raw_encdm, raw_rgph),
        chart_height=CHART_H_TALL + 40,
    )

    plot_row([
        (
            "Weighted Poverty Split",
            "National household-weighted share of poor, vulnerable, and non-poor statuses after cleaning.",
            fig_poverty_breakdown(encdm),
        ),
        (
            "Vulnerability Split",
            "Complementary view of vulnerability classification weighted by survey design.",
            fig_vulnerability_breakdown(encdm),
        ),
    ])

    plot_row([
        (
            "National Education Structure",
            "Weighted education distribution — a structural socioeconomic indicator.",
            fig_national_education(encdm, label_maps),
        ),
        (
            "National Employment Mix",
            "Top employment categories by weighted household count across Morocco.",
            fig_national_employment(encdm, label_maps),
        ),
    ], chart_height=CHART_H_TALL)

    plot_row([
        (
            "Household Size Distribution",
            "Inverse-scaled ENCDM household sizes — demographic structure independent of poverty labels.",
            fig_household_size_national(encdm),
        ),
        (
            "RGPH Asset Access (National)",
            "Census-wide rates of electricity, internet, vehicles, and refrigeration.",
            fig_rgph_infrastructure(rgph),
        ),
    ])

    with st.container(border=True):
        st.markdown('<p class="section-heading">Imputation Dependencies (ENCDM)</p>', unsafe_allow_html=True)
        st.markdown(
            '<p class="plot-desc">Each target column is imputed only from theoretically related predictors — '
            "the same schema Simpute generalizes into adaptive routing.</p>",
            unsafe_allow_html=True,
        )
        dep_lines = []
        for target, deps in deps_encdm.items():
            if isinstance(deps, list) and deps:
                dep_lines.append(f"`{target}` ← {', '.join(str(d) for d in deps[:4])}")
        st.markdown("\n\n".join(dep_lines[:8]) or "No dependency graph loaded.")

    st.markdown('<p class="card-eyebrow">Clean Data Preview</p>', unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["ENCDM Household Survey", "RGPH Census"])
    preview_cols = [
        "Région_12", "Milieu", "Sexe_CM", "Age_CM", "Niveau_scolaire_agreg_CM",
        "Taille_ménage", "Pauvre", "Vulnérable",
    ]
    with tab1:
        with st.container(border=True):
            prev = inverse_scale_encdm(encdm[preview_cols].head(3000).copy())
            for col in ["Région_12", "Milieu", "Sexe_CM", "Niveau_scolaire_agreg_CM"]:
                prev[col] = translate(prev[col], col, label_maps)
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
# 2 · REGIONAL ANALYTICS
# ================================================================
elif nav == "Regional Analytics":
    st.markdown('<h1 class="main-title">Regional Analytics</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Spatial Socioeconomic Profiles Across Morocco</p>', unsafe_allow_html=True)

    st.markdown(
        '<p class="prose-block">Morocco\'s twelve administrative regions exhibit sharp heterogeneity in poverty, '
        "urbanization, education, and housing quality. Use the interactive choropleth below to select a region — "
        "three localized diagnostic plots and comparative panels update automatically.</p>",
        unsafe_allow_html=True,
    )

    sel_code = int(st.session_state.selected_region_code)
    sel = region_stats.get(sel_code, next(iter(region_stats.values())))
    sel_name = sel["name"]

    region_map_fragment(geojson, region_stats, geo_id_map, sel_code)

    st.markdown(
        f'<p class="plot-desc" style="margin-top:0.5rem;"><strong>Selected:</strong> {sel_name}</p>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div class="kpi-strip">'
        f'<div class="kpi-cell"><div class="metric-label">Poverty Rate</div>'
        f'<div class="metric-value accent">{sel["poverty_rate"]}%</div></div>'
        f'<div class="kpi-cell"><div class="metric-label">Households</div>'
        f'<div class="metric-value">{sel["households"]:,}</div></div>'
        f'<div class="kpi-cell"><div class="metric-label">Urban Share</div>'
        f'<div class="metric-value">{sel["urban_pct"]}%</div></div>'
        f'<div class="kpi-cell"><div class="metric-label">Avg Age</div>'
        f'<div class="metric-value">{sel["avg_age"]}</div></div>'
        f'<div class="kpi-cell"><div class="metric-label">Avg HH Size</div>'
        f'<div class="metric-value">{sel["avg_size"]}</div></div>'
        f"</div>",
        unsafe_allow_html=True,
    )

    st.markdown(f'<p class="card-eyebrow">Regional Profile — {sel_name}</p>', unsafe_allow_html=True)
    plot_row([
        (
            "Age Distribution",
            f"Individual ages (inverse-scaled) for respondents in {sel_name}.",
            fig_age_distribution(encdm, sel_code),
        ),
        (
            "Education Structure",
            f"Weighted education levels in {sel_name} — structural composition.",
            fig_region_education(encdm, sel_code, label_maps),
        ),
        (
            "Amenity Access (RGPH)",
            f"Census-based electricity, water, and internet rates in {sel_name}.",
            fig_region_amenities(rgph, sel_code),
        ),
    ])

    plot_row([
        (
            "Urban / Rural Poverty",
            "Weighted poverty rates by milieu across all twelve regions.",
            fig_urban_rural_poverty(encdm, region_names),
        ),
        (
            "Poverty Rate by Region",
            "National ranking of regions by household-weighted poverty.",
            fig_region_poverty(region_stats),
        ),
    ], chart_height=CHART_H_TALL)

    plot_row([
        (
            "Employment Mix",
            f"Weighted employment-status composition in {sel_name}.",
            fig_employment_mix(encdm, sel_code, label_maps),
        ),
        (
            "Urban–Rural Composition",
            f"Survey-weighted urban vs rural household share in {sel_name}.",
            fig_milieu_split(encdm, sel_code),
        ),
        (
            "Household Size",
            f"Distribution of household sizes in {sel_name}.",
            fig_region_household_size(encdm, sel_code),
        ),
    ])

    plot_row([
        (
            "Gender Representation",
            f"Male vs female household-head weighted shares in {sel_name}.",
            fig_region_gender(encdm, sel_code),
        ),
        (
            "Housing — Room Count",
            f"RGPH rooms-per-dwelling distribution in {sel_name}.",
            fig_rgph_rooms(rgph, sel_code),
        ),
    ])

    st.markdown('<p class="card-eyebrow">Imputation Schema Networks</p>', unsafe_allow_html=True)
    n1, n2 = st.columns(2)
    with n1:
        with st.container(border=True):
            st.markdown('<p class="section-heading">ENCDM Dependencies</p>', unsafe_allow_html=True)
            st.markdown(
                '<p class="plot-desc">KNN imputation routing for survey variables. '
                "Focus shows only the selected node and its direct neighbors.</p>",
                unsafe_allow_html=True,
            )
            encdm_net = build_encdm_network()
            encdm_ids = encdm_net[1] if encdm_net else []
            encdm_opts = ["(overview)"] + encdm_ids
            enc_idx = encdm_opts.index(st.session_state.encdm_node) if st.session_state.encdm_node in encdm_opts else 0
            choice = st.selectbox("Focus node (ENCDM)", encdm_opts, index=enc_idx, key="encdm_node_sel")
            st.session_state.encdm_node = None if choice == "(overview)" else choice
            encdm_focused = build_encdm_network(highlight_node=st.session_state.encdm_node)
            if encdm_focused:
                st.plotly_chart(encdm_focused[0], use_container_width=True, height=CHART_H_TALL)
    with n2:
        with st.container(border=True):
            st.markdown('<p class="section-heading">RGPH Dependencies</p>', unsafe_allow_html=True)
            st.markdown(
                '<p class="plot-desc">LightGBM imputation routing for census variables. '
                "Focus shows only the selected node and its direct neighbors.</p>",
                unsafe_allow_html=True,
            )
            rgph_net = build_rgph_network()
            rgph_ids = rgph_net[1] if rgph_net else []
            rgph_opts = ["(overview)"] + rgph_ids
            rg_idx = rgph_opts.index(st.session_state.rgph_node) if st.session_state.rgph_node in rgph_opts else 0
            choice = st.selectbox("Focus node (RGPH)", rgph_opts, index=rg_idx, key="rgph_node_sel")
            st.session_state.rgph_node = None if choice == "(overview)" else choice
            rgph_focused = build_rgph_network(highlight_node=st.session_state.rgph_node)
            if rgph_focused:
                st.plotly_chart(rgph_focused[0], use_container_width=True, height=CHART_H_TALL)


# ================================================================
# 3 · PREDICTIVE ENGINE
# ================================================================
elif nav == "Predictive Engine":
    st.markdown('<h1 class="main-title">Predictive Engine</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Dual-Model Inference · LightGBM & Hypernetwork</p>', unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown('<p class="section-heading">What the Inference Sandbox Does</p>', unsafe_allow_html=True)
        st.markdown(
            '<p class="prose-block">The predictive engine answers a policy-facing question: <em>given a household\'s '
            "demographic and geographic profile, what is the estimated probability of poverty and vulnerability?</em> "
            "When you submit the form, the backend executes a reproducible pipeline:</p>"
            "<ol class='prose-block'>"
            "<li><strong>Encode</strong> — human-readable labels (region, milieu, education…) map to numeric survey codes.</li>"
            "<li><strong>Scale</strong> — persisted <code>StandardScaler</code> objects transform age and household size "
            "to the same units used during training.</li>"
            "<li><strong>LightGBM inference</strong> — two gradient-boosted classifiers (<code>Pauvre</code>, "
            "<code>Vulnérable</code>) return calibrated probabilities from their learned feature space.</li>"
            "<li><strong>Hypernetwork inference</strong> — a geography-conditioned deep model generates context-specific "
            "weights from RGPH strata embeddings, then predicts the same two targets through a dynamically parameterized network.</li>"
            "<li><strong>Explain</strong> — global feature importances visualize which inputs most influenced the LightGBM poverty estimate.</li>"
            "</ol>",
            unsafe_allow_html=True,
        )

    with st.container(border=True):
        st.markdown('<p class="section-heading">Hypernetwork Architecture (<code>hypernet.py</code>)</p>', unsafe_allow_html=True)
        st.markdown(
            '<p class="prose-block">The hypernetwork is the most complex artifact in this repository. Unlike LightGBM — '
            "which learns one fixed set of split rules — the hypernet <strong>generates a fresh target-network "
            "for every geographic stratum</strong>. Its design has four layers of complexity:</p>"
            "<ul class='prose-block'>"
            "<li><strong>Dual embeddings</strong> — separate <code>MultiEmbedding</code> modules encode RGPH census categoricals "
            "and ENCDM survey categoricals into dense vectors, preserving ordinal structure without one-hot explosion.</li>"
            "<li><strong>Strata context</strong> — for each Region × Milieu pair, RGPH micro-records are aggregated into a "
            "representative housing-quality context (modal categories + mean numerics). This is the \"macro\" signal.</li>"
            "<li><strong>Weight generation</strong> — a hypernetwork MLP reads the RGPH embedding and outputs all parameters "
            "(W₁, b₁, W₂, b₂) of a small feed-forward <em>target network</em> via <code>FunctionalTargetNet</code>.</li>"
            "<li><strong>Dynamic forward pass</strong> — household ENCDM features pass through the generated weights to produce "
            "poverty and vulnerability logits, sigmoid-transformed into probabilities. The optional <em>rural transfer</em> "
            "counterfactual swaps the RGPH stratum to rural while holding household traits fixed.</li>"
            "</ul>"
            '<p class="prose-block">This architecture captures spatial heterogeneity: the same household profile can yield '
            "different risk estimates when the underlying regional housing context changes — mirroring how infrastructure "
            "and living conditions modulate poverty dynamics across Morocco.</p>",
            unsafe_allow_html=True,
        )

    plot_row([
        (
            "Feature Importance — Poverty",
            "Global LightGBM gain-based importances for the Pauvre classifier.",
            fig_feature_importance(bundles, "Pauvre"),
        ),
        (
            "Feature Importance — Vulnerability",
            "Global LightGBM gain-based importances for the Vulnérable classifier.",
            fig_feature_importance(bundles, "Vulnérable"),
        ),
    ], chart_height=CHART_H_TALL)

    plot_row([
        (
            "ROC Curves — LightGBM",
            "Discrimination on a held sample; curves are monotonicized for stable visualization.",
            fig_roc_curves(encdm, bundles),
        ),
        (
            "Hypernetwork Training Loss",
            "Loss trajectory stored in the Hypernet.pt checkpoint across training epochs.",
            fig_hypernet_loss(),
        ),
    ])

    st.markdown('<p class="card-eyebrow">Scenario Simulation</p>', unsafe_allow_html=True)

    region_labels = list(code_maps["Région_12"].keys())
    milieu_labels = list(code_maps["Milieu"].keys())
    sexe_labels = list(code_maps["Sexe_CM"].keys())
    edu_labels = list(code_maps["Niveau_scolaire_agreg_CM"].keys())
    sit_labels = list(code_maps["Situation_profession_agreg_CM"].keys())
    taille_agg_labels = list(code_maps["Taille_agregée"].keys())

    with st.container(border=True):
        st.markdown('<p class="section-heading">Household Parameters</p>', unsafe_allow_html=True)
        st.markdown(
            '<p class="plot-desc">Set region, milieu, demographics, and employment. '
            "Rural transfer re-conditions the hypernetwork on a rural RGPH stratum for spatial counterfactuals.</p>",
            unsafe_allow_html=True,
        )
        with st.form("inference_form"):
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
        try:
            results = run_dual_prediction(form_inputs, code_maps, rgph, rural_transfer=rural_transfer)
            lgbm, hyper = results["lgbm"], results["hypernet"]

            with st.container(border=True):
                st.markdown('<p class="section-heading">Dual Inference Results</p>', unsafe_allow_html=True)
                st.markdown(
                    '<p class="plot-desc">Probabilities below are <em>estimates for policy research</em> — '
                    "not individual classifications. Thresholds are learned from training artifacts.</p>",
                    unsafe_allow_html=True,
                )
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("LGBM · Poverty", f"{lgbm['Pauvre']['probability']:.1%}")
                c2.metric("LGBM · Vulnerability", f"{lgbm['Vulnérable']['probability']:.1%}")
                hpp = hyper["Pauvre"]["probability"]
                hvp = hyper["Vulnérable"]["probability"]
                c3.metric("Hypernet · Poverty", f"{hpp:.1%}" if hpp is not None else "N/A")
                c4.metric("Hypernet · Vulnerability", f"{hvp:.1%}" if hvp is not None else "N/A")

            plot_row([
                (
                    "Model Comparison",
                    "Side-by-side poverty and vulnerability probabilities from both engines.",
                    fig_dual_comparison(lgbm, hyper),
                ),
                (
                    "Feature Contributions (LGBM)",
                    "Relative importance of top features for the poverty model on this profile.",
                    fig_contribution_waterfall(results["contributions"]),
                ),
            ])

            with st.expander("Encoded feature vector (diagnostics)"):
                diag = pd.DataFrame(
                    {"Raw": results["feature_row"], "Scaled (LGBM)": results["scaled_row"]}
                ).T
                st.dataframe(diag, use_container_width=True)

            if rural_transfer:
                st.info(
                    "Rural transfer shifts the Hypernetwork RGPH context to a rural stratum "
                    "while holding household characteristics constant — useful for spatial counterfactuals."
                )
        except Exception as exc:
            st.error(f"Inference failed: {exc}")
    else:
        st.info("Configure household characteristics above and run dual inference to see live model outputs.")


st.markdown(
    """
<div class="footer">
    <p><b>MOROCCO SOCIOECONOMIC INTELLIGENCE PLATFORM</b></p>
    <p>HCP Data Fusion · ENCDM 2019–2020 · RGPH 2014 · LightGBM + Hypernetwork · Simpute</p>
    <p style="font-size:0.65rem;opacity:0.75;">Models are probabilistic estimates for policy research — not individual classifications.</p>
</div>
""",
    unsafe_allow_html=True,
)
