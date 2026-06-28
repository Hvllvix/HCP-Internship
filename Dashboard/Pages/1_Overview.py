"""Overview page module."""
import streamlit as st

from Utils.plots import (
    CHART_H_TALL,
    fig_dataset_dims,
    fig_null_counts,
    fig_raw_missing_values,
)
from Utils.utils import dep_cards, pkg_cards, render_mermaid, render_tree
from Utils.theme import metric_row
from Utils.data_loader import inverse_scale_encdm, translate, get_label
from Utils.sandbox import run_dual_prediction


def render(encdm, rgph, raw_encdm, raw_rgph, nullreport, wpov):
    st.markdown('<h1 class="main-title">Overview</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Dataset Metadata and Pre-Processing Profile</p>', unsafe_allow_html=True)

    raw_e = nullreport["encdm_raw"]
    raw_r = nullreport["rgph_raw"]

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

    # Side-by-side: Tree + Git Graph
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        st.markdown('<p class="section-heading">Repository Structure</p>', unsafe_allow_html=True)
        with st.expander("View Repository Structure", expanded=False):
            render_tree()
    with c2:
        st.markdown('<p class="section-heading">Git Graph</p>', unsafe_allow_html=True)
        try:
            from pathlib import Path
            dot_text = Path("Others/git_graph.txt").read_text(encoding="utf-8")
            st.graphviz_chart(dot_text)
        except Exception:
            st.info("Git graph unavailable")

    from Utils.utils import plot_row, plot_block
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
        from Utils.data_loader import load_deps_encdm
        deps = load_deps_encdm()
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
    from Utils.data_loader import build_label_maps
    labels = build_label_maps()
    with tab1:
        with st.container(border=True):
            prev = inverse_scale_encdm(encdm[preview].head(3000).copy())
            for col in ["Région_12", "Milieu", "Sexe_CM", "Niveau_scolaire_agreg_CM"]:
                prev[col] = translate(prev[col], col, labels)
            prev["Pauvre"] = prev["Pauvre"].map({1: "Poor", 0: "Non-Poor"})
            prev["Vulnérable"] = prev["Vulnérable"].map({1: "Vulnerable", 0: "Not Vulnerable"})
            prev.columns = [get_label(c) for c in prev.columns]
            st.dataframe(prev, width='stretch', height=360)
    with tab2:
        with st.container(border=True):
            cols2 = ["REG", "MIL", "TAILLE", "PIECES", "ELEC", "NET", "EAU.MODE"]
            prev2 = rgph[cols2].head(3000).copy()
            prev2["MIL"] = prev2["MIL"].map({0: "Urban", 1: "Rural"})
            prev2["ELEC"] = prev2["ELEC"].map({0: "No", 1: "Yes"})
            prev2["NET"] = prev2["NET"].map({0: "No", 1: "Yes"})
            prev2.columns = [get_label(c) for c in prev2.columns]
            st.dataframe(prev2, width='stretch', height=360)
