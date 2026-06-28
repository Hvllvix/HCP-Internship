"""Predictive Engine page module."""
import pandas as pd
import streamlit as st

from Utils.plots import (
    CHART_H_TALL,
    fig_contribution_waterfall,
    fig_dual_comparison,
    fig_feature_importance,
    fig_hypernet_loss,
    fig_roc_curves,
)
from Utils.sandbox import InferenceError, run_dual_prediction
from Utils.theme import metric_row
from Utils.utils import plot_row
from Utils.data_loader import load_all_lgbm, load_scalers_encdm, load_encdm


def render(encdm, rgph, codes, bundles):
    st.markdown('<h1 class="main-title">Predictive Engine</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Dual-Model Inference · LightGBM and Hypernetwork</p>', unsafe_allow_html=True)

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
            go_btn = st.form_submit_button("Run dual inference", width='stretch')

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

        with st.expander("Encoded feature vector", expanded=False):
            diag = pd.DataFrame({"Raw": res["feature_row"], "Scaled": res["scaled_row"]}).T
            st.dataframe(diag, width='stretch')

        if st.session_state.rural_xfer:
            st.info("Rural transfer: Hypernetwork RGPH context shifted to rural stratum.")
    elif not go_btn:
        st.info("Configure household characteristics and run dual inference.")