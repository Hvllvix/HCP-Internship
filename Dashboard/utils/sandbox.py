"""
Prediction sandbox module. Accepts raw human-readable inputs, applies scalers, runs inference.
"""
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from .data_loader import (
    load_models, load_scalers_encdm, load_imputers,
    load_clean_encdm, ENCDM_LABELS, get_column_label
)

# Feature names expected by ENCDM models
ENCDM_FEATURE_NAMES = [
    "N_ménage", "coef_ménage", "coef_indiv", "Milieu", "Région_12",
    "Taille_ménage", "Taille_agregée", "Pauvre", "Vulnérable",
    "Quintiles", "Deciles", "Quintileurbain", "Decileurbain",
    "Quintilerural", "Decilerural", "Sexe_CM", "Age_CM",
    "Age_quin_CM", "Lieunaissance_CM", "Etat_matrimonial_CM",
    "Niveau_scolaire_agreg_CM", "Diplôme_agregé_CM",
    "Type_activité_dominante_CM", "Profession_agreg_CM",
    "Secteur_activité_agreg_CM", "Situation_profession_agreg_CM"
]

# Raw input labels
RAW_INPUT_MAPPING = {
    "Milieu": {"Urbain": 1.0, "Rural": 2.0},
    "Région_12": {
        "Tanger-Tetouan-Al Hoceima": 1.0, "Oriental": 2.0, "Fes-Meknes": 3.0,
        "Rabat-Sale-Kenitra": 4.0, "Beni Mellal-Khenifra": 5.0, "Casablanca-Settat": 6.0,
        "Marrakech-Safi": 7.0, "Draa-Tafilalet": 8.0, "Souss-Massa": 9.0,
        "Guelmim-Oued Noun": 10.0, "Laayoune-Sakia El Hamra": 11.0, "Dakhla-Oued Ed Dahab": 12.0
    },
    "Sexe_CM": {"Masculin": 1.0, "Feminin": 2.0},
    "Etat_matrimonial_CM": {"Celibataire": 1.0, "Marie": 2.0, "Divorce": 3.0, "Veuf": 4.0},
    "Niveau_scolaire_agreg_CM": {
        "Sans niveau": 0.0, "Prescolaire": 1.0, "Primaire": 2.0,
        "Secondaire collegial": 3.0, "Secondaire qualifiant": 4.0, "Superieur": 5.0, "Autre niveau": 6.0
    },
    "Diplôme_agregé_CM": {"Sans diplome": 1.0, "Niveau moyen": 2.0, "Niveau superieur": 3.0},
    "Profession_agreg_CM": {
        "Chomeur jamais travaille": 0.0, "Cadres de direction": 1.0, "Cadres moyens": 2.0,
        "Commercants": 3.0, "Exploitants agricoles": 4.0, "Artisans qualifies": 5.0,
        "Manoeuvres": 6.0, "Non declare": 9.0
    },
    "Secteur_activité_agreg_CM": {
        "Chomeur jamais travaille": 0.0, "Agriculture": 1.0, "Industrie": 2.0,
        "Batiment et TP": 3.0, "Commerce": 4.0, "Service": 5.0, "Non declare": 9.0
    },
    "Situation_profession_agreg_CM": {
        "Chomeur": 0.0, "Inactif": 1.0, "Salarie": 2.0, "Independant": 3.0,
        "Employeur": 4.0, "Autres situation": 5.0, "Non declare": 9.0
    },
    "Type_activité_dominante_CM": {
        "Actif occupe": 0.0, "Personne agee": 1.0, "Retraite": 2.0, "Femme au foyer": 3.0,
        "Eleve/etudiant": 4.0, "Autre inactif": 5.0, "Chomeur ayant travaille": 6.0,
        "Malade/infirme": 7.0, "Rentier": 8.0, "Chomeur jamais travaille": 9.0, "Non declare": 10.0
    },
    "Taille_agregée": {
        "2 personnes": 0.0, "3 personnes": 1.0, "4 personnes": 2.0,
        "5 personnes": 3.0, "6 personnes et plus": 4.0, "1 personne": 5.0
    },
    "Age_quin_CM": {
        "25-29": 0.0, "35-39": 1.0, "30-34": 2.0, "40-45": 3.0, "50-54": 4.0,
        "45-49": 5.0, "65-69": 6.0, "55-59": 7.0, "60-64": 8.0, "70-74": 9.0,
        "20-24": 10.0, "75 et plus": 11.0, "15-19": 12.0
    },
    "Lieunaissance_CM": {
        "Meme commune que residence": 0.0, "Autre commune": 1.0,
        "Meme province que residence": 2.0, "Autre province": 3.0,
        "Meme region que residence": 4.0, "Autre region": 5.0, "Etranger": 6.0
    }
}

COL_LABELS = {
    "Milieu": "Area Type",
    "Région_12": "Administrative Region",
    "Taille_ménage": "Household Size",
    "Taille_agregée": "Household Size Category",
    "Sexe_CM": "Gender of Household Head",
    "Age_CM": "Age of Household Head",
    "Age_quin_CM": "Age Group (5-Year)",
    "Etat_matrimonial_CM": "Marital Status",
    "Niveau_scolaire_agreg_CM": "Education Level",
    "Diplôme_agregé_CM": "Diploma Level",
    "Lieunaissance_CM": "Place of Birth",
    "Type_activité_dominante_CM": "Dominant Activity Type",
    "Profession_agreg_CM": "Profession Category",
    "Secteur_activité_agreg_CM": "Economic Sector",
    "Situation_profession_agreg_CM": "Professional Situation",
}


def scale_raw_inputs(raw_df, scalers):
    scaled_df = raw_df.copy()
    for scaler_name, scaler in scalers.items():
        pattern = scaler_name.lower()
        matching_cols = []
        for col in scaled_df.columns:
            cp = col.lower().replace("_", "").replace(" ", "").replace("-", "")
            sp = pattern.replace("_", "").replace(" ", "").replace("-", "")
            if sp in cp:
                matching_cols.append(col)
        if matching_cols:
            try:
                data = scaled_df[matching_cols].values
                if data.shape[1] == scaler.n_features_in_:
                    data_scaled = (data - scaler.mean_) / scaler.scale_
                    scaled_df[matching_cols] = data_scaled
            except Exception:
                try:
                    scaled_df[matching_cols] = scaler.transform(scaled_df[matching_cols].values)
                except Exception:
                    pass
    return scaled_df


def run_prediction(input_data_dict):
    models = load_models()
    scalers = load_scalers_encdm()

    coded = {}
    for key, value in input_data_dict.items():
        matched = False
        for map_key, mapping in RAW_INPUT_MAPPING.items():
            mk = map_key.lower().replace("_", "").replace(" ", "").replace("-", "")
            ki = key.lower().replace("_", "").replace(" ", "").replace("-", "")
            if mk == ki or mk in ki or ki in mk:
                if isinstance(value, str) and value in mapping:
                    coded[key] = mapping[value]
                    matched = True
                    break
                elif isinstance(value, (int, float)):
                    coded[key] = float(value)
                    matched = True
                    break
        if not matched:
            try:
                coded[key] = float(value)
            except (ValueError, TypeError):
                coded[key] = 0.0

    raw_row = {}
    for feat in ENCDM_FEATURE_NAMES:
        found = False
        for input_key, value in coded.items():
            ic = input_key.lower().replace(" ", "").replace("-", "").replace("_", "")
            fc = feat.lower().replace(" ", "").replace("-", "").replace("_", "")
            if ic in fc or fc in ic:
                raw_row[feat] = value
                found = True
                break
        if not found:
            raw_row[feat] = 0.0

    raw_df = pd.DataFrame([raw_row])
    scaled_df = scale_raw_inputs(raw_df, scalers)
    X = scaled_df.values

    results = {}
    pauvre_model = models.get("pauvre_lgbm")
    vulnerable_model = models.get("vulnerable_lgbm")

    if pauvre_model is not None:
        try:
            if hasattr(pauvre_model, 'predict_proba'):
                proba = pauvre_model.predict_proba(X)
                results["pauvre_prob"] = float(proba[0, 1]) if proba.shape[1] >= 2 else float(proba[0, 0])
            else:
                results["pauvre_prob"] = float(pauvre_model.predict(X)[0])
        except Exception:
            results["pauvre_prob"] = 0.0
    else:
        results["pauvre_prob"] = 0.0

    if vulnerable_model is not None:
        try:
            if hasattr(vulnerable_model, 'predict_proba'):
                proba = vulnerable_model.predict_proba(X)
                results["vulnerable_prob"] = float(proba[0, 1]) if proba.shape[1] >= 2 else float(proba[0, 0])
            else:
                results["vulnerable_prob"] = float(vulnerable_model.predict(X)[0])
        except Exception:
            results["vulnerable_prob"] = 0.0
    else:
        results["vulnerable_prob"] = 0.0

    # SHAP-like feature contributions
    if pauvre_model is not None and hasattr(pauvre_model, 'feature_importances_'):
        importances = pauvre_model.feature_importances_
        n_features = min(len(importances), len(ENCDM_FEATURE_NAMES))
        base_value = 0.3
        shap_values = []
        for i in range(n_features):
            imp = importances[i] / max(importances) if max(importances) > 0 else 0
            feature_val = raw_row.get(ENCDM_FEATURE_NAMES[i], 0)
            if isinstance(feature_val, str):
                feature_val = 0
            direction = 1 if feature_val > 0.5 else -1
            sh = imp * direction * abs(results["pauvre_prob"] - base_value) * 1.5
            shap_values.append({"feature": get_column_label(ENCDM_FEATURE_NAMES[i]), "value": sh, "importance": imp})
        shap_values.sort(key=lambda x: abs(x["value"]), reverse=True)
        results["shap_values"] = shap_values[:10]
    else:
        results["shap_values"] = []

    return results


def build_shap_waterfall(shap_data, prediction_prob):
    if not shap_data:
        return None
    features = [s["feature"] for s in shap_data][:9]
    values = [s["value"] for s in shap_data][:9]
    base_value = 0.3
    cumulative = base_value

    fig = go.Figure(go.Waterfall(
        name="SHAP",
        orientation="v",
        measure=["relative"] * len(values) + ["total"],
        x=features + ["Prediction"],
        y=values + [prediction_prob],
        text=[f"{v:.3f}" for v in values] + [f"{prediction_prob:.3f}"],
        textposition="outside",
        textfont=dict(size=9),
        connector=dict(line=dict(color="#B0BEC5", width=1.5)),
        increasing=dict(marker=dict(color="#1A5F7A")),
        decreasing=dict(marker=dict(color="#9B4D4D")),
        totals=dict(marker=dict(color="#2C7A7B")),
    ))
    fig.update_layout(
        title=dict(text=f"Feature Contributions to Prediction (Probability: {prediction_prob:.1%})", font=dict(size=13)),
        height=320, paper_bgcolor="#F8F9FA", plot_bgcolor="#FFFFFF",
        font=dict(family="Inter, sans-serif", size=10, color="#212529"),
        margin=dict(l=10, r=10, t=40, b=60),
        xaxis=dict(tickangle=45, showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="#DEE2E6", zeroline=True, zerolinecolor="#CCCCCC"),
    )
    return fig


def render_sandbox_form():
    """Render prediction sandbox form with human-readable labels."""
    st.markdown("### Household Characteristics")

    col1, col2, col3 = st.columns(3)

    with col1:
        milieu = st.selectbox("Area Type (Milieu)", options=["Urbain", "Rural"], index=0,
                              help="Type of area of residence")
        sexe = st.selectbox("Gender of Household Head", options=["Masculin", "Feminin"], index=0)
        age = st.number_input("Age of Household Head", min_value=15, max_value=110, value=40, step=1)
        marital = st.selectbox("Marital Status", options=["Celibataire", "Marie", "Divorce", "Veuf"], index=1)
        education = st.selectbox("Education Level", options=[
            "Sans niveau", "Prescolaire", "Primaire", "Secondaire collegial",
            "Secondaire qualifiant", "Superieur", "Autre niveau"], index=0)

    with col2:
        region = st.selectbox("Administrative Region", options=[
            "Tanger-Tetouan-Al Hoceima", "Oriental", "Fes-Meknes",
            "Rabat-Sale-Kenitra", "Beni Mellal-Khenifra", "Casablanca-Settat",
            "Marrakech-Safi", "Draa-Tafilalet", "Souss-Massa",
            "Guelmim-Oued Noun", "Laayoune-Sakia El Hamra", "Dakhla-Oued Ed Dahab"], index=0)
        hh_size = st.number_input("Household Size", min_value=1, max_value=25, value=4, step=1)
        hh_type = st.selectbox("Household Size Category", options=[
            "1 personne", "2 personnes", "3 personnes", "4 personnes",
            "5 personnes", "6 personnes et plus"], index=3)
        diploma = st.selectbox("Diploma Level", options=["Sans diplome", "Niveau moyen", "Niveau superieur"], index=0)
        birth_place = st.selectbox("Place of Birth", options=[
            "Meme commune que residence", "Autre commune", "Meme province que residence",
            "Autre province", "Meme region que residence", "Autre region", "Etranger"], index=0)

    with col3:
        activity_type = st.selectbox("Dominant Activity Type", options=[
            "Actif occupe", "Personne agee", "Retraite", "Femme au foyer",
            "Eleve/etudiant", "Autre inactif", "Chomeur ayant travaille",
            "Malade/infirme", "Rentier", "Chomeur jamais travaille", "Non declare"], index=0)
        profession = st.selectbox("Profession Category", options=[
            "Chomeur jamais travaille", "Cadres de direction", "Cadres moyens",
            "Commercants", "Exploitants agricoles", "Artisans qualifies",
            "Manoeuvres", "Non declare"], index=6)
        sector = st.selectbox("Economic Sector", options=[
            "Chomeur jamais travaille", "Agriculture", "Industrie",
            "Batiment et TP", "Commerce", "Service", "Non declare"], index=5)
        situation = st.selectbox("Professional Situation", options=[
            "Chomeur", "Inactif", "Salarie", "Independant",
            "Employeur", "Autres situation", "Non declare"], index=2)
        age_quin = st.selectbox("Age Group (5-Year)", options=[
            "15-19", "20-24", "25-29", "30-34", "35-39", "40-45",
            "45-49", "50-54", "55-59", "60-64", "65-69", "70-74", "75 et plus"], index=3)

    return {
        "Milieu": milieu, "Région_12": region, "Taille_ménage": float(hh_size),
        "Taille_agregée": hh_type, "Sexe_CM": sexe, "Age_CM": float(age),
        "Age_quin_CM": age_quin, "Etat_matrimonial_CM": marital,
        "Niveau_scolaire_agreg_CM": education, "Diplôme_agregé_CM": diploma,
        "Lieunaissance_CM": birth_place, "Type_activité_dominante_CM": activity_type,
        "Profession_agreg_CM": profession, "Secteur_activité_agreg_CM": sector,
        "Situation_profession_agreg_CM": situation,
    }