"""
Dual-inference prediction sandbox: LightGBM + Hypernetwork with imputation pipeline.
"""
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path
from .data_loader import (
    load_models, load_scalers_encdm, load_imputers,
    load_clean_encdm, load_clean_rgph, get_column_label,
)
from .translations import REGION_NAME_TO_CODE
from .hypernet import get_hypernet_engine

ENCDM_FEATURE_NAMES = [
    "N_ménage", "coef_ménage", "coef_indiv", "Milieu", "Région_12",
    "Taille_ménage", "Taille_agregée", "Pauvre", "Vulnérable",
    "Quintiles", "Deciles", "Quintileurbain", "Decileurbain",
    "Quintilerural", "Decilerural", "Sexe_CM", "Age_CM",
    "Age_quin_CM", "Lieunaissance_CM", "Etat_matrimonial_CM",
    "Niveau_scolaire_agreg_CM", "Diplôme_agregé_CM",
    "Type_activité_dominante_CM", "Profession_agreg_CM",
    "Secteur_activité_agreg_CM", "Situation_profession_agreg_CM",
]

RAW_INPUT_MAPPING = {
    "Milieu": {"Urbain": 1.0, "Rural": 2.0},
    "Région_12": {
        "Tanger-Tetouan-Al Hoceima": 1.0, "Oriental": 2.0, "Fes-Meknes": 3.0,
        "Rabat-Sale-Kenitra": 4.0, "Beni Mellal-Khenifra": 5.0, "Casablanca-Settat": 6.0,
        "Marrakech-Safi": 7.0, "Draa-Tafilalet": 8.0, "Souss-Massa": 9.0,
        "Guelmim-Oued Noun": 10.0, "Laayoune-Sakia El Hamra": 11.0, "Dakhla-Oued Ed Dahab": 12.0,
    },
    "Sexe_CM": {"Masculin": 1.0, "Feminin": 2.0},
    "Etat_matrimonial_CM": {"Celibataire": 1.0, "Marie": 2.0, "Divorce": 3.0, "Veuf": 4.0},
    "Niveau_scolaire_agreg_CM": {
        "Sans niveau": 0.0, "Prescolaire": 1.0, "Primaire": 2.0,
        "Secondaire collegial": 3.0, "Secondaire qualifiant": 4.0, "Superieur": 5.0, "Autre niveau": 6.0,
    },
    "Diplôme_agregé_CM": {"Sans diplome": 1.0, "Niveau moyen": 2.0, "Niveau superieur": 3.0},
    "Profession_agreg_CM": {
        "Chomeur jamais travaille": 0.0, "Cadres de direction": 1.0, "Cadres moyens": 2.0,
        "Commercants": 3.0, "Exploitants agricoles": 4.0, "Artisans qualifies": 5.0,
        "Manoeuvres": 6.0, "Non declare": 9.0,
    },
    "Secteur_activité_agreg_CM": {
        "Chomeur jamais travaille": 0.0, "Agriculture": 1.0, "Industrie": 2.0,
        "Batiment et TP": 3.0, "Commerce": 4.0, "Service": 5.0, "Non declare": 9.0,
    },
    "Situation_profession_agreg_CM": {
        "Chomeur": 0.0, "Inactif": 1.0, "Salarie": 2.0, "Independant": 3.0,
        "Employeur": 4.0, "Autres situation": 5.0, "Non declare": 9.0,
    },
    "Type_activité_dominante_CM": {
        "Actif occupe": 0.0, "Personne agee": 1.0, "Retraite": 2.0, "Femme au foyer": 3.0,
        "Eleve/etudiant": 4.0, "Autre inactif": 5.0, "Chomeur ayant travaille": 6.0,
        "Malade/infirme": 7.0, "Rentier": 8.0, "Chomeur jamais travaille": 9.0, "Non declare": 10.0,
    },
    "Taille_agregée": {
        "2 personnes": 0.0, "3 personnes": 1.0, "4 personnes": 2.0,
        "5 personnes": 3.0, "6 personnes et plus": 4.0, "1 personne": 5.0,
    },
    "Age_quin_CM": {
        "25-29": 0.0, "35-39": 1.0, "30-34": 2.0, "40-45": 3.0, "50-54": 4.0,
        "45-49": 5.0, "65-69": 6.0, "55-59": 7.0, "60-64": 8.0, "70-74": 9.0,
        "20-24": 10.0, "75 et plus": 11.0, "15-19": 12.0,
    },
    "Lieunaissance_CM": {
        "Meme commune que residence": 0.0, "Autre commune": 1.0,
        "Meme province que residence": 2.0, "Autre province": 3.0,
        "Meme region que residence": 4.0, "Autre region": 5.0, "Etranger": 6.0,
    },
}


def _encode_inputs(input_data_dict):
    coded = {}
    for key, value in input_data_dict.items():
        matched = False
        for map_key, mapping in RAW_INPUT_MAPPING.items():
            mk = map_key.lower().replace("_", "").replace(" ", "").replace("-", "").replace("é", "e")
            ki = key.lower().replace("_", "").replace(" ", "").replace("-", "").replace("é", "e")
            if mk == ki or mk in ki or ki in mk:
                if isinstance(value, str) and value in mapping:
                    coded[key] = mapping[value]
                    matched = True
                    break
                if isinstance(value, (int, float)):
                    coded[key] = float(value)
                    matched = True
                    break
        if not matched:
            try:
                coded[key] = float(value)
            except (ValueError, TypeError):
                coded[key] = 0.0
    return coded


def _build_feature_row(coded):
    raw_row = {}
    for feat in ENCDM_FEATURE_NAMES:
        found = False
        for input_key, value in coded.items():
            ic = input_key.lower().replace(" ", "").replace("-", "").replace("_", "").replace("é", "e")
            fc = feat.lower().replace(" ", "").replace("-", "").replace("_", "").replace("é", "e")
            if ic in fc or fc in ic:
                raw_row[feat] = value
                found = True
                break
        if not found:
            raw_row[feat] = 0.0
    return raw_row


def _apply_imputers(raw_row, imputers):
    """Fill zero/missing slots using trained KNN imputers where available."""
    row = raw_row.copy()
    df_ref = load_clean_encdm()
    if df_ref is None:
        return row

    impute_targets = ["Quintileurbain", "Decileurbain", "Quintilerural", "Decilerural",
                      "Secteur_activité_agreg_CM", "Profession_agreg_CM", "Lieunaissance_CM"]
    for target in impute_targets:
        if row.get(target, 0) != 0:
            continue
        key = f"KNN_{target}"
        if key not in imputers:
            key = f"ENCDM/KNN_{target}"
        model = imputers.get(f"KNN_{target}") or imputers.get(key.split("/")[-1] if "/" in key else key)
        if model is None:
            for k, m in imputers.items():
                if target.replace("é", "e") in k or target in k:
                    model = m
                    break
        if model is None:
            continue
        try:
            feat_cols = [c for c in df_ref.columns if c in row and c != target][:8]
            if feat_cols:
                X_imp = pd.DataFrame([[row[c] for c in feat_cols]], columns=feat_cols)
                row[target] = float(model.predict(X_imp)[0])
        except Exception:
            pass
    return row


def scale_raw_inputs(raw_df, scalers):
    scaled_df = raw_df.copy()
    for scaler_name, scaler in scalers.items():
        matching_cols = []
        pattern = scaler_name.lower().replace("_", "").replace("é", "e")
        for col in scaled_df.columns:
            cp = col.lower().replace("_", "").replace(" ", "").replace("-", "").replace("é", "e")
            if pattern in cp or cp in pattern:
                matching_cols.append(col)
        if matching_cols:
            try:
                vals = scaled_df[matching_cols].values.astype(float)
                if vals.shape[1] == getattr(scaler, "n_features_in_", vals.shape[1]):
                    scaled_df[matching_cols] = scaler.transform(vals)
            except Exception:
                pass
    return scaled_df


def _lgbm_predict(models, X_df, feature_list=None):
    results = {"pauvre_prob": 0.0, "vulnerable_prob": 0.0}
    pauvre_model = models.get("pauvre_lgbm")
    vuln_model = models.get("vulnerable_lgbm")

    if pauvre_model is not None:
        try:
            feats = models.get("pauvre_lgbm_features", feature_list or [])
            X_use = X_df[feats] if feats and all(f in X_df.columns for f in feats) else X_df
            if hasattr(pauvre_model, "predict_proba"):
                proba = pauvre_model.predict_proba(X_use)
                results["pauvre_prob"] = float(proba[0, 1]) if proba.shape[1] >= 2 else float(proba[0, 0])
            else:
                results["pauvre_prob"] = float(pauvre_model.predict(X_use)[0])
        except Exception:
            pass

    if vuln_model is not None:
        try:
            feats = models.get("vulnerable_lgbm_features", feature_list or [])
            X_use = X_df[feats] if feats and all(f in X_df.columns for f in feats) else X_df
            if hasattr(vuln_model, "predict_proba"):
                proba = vuln_model.predict_proba(X_use)
                results["vulnerable_prob"] = float(proba[0, 1]) if proba.shape[1] >= 2 else float(proba[0, 0])
            else:
                results["vulnerable_prob"] = float(vuln_model.predict(X_use)[0])
        except Exception:
            pass
    return results


def run_dual_prediction(input_data_dict, rural_transfer=False):
    models = load_models()
    scalers = load_scalers_encdm()
    imputers = load_imputers()
    rgph_df = load_clean_rgph()

    coded = _encode_inputs(input_data_dict)
    raw_row = _build_feature_row(coded)
    raw_row = _apply_imputers(raw_row, imputers)

    raw_df = pd.DataFrame([raw_row])
    scaled_df = scale_raw_inputs(raw_df, scalers)

    feats = models.get("pauvre_lgbm_features", [])
    if feats:
        row_scaled = {}
        for f in feats:
            if f in scaled_df.columns:
                row_scaled[f] = scaled_df[f].iloc[0]
            else:
                for col in scaled_df.columns:
                    if col.lower().replace("é", "e") == f.lower().replace("é", "e"):
                        row_scaled[f] = scaled_df[col].iloc[0]
                        break
                else:
                    row_scaled[f] = raw_row.get(f, 0.0)
        X_df = pd.DataFrame([row_scaled], columns=feats).astype(np.float32)
    else:
        X_df = scaled_df.astype(np.float32)

    lgbm_results = _lgbm_predict(models, X_df, feats)

    region_name = input_data_dict.get("Région_12", input_data_dict.get("Region_12", "Guelmim-Oued Noun"))
    if isinstance(region_name, str):
        region_code = REGION_NAME_TO_CODE.get(region_name, 10.0)
    else:
        region_code = float(region_name)
    milieu_code = raw_row.get("Milieu", coded.get("Milieu", 1.0))
    hh_size = raw_row.get("Taille_ménage", coded.get("Taille_ménage", 4.0))

    engine = get_hypernet_engine()
    hyper_results = engine.predict(
        raw_row, region_code, milieu_code, hh_size,
        rgph_df=rgph_df, rural_transfer=rural_transfer,
    )

    shap_values = []
    pauvre_model = models.get("pauvre_lgbm")
    if pauvre_model is not None and hasattr(pauvre_model, "feature_importances_"):
        importances = pauvre_model.feature_importances_
        feat_names = feats or ENCDM_FEATURE_NAMES[:len(importances)]
        for i, imp in enumerate(importances[:len(feat_names)]):
            fv = raw_row.get(feat_names[i], 0)
            direction = 1 if float(fv) > 0.5 else -1
            sh = (imp / max(importances.max(), 1e-9)) * direction * abs(lgbm_results["pauvre_prob"] - 0.3)
            shap_values.append({
                "feature": get_column_label(str(feat_names[i])),
                "value": sh, "importance": float(imp),
            })
        shap_values.sort(key=lambda x: abs(x["value"]), reverse=True)

    return {
        "lgbm": lgbm_results,
        "hypernet": hyper_results,
        "raw_row": raw_row,
        "shap_values": shap_values[:10],
        "rural_transfer": rural_transfer,
    }


def build_shap_waterfall(shap_data, prediction_prob):
    if not shap_data:
        return None
    features = [s["feature"] for s in shap_data][:9]
    values = [s["value"] for s in shap_data][:9]
    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=["relative"] * len(values) + ["total"],
        x=features + ["Prediction"],
        y=values + [prediction_prob],
        text=[f"{v:.3f}" for v in values] + [f"{prediction_prob:.3f}"],
        textposition="outside",
        connector=dict(line=dict(color="#B0BEC5", width=1.5)),
        increasing=dict(marker=dict(color="#1A3A5C")),
        decreasing=dict(marker=dict(color="#9B4D4D")),
        totals=dict(marker=dict(color="#2C7A7B")),
    ))
    fig.update_layout(
        title=dict(text=f"LGBM Feature Contributions ({prediction_prob:.1%})", font=dict(size=13)),
        height=320, paper_bgcolor="#F8F9FA", plot_bgcolor="#FFFFFF",
        margin=dict(l=10, r=10, t=40, b=60),
        xaxis=dict(tickangle=45), yaxis=dict(showgrid=True, gridcolor="#DEE2E6"),
    )
    return fig


def render_sandbox_form(default_region="Guelmim-Oued Noun"):
    col1, col2, col3 = st.columns(3)
    region_options = list(RAW_INPUT_MAPPING["Région_12"].keys())
    region_idx = region_options.index(default_region) if default_region in region_options else 9

    with col1:
        milieu = st.selectbox("Area Type", options=["Urbain", "Rural"], index=0)
        sexe = st.selectbox("Gender of Household Head", options=["Masculin", "Feminin"], index=0)
        age = st.number_input("Age of Household Head", min_value=15, max_value=110, value=40, step=1)
        marital = st.selectbox("Marital Status", options=["Celibataire", "Marie", "Divorce", "Veuf"], index=1)
        education = st.selectbox("Education Level", options=[
            "Sans niveau", "Prescolaire", "Primaire", "Secondaire collegial",
            "Secondaire qualifiant", "Superieur", "Autre niveau"], index=0)

    with col2:
        region = st.selectbox("Administrative Region", options=region_options, index=region_idx)
        hh_size = st.number_input("Household Size (integer)", min_value=1, max_value=25, value=4, step=1)
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
        "Milieu": milieu, "Région_12": region, "Taille_ménage": int(hh_size),
        "Taille_agregée": hh_type, "Sexe_CM": sexe, "Age_CM": float(age),
        "Age_quin_CM": age_quin, "Etat_matrimonial_CM": marital,
        "Niveau_scolaire_agreg_CM": education, "Diplôme_agregé_CM": diploma,
        "Lieunaissance_CM": birth_place, "Type_activité_dominante_CM": activity_type,
        "Profession_agreg_CM": profession, "Secteur_activité_agreg_CM": sector,
        "Situation_profession_agreg_CM": situation,
    }


# Backward compatibility
def run_prediction(input_data_dict):
    r = run_dual_prediction(input_data_dict)
    return {
        "pauvre_prob": r["lgbm"]["pauvre_prob"],
        "vulnerable_prob": r["lgbm"]["vulnerable_prob"],
        "shap_values": r["shap_values"],
    }
