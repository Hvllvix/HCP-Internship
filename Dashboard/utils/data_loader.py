"""
Data loading and caching module for the HCP Poverty Dashboard.
Handles .parquet files, .sav files, scalers, imputers, and models.
"""
import pandas as pd
import numpy as np
import streamlit as st
import joblib
import os
import json
import torch
from pathlib import Path

# --- Path Constants ---
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_PROCESSED = BASE_DIR / "Data" / "Processed"
DATA_RAW = BASE_DIR / "Data" / "Raw"
MODELS_DIR = BASE_DIR / "Models"
SCALERS_DIR = MODELS_DIR / "Scalers"
IMPUTERS_DIR = MODELS_DIR / "Imputers"
CLASSIFIER_DIR = MODELS_DIR / "Classifier"
ASSETS_DIR = BASE_DIR / "Assets"
MAPS_DIR = ASSETS_DIR / "Maps"
DEPS_DIR = ASSETS_DIR / "Dependencies"

# ============================================================================
# COMPREHENSIVE COLUMN NAME MAPPING
# Maps raw database column names to clean, professional, human-readable labels
# ============================================================================
ENCDM_COLUMN_LABELS = {
    "N_ménage": "Household ID",
    "coef_ménage": "Household Sampling Weight",
    "coef_indiv": "Individual Sampling Weight",
    "Milieu": "Area Type",
    "Région_12": "Administrative Region",
    "Region_12": "Administrative Region",
    "Taille_ménage": "Household Size",
    "Taille_agregée": "Household Size Category",
    "Taille_agreee": "Household Size Category",
    "Pauvre": "Poverty Status",
    "Vulnérable": "Vulnerability Status",
    "Vulnerable": "Vulnerability Status",
    "Quintiles": "Expenditure Quintile",
    "Deciles": "Expenditure Decile",
    "Quintileurbain": "Urban Expenditure Quintile",
    "Decileurbain": "Urban Expenditure Decile",
    "Quintilerural": "Rural Expenditure Quintile",
    "Decilerural": "Rural Expenditure Decile",
    "Sexe_CM": "Gender of Household Head",
    "Age_CM": "Age of Household Head",
    "Age_quin_CM": "Age Group (5-Year)",
    "Lieunaissance_CM": "Place of Birth",
    "Etat_matrimonial_CM": "Marital Status",
    "Niveau_scolaire_agreg_CM": "Education Level",
    "Diplôme_agregé_CM": "Diploma Level",
    "Diplome_agrege_CM": "Diploma Level",
    "Type_activité_dominante_CM": "Dominant Activity Type",
    "Type_activite_dominante_CM": "Dominant Activity Type",
    "Profession_agreg_CM": "Profession Category",
    "Secteur_activité_agreg_CM": "Economic Sector",
    "Secteur_activite_agreg_CM": "Economic Sector",
    "Situation_profession_agreg_CM": "Professional Situation",
    "DAM": "Total Annual Expenditure",
    "DAM_G1": "Food Expenditure",
    "DAM_G2": "Housing Expenditure",
    "DAM_G3": "Health Expenditure",
    "DAM_G4": "Education Expenditure",
    "DAM_G5": "Transport Expenditure",
    "DAM_G6": "Communication Expenditure",
    "DAM_G7": "Leisure Expenditure",
    "DAM_G8": "Hygiene Expenditure",
    "DAM_G9": "Other Expenditure",
    "DAM_hygiene": "Hygiene Products Expenditure",
    "DAM_soins_medicaux": "Medical Care Expenditure",
    "DAP": "Per Capita Annual Expenditure",
    "DAP_G1": "Per Capita Food Expenditure",
    "DAP_G2": "Per Capita Housing Expenditure",
}

RGPH_COLUMN_LABELS = {
    "REG": "Administrative Region",
    "PRO": "Province Code",
    "MIL": "Area Type",
    "MEN.PRO": "Province Household Code",
    "TAILLE": "Household Size",
    "TYPE.LOG": "Dwelling Type",
    "MURS": "Wall Construction Material",
    "TOIT": "Roof Construction Material",
    "SOL": "Floor Construction Material",
    "AGE.LOG": "Dwelling Age",
    "PIECES": "Number of Rooms",
    "STAT.OCC": "Occupancy Status",
    "CUIS": "Kitchen Type",
    "WC": "Toilet Type",
    "BD": "Bathroom Type",
    "BLOC": "Block/Building Type",
    "ECL.MODE": "Lighting Mode",
    "EAU.MODE": "Water Supply Mode",
    "EAU.DIST": "Water Distance",
    "EAU.DUR": "Water Duration",
    "EAUX.US": "Used Water Evacuation",
    "DECH": "Waste Disposal Mode",
    "GAZ": "Gas Usage",
    "ELEC": "Electricity Usage",
    "CHAR": "Charcoal Usage",
    "BOIS": "Wood Usage",
    "DEJ.ANIM": "Domestic Animal Presence",
    "TELE": "Television Access",
    "RADIO": "Radio Access",
    "TEL.PORT": "Mobile Phone Access",
    "TEL.FIXE": "Fixed Line Phone Access",
    "NET": "Internet Access",
    "PC": "Computer Access",
    "PARAB": "Satellite Dish Access",
    "FRIGO": "Refrigerator Access",
    "CAM": "Camera Access",
    "VOIT": "Car Ownership",
    "TRACT": "Tractor Ownership",
    "MOTO": "Motorcycle Ownership",
    "ROUTE.DIST": "Distance to Paved Road",
    "MEN.TYPE": "Household Type",
    "PDS": "Sampling Weight",
}

def get_column_label(col_name, dataset="encdm"):
    """Return human-readable label for a column name."""
    if dataset == "encdm":
        return ENCDM_COLUMN_LABELS.get(col_name, col_name)
    return RGPH_COLUMN_LABELS.get(col_name, col_name)

def get_column_labels(dataset="encdm"):
    """Return the full mapping dict."""
    return ENCDM_COLUMN_LABELS if dataset == "encdm" else RGPH_COLUMN_LABELS

# Column name aliases (accented vs non-accented)
COLUMN_ALIASES = {
    "Region_12": ["Région_12", "Region_12"],
    "Vulnerable": ["Vulnérable", "Vulnerable"],
    "Taille_agreee": ["Taille_agregée", "Taille_agreee"],
    "Diplome_agrege_CM": ["Diplôme_agregé_CM", "Diplome_agrege_CM"],
    "Type_activite_dominante_CM": ["Type_activité_dominante_CM", "Type_activite_dominante_CM"],
    "Secteur_activite_agreg_CM": ["Secteur_activité_agreg_CM", "Secteur_activite_agreg_CM"],
    "coef_menage": ["coef_ménage", "coef_menage"],
    "N_menage": ["N_ménage", "N_menage"],
    "Taille_menage": ["Taille_ménage", "Taille_menage"],
}

def resolve_colname(df, name):
    """Find actual column name in dataframe given an alias."""
    if name in df.columns:
        return name
    aliases = COLUMN_ALIASES.get(name, [name])
    for alias in aliases:
        if alias in df.columns:
            return alias
    for col in df.columns:
        if col.lower().replace(" ", "").replace("-", "").replace("_", "") == \
           name.lower().replace(" ", "").replace("-", "").replace("_", ""):
            return col
    return name

def get_encdm_raw_column(encdm_raw_df, col_name):
    """Get column from raw ENCDM with alias fallback."""
    if encdm_raw_df is not None and col_name in encdm_raw_df.columns:
        return encdm_raw_df[col_name]
    for alt_name in COLUMN_ALIASES.get(col_name, []):
        if encdm_raw_df is not None and alt_name in encdm_raw_df.columns:
            return encdm_raw_df[alt_name]
    return None

# Value Labels from Raw Data
ENCDM_LABELS = {
    "Milieu": {1.0: "Urbain", 2.0: "Rural"},
    "Region_12": {
        1.0: "Tanger-Tetouan-Al Hoceima", 2.0: "Oriental", 3.0: "Fes-Meknes",
        4.0: "Rabat-Sale-Kenitra", 5.0: "Beni Mellal-Khenifra", 6.0: "Casablanca-Settat",
        7.0: "Marrakech-Safi", 8.0: "Draa-Tafilalet", 9.0: "Souss-Massa",
        10.0: "Guelmim-Oued Noun", 11.0: "Laayoune-Sakia El Hamra", 12.0: "Dakhla-Oued Ed Dahab"
    },
    "Pauvre": {0.0: "Non pauvre", 1.0: "Pauvre"},
    "Vulnerable": {1.0: "Vulnerable", 2.0: "Non vulnerable"},
    "Sexe_CM": {1.0: "Masculin", 2.0: "Feminin"},
    "Etat_matrimonial_CM": {1.0: "Celibataire", 2.0: "Marie", 3.0: "Divorce", 4.0: "Veuf"},
    "Niveau_scolaire_agreg_CM": {
        0.0: "Sans niveau", 1.0: "Prescolaire", 2.0: "Primaire",
        3.0: "Secondaire collegial", 4.0: "Secondaire qualifiant", 5.0: "Superieur", 6.0: "Autre niveau"
    },
    "Diplome_agrege_CM": {1.0: "Sans diplome", 2.0: "Niveau moyen", 3.0: "Niveau superieur"},
    "Profession_agreg_CM": {
        0.0: "Chomeur jamais travaille", 1.0: "Cadres de direction",
        2.0: "Cadres moyens", 3.0: "Commercants", 4.0: "Exploitants agricoles",
        5.0: "Artisans qualifies", 6.0: "Manoeuvres", 9.0: "Non declare"
    },
    "Secteur_activite_agreg_CM": {
        0.0: "Chomeur jamais travaille", 1.0: "Agriculture",
        2.0: "Industrie", 3.0: "Batiment et TP", 4.0: "Commerce", 5.0: "Service", 9.0: "Non declare"
    },
    "Situation_profession_agreg_CM": {
        0.0: "Chomeur", 1.0: "Inactif", 2.0: "Salarie", 3.0: "Independant",
        4.0: "Employeur", 5.0: "Autres situation", 9.0: "Non declare"
    },
    "Type_activite_dominante_CM": {
        0.0: "Actif occupe", 1.0: "Personne agee", 2.0: "Retraite", 3.0: "Femme au foyer",
        4.0: "Eleve/etudiant", 5.0: "Autre inactif", 6.0: "Chomeur ayant travaille",
        7.0: "Malade/infirme", 8.0: "Rentier", 9.0: "Chomeur jamais travaille", 10.0: "Non declare"
    }
}

RGPH_LABELS = {
    "REG": {1.0: "Tanger-Tetouan-Al Hoceima", 2.0: "Oriental", 3.0: "Fes-Meknes",
            4.0: "Rabat-Sale-Kenitra", 5.0: "Beni Mellal-Khenifra", 6.0: "Casablanca-Settat",
            7.0: "Marrakech-Safi", 8.0: "Draa-Tafilalet", 9.0: "Souss-Massa",
            10.0: "Guelmim-Oued Noun", 11.0: "Laayoune-Sakia El Hamra", 12.0: "Dakhla-Oued Ed Dahab"},
    "MIL": {1.0: "Urbain", 2.0: "Rural"},
    "TYPE.LOG": {1.0: "Villa", 2.0: "Appartement", 3.0: "Maison marocaine", 4.0: "Maison sommaire", 5.0: "Logement rural", 6.0: "Autre"},
    "MURS": {1.0: "Beton/Brique", 2.0: "Pierre mortier", 3.0: "Bois", 4.0: "Pierre terre", 5.0: "Pise", 6.0: "Autre"},
    "STAT.OCC": {1.0: "Proprietaire", 2.0: "Locataire", 3.0: "Fonction", 4.0: "Gratuit", 5.0: "Autre"},
    "EAU.MODE": {1.0: "Reseau public (prive)", 2.0: "Reseau public (partage)", 3.0: "Fontaine/Puits equipe", 4.0: "Vendeur eau potable", 5.0: "Puits non equipe", 6.0: "Source/Oued", 7.0: "Autre"},
    "GAZ": {1.0: "Frequent", 2.0: "Occasionnel", 3.0: "Non utilise"},
    "ELEC": {1.0: "Frequent", 2.0: "Occasionnel", 3.0: "Non utilise"},
}

# --- Data Loading Functions ---

@st.cache_resource
def load_clean_encdm():
    path = DATA_PROCESSED / "CleanENCDM.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return None

@st.cache_resource
def load_clean_rgph():
    path = DATA_PROCESSED / "CleanRGPH.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return None

@st.cache_resource
def load_raw_encdm():
    path = DATA_RAW / "ENCDM.sav"
    if path.exists():
        import pyreadstat
        df, meta = pyreadstat.read_sav(path)
        return df, meta
    return None, None

@st.cache_resource
def load_raw_rgph():
    path = DATA_RAW / "RGPH.sav"
    if path.exists():
        import pyreadstat
        df, meta = pyreadstat.read_sav(path)
        return df, meta
    return None, None

@st.cache_resource
def load_encdm_map():
    path = MAPS_DIR / "MapENCDM.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

@st.cache_resource
def load_rgph_map():
    path = MAPS_DIR / "MapRGPH.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

@st.cache_resource
def load_encdm_dependencies():
    path = DEPS_DIR / "DependenciesENCDM.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

@st.cache_resource
def load_rgph_dependencies():
    path = DEPS_DIR / "DependenciesRGPH.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

@st.cache_resource
def load_morocco_geojson():
    path = Path(__file__).parent / "MoroccoGeoMap.geojson"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def _unwrap_model(model_obj):
    if isinstance(model_obj, dict) and "model" in model_obj:
        return model_obj["model"], model_obj.get("features", []), model_obj.get("threshold", 0.5)
    return model_obj, [], 0.5

@st.cache_resource
def load_models():
    models = {}
    try:
        path = CLASSIFIER_DIR / "ENCDM_LGBM_Pauvre.joblib"
        if path.exists():
            raw = joblib.load(path)
            m, f, t = _unwrap_model(raw)
            models["pauvre_lgbm"] = m
            models["pauvre_lgbm_features"] = f
            models["pauvre_lgbm_threshold"] = t
    except Exception:
        models["pauvre_lgbm"] = None
    try:
        p1 = CLASSIFIER_DIR / "ENCDM_LGBM_Vulnerable.joblib"
        p2 = CLASSIFIER_DIR / "ENCDM_LGBM_Vulnerable.joblib"
        path = p1 if p1.exists() else p2
        if not path.exists():
            # Try with accent
            path = CLASSIFIER_DIR / "ENCDM_LGBM_Vulnérable.joblib"
        if path.exists():
            raw = joblib.load(path)
            m, f, t = _unwrap_model(raw)
            models["vulnerable_lgbm"] = m
            models["vulnerable_lgbm_features"] = f
            models["vulnerable_lgbm_threshold"] = t
    except Exception:
        models["vulnerable_lgbm"] = None
    try:
        path = CLASSIFIER_DIR / "Transfer_LGBM_Pauvre.joblib"
        if path.exists():
            raw = joblib.load(path)
            m, f, t = _unwrap_model(raw)
            models["transfer_pauvre"] = m
    except Exception:
        models["transfer_pauvre"] = None
    try:
        path = CLASSIFIER_DIR / "Hypernet.pt"
        if path.exists():
            models["hypernet"] = str(path)
    except Exception:
        models["hypernet"] = None
    return models

@st.cache_resource
def load_scalers_encdm():
    scalers = {}
    scaler_dir = SCALERS_DIR / "ENCDM"
    if scaler_dir.exists():
        for f in scaler_dir.glob("*.joblib"):
            try:
                key = f.stem.replace("SS_", "")
                scalers[key] = joblib.load(f)
            except Exception:
                pass
    return scalers

@st.cache_resource
def load_imputers():
    imputers = {}
    if IMPUTERS_DIR.exists():
        for f in IMPUTERS_DIR.glob("*.joblib"):
            try:
                imputers[f.stem] = joblib.load(f)
            except Exception:
                pass
    return imputers

def get_encdm_feature_stats():
    df = load_clean_encdm()
    if df is None:
        return {}
    region_col = resolve_colname(df, "Region_12")
    vulnerable_col = resolve_colname(df, "Vulnerable")
    stats = {
        "n_households": len(df),
        "n_features": len(df.columns),
        "poverty_rate": float(df["Pauvre"].mean() * 100) if "Pauvre" in df.columns else 0,
        "vulnerable_rate": float(df[vulnerable_col].mean() * 100) if vulnerable_col in df.columns else 0,
        "missing_rate": float(df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100),
        "n_regions": int(df[region_col].nunique()) if region_col in df.columns else 0,
    }
    return stats

def get_rgph_feature_stats():
    df = load_clean_rgph()
    if df is None:
        return {}
    return {
        "n_households": len(df),
        "n_features": len(df.columns),
        "missing_rate": float(df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100),
        "n_regions": int(df["REG"].nunique()) if "REG" in df.columns else 0,
    }