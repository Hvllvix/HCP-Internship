"""
Loads project data, mappings, and persisted model artifacts.
"""
import json
import pathlib

import joblib
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]

ENCDM_CONFIG = {
    "categorical": [
        "Milieu", "Région_12", "Taille_agregée", "Sexe_CM",
        "Niveau_scolaire_agreg_CM", "Situation_profession_agreg_CM",
    ],
    "numerical": ["Taille_ménage", "Age_CM"],
    "target": ["Pauvre", "Vulnérable"],
    "weight": "coef_indiv",
}

RGPH_CONFIG = {
    "categorical": [
        "REG", "MIL", "TYPE.LOG", "MURS", "TOIT", "SOL",
        "EAU.MODE", "ELEC", "NET", "VOIT",
    ],
    "numerical": ["TAILLE", "PIECES", "ROUTE.DIST"],
    "weight": "PDS",
}

GEOJSON_REGIONS = {
    "Laayoune-Saguia Hamra": 1,
    "Rabat-Sale-Kenitra": 5,
    "Beni Mellal-Khenifra": 7,
    "Dakhla-Oued Eddahab": 0,
    "Tanger-Tetouan-Hoceima": 11,
    "Marrakech-Safi": 8,
    "Daraa-Tafilelt": 4,
    "Guelmim-Oued Noun": 2,
    "Fes-Meknes": 10,
    "Oriental": 9,
    "Casablanca-Settat": 6,
    "Souss Massa": 3,
}

COLUMN_LABELS = {
    "N_ménage": "Household ID",
    "coef_ménage": "Household Weight",
    "coef_indiv": "Individual Weight",
    "Milieu": "Area Type",
    "Région_12": "Region",
    "Taille_ménage": "Household Size",
    "Taille_agregée": "Size Category",
    "Pauvre": "Poverty Status",
    "Vulnérable": "Vulnerability",
    "Sexe_CM": "Gender",
    "Age_CM": "Age",
    "Niveau_scolaire_agreg_CM": "Education Level",
    "Situation_profession_agreg_CM": "Employment Status",
    "REG": "Region Code",
    "MIL": "Area Type (RGPH)",
    "TAILLE": "HH Size (RGPH)",
    "PIECES": "Rooms",
    "ELEC": "Electricity",
    "NET": "Internet",
    "EAU.MODE": "Water Source",
}


def load_encdm():
    return pd.read_parquet(ROOT / "Data/Processed/CleanENCDM.parquet")


def load_rgph():
    return pd.read_parquet(ROOT / "Data/Processed/CleanRGPH.parquet")


def load_geojson():
    with open(ROOT / "Assets/Maps/Morocco-Regions.geojson", encoding="utf-8") as f:
        return json.load(f)


def load_mapping_encdm():
    with open(ROOT / "Assets/Maps/MapENCDM.json", encoding="utf-8") as f:
        return json.load(f)


def load_mapping_rgph():
    with open(ROOT / "Assets/Maps/MapRGPH.json", encoding="utf-8") as f:
        return json.load(f)


def load_deps_encdm():
    with open(ROOT / "Assets/Dependencies/DependenciesENCDM.json", encoding="utf-8") as f:
        return json.load(f)


def load_deps_rgph():
    with open(ROOT / "Assets/Dependencies/DependenciesRGPH.json", encoding="utf-8") as f:
        return json.load(f)


def build_label_maps():
    mapping = load_mapping_encdm()
    return {
        col: {v: k for k, v in items.items() if v is not None}
        for col, items in mapping.items()
    }


def build_region_name_map():
    raw = load_mapping_encdm()["Région_12"]
    return {v: k for k, v in raw.items()}


def build_code_maps():
    """Human label -> numeric code per ENCDM categorical field."""
    mapping = load_mapping_encdm()
    return {
        col: {k: v for k, v in items.items() if v is not None}
        for col, items in mapping.items()
    }


REGION_CODES = {v: k for k, v in GEOJSON_REGIONS.items()}


def get_label(col):
    return COLUMN_LABELS.get(col, col)


def translate(series, colname, label_maps):
    mapping = label_maps.get(colname, {})
    return series.map(mapping).fillna(series)


def inverse_scale_encdm(df):
    """Restore native units for scaled ENCDM columns (matches Modeling.ipynb)."""
    out = df.copy()
    scaler_dir = ROOT / "Models/Scalers/ENCDM"
    if not scaler_dir.exists():
        return out
    for path in scaler_dir.glob("SS_*.joblib"):
        col = path.stem.replace("SS_", "")
        if col in out.columns:
            scaler = joblib.load(path)
            out[col] = scaler.inverse_transform(out[[col]]).ravel()
    return out


def load_lgbm_bundle(target):
    path = ROOT / "Models/Classifier" / f"ENCDM_LGBM_{target}.joblib"
    return joblib.load(path)


def load_all_lgbm():
    bundles = {}
    for target in ENCDM_CONFIG["target"]:
        bundles[target] = load_lgbm_bundle(target)
    return bundles


def load_scalers_encdm():
    scalers = {}
    for path in (ROOT / "Models/Scalers/ENCDM").glob("SS_*.joblib"):
        col = path.stem.replace("SS_", "")
        scalers[col] = joblib.load(path)
    return scalers


def build_region_bridge():
    """Map ENCDM Région_12 codes -> RGPH REG codes (Modeling.ipynb)."""
    map_encdm = load_mapping_encdm()["Région_12"]
    map_rgph = load_mapping_rgph()["REG"]
    inv_rgph = {name: code for name, code in map_rgph.items()}
    bridge = {}
    for enc_code, enc_name in map_encdm.items():
        if enc_name in inv_rgph:
            bridge[enc_code] = inv_rgph[enc_name]
        else:
            for rgph_name, rgph_code in inv_rgph.items():
                if enc_name.replace("é", "e").replace("â", "a") in rgph_name.replace("é", "e").replace("â", "a"):
                    bridge[enc_code] = rgph_code
                    break
    return bridge
