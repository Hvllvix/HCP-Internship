"""
Dual-engine inference: LightGBM + Hypernetwork.
"""
import numpy as np
import pandas as pd

from data_loader import ENCDM_CONFIG, load_all_lgbm
from hypernet import get_hypernet_engine


def build_feature_row(form_inputs, code_maps):
    """Map UI labels to numeric feature dict."""
    row = {}
    for col in ENCDM_CONFIG["categorical"]:
        label = form_inputs.get(col)
        mapping = code_maps.get(col, {})
        if label in mapping:
            row[col] = mapping[label]
        elif isinstance(label, (int, float)):
            row[col] = int(label)
        else:
            row[col] = 0
    row["Taille_ménage"] = float(form_inputs.get("Taille_ménage", 4))
    row["Age_CM"] = float(form_inputs.get("Age_CM", 35))
    return row


def row_to_model_frame(row, features):
    return pd.DataFrame(
        [[row[f] for f in features]], columns=features, dtype="float32"
    )


def run_lgbm_inference(feature_row, bundles):
    results = {}
    for target, bundle in bundles.items():
        model = bundle["model"]
        feats = bundle["features"]
        X = row_to_model_frame(feature_row, feats)
        proba = model.predict_proba(X)[0, 1]
        threshold = bundle.get("threshold", 0.5)
        results[target] = {
            "probability": float(proba),
            "label": int(proba >= threshold),
            "threshold": float(threshold),
        }
    return results


def run_hypernet_inference(feature_row, region_code, milieu_code, rgph_df, rural_transfer=False):
    engine = get_hypernet_engine(rgph_df)
    probs = engine.predict(feature_row, region_code, milieu_code, rural_transfer)
    results = {}
    for target in ENCDM_CONFIG["target"]:
        p = probs.get(target)
        if p is None:
            results[target] = {"probability": None, "label": None, "threshold": None}
            continue
        threshold = engine.thresholds.get(target, 0.5)
        results[target] = {
            "probability": p,
            "label": int(p >= threshold),
            "threshold": threshold,
        }
    return results


def feature_contributions(bundles, feature_row):
    """SHAP-inspired bars from LightGBM feature importances."""
    target = "Pauvre"
    bundle = bundles.get(target)
    if bundle is None:
        return []

    model = bundle["model"]
    feats = bundle["features"]
    if not hasattr(model, "feature_importances_"):
        return []

    imps = pd.Series(model.feature_importances_, index=feats)
    total = imps.sum() or 1.0
    imps = (imps / total * 100).sort_values(ascending=False)

    contributions = []
    for feat, imp in imps.head(8).items():
        val = feature_row.get(feat, 0)
        contributions.append({"feature": feat, "importance": float(imp), "value": val})
    return contributions


def run_dual_prediction(form_inputs, code_maps, rgph_df, rural_transfer=False):
    row = build_feature_row(form_inputs, code_maps)
    bundles = load_all_lgbm()

    lgbm = run_lgbm_inference(row, bundles)
    region_code = row.get("Région_12", 2)
    milieu_code = row.get("Milieu", 0)
    hyper = run_hypernet_inference(
        row, region_code, milieu_code, rgph_df, rural_transfer=rural_transfer
    )
    contributions = feature_contributions(bundles, row)

    return {
        "feature_row": row,
        "lgbm": lgbm,
        "hypernet": hyper,
        "contributions": contributions,
    }
