"""Dual-engine inference: LightGBM + Hypernetwork."""
import pandas as pd

from data_loader import ENCDM_CONFIG, load_all_lgbm, load_encdm, load_scalers_encdm
from hypernet import HypernetInferenceError, get_hypernet_engine


class InferenceError(Exception):
    """Base inference failure."""


def build_feature_row(form_inputs, code_maps):
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


def scale_feature_row(row, scalers):
    scaled = dict(row)
    df = pd.DataFrame([row])
    for col, scaler in scalers.items():
        if col not in df.columns:
            continue
        try:
            scaled[col] = float(scaler.transform(df[[col]]).ravel()[0])
        except Exception:
            scaled[col] = float(row[col])
    return scaled


def row_to_model_frame(row, features):
    return pd.DataFrame(
        [[row[f] for f in features]], columns=features, dtype="float32"
    )


def run_lgbm_inference(feature_row, bundles, scalers):
    scaled_row = scale_feature_row(feature_row, scalers)
    results = {}
    for target, bundle in bundles.items():
        model = bundle["model"]
        feats = bundle["features"]
        X = row_to_model_frame(scaled_row, feats)
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
    if not engine._ready:
        return {
            t: {"probability": None, "label": None, "threshold": None}
            for t in ENCDM_CONFIG["target"]
        }
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


def ood_flags(feature_row, encdm_df=None):
    if encdm_df is None:
        encdm_df = load_encdm()
    flags = []
    if "Age_CM" in encdm_df.columns:
        ages = encdm_df["Age_CM"]
        q05, q95 = ages.quantile(0.05), ages.quantile(0.95)
        age = feature_row.get("Age_CM")
        if age is not None and (age < q05 or age > q95):
            flags.append(f"Age ({age:.0f}) is outside the central training range ({q05:.0f}-{q95:.0f}).")
    if "Taille_ménage" in encdm_df.columns:
        sizes = encdm_df["Taille_ménage"]
        q05, q95 = sizes.quantile(0.05), sizes.quantile(0.95)
        size = feature_row.get("Taille_ménage")
        if size is not None and (size < q05 or size > q95):
            flags.append(
                f"Household size ({size:.0f}) is outside the central training range ({q05:.0f}-{q95:.0f})."
            )
    return flags


def run_dual_prediction(form_inputs, code_maps, rgph_df, rural_transfer=False):
    row = build_feature_row(form_inputs, code_maps)
    bundles = load_all_lgbm()
    scalers = load_scalers_encdm()

    lgbm = run_lgbm_inference(row, bundles, scalers)
    region_code = row.get("Région_12", 2)
    milieu_code = row.get("Milieu", 0)
    try:
        hyper = run_hypernet_inference(
            row, region_code, milieu_code, rgph_df, rural_transfer=rural_transfer
        )
    except HypernetInferenceError as exc:
        raise InferenceError(str(exc)) from exc

    encdm_df = load_encdm()
    return {
        "feature_row": row,
        "scaled_row": scale_feature_row(row, scalers),
        "lgbm": lgbm,
        "hypernet": hyper,
        "contributions": feature_contributions(bundles, row),
        "ood_flags": ood_flags(row, encdm_df),
    }
