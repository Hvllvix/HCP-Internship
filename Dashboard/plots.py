"""
Plot helpers for dashboard sections.
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_auc_score, roc_curve

from data_loader import ENCDM_CONFIG, get_label, inverse_scale_encdm
from theme import PALETTE, plotly_layout


def fig_missing_values(encdm, top_n=10):
    miss = encdm.isna().sum()
    miss = miss[miss > 0].sort_values(ascending=False)
    if len(miss) == 0:
        return None
    md = pd.DataFrame({
        "Feature": [get_label(c) for c in miss.index],
        "Missing %": (miss / len(encdm) * 100).round(2),
    }).head(top_n)
    fig = px.bar(
        md, x="Feature", y="Missing %", color="Missing %",
        color_continuous_scale=[PALETTE["gray"], PALETTE["amber"], PALETTE["navy"]],
    )
    fig.update_layout(**plotly_layout(height=280, showlegend=False))
    fig.update_xaxes(tickangle=-35)
    return fig


def fig_poverty_breakdown(encdm):
    poor_w = (encdm[encdm["Pauvre"] == 1]["coef_ménage"]).sum()
    vuln_w = (encdm[encdm["Vulnérable"] == 1]["coef_ménage"]).sum()
    nonpoor_w = (
        encdm[(encdm["Pauvre"] == 0) & (encdm["Vulnérable"] == 0)]["coef_ménage"]
    ).sum()
    sd = pd.DataFrame({
        "Status": ["Non-Poor", "Vulnerable", "Poor"],
        "Weight": [nonpoor_w, vuln_w, poor_w],
    })
    colors = {"Non-Poor": PALETTE["navy"], "Vulnerable": PALETTE["amber"], "Poor": PALETTE["danger"]}
    fig = px.pie(sd, values="Weight", names="Status", color="Status", color_discrete_map=colors)
    fig.update_layout(**plotly_layout(height=280))
    return fig


def fig_region_poverty(region_stats):
    df = pd.DataFrame([
        {"Region": r["name"], "Poverty Rate": r["poverty_rate"]}
        for r in region_stats.values()
    ]).sort_values("Poverty Rate")
    fig = px.bar(
        df, x="Poverty Rate", y="Region", orientation="h", color="Poverty Rate",
        color_continuous_scale=[PALETTE["amber_soft"], PALETTE["amber"], PALETTE["navy"]],
        text_auto=".1f",
    )
    fig.update_layout(**plotly_layout(height=400, coloraxis_showscale=False))
    fig.update_traces(textposition="outside")
    return fig


def fig_choropleth(geojson, region_stats, geojson_regions, selected_code=None):
    z_data, locations, hover = [], [], []
    sel_id = None
    for feat in geojson["features"]:
        props = feat["properties"]
        gname = props["region"]
        cid = props["cartodb_id"]
        code = geojson_regions.get(gname)
        locations.append(cid)
        if code is not None and code in region_stats:
            rate = region_stats[code]["poverty_rate"]
            z_data.append(rate)
            hover.append(region_stats[code]["name"])
            if code == selected_code:
                sel_id = cid
        else:
            z_data.append(None)
            hover.append(gname)

    fig = go.Figure()
    fig.add_trace(go.Choropleth(
        geojson=geojson,
        featureidkey="properties.cartodb_id",
        locations=locations,
        z=z_data,
        text=hover,
        hovertemplate="<b>%{text}</b><br>Poverty: %{z:.1f}%<extra></extra>",
        colorscale=[
            [0, PALETTE["white"]],
            [0.25, PALETTE["amber_soft"]],
            [0.5, PALETTE["amber"]],
            [0.75, PALETTE["navy_light"]],
            [1, PALETTE["navy"]],
        ],
        zmin=5, zmax=45,
        marker=dict(line=dict(width=1.2, color=PALETTE["navy"])),
        colorbar=dict(title="Poverty %", thickness=12, len=0.55),
    ))
    if sel_id is not None and selected_code in region_stats:
        fig.add_trace(go.Choropleth(
            geojson=geojson,
            featureidkey="properties.cartodb_id",
            locations=[sel_id],
            z=[region_stats[selected_code]["poverty_rate"]],
            colorscale=[[0, PALETTE["amber"]], [1, PALETTE["amber"]]],
            marker=dict(line=dict(width=3, color=PALETTE["black"])),
            showscale=False,
            hovertemplate="<b>SELECTED</b><br>%{z:.1f}%<extra></extra>",
        ))
    fig.update_layout(
        **plotly_layout(height=480, margin=dict(l=0, r=0, t=0, b=0)),
        geo=dict(
            projection_type="mercator",
            showland=True,
            landcolor=PALETTE["gray"],
            coastlinecolor=PALETTE["white"],
            showcountries=False,
            showframe=False,
            lonaxis_range=[-17, -1],
            lataxis_range=[21, 36],
        ),
    )
    return fig


def fig_feature_importance(bundles, target="Pauvre"):
    bundle = bundles.get(target)
    if bundle is None:
        return None
    model = bundle["model"]
    feats = bundle["features"]
    imps = pd.Series(model.feature_importances_, index=feats).sort_values()
    fig = px.bar(
        imps, x=imps.values, y=imps.index, orientation="h",
        color=imps.values,
        color_continuous_scale=[PALETTE["gray"], PALETTE["amber"], PALETTE["navy"]],
        labels={"x": "Importance", "y": ""},
    )
    fig.update_layout(**plotly_layout(height=320, showlegend=False, coloraxis_showscale=False))
    return fig


def fig_roc_curves(encdm, bundles, sample_n=2000):
    """Weighted ROC from a sample — real model evaluation."""
    rng = np.random.default_rng(42)
    idx = rng.choice(len(encdm), size=min(sample_n, len(encdm)), replace=False)
    sample = encdm.iloc[idx]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines", name="Random",
        line=dict(color=PALETTE["gray"], dash="dash"),
    ))

    for target in ENCDM_CONFIG["target"]:
        bundle = bundles.get(target)
        if bundle is None:
            continue
        X = sample[bundle["features"]].astype("float32")
        y = sample[target].astype(int)
        w = sample["coef_indiv"]
        prob = bundle["model"].predict_proba(X)[:, 1]
        fpr, tpr, _ = roc_curve(y, prob, sample_weight=w)
        auc = roc_auc_score(y, prob, sample_weight=w)
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines",
            name=f"LGBM {target} (AUC {auc:.3f})",
            line=dict(width=2.5, color=PALETTE["navy"] if target == "Pauvre" else PALETTE["amber"]),
        ))

    fig.update_layout(**plotly_layout(
        height=300,
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        legend=dict(orientation="h", y=1.15),
    ))
    return fig


def fig_contribution_waterfall(contributions):
    if not contributions:
        return None
    df = pd.DataFrame(contributions)
    fig = px.bar(
        df, x="importance", y="feature", orientation="h",
        color="importance",
        color_continuous_scale=[PALETTE["gray"], PALETTE["amber"]],
        text_auto=".1f",
    )
    fig.update_layout(**plotly_layout(height=280, showlegend=False, coloraxis_showscale=False))
    fig.update_traces(textposition="outside")
    return fig


def fig_dual_comparison(lgbm, hyper):
    rows = []
    for engine, res in [("LightGBM", lgbm), ("Hypernet", hyper)]:
        for target in ENCDM_CONFIG["target"]:
            p = res[target]["probability"]
            if p is not None:
                rows.append({"Model": engine, "Target": target, "Probability": p})
    if not rows:
        return None
    df = pd.DataFrame(rows)
    colors = {"Pauvre": PALETTE["navy"], "Vulnérable": PALETTE["amber"]}
    fig = px.bar(
        df, x="Model", y="Probability", color="Target", barmode="group",
        color_discrete_map=colors, text_auto=".0%",
    )
    fig.update_layout(**plotly_layout(height=280, yaxis=dict(range=[0, 1]), legend=dict(orientation="h", y=1.12)))
    return fig


def fig_region_education(encdm, region_code, label_maps):
    sub = encdm[encdm["Région_12"] == region_code].copy()
    if len(sub) == 0:
        return None
    sub["edu_label"] = translate_series(sub["Niveau_scolaire_agreg_CM"], "Niveau_scolaire_agreg_CM", label_maps)
    sub["status"] = sub["Pauvre"].map({1: "Poor", 0: "Non-Poor"})
    grouped = (
        sub.groupby(["edu_label", "status"])["coef_ménage"]
        .sum()
        .reset_index()
    )
    fig = px.bar(
        grouped, x="edu_label", y="coef_ménage", color="status", barmode="group",
        color_discrete_map={"Poor": PALETTE["danger"], "Non-Poor": PALETTE["navy"]},
        labels={"coef_ménage": "Weighted Households", "edu_label": "Education"},
    )
    fig.update_layout(**plotly_layout(height=280, legend=dict(orientation="h", y=1.12)))
    fig.update_xaxes(tickangle=-25)
    return fig


def fig_region_gender(encdm, region_code):
    sub = encdm[encdm["Région_12"] == region_code]
    if len(sub) == 0:
        return None
    rows = []
    for sexe, label in [(1, "Male"), (2, "Female")]:
        s = sub[sub["Sexe_CM"] == sexe]
        w = s["coef_ménage"].sum()
        if w == 0:
            continue
        rate = (s["Pauvre"] * s["coef_ménage"]).sum() / w * 100
        rows.append({"Gender": label, "Poverty Rate (%)": round(rate, 1)})
    if not rows:
        return None
    df = pd.DataFrame(rows)
    fig = px.bar(
        df, x="Gender", y="Poverty Rate (%)", color="Gender",
        color_discrete_map={"Male": PALETTE["navy"], "Female": PALETTE["amber"]},
        text_auto=".1f",
    )
    fig.update_layout(**plotly_layout(height=260, showlegend=False))
    return fig


def fig_urban_rural_poverty(encdm, region_names):
    rows = []
    for code in sorted(encdm["Région_12"].unique()):
        sub = encdm[encdm["Région_12"] == code]
        for milieu, label in [(0, "Urban"), (1, "Rural")]:
            s = sub[sub["Milieu"] == milieu]
            w = s["coef_ménage"].sum()
            if w == 0:
                continue
            rate = (s["Pauvre"] * s["coef_ménage"]).sum() / w * 100
            rows.append({
                "Region": region_names.get(code, str(code)),
                "Milieu": label,
                "Poverty Rate (%)": round(rate, 1),
            })
    if not rows:
        return None
    df = pd.DataFrame(rows)
    fig = px.bar(
        df, x="Region", y="Poverty Rate (%)", color="Milieu", barmode="group",
        color_discrete_map={"Urban": PALETTE["navy"], "Rural": PALETTE["amber"]},
    )
    fig.update_layout(**plotly_layout(height=360, legend=dict(orientation="h", y=1.08)))
    fig.update_xaxes(tickangle=-35)
    return fig


def fig_rgph_housing_index(rgph, region_code):
    sub = rgph[rgph["REG"] == region_code]
    if len(sub) == 0:
        return None
    indicators = {
        "Electricity": sub["ELEC"].mean() * 100,
        "Internet": sub["NET"].mean() * 100,
        "Vehicle": sub["VOIT"].mean() * 100 if "VOIT" in sub.columns else 0,
        "Fridge": sub["FRIGO"].mean() * 100 if "FRIGO" in sub.columns else 0,
    }
    df = pd.DataFrame({"Amenity": list(indicators.keys()), "Access (%)": list(indicators.values())})
    fig = px.bar(
        df, x="Amenity", y="Access (%)", color="Access (%)",
        color_continuous_scale=[PALETTE["gray"], PALETTE["navy"]],
        text_auto=".1f", range_color=[0, 100],
    )
    fig.update_layout(**plotly_layout(height=260, coloraxis_showscale=False))
    return fig


def fig_hypernet_loss():
    import torch
    from data_loader import ROOT

    ckpt_path = ROOT / "Models/Classifier/Hypernet.pt"
    if not ckpt_path.exists():
        return None
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    history = ckpt.get("history")
    if not history:
        return None
    epochs = list(range(1, len(history) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=epochs, y=history, mode="lines+markers", name="Training Loss",
        line=dict(color=PALETTE["navy"], width=2.5),
        marker=dict(size=6, color=PALETTE["amber"]),
    ))
    fig.update_layout(**plotly_layout(
        height=280,
        xaxis_title="Epoch",
        yaxis_title="Loss",
        legend=dict(orientation="h", y=1.12),
    ))
    return fig


def translate_series(series, colname, label_maps):
    mapping = label_maps.get(colname, {})
    return series.map(mapping).fillna(series.astype(str))


def compute_region_stats(encdm, region_names):
    stats = {}
    for code in sorted(encdm["Région_12"].unique()):
        sub = encdm[encdm["Région_12"] == code]
        w = sub["coef_ménage"].sum()
        if w == 0:
            continue
        stats[int(code)] = {
            "name": region_names.get(code, f"Region {code}"),
            "poverty_rate": round((sub["Pauvre"] * sub["coef_ménage"]).sum() / w * 100, 2),
            "households": len(sub),
            "urban_pct": round(
                sub[sub["Milieu"] == 0]["coef_ménage"].sum() / w * 100, 1
            ) if "Milieu" in sub.columns else 0,
            "avg_age": round(inverse_scale_encdm(sub)["Age_CM"].mean(), 1),
            "avg_size": round(inverse_scale_encdm(sub)["Taille_ménage"].mean(), 1),
        }
    return stats
