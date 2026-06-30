"""Plot helpers for dashboard sections."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import roc_auc_score, roc_curve

from Utils.data_loader import ENCDM_CONFIG, get_label, household_weights, inverse_scale_encdm, weighted_poverty_rate
from Utils.theme import PALETTE, plotly_layout

CHART_H = 300
CHART_H_TALL = 380
LOG_EPS = 1e-4


def _layout(height=None, **kwargs):
    return plotly_layout(height=height or CHART_H, **kwargs)


def _log_prob(p):
    return float(np.log10(max(p, LOG_EPS)))


def fig_dataset_dims(raw_encdm, raw_rgph, clean_encdm=None, clean_rgph=None):
    rows = []
    for name, df in [("ENCDM Raw", raw_encdm), ("RGPH Raw", raw_rgph)]:
        if df is not None and len(df):
            rows.append({"Dataset": name, "Metric": "Rows", "Value": len(df)})
            rows.append({"Dataset": name, "Metric": "Columns", "Value": len(df.columns)})
    if clean_encdm is not None:
        rows.append({"Dataset": "ENCDM Clean", "Metric": "Rows", "Value": len(clean_encdm)})
        rows.append({"Dataset": "ENCDM Clean", "Metric": "Columns", "Value": len(clean_encdm.columns)})
    if clean_rgph is not None:
        rows.append({"Dataset": "RGPH Clean", "Metric": "Rows", "Value": len(clean_rgph)})
        rows.append({"Dataset": "RGPH Clean", "Metric": "Columns", "Value": len(clean_rgph.columns)})
    if not rows:
        return None
    df = pd.DataFrame(rows)
    fig = px.bar(
        df, x="Dataset", y="Value", color="Metric", barmode="group",
        color_discrete_map={"Rows": PALETTE["navy"], "Columns": PALETTE["amber"]},
        text_auto=",",
    )
    fig.update_layout(**_layout(CHART_H, legend=dict(orientation="h", y=1.12)))
    fig.update_traces(textposition="outside")
    return fig


def fig_null_counts(raw_encdm, raw_rgph):
    rows = []
    for name, df in [("ENCDM", raw_encdm), ("RGPH", raw_rgph)]:
        if df is None or len(df) == 0:
            continue
        nulls = int(df.isna().sum().sum())
        cells = int(df.size)
        rows.append({"Dataset": name, "Null Cells": nulls, "Filled Cells": cells - nulls})
    if not rows:
        return None
    df = pd.DataFrame(rows)
    fig = px.bar(
        df, x="Dataset", y=["Null Cells", "Filled Cells"], barmode="stack",
        color_discrete_map={"Null Cells": PALETTE["amber"], "Filled Cells": PALETTE["navy"]},
    )
    fig.update_layout(**_layout(CHART_H, legend=dict(orientation="h", y=1.12)))
    return fig


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
        color_continuous_scale=[PALETTE["zinc400"], PALETTE["amber"], PALETTE["navy"]],
    )
    fig.update_layout(**_layout())
    fig.update_xaxes(tickangle=-35)
    return fig


def fig_raw_missing_values(raw_encdm, raw_rgph, top_n=12):
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("ENCDM Raw", "RGPH Raw"),
        horizontal_spacing=0.12,
    )
    has_trace = False
    for col_idx, df in enumerate([raw_encdm, raw_rgph], 1):
        if df is None or len(df) == 0:
            continue
        miss_pct = (df.isna().sum() / len(df) * 100)
        miss_pct = miss_pct[miss_pct > 0].sort_values(ascending=False).head(top_n)
        if len(miss_pct) == 0:
            continue
        has_trace = True
        colors = [
            PALETTE["danger"] if v > 5 else PALETTE["amber"] if v > 1 else PALETTE["navy"]
            for v in miss_pct.values
        ]
        fig.add_trace(
            go.Bar(
                x=miss_pct.values,
                y=[get_label(c) if get_label(c) != c else str(c)[:28] for c in miss_pct.index],
                orientation="h",
                marker_color=colors,
                text=[f"{v:.1f}%" for v in miss_pct.values],
                textposition="outside",
                showlegend=False,
            ),
            row=1, col=col_idx,
        )
        fig.update_xaxes(
            title_text="Missing %",
            range=[0, min(105, miss_pct.max() * 1.15 + 5)],
            row=1, col=col_idx,
        )
    if not has_trace:
        return None
    fig.update_layout(**_layout(CHART_H_TALL, title="Raw Dataset Missingness (Pre-Imputation)"))
    return fig


def fig_vulnerability_breakdown(encdm):
    w = household_weights(encdm)
    tmp = encdm.assign(_w=w)
    vuln = tmp.loc[tmp["Vulnérable"] == 1, "_w"].sum()
    nonvuln = tmp.loc[tmp["Vulnérable"] == 0, "_w"].sum()
    df = pd.DataFrame({"Status": ["Not Vulnerable", "Vulnerable"], "Weight": [nonvuln, vuln]})
    fig = px.pie(
        df, values="Weight", names="Status",
        color_discrete_sequence=[PALETTE["navy"], PALETTE["amber"]],
    )
    fig.update_layout(**_layout())
    return fig


def fig_national_employment(encdm, label_maps):
    w = household_weights(encdm)
    sub = encdm.copy()
    sub["employment"] = translate_series(
        sub["Situation_profession_agreg_CM"], "Situation_profession_agreg_CM", label_maps
    )
    grouped = sub.assign(_w=w).groupby("employment")["_w"].sum().sort_values(ascending=False).head(10).reset_index()
    grouped.columns = ["Employment", "Weight"]
    fig = px.bar(grouped, x="Weight", y="Employment", orientation="h", color_discrete_sequence=[PALETTE["navy"]])
    fig.update_layout(**_layout(CHART_H_TALL, showlegend=False))
    return fig


def fig_household_size_national(encdm):
    sub = inverse_scale_encdm(encdm)
    fig = px.histogram(sub, x="Taille_ménage", nbins=14, color_discrete_sequence=[PALETTE["amber"]])
    fig.update_layout(**_layout(xaxis_title="Household size", yaxis_title="Count"))
    return fig


def fig_rgph_infrastructure(rgph):
    cols = {"ELEC": "Electricity", "NET": "Internet", "VOIT": "Vehicle", "FRIGO": "Fridge"}
    rows = []
    for col, label in cols.items():
        if col in rgph.columns:
            rows.append({"Asset": label, "Access (%)": round(rgph[col].mean() * 100, 1)})
    if not rows:
        return None
    df = pd.DataFrame(rows)
    fig = px.bar(
        df, x="Asset", y="Access (%)", color="Access (%)",
        color_continuous_scale=[PALETTE["zinc400"], PALETTE["navy"]],
        text_auto=".1f", range_color=[0, 100],
    )
    fig.update_layout(**_layout(coloraxis_showscale=False))
    return fig


def fig_region_household_size(encdm, region_code):
    sub = inverse_scale_encdm(encdm[encdm["Région_12"] == region_code])
    if len(sub) == 0:
        return None
    fig = px.histogram(sub, x="Taille_ménage", nbins=12, color_discrete_sequence=[PALETTE["amber"]])
    fig.update_layout(**_layout(xaxis_title="Household size", yaxis_title="Count"))
    return fig


def fig_poverty_breakdown(encdm):
    w = household_weights(encdm)
    tmp = encdm.assign(_w=w)
    poor_w = tmp.loc[tmp["Pauvre"] == 1, "_w"].sum()
    vuln_w = tmp.loc[tmp["Vulnérable"] == 1, "_w"].sum()
    nonpoor_w = tmp.loc[(tmp["Pauvre"] == 0) & (tmp["Vulnérable"] == 0), "_w"].sum()
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
    z_data, locations, hover, custom_ids = [], [], [], []
    sel_id = None
    for feat in geojson["features"]:
        props = feat["properties"]
        gname = props["region"]
        cid = props["cartodb_id"]
        code = geojson_regions.get(gname)
        locations.append(cid)
        custom_ids.append(str(cid))
        if code is not None and code in region_stats:
            rate = region_stats[code]["poverty_rate"]
            z_data.append(rate)
            hover.append(region_stats[code]["name"])
            if code == selected_code:
                sel_id = cid
        else:
            z_data.append(None)
            hover.append(gname)

    # Apply log scaling to make regional differences more visible
    z_scaled = [np.log1p(v) if v is not None else None for v in z_data]

    bg = PALETTE["bg"]
    fig = go.Figure()
    fig.add_trace(go.Choropleth(
        geojson=geojson,
        featureidkey="properties.cartodb_id",
        locations=locations,
        z=z_scaled,
        text=hover,
        customdata=custom_ids,
        hovertemplate="<b>%{text}</b><br>Poverty: %{z:.1f}%<extra></extra>",
        colorscale=[
            [0, PALETTE["white"]],
            [0.25, PALETTE["amber_soft"]],
            [0.5, PALETTE["amber"]],
            [0.75, PALETTE["navy_light"]],
            [1, PALETTE["navy"]],
        ],
        zmin=5, zmax=45,
        marker=dict(line=dict(width=1.2, color=PALETTE["zinc700"])),
        colorbar=dict(title="Poverty %", thickness=12, len=0.55),
        name="regions",
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
            name="selected",
        ))
    fig.update_layout(
        **plotly_layout(
            height=480,
            margin=dict(l=0, r=0, t=30, b=0),
            title="Click a region to filter panels below",
            paper_bgcolor=bg,
            plot_bgcolor=bg,
        ),
        geo=dict(
            scope="africa",
            projection_type="mercator",
            showland=False,
            showocean=False,
            showcountries=False,
            showcoastlines=False,
            showlakes=False,
            showrivers=False,
            showframe=False,
            bgcolor=bg,
            lonaxis=dict(range=[-17.2, -0.8]),
            lataxis=dict(range=[21.0, 36.2]),
        ),
        clickmode="event+select",
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
        color_continuous_scale=[PALETTE["zinc400"], PALETTE["amber"], PALETTE["navy"]],
        labels={"x": "Importance", "y": ""},
    )
    fig.update_layout(**plotly_layout(height=320, showlegend=False, coloraxis_showscale=False))
    return fig


def fig_roc_curves(encdm, bundles, sample_n=2000):
    rng = np.random.default_rng(42)
    idx = rng.choice(len(encdm), size=min(sample_n, len(encdm)), replace=False)
    sample = encdm.iloc[idx]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines", name="Random",
        line=dict(color=PALETTE["zinc400"], dash="dash"),
    ))
    for target in ENCDM_CONFIG["target"]:
        bundle = bundles.get(target)
        if bundle is None:
            continue
        X = sample[bundle["features"]].astype("float32")
        y = sample[target].astype(int)
        prob = bundle["model"].predict_proba(X)[:, 1]
        try:
            auc = roc_auc_score(y, prob)
        except ValueError:
            auc = float("nan")
        fpr, tpr, _ = roc_curve(y, prob)
        order = np.argsort(fpr)
        fpr_s = np.clip(fpr[order], 0, 1)
        tpr_s = np.maximum.accumulate(tpr[order])
        fig.add_trace(go.Scatter(
            x=fpr_s, y=tpr_s, mode="lines",
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
        color_continuous_scale=[PALETTE["zinc400"], PALETTE["amber"]],
        text_auto=".1f",
    )
    fig.update_layout(**plotly_layout(height=280, showlegend=False, coloraxis_showscale=False))
    fig.update_traces(textposition="outside")
    return fig


def fig_dual_comparison(lgbm, hyper, log_scale=True):
    rows = []
    for engine, res in [("LightGBM", lgbm), ("Hypernet", hyper)]:
        for target in ENCDM_CONFIG["target"]:
            p = res[target]["probability"]
            if p is not None:
                val = _log_prob(p) if log_scale else p
                rows.append({
                    "Model": engine,
                    "Target": target,
                    "Value": val,
                    "Raw": p,
                })
    if not rows:
        return None
    df = pd.DataFrame(rows)
    colors = {"Pauvre": PALETTE["navy"], "Vulnérable": PALETTE["amber"]}
    ylabel = "log10(probability)" if log_scale else "Probability"
    fig = px.bar(
        df, x="Model", y="Value", color="Target", barmode="group",
        color_discrete_map=colors,
        text=[f"{r['Raw']:.2%}" for r in rows],
    )
    fig.update_layout(**plotly_layout(
        height=300,
        yaxis_title=ylabel,
        legend=dict(orientation="h", y=1.12),
    ))
    fig.update_traces(textposition="outside")
    return fig


def fig_region_education(encdm, region_code, label_maps):
    sub = encdm[encdm["Région_12"] == region_code].copy()
    if len(sub) == 0:
        return None
    w = household_weights(sub)
    sub["edu_label"] = translate_series(sub["Niveau_scolaire_agreg_CM"], "Niveau_scolaire_agreg_CM", label_maps)
    grouped = sub.assign(_w=w).groupby("edu_label")["_w"].sum().reset_index()
    grouped.columns = ["Education", "Weighted Households"]
    fig = px.bar(grouped, x="Education", y="Weighted Households", color_discrete_sequence=[PALETTE["navy"]])
    fig.update_layout(**plotly_layout(height=280, showlegend=False))
    fig.update_xaxes(tickangle=-25)
    return fig


def fig_region_gender(encdm, region_code):
    """Sexe_CM: 0=Masculin, 1=Féminin per MapENCDM.json."""
    sub = encdm[encdm["Région_12"] == region_code]
    if len(sub) == 0:
        return None
    w = household_weights(sub)
    total = w.sum()
    if total <= 0:
        return None
    rows = []
    for code, label in [(0, "Male"), (1, "Female")]:
        mask = sub["Sexe_CM"] == code
        share = round(w[mask].sum() / total * 100, 1)
        rows.append({"Gender": label, "Weighted Share (%)": share})
    df = pd.DataFrame(rows)
    fig = px.bar(
        df, x="Gender", y="Weighted Share (%)", color="Gender",
        color_discrete_map={"Male": PALETTE["navy"], "Female": PALETTE["amber"]},
        text_auto=".1f",
    )
    fig.update_layout(**plotly_layout(height=260, showlegend=False))
    return fig


def fig_region_amenities(rgph, region_code):
    sub = rgph[rgph["REG"] == region_code]
    if len(sub) == 0:
        return None
    has_water = sub["EAU.MODE"].isin([1, 2]).mean() * 100
    df = pd.DataFrame({
        "Amenity": ["Electricity", "Improved Water", "Internet"],
        "Access (%)": [sub["ELEC"].mean() * 100, has_water, sub["NET"].mean() * 100],
    })
    fig = px.bar(
        df, x="Amenity", y="Access (%)", color="Access (%)",
        color_continuous_scale=[PALETTE["zinc400"], PALETTE["navy"]],
        text_auto=".1f", range_color=[0, 100],
    )
    fig.update_layout(**plotly_layout(height=280, coloraxis_showscale=False))
    return fig


def fig_urban_rural_poverty(encdm, region_names):
    rows = []
    for code in sorted(encdm["Région_12"].unique()):
        sub = encdm[encdm["Région_12"] == code]
        w = household_weights(sub)
        for milieu, label in [(0, "Urban"), (1, "Rural")]:
            mask = sub["Milieu"] == milieu
            wt = w[mask].sum()
            if wt == 0:
                continue
            rate = (sub.loc[mask, "Pauvre"] * w[mask]).sum() / wt * 100
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
        color_continuous_scale=[PALETTE["zinc400"], PALETTE["navy"]],
        text_auto=".1f", range_color=[0, 100],
    )
    fig.update_layout(**plotly_layout(height=260, coloraxis_showscale=False))
    return fig


def fig_hypernet_loss():
    import torch
    from Utils.data_loader import ROOT

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
        yaxis_type="log",
        legend=dict(orientation="h", y=1.12),
    ))
    return fig


def translate_series(series, colname, label_maps):
    mapping = label_maps.get(colname, {})
    return series.map(mapping).fillna(series.astype(str))


def compute_region_stats(encdm, region_names):
    stats = {}
    w_all = household_weights(encdm)
    for code in sorted(encdm["Région_12"].unique()):
        mask = encdm["Région_12"] == code
        sub = encdm[mask]
        w = w_all[mask]
        total = w.sum()
        if total <= 0:
            continue
        stats[int(code)] = {
            "name": region_names.get(code, f"Region {code}"),
            "poverty_rate": round(weighted_poverty_rate(sub, w), 2),
            "households": len(sub),
            "urban_pct": round(float(w[sub["Milieu"] == 0].sum() / total * 100), 1) if "Milieu" in sub.columns else 0,
            "avg_age": round(float(inverse_scale_encdm(sub)["Age_CM"].mean()), 1),
            "avg_size": round(float(inverse_scale_encdm(sub)["Taille_ménage"].mean()), 1),
        }
    return stats


def fig_employment_mix(encdm, region_code, label_maps):
    sub = encdm[encdm["Région_12"] == region_code].copy()
    if len(sub) == 0:
        return None
    w = household_weights(sub)
    sub["employment"] = translate_series(
        sub["Situation_profession_agreg_CM"], "Situation_profession_agreg_CM", label_maps
    )
    grouped = sub.assign(_w=w).groupby("employment")["_w"].sum().sort_values(ascending=False).head(8).reset_index()
    grouped.columns = ["Employment", "Weight"]
    fig = px.bar(grouped, x="Weight", y="Employment", orientation="h", color_discrete_sequence=[PALETTE["amber"]])
    fig.update_layout(**plotly_layout(height=300, showlegend=False))
    return fig


def fig_milieu_split(encdm, region_code):
    sub = encdm[encdm["Région_12"] == region_code]
    if len(sub) == 0:
        return None
    w = household_weights(sub)
    urban = w[sub["Milieu"] == 0].sum()
    rural = w[sub["Milieu"] == 1].sum()
    df = pd.DataFrame({"Area": ["Urban", "Rural"], "Weight": [urban, rural]})
    fig = px.pie(df, values="Weight", names="Area", color_discrete_sequence=[PALETTE["navy"], PALETTE["amber"]])
    fig.update_layout(**plotly_layout(height=280))
    return fig


def fig_age_distribution(encdm, region_code):
    sub = inverse_scale_encdm(encdm[encdm["Région_12"] == region_code])
    if len(sub) == 0:
        return None
    fig = px.histogram(sub, x="Age_CM", nbins=28, color_discrete_sequence=[PALETTE["navy"]])
    fig.update_layout(**plotly_layout(height=280, xaxis_title="Age (years)", yaxis_title="Individuals"))
    return fig


def fig_rgph_rooms(rgph, region_code):
    sub = rgph[rgph["REG"] == region_code]
    if len(sub) == 0 or "PIECES" not in sub.columns:
        return None
    fig = px.histogram(sub, x="PIECES", nbins=12, color_discrete_sequence=[PALETTE["navy_light"]])
    fig.update_layout(**plotly_layout(height=280, xaxis_title="Rooms", yaxis_title="Households"))
    return fig


def fig_region_poverty_vs_vulnerability(encdm, region_code):
    """Scatter of poverty vs vulnerability rates for a region."""
    sub = encdm[encdm["Région_12"] == region_code]
    if len(sub) == 0:
        return None
    w = household_weights(sub)
    df = sub.assign(_w=w)
    rates = df.groupby("Taille_agregée").apply(
        lambda g: pd.Series({
            "Poverty": (g["Pauvre"] * g["_w"]).sum() / g["_w"].sum() * 100,
            "Vulnerability": (g["Vulnérable"] * g["_w"]).sum() / g["_w"].sum() * 100,
            "Weight": g["_w"].sum(),
        })
    ).reset_index()
    rates.columns = ["Size Category", "Poverty Rate (%)", "Vulnerability Rate (%)", "Weight"]
    fig = px.scatter(
        rates, x="Poverty Rate (%)", y="Vulnerability Rate (%)",
        size="Weight", color="Size Category",
        color_discrete_sequence=[PALETTE["navy"], PALETTE["amber"], PALETTE["navy_light"], PALETTE["zinc400"]],
        hover_name="Size Category",
    )
    fig.update_layout(**_layout(height=280, showlegend=False))
    return fig


def fig_region_education_gender(encdm, region_code, label_maps):
    """Education distribution by gender for a region."""
    sub = encdm[encdm["Région_12"] == region_code].copy()
    if len(sub) == 0:
        return None
    w = household_weights(sub)
    sub["edu"] = translate_series(sub["Niveau_scolaire_agreg_CM"], "Niveau_scolaire_agreg_CM", label_maps)
    sub["gender"] = sub["Sexe_CM"].map({0: "Male", 1: "Female"})
    grouped = sub.assign(_w=w).groupby(["edu", "gender"])["_w"].sum().reset_index()
    grouped.columns = ["Education", "Gender", "Weight"]
    fig = px.bar(
        grouped, x="Education", y="Weight", color="Gender", barmode="group",
        color_discrete_map={"Male": PALETTE["navy"], "Female": PALETTE["amber"]},
    )
    fig.update_layout(**_layout(height=280, legend=dict(orientation="h", y=1.12)))
    fig.update_xaxes(tickangle=-25)
    return fig


def fig_national_region_comparison(regstats):
    """All regions compared on urban share, avg age, and hh size."""
    df = pd.DataFrame([
        {
            "Region": r["name"],
            "Urban Share (%)": r["urban_pct"],
            "Avg Age": r["avg_age"],
            "Avg HH Size": r["avg_size"],
        }
        for r in regstats.values()
    ]).sort_values("Urban Share (%)", ascending=False)

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Urban Share (%)", "Average Age", "Avg Household Size"),
        horizontal_spacing=0.08,
    )
    metrics = [
        ("Urban Share (%)", PALETTE["navy"]),
        ("Avg Age", PALETTE["amber"]),
        ("Avg HH Size", PALETTE["navy_light"]),
    ]
    for i, (metric, color) in enumerate(metrics, 1):
        sorted_df = df.sort_values(metric, ascending=False)
        fig.add_trace(
            go.Bar(
                x=sorted_df["Region"], y=sorted_df[metric],
                marker_color=color,
                text=[f"{v:.1f}" for v in sorted_df[metric]],
                textposition="outside",
                showlegend=False,
            ),
            row=1, col=i,
        )
        fig.update_xaxes(tickangle=-35, row=1, col=i)
    fig.update_layout(**_layout(height=380))
    return fig


def fig_national_education(encdm, label_maps):
    w = household_weights(encdm)
    sub = encdm.copy()
    sub["edu"] = translate_series(sub["Niveau_scolaire_agreg_CM"], "Niveau_scolaire_agreg_CM", label_maps)
    grouped = sub.assign(_w=w).groupby("edu")["_w"].sum().sort_values(ascending=False).reset_index()
    grouped.columns = ["Education", "Weight"]
    fig = px.bar(grouped, x="Education", y="Weight", color_discrete_sequence=[PALETTE["navy"]])
    fig.update_layout(**plotly_layout(height=320, showlegend=False))
    fig.update_xaxes(tickangle=-30)
    return fig
