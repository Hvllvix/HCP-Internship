"""
Interactive mapping module — unified Morocco choropleth with regional sync panels.
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from .data_loader import (
    load_morocco_geojson, load_clean_encdm, load_clean_rgph,
    load_raw_encdm, ENCDM_LABELS, RGPH_LABELS, get_column_label,
    resolve_colname
)
from .translations import (
    REGION_NAMES, DEFAULT_REGION_CODE, DEFAULT_REGION_NAME, REGION_ORDER,
    REGION_NAME_TO_CODE, GEOJSON_TO_CODE, translate_value,
)

GEOJSON_TO_ENCDM = {
    "Tanger-Tetouan-Hoceima": "Tanger-Tetouan-Al Hoceima",
    "Oriental": "Oriental",
    "Fes-Meknes": "Fes-Meknes",
    "Rabat-Sale-Kenitra": "Rabat-Sale-Kenitra",
    "Beni Mellal-Khenifra": "Beni Mellal-Khenifra",
    "Casablanca-Settat": "Casablanca-Settat",
    "Marrakech-Safi": "Marrakech-Safi",
    "Daraa-Tafilelt": "Draa-Tafilalet",
    "Souss Massa": "Souss-Massa",
    "Guelmim-Oued Noun": "Guelmim-Oued Noun",
    "Laayoune-Saguia Hamra": "Laayoune-Sakia El Hamra",
    "Dakhla-Oued Eddahab": "Dakhla-Oued Ed Dahab",
}

ENCDM_CODE_TO_GEOJSON = {
    1: "Tanger-Tetouan-Hoceima", 2: "Oriental", 3: "Fes-Meknes",
    4: "Rabat-Sale-Kenitra", 5: "Beni Mellal-Khenifra", 6: "Casablanca-Settat",
    7: "Marrakech-Safi", 8: "Daraa-Tafilelt", 9: "Souss Massa",
    10: "Guelmim-Oued Noun", 11: "Laayoune-Saguia Hamra", 12: "Dakhla-Oued Eddahab",
}

ENCDM_NAME_TO_CODE = REGION_NAME_TO_CODE


def compute_region_data(region_key=None):
    try:
        clean_df = load_clean_encdm()
        if clean_df is None:
            return {} if region_key is None else None
    except Exception:
        return {} if region_key is None else None

    region_col = resolve_colname(clean_df, "Region_12")
    if region_col not in clean_df.columns:
        return {} if region_key is None else None

    cols_available = [region_col, "Pauvre"]
    milieu_col = resolve_colname(clean_df, "Milieu")
    sexe_col = resolve_colname(clean_df, "Sexe_CM")
    age_col = resolve_colname(clean_df, "Age_CM")
    edu_col = resolve_colname(clean_df, "Niveau_scolaire_agreg_CM")
    taille_col = resolve_colname(clean_df, "Taille_ménage")
    dam_col = "DAM" if "DAM" in clean_df.columns else None

    for c in [milieu_col, sexe_col, age_col, edu_col, taille_col, dam_col]:
        if c and c in clean_df.columns:
            cols_available.append(c)

    df = clean_df[cols_available].dropna(subset=[region_col, "Pauvre"]).copy()
    if region_key is not None:
        df = df[df[region_col] == region_key]
    if len(df) == 0:
        return None if region_key is not None else {}

    result = {
        "poverty_rate": float(df["Pauvre"].mean() * 100),
        "household_count": len(df),
        "urban_pct": float(df[df[milieu_col] == 1.0].shape[0] / len(df) * 100) if milieu_col in df.columns else 0,
        "female_headed_pct": float(df[df[sexe_col] == 2.0].shape[0] / len(df) * 100) if sexe_col in df.columns else 0,
        "avg_age": float(df[age_col].mean()) if age_col in df.columns else 0,
        "avg_hh_size": float(df[taille_col].mean()) if taille_col in df.columns else 0,
        "avg_expenditure": float(df[dam_col].mean()) if dam_col and dam_col in df.columns else 0,
        "region_name": REGION_NAMES.get(float(region_key), "Morocco") if region_key else "Morocco",
    }

    if edu_col and edu_col in df.columns:
        edu = df[edu_col].value_counts(normalize=True) * 100
        result["education_no_schooling"] = float(edu.get(0.0, 0))
        result["education_primary"] = float(edu.get(2.0, 0))
        result["education_secondary"] = float(edu.get(3.0, 0) + edu.get(4.0, 0))
        result["education_higher"] = float(edu.get(5.0, 0))

    return result


def compute_all_region_data():
    all_data = {}
    for code in range(1, 13):
        geojson_name = ENCDM_CODE_TO_GEOJSON.get(code)
        if geojson_name:
            data = compute_region_data(region_key=code)
            if data:
                all_data[geojson_name] = data
    return all_data


def get_region_narrative(region_code):
    name = REGION_NAMES.get(float(region_code), DEFAULT_REGION_NAME)
    data = compute_region_data(region_key=region_code) or {}
    pr = data.get("poverty_rate", 0)
    urban = data.get("urban_pct", 0)
    edu_low = data.get("education_no_schooling", 0)

    if region_code == 10:
        intro = (
            f"<strong>{name}</strong> anchors this dashboard as the internship host region. "
            "Situated at the gateway between Atlantic Morocco and the Sahara, its economy blends "
            "pastoralism, phosphate logistics, and emerging tourism along the Guelmim corridor."
        )
    else:
        intro = (
            f"<strong>{name}</strong> exhibits a distinct socioeconomic profile within Morocco's "
            "twelve-region administrative framework, shaped by urbanization gradients, sectoral "
            "employment, and historical investment in infrastructure."
        )

    drivers = (
        f"With a measured poverty rate of <strong>{pr:.1f}%</strong>, an urban share of "
        f"<strong>{urban:.0f}%</strong>, and <strong>{edu_low:.0f}%</strong> of household heads "
        "reporting no formal schooling, local outcomes reflect the interplay of labor-market "
        "informality, household dependency ratios, and spatial access to services. "
        "RGPH housing-quality indicators further differentiate material living standards "
        "across rural communes and provincial centers."
    )
    return intro + " " + drivers


def build_morocco_choropleth(selected_region_code=DEFAULT_REGION_CODE):
    geojson = load_morocco_geojson()
    if geojson is None:
        return None

    region_data = compute_all_region_data()
    selected_geo = ENCDM_CODE_TO_GEOJSON.get(int(selected_region_code), "Guelmim-Oued Noun")

    locations, z_values, hover_texts, line_widths, line_colors = [], [], [], [], []
    for feat in geojson["features"]:
        props = feat["properties"]
        region_name = props.get("region", "")
        info = region_data.get(region_name, {})
        locations.append(region_name)
        z_values.append(info.get("poverty_rate", 0))
        hover_texts.append(
            f"<b>{region_name}</b><br>"
            f"Poverty: {info.get('poverty_rate', 0):.1f}% | "
            f"Households: {info.get('household_count', 0):,} | "
            f"Urban: {info.get('urban_pct', 0):.0f}%"
        )
        is_selected = region_name == selected_geo
        line_widths.append(2 if is_selected else 0.8)
        line_colors.append("#1A1A1A" if is_selected else "#FFFFFF")

    fig = go.Figure(go.Choropleth(
        geojson=geojson,
        locations=locations,
        z=z_values,
        featureidkey="properties.region",
        colorscale=[
            [0, "#E8E8E3"], [0.2, "#B8C5D0"], [0.4, "#7A9BA8"],
            [0.6, "#4A6B7A"], [0.8, "#2E3A6D"], [1, "#1A1A1A"],
        ],
        zmin=0,
        zmax=max(z_values) * 1.15 if z_values else 25,
        marker=dict(line=dict(width=line_widths, color=line_colors)),
        colorbar=dict(
            title=dict(text="Poverty %", font=dict(size=11)),
            thickness=14, len=0.55,
            tickfont=dict(size=10),
        ),
        text=hover_texts,
        hovertemplate="%{text}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(
            text=f"Regional Poverty Rate — {selected_geo}",
            font=dict(size=14, color="#1A1A1A", family="Inter, sans-serif"), x=0.02, xanchor="left",
        ),
        geo=dict(
            fitbounds="locations",
            projection=dict(type="mercator"),
            showframe=False,
            showcoastlines=False,
            showland=True,
            landcolor="#F5F5F0",
            bgcolor="#F5F5F0",
        ),
        height=520,
        paper_bgcolor="#F5F5F0",
        plot_bgcolor="#F5F5F0",
        font=dict(family="Inter, sans-serif", size=11, color="#1A1A1A"),
        margin=dict(l=0, r=0, t=40, b=0),
        dragmode=False,
    )
    return fig


def build_region_income_profile(region_code):
    df = load_clean_encdm()
    if df is None:
        return None
    region_col = resolve_colname(df, "Region_12")
    dam_col = "DAM" if "DAM" in df.columns else None
    if not dam_col or region_col not in df.columns:
        return None

    subset = df[df[region_col] == region_code][[dam_col, "Pauvre"]].dropna()
    if len(subset) == 0:
        return None

    subset = subset.copy()
    subset["status"] = subset["Pauvre"].map({0: "Non pauvre", 1: "Pauvre"})

    fig = px.histogram(
        subset, x=dam_col, color="status", nbins=30,
        barmode="overlay", opacity=0.8,
        color_discrete_map={"Non pauvre": "#2563EB", "Pauvre": "#BE123C"},
        labels={dam_col: "Annual Expenditure (MAD)", "status": "Status"},
    )
    name = REGION_NAMES.get(float(region_code), "")
    fig.update_layout(
        title=dict(text=f"Spending Profile — {name}", font=dict(size=13, family="Inter", color="#1A1A1A"), x=0.02),
        height=320,
        paper_bgcolor="#F5F5F0", plot_bgcolor="#FFFFFF",
        font=dict(family="Inter, sans-serif", size=10, color="#1A1A1A"),
        legend=dict(font=dict(size=10)),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#E0E0E0", linecolor="#E0E0E0", tickfont=dict(size=9))
    fig.update_yaxes(showgrid=True, gridcolor="#E0E0E0", linecolor="#E0E0E0", tickfont=dict(size=9))
    return fig


def build_region_amenity_bars(region_code):
    df = load_clean_rgph()
    if df is None or "REG" not in df.columns:
        return None
    subset = df[df["REG"] == region_code]
    if len(subset) == 0:
        return None

    amenity_cols = [c for c in ["ELEC", "EAU.MODE", "NET", "FRIGO", "VOIT", "TEL.PORT"] if c in subset.columns]
    if not amenity_cols:
        return None

    rates = []
    for col in amenity_cols:
        count = len(subset)
        if count == 0:
            continue
        if col == "ELEC":
            rate = float((subset[col] == 1.0).mean() * 100)
        elif col == "EAU.MODE":
            rate = float((subset[col].isin([1.0, 2.0])).mean() * 100)
        elif col == "VOIT":
            rate = float((subset[col] >= 1.0).mean() * 100)
        else:
            rate = float((subset[col] == 1.0).mean() * 100)
        rates.append({"amenity": get_column_label(col, dataset="rgph"), "access_rate": rate})

    if not rates:
        return None

    rate_df = pd.DataFrame(rates).sort_values("access_rate", ascending=True)
    fig = go.Figure(go.Bar(
        x=rate_df["access_rate"], y=rate_df["amenity"], orientation="h",
        marker=dict(color="#0D9488"),
        text=[f"{v:.0f}%" for v in rate_df["access_rate"]], textposition="outside",
        textfont=dict(size=9),
    ))
    name = REGION_NAMES.get(float(region_code), "")
    fig.update_layout(
        title=dict(text=f"Household Amenity Access — {name}", font=dict(size=13, family="Inter", color="#1A1A1A"), x=0.02),
        xaxis_title="Access Rate (%)", height=320,
        paper_bgcolor="#F5F5F0", plot_bgcolor="#FFFFFF",
        font=dict(family="Inter, sans-serif", size=10, color="#1A1A1A"),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    fig.update_xaxes(range=[0, 105], showgrid=True, gridcolor="#E0E0E0", linecolor="#E0E0E0", tickfont=dict(size=9))
    fig.update_yaxes(tickfont=dict(size=9))
    return fig


def build_region_education_profile(region_code):
    df = load_clean_encdm()
    if df is None:
        return None
    region_col = resolve_colname(df, "Region_12")
    edu_col = resolve_colname(df, "Niveau_scolaire_agreg_CM")
    if region_col not in df.columns or edu_col not in df.columns:
        return None

    subset = df[df[region_col] == region_code][[edu_col, "Pauvre"]].dropna()
    if len(subset) == 0:
        return None

    cross = subset.groupby([edu_col, "Pauvre"]).size().unstack(fill_value=0)
    cross.index = cross.index.map(lambda x: ENCDM_LABELS.get("Niveau_scolaire_agreg_CM", {}).get(x, str(x)))
    cross.columns = ["Non pauvre", "Pauvre"]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=cross.index, y=cross["Non pauvre"], name="Non pauvre",
        marker_color="#2563EB", text=cross["Non pauvre"], textposition="outside",
        textfont=dict(size=9), hovertemplate="<b>%{x}</b><br>Non pauvre: %{y:,.0f}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=cross.index, y=cross["Pauvre"], name="Pauvre",
        marker_color="#BE123C", text=cross["Pauvre"], textposition="outside",
        textfont=dict(size=9), hovertemplate="<b>%{x}</b><br>Pauvre: %{y:,.0f}<extra></extra>",
    ))

    name = REGION_NAMES.get(float(region_code), "")
    fig.update_layout(
        title=dict(text=f"Education vs Poverty — {name}", font=dict(size=13, family="Inter", color="#1A1A1A"), x=0.02),
        barmode="group", height=320,
        paper_bgcolor="#F5F5F0", plot_bgcolor="#FFFFFF",
        font=dict(family="Inter, sans-serif", size=10, color="#1A1A1A"),
        legend=dict(font=dict(size=10)),
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(tickangle=30, tickfont=dict(size=9)),
    )
    fig.update_yaxes(showgrid=True, gridcolor="#E0E0E0", linecolor="#E0E0E0", tickfont=dict(size=9))
    return fig


def build_urban_rural_map():
    try:
        raw_df, _ = load_raw_encdm()
        if raw_df is None:
            return None
        region_col = resolve_colname(raw_df, "Region_12")
        milieu_col = resolve_colname(raw_df, "Milieu")
        df_plot = raw_df[[region_col, milieu_col]].dropna().copy()
        df_plot["region"] = df_plot[region_col].map(ENCDM_LABELS["Region_12"])
        df_plot["milieu"] = df_plot[milieu_col].map(ENCDM_LABELS["Milieu"])
    except Exception:
        return None

    cross = df_plot.groupby(["region", "milieu"]).size().reset_index(name="count")
    total = cross.groupby("region")["count"].sum().reset_index(name="total")
    cross = cross.merge(total, on="region")
    cross["pct"] = cross["count"] / cross["total"] * 100

    fig = go.Figure()
    for milieu, color in [("Urbain", "#1A3A5C"), ("Rural", "#2C7A7B")]:
        subset = cross[cross["milieu"] == milieu]
        fig.add_trace(go.Bar(x=subset["region"], y=subset["pct"], name=milieu,
            marker_color=color, text=[f"{v:.0f}%" for v in subset["pct"]],
            textposition="inside", textfont=dict(size=9, color="white")))
    fig.update_layout(title=dict(text="Urban/Rural Composition by Region", font=dict(size=14)),
        barmode="stack", height=400, paper_bgcolor="#F8F9FA", plot_bgcolor="#FFFFFF",
        font=dict(family="Inter, sans-serif", size=11, color="#212529"),
        xaxis=dict(tickangle=45), margin=dict(b=80))
    return fig


def build_region_gender_profile(region_code):
    df = load_clean_encdm()
    if df is None:
        return None
    region_col = resolve_colname(df, "Region_12")
    sexe_col = resolve_colname(df, "Sexe_CM")
    if region_col not in df.columns or sexe_col not in df.columns:
        return None

    subset = df[df[region_col] == region_code][[sexe_col, "Pauvre"]].dropna()
    if len(subset) == 0:
        return None

    cross = subset.groupby([sexe_col, "Pauvre"]).size().unstack(fill_value=0)
    cross.index = cross.index.map(lambda x: ENCDM_LABELS.get("Sexe_CM", {}).get(x, str(x)))
    cross.columns = ["Non pauvre", "Pauvre"]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=cross.index, y=cross["Non pauvre"], name="Non pauvre",
        marker_color="#2563EB", text=cross["Non pauvre"], textposition="outside",
        textfont=dict(size=9),
    ))
    fig.add_trace(go.Bar(
        x=cross.index, y=cross["Pauvre"], name="Pauvre",
        marker_color="#BE123C", text=cross["Pauvre"], textposition="outside",
        textfont=dict(size=9),
    ))

    name = REGION_NAMES.get(float(region_code), "")
    fig.update_layout(
        title=dict(text=f"Gender and Poverty — {name}", font=dict(size=13, family="Inter", color="#1A1A1A"), x=0.02),
        barmode="group", height=320,
        paper_bgcolor="#F5F5F0", plot_bgcolor="#FFFFFF",
        font=dict(family="Inter, sans-serif", size=10, color="#1A1A1A"),
        legend=dict(font=dict(size=10)),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    fig.update_yaxes(showgrid=True, gridcolor="#E0E0E0", linecolor="#E0E0E0", tickfont=dict(size=9))
    return fig


def build_rgph_housing_map():
    df = load_clean_rgph()
    if df is None or "REG" not in df.columns:
        return None
    quality_cols = [c for c in ["TYPE.LOG", "MURS", "TOIT", "SOL", "GAZ", "ELEC", "EAU.MODE"] if c in df.columns]
    if not quality_cols:
        return None
    region_stats = df.groupby("REG")[quality_cols].mean().reset_index()
    region_stats["quality_index"] = region_stats[quality_cols].mean(axis=1)
    region_stats["region"] = region_stats["REG"].map(RGPH_LABELS["REG"])
    region_stats = region_stats.sort_values("quality_index", ascending=True)
    colors = ["#9B4D4D" if v < region_stats["quality_index"].median() else "#1A3A5C"
              for v in region_stats["quality_index"]]
    fig = go.Figure(go.Bar(x=region_stats["region"], y=region_stats["quality_index"],
        marker_color=colors, text=[f"{v:.2f}" for v in region_stats["quality_index"]], textposition="outside"))
    fig.update_layout(
        title=dict(text="Housing Quality Index by Region", font=dict(size=14, family="Inter", color="#1A1A1A"), x=0.02),
        height=360, paper_bgcolor="#F5F5F0", plot_bgcolor="#FFFFFF",
        font=dict(family="Inter, sans-serif", size=10, color="#1A1A1A"),
        xaxis=dict(tickangle=45, tickfont=dict(size=9)), margin=dict(b=60),
    )
    fig.update_yaxes(showgrid=True, gridcolor="#E0E0E0", linecolor="#E0E0E0", tickfont=dict(size=9))
    return fig


def get_all_maps():
    return [
        ("Morocco Regional Poverty Map", build_morocco_choropleth,
         "Unified choropleth — defaults to Guelmim-Oued Noun"),
        ("Urban/Rural Composition", build_urban_rural_map,
         "Urban vs Rural distribution across Moroccan regions"),
        ("RGPH Housing Quality", build_rgph_housing_map,
         "Composite housing quality index by region from census data"),
    ]
