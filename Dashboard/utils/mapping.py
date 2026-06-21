"""
Interactive mapping module for the HCP Poverty Dashboard.
Includes interactive Morocco choropleth map using MoroccoGeoMap.geojson with region click callbacks.
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from .data_loader import (
    load_morocco_geojson, load_clean_encdm, load_clean_rgph,
    load_raw_encdm, ENCDM_LABELS, RGPH_LABELS, get_column_label,
    resolve_colname
)

# Region name normalization between GeoJSON properties and ENCDM codes
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


def _get_raw_col(raw_df, name):
    """Get column from raw ENCDM, resolving accented names."""
    if raw_df is None:
        return None
    if name in raw_df.columns:
        return name
    # Try direct match with accented chars (normalized comparison)
    name_norm = name.lower().replace(" ", "").replace("-", "").replace("_", "").replace("é", "e").replace("è", "e").replace("ê", "e")
    for col in raw_df.columns:
        col_norm = col.lower().replace(" ", "").replace("-", "").replace("_", "").replace("é", "e").replace("è", "e").replace("ê", "e")
        if col_norm == name_norm:
            return col
    return name


def compute_region_data(region_key=None):
    """
    Compute socioeconomic data for all (or one) region(s) using clean ENCDM parquet.
    If region_key is None, returns data for all regions.
    """
    try:
        clean_df = load_clean_encdm()
        if clean_df is None:
            return {} if region_key is None else None
    except Exception:
        return {} if region_key is None else None

    region_col = resolve_colname(clean_df, "Region_12")
    if region_col not in clean_df.columns:
        return {} if region_key is None else None

    # Use clean parquet data - all numeric, properly formatted
    cols_available = [region_col, "Pauvre"]
    milieu_col = resolve_colname(clean_df, "Milieu")
    if milieu_col in clean_df.columns:
        cols_available.append(milieu_col)
    sexe_col = resolve_colname(clean_df, "Sexe_CM")
    if sexe_col in clean_df.columns:
        cols_available.append(sexe_col)
    age_col = resolve_colname(clean_df, "Age_CM")
    if age_col in clean_df.columns:
        cols_available.append(age_col)
    edu_col = resolve_colname(clean_df, "Niveau_scolaire_agreg_CM")
    if edu_col in clean_df.columns:
        cols_available.append(edu_col)

    df = clean_df[cols_available].dropna(subset=[region_col, "Pauvre"]).copy()

    if region_key is not None:
        df = df[df[region_col] == region_key]

    if len(df) == 0:
        return None if region_key is not None else {}

    result = {
        "poverty_rate": float(df["Pauvre"].mean() * 100),
        "household_count": len(df),
        "urban_pct": float(df[df[milieu_col] == 1.0].shape[0] / len(df) * 100) if milieu_col and milieu_col in df.columns else 0,
        "female_headed_pct": float(df[df[sexe_col] == 2.0].shape[0] / len(df) * 100) if sexe_col and sexe_col in df.columns else 0,
        "avg_age": float(df[age_col].mean()) if age_col and age_col in df.columns else 0,
    }

    if edu_col and edu_col in df.columns:
        edu = df[edu_col].value_counts(normalize=True) * 100
        result["education_no_schooling"] = float(edu.get(0.0, 0))
        result["education_primary"] = float(edu.get(2.0, 0))
        result["education_secondary"] = float(edu.get(3.0, 0) + edu.get(4.0, 0))
        result["education_higher"] = float(edu.get(5.0, 0))

    return result


def compute_all_region_data():
    """Compute data for all 12 regions."""
    all_data = {}
    for code in range(1, 13):
        geojson_name = ENCDM_CODE_TO_GEOJSON.get(code)
        if geojson_name:
            data = compute_region_data(region_key=code)
            if data:
                all_data[geojson_name] = data
    return all_data


@st.cache_resource
def build_morocco_choropleth():
    """Build interactive choropleth map of Morocco using GeoJSON."""
    geojson = load_morocco_geojson()
    if geojson is None:
        return None

    # Compute data for each region
    region_data = compute_all_region_data()

    # Build features for map
    geojson_features = []
    for feat in geojson["features"]:
        props = feat["properties"]
        region_name = props.get("region", "")
        region_info = region_data.get(region_name, {})
        feat_copy = dict(feat)
        # Copy in poverty data as properties for coloring
        feat_copy["properties"] = {
            "region": region_name,
            "poverty_rate": region_info.get("poverty_rate", 0),
            "household_count": region_info.get("household_count", 0),
            "urban_pct": region_info.get("urban_pct", 0),
            "avg_age": region_info.get("avg_age", 0),
        }
        geojson_features.append(feat_copy)

    # Build the choropleth
    locations = []
    z_values = []
    hover_texts = []
    ids = []

    for feat in geojson_features:
        props = feat["properties"]
        ids.append(props["region"])
        locations.append(props["region"])
        z_values.append(props["poverty_rate"])
        hover_texts.append(
            f"<b>{props['region']}</b><br>"
            f"Poverty Rate: {props['poverty_rate']:.1f}%<br>"
            f"Households: {props['household_count']:,}<br>"
            f"Urban: {props['urban_pct']:.0f}% | Avg Age: {props['avg_age']:.0f}"
        )

    fig = go.Figure()

    fig.add_trace(go.Choropleth(
        geojson=geojson,
        locations=locations,
        z=z_values,
        featureidkey="properties.region",
        colorscale=[
            [0, "#C4D7E7"],
            [0.1, "#8BB3C4"],
            [0.2, "#5B8FA0"],
            [0.3, "#3B6E7E"],
            [0.5, "#1A5F7A"],
            [0.7, "#9B4D4D"],
            [1, "#7A3030"],
        ],
        zmin=0,
        zmax=max(z_values) * 1.2 if z_values else 20,
        marker=dict(line=dict(width=1.5, color="#FFFFFF")),
        colorbar=dict(
            title="Poverty Rate (%)",
            title_font=dict(size=11, color="#212529"),
            tickfont=dict(size=10, color="#212529"),
            thickness=15,
            len=0.6,
        ),
        text=hover_texts,
        hovertemplate="%{text}<extra></extra>",
        customdata=ids,
    ))

    fig.update_layout(
        title=dict(
            text="Morocco Regional Poverty Map (Click a region for details)",
            font=dict(size=14, color="#212529"),
            x=0.02, xanchor="left"
        ),
        geo=dict(
            fitbounds="locations",
            projection=dict(type="mercator"),
            showframe=False,
            showcoastlines=True,
            coastlinecolor="#DEE2E6",
            coastlinewidth=1,
            showland=True,
            landcolor="#F8F9FA",
            bgcolor="#F8F9FA",
        ),
        height=520,
        paper_bgcolor="#F8F9FA",
        font=dict(family="Inter, sans-serif", size=11, color="#212529"),
        margin=dict(l=10, r=10, t=50, b=10),
        clickmode="event+select",
    )

    return fig


def build_urban_rural_map():
    """Build urban/rural composition by region."""
    try:
        raw_df, _ = load_raw_encdm()
        if raw_df is None:
            return None
        region_col = _get_raw_col(raw_df, "Region_12")
        milieu_col = _get_raw_col(raw_df, "Milieu")
        if region_col is None or milieu_col is None:
            return None
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
    for milieu, color in [("Urbain", "#1A5F7A"), ("Rural", "#2C7A7B")]:
        subset = cross[cross["milieu"] == milieu]
        fig.add_trace(go.Bar(
            x=subset["region"], y=subset["pct"], name=milieu,
            marker_color=color,
            text=[f"{v:.0f}%" for v in subset["pct"]],
            textposition="inside", textfont=dict(size=9, color="white"),
            hovertemplate="<b>%{x}</b><br>%{legend}: %{y:.1f}%<extra></extra>",
        ))

    fig.update_layout(
        title=dict(text="Urban/Rural Composition by Region", font=dict(size=14)),
        xaxis_title="Region", yaxis_title="Percentage (%)",
        barmode="stack", height=400,
        paper_bgcolor="#F8F9FA", plot_bgcolor="#FFFFFF",
        font=dict(family="Inter, sans-serif", size=11, color="#212529"),
        margin=dict(l=10, r=10, t=50, b=80),
        xaxis=dict(tickangle=45, showgrid=True, gridcolor="#DEE2E6", zeroline=False),
        yaxis=dict(showgrid=True, gridcolor="#DEE2E6", zeroline=False, ticksuffix="%"),
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
    )
    return fig


def build_rgph_housing_map():
    """Build RGPH housing quality by region."""
    df = load_clean_rgph()
    if df is None or "REG" not in df.columns:
        return None

    quality_cols = []
    for col in ["TYPE.LOG", "MURS", "TOIT", "SOL", "GAZ", "ELEC", "EAU.MODE"]:
        if col in df.columns:
            quality_cols.append(col)

    if not quality_cols:
        return None

    region_stats = df.groupby("REG")[quality_cols].mean().reset_index()
    region_stats["quality_index"] = region_stats[quality_cols].mean(axis=1)
    region_stats["region"] = region_stats["REG"].map(RGPH_LABELS["REG"])
    region_stats = region_stats.sort_values("quality_index", ascending=True)

    colors = ["#9B4D4D" if v < region_stats["quality_index"].median() else "#1A5F7A"
              for v in region_stats["quality_index"]]

    fig = go.Figure(go.Bar(
        x=region_stats["region"], y=region_stats["quality_index"],
        marker_color=colors,
        text=[f"{v:.2f}" for v in region_stats["quality_index"]],
        textposition="outside", textfont=dict(size=9),
        hovertemplate="<b>%{x}</b><br>Housing Quality Index: %{y:.2f}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(text="Housing Quality Index by Region (RGPH)", font=dict(size=14)),
        xaxis_title="Region", yaxis_title="Housing Quality Index",
        height=420, paper_bgcolor="#F8F9FA", plot_bgcolor="#FFFFFF",
        font=dict(family="Inter, sans-serif", size=11, color="#212529"),
        margin=dict(l=10, r=10, t=50, b=80),
        xaxis=dict(tickangle=45, showgrid=True, gridcolor="#DEE2E6", zeroline=False),
        yaxis=dict(showgrid=True, gridcolor="#DEE2E6", zeroline=False),
    )
    return fig


def get_all_maps():
    return [
        ("Morocco Regional Poverty Map", build_morocco_choropleth,
         "Interactive choropleth map - click a region for detailed statistics"),
        ("Urban/Rural Composition", build_urban_rural_map,
         "Urban vs Rural distribution across Moroccan regions"),
        ("RGPH Housing Quality", build_rgph_housing_map,
         "Composite housing quality index by region from census data"),
    ]