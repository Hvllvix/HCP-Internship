"""Regional Analysis page module."""
import streamlit as st

from Utils.data_loader import GEOJSON_REGIONS
from Utils.plots import (
    CHART_H_TALL,
    compute_region_stats,
    fig_age_distribution,
    fig_choropleth,
    fig_milieu_split,
    fig_national_education,
    fig_national_employment,
    fig_poverty_breakdown,
    fig_region_amenities,
    fig_region_education,
    fig_region_education_gender,
    fig_region_gender,
    fig_region_household_size,
    fig_region_poverty,
    fig_rgph_infrastructure,
    fig_rgph_rooms,
    fig_urban_rural_poverty,
)
from Utils.utils import plot_block, safe_plot_row, parse_map_click
from Utils.theme import metric_row


def render(encdm, rgph, geojson, regstats, codes, labels, regions, geoidmap):
    st.markdown('<h1 class="main-title">Regional Analytics</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Spatial Socioeconomic Profiles Across Morocco</p>', unsafe_allow_html=True)

    st.markdown(
        '<div class="page-description">'
        "Explore <strong>regional poverty and vulnerability</strong> across Morocco's 12 administrative regions. "
        "Click any region on the interactive choropleth to reveal localized demographic, education, employment, "
        "housing, and infrastructure indicators. National-level summaries provide context for comparing "
        "regional disparities in weighted poverty shares, education attainment, and household characteristics."
        "</div>",
        unsafe_allow_html=True,
    )

    try:
        sel = int(st.session_state.sel_reg)
        if sel not in regstats:
            sel = next(iter(regstats))
            st.session_state.sel_reg = sel
        selinfo = regstats[sel]
        selname = selinfo["name"]

        name_to_code = {regstats[c]["name"]: c for c in regstats}
        pick = st.selectbox(
            "Region selector",
            list(name_to_code.keys()),
            index=list(name_to_code.values()).index(sel),
            key="region_picker",
        )
        if name_to_code[pick] != sel:
            st.session_state.sel_reg = name_to_code[pick]
            st.rerun()

        # ── Morocco Choropleth Map ──
        with st.container(border=True):
            st.markdown('<p class="section-heading">Morocco Poverty Choropleth</p>', unsafe_allow_html=True)
            mapfig = fig_choropleth(geojson, regstats, GEOJSON_REGIONS, selected_code=sel)
            mapfig.update_layout(dragmode=False)
            mapevt = st.plotly_chart(
                mapfig,
                width='stretch',
                height=CHART_H_TALL + 80,
                on_select="rerun",
                key="morocco_map",
                selection_mode="points",
            )
            clicked = parse_map_click(mapevt, geoidmap)
            if clicked is not None and clicked != sel:
                st.session_state.sel_reg = clicked
                st.rerun()

        st.markdown(f'<p class="plot-desc"><strong>Selected:</strong> {selname}</p>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="kpi-strip">'
            f'<div class="kpi-cell"><div class="metric-label">Poverty Rate</div>'
            f'<div class="metric-value accent">{selinfo["poverty_rate"]}%</div></div>'
            f'<div class="kpi-cell"><div class="metric-label">Households</div>'
            f'<div class="metric-value">{selinfo["households"]:,}</div></div>'
            f'<div class="kpi-cell"><div class="metric-label">Urban Share</div>'
            f'<div class="metric-value">{selinfo["urban_pct"]}%</div></div>'
            f'<div class="kpi-cell"><div class="metric-label">Avg Age</div>'
            f'<div class="metric-value">{selinfo["avg_age"]}</div></div>'
            f'<div class="kpi-cell"><div class="metric-label">Avg HH Size</div>'
            f'<div class="metric-value">{selinfo["avg_size"]}</div></div>'
            f"</div>",
            unsafe_allow_html=True,
        )

        # ── Dynamic Plots (update with region selection, right below map) ──
        st.markdown(f'<h2 class="main-title" style="font-size:1.3rem;">Regional Profile — {selname}</h2>', unsafe_allow_html=True)
        st.markdown(
            '<p class="prose-block">The panels below update dynamically based on the region selected in the choropleth map above. '
            'Explore age structure, education attainment, infrastructure access, and socioeconomic composition for '
            f'<strong>{selname}</strong>.</p>',
            unsafe_allow_html=True,
        )
        safe_plot_row([
            ("Age Distribution", f"Ages in {selname}.", lambda: fig_age_distribution(encdm, sel)),
            ("Education Structure", f"Education in {selname}.", lambda: fig_region_education(encdm, sel, labels)),
        ], ctx=f"Regional Profile {selname}")
        safe_plot_row([
            ("Amenity Access (RGPH)", f"Infrastructure in {selname}.", lambda: fig_region_amenities(rgph, sel)),
            ("Education by Gender", f"Education split by gender in {selname}.", lambda: fig_region_education_gender(encdm, sel, labels)),
        ], ctx=f"Regional Profile {selname}")
        safe_plot_row([
            ("Urban-Rural Composition", f"Milieu split in {selname}.", lambda: fig_milieu_split(encdm, sel)),
            ("Gender Representation", f"Gender shares in {selname}.", lambda: fig_region_gender(encdm, sel)),
        ], ctx=f"Regional Profile {selname}")
        safe_plot_row([
            ("Household Size", f"HH sizes in {selname}.", lambda: fig_region_household_size(encdm, sel)),
            ("Housing - Room Count", f"Rooms in {selname}.", lambda: fig_rgph_rooms(rgph, sel)),
        ], ctx=f"Regional Profile {selname}")

        # ── Static Plots (national overview, separated) ──
        st.markdown('<hr style="margin:1.5rem 0;border-color:var(--zinc200);">', unsafe_allow_html=True)
        st.markdown('<h2 class="main-title" style="font-size:1.3rem;">National & Cross-Region Overview</h2>', unsafe_allow_html=True)
        st.markdown(
            '<p class="prose-block">Static summaries of national poverty distribution, education attainment, employment categories, '
            'household characteristics, and regional rankings. These panels provide context-independent benchmarks for '
            'comparing against the selected region\'s profile above.</p>',
            unsafe_allow_html=True,
        )
        safe_plot_row([
            ("Weighted Poverty Split", "National weighted poverty shares.", lambda: fig_poverty_breakdown(encdm)),
            ("National Education", "Weighted education distribution.", lambda: fig_national_education(encdm, labels)),
        ])
        safe_plot_row([
            ("National Employment", "Top employment categories.", lambda: fig_national_employment(encdm, labels)),
            ("Household Size (National)", "Inverse-scaled ENCDM sizes.", lambda: __import__('Utils.plots', fromlist=['fig_household_size_national']).fig_household_size_national(encdm)),
        ])
        safe_plot_row([
            ("RGPH Asset Access", "National infrastructure rates.", lambda: fig_rgph_infrastructure(rgph)),
            ("Poverty Rate by Region", "Regional ranking.", lambda: fig_region_poverty(regstats)),
        ])
        safe_plot_row([
            ("Urban / Rural Poverty by Region", "Poverty by milieu across regions.", lambda: fig_urban_rural_poverty(encdm, regions)),
        ])

    except Exception as exc:
        st.error(f"Regional Analytics failed for region code {st.session_state.sel_reg}: {exc}")
