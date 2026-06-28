"""Regional Analysis page module."""
import streamlit as st

from Utils.data_loader import GEOJSON_REGIONS
from Utils.plots import (
    CHART_H_TALL,
    compute_region_stats,
    fig_age_distribution,
    fig_choropleth,
    fig_employment_mix,
    fig_milieu_split,
    fig_national_education,
    fig_national_employment,
    fig_poverty_breakdown,
    fig_raw_missing_values,
    fig_region_amenities,
    fig_region_education,
    fig_region_gender,
    fig_region_household_size,
    fig_region_poverty,
    fig_rgph_infrastructure,
    fig_rgph_rooms,
    fig_urban_rural_poverty,
    fig_vulnerability_breakdown,
)
from Utils.utils import safe_plot_row, parse_map_click
from Utils.theme import metric_row


def render(encdm, rgph, geojson, regstats, codes, regions, geoidmap):
    st.markdown('<h1 class="main-title">Regional Analytics</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Spatial Socioeconomic Profiles Across Morocco</p>', unsafe_allow_html=True)

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

        st.markdown('<p class="card-eyebrow">National Structure</p>', unsafe_allow_html=True)
        safe_plot_row([
            ("Weighted Poverty Split", "National weighted poverty shares.", lambda: fig_poverty_breakdown(encdm)),
            ("Vulnerability Split", "National vulnerability shares.", lambda: fig_vulnerability_breakdown(encdm)),
        ])
        safe_plot_row([
            ("National Education", "Weighted education distribution.", lambda: fig_national_education(encdm, codes)),
            ("National Employment", "Top employment categories.", lambda: fig_national_employment(encdm, codes)),
        ], h=CHART_H_TALL)
        safe_plot_row([
            ("Household Size (National)", "Inverse-scaled ENCDM sizes.", lambda: __import__('Utils.plots', fromlist=['fig_household_size_national']).fig_household_size_national(encdm)),
            ("RGPH Asset Access", "National infrastructure rates.", lambda: fig_rgph_infrastructure(rgph)),
        ])

        st.markdown(f'<p class="card-eyebrow">Regional Profile - {selname}</p>', unsafe_allow_html=True)
        safe_plot_row([
            ("Age Distribution", f"Ages in {selname}.", lambda: fig_age_distribution(encdm, sel)),
            ("Education Structure", f"Education in {selname}.", lambda: fig_region_education(encdm, sel, codes)),
            ("Amenity Access (RGPH)", f"Infrastructure in {selname}.", lambda: fig_region_amenities(rgph, sel)),
        ], ctx=f"Regional Profile {selname}")
        safe_plot_row([
            ("Urban / Rural Poverty", "Poverty by milieu.", lambda: fig_urban_rural_poverty(encdm, regions)),
            ("Poverty Rate by Region", "Regional ranking.", lambda: fig_region_poverty(regstats)),
        ], h=CHART_H_TALL, ctx=f"Regional Compare {selname}")
        safe_plot_row([
            ("Employment Mix", f"Employment in {selname}.", lambda: fig_employment_mix(encdm, sel, codes)),
            ("Urban-Rural Composition", f"Milieu split in {selname}.", lambda: fig_milieu_split(encdm, sel)),
            ("Household Size", f"HH sizes in {selname}.", lambda: fig_region_household_size(encdm, sel)),
        ], ctx=f"Regional Demographics {selname}")
        safe_plot_row([
            ("Gender Representation", f"Gender shares in {selname}.", lambda: fig_region_gender(encdm, sel)),
            ("Housing - Room Count", f"Rooms in {selname}.", lambda: fig_rgph_rooms(rgph, sel)),
        ], ctx=f"Regional Housing {selname}")

    except Exception as exc:
        st.error(f"Regional Analytics failed for region code {st.session_state.sel_reg}: {exc}")