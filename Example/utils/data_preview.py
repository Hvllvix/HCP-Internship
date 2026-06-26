"""
Paginated, filterable parquet preview tables with human-readable column labels.
"""
import streamlit as st
import pandas as pd
from .translations import rename_columns_readable, translate_series, REGION_NAME_TO_CODE
from .data_loader import resolve_colname


def render_paginated_preview(df, dataset="encdm", key_prefix="preview"):
    if df is None or df.empty:
        st.caption("Dataset unavailable.")
        return

    st.markdown('<div class="card"><div class="card-body">', unsafe_allow_html=True)

    search = st.text_input("Search across all columns", "", key=f"{key_prefix}_search")
    page_size = st.selectbox("Rows per page", [25, 50, 100, 200], index=1, key=f"{key_prefix}_pagesize")

    if dataset == "encdm":
        rc = resolve_colname(df, "Region_12")
        regions = ["All"] + sorted(df[rc].dropna().unique().tolist()) if rc in df.columns else ["All"]
        region_labels = {c: str(c) for c in regions}
        if rc in df.columns:
            from .translations import REGION_NAMES
            region_labels = {"All": "All"} | {c: REGION_NAMES.get(float(c), str(c)) for c in regions if c != "All"}
        filter_region = st.selectbox(
            "Filter by region code",
            options=regions,
            format_func=lambda x: region_labels.get(x, str(x)),
            key=f"{key_prefix}_region",
        )
    else:
        filter_region = "All"
        rc = None

    sort_col = st.selectbox(
        "Sort column",
        options=list(df.columns),
        format_func=lambda c: rename_columns_readable(pd.DataFrame(columns=[c]), dataset=dataset).columns[0],
        key=f"{key_prefix}_sortcol",
    )
    sort_asc = st.checkbox("Ascending sort", value=True, key=f"{key_prefix}_sortasc")

    working = df.copy()
    if filter_region != "All" and rc in working.columns:
        working = working[working[rc] == filter_region]

    if search.strip():
        mask = working.astype(str).apply(lambda row: row.str.contains(search, case=False, na=False).any(), axis=1)
        working = working[mask]

    working = working.sort_values(sort_col, ascending=sort_asc, na_position="last")
    total = len(working)
    total_pages = max(1, (total + page_size - 1) // page_size)
    page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1, key=f"{key_prefix}_page")
    start = (page - 1) * page_size
    page_df = working.iloc[start:start + page_size].copy()

    display_df = rename_columns_readable(page_df, dataset=dataset)
    for col in page_df.select_dtypes(include=["float64", "int64"]).columns:
        if col in ("Pauvre", "Milieu", "REG", "MIL") or "Region" in col or "Région" in col:
            label_col = rename_columns_readable(pd.DataFrame(columns=[col]), dataset=dataset).columns[0]
            if label_col in display_df.columns:
                display_df[label_col] = translate_series(page_df[col], col, dataset=dataset)

    st.caption(f"Showing rows {start + 1}-{min(start + page_size, total)} of {total:,} (filtered)")
    st.dataframe(display_df, use_container_width=True, height=320)
    st.markdown('</div></div>', unsafe_allow_html=True)
