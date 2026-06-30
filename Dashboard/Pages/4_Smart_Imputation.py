"""Smart Imputation page - Simpute companion module."""
import io

import pandas as pd
import streamlit as st

from Utils.data_loader import get_label
from Utils.theme import inject


def render():
    inject()
    st.markdown('<h1 class="main-title">Smart Imputation</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Adaptive Per-Column Imputation via Simpute</p>', unsafe_allow_html=True)

    st.markdown(
        '<div class="page-description">'
        "Upload a CSV dataset with missing values and let <strong>Simpute</strong> — the adaptive per-column "
        "imputation library — intelligently handle them. For each column with gaps, Simpute selects the optimal "
        "strategy among <strong>KNN</strong>, <strong>LightGBM</strong>, or statistical imputation based on "
        "data characteristics. Review the imputation results, inspect before/after metrics, and download the "
        "cleaned dataset for downstream analysis."
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown('<p class="card-eyebrow">Simpute Companion Project</p>', unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown(
            '<p class="prose-block">Adaptive per-column imputation library generalizing the internship pipeline. '
            '<a href="https://pypi.org/project/simpute/">PyPI</a> · '
            '<a href="https://github.com/Hvllvix/Simpute">GitHub</a>.</p>',
            unsafe_allow_html=True,
        )
    st.markdown('<p class="card-eyebrow">Simpute API Reference</p>', unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown(
            """<table style="width:100%; border-collapse: collapse; font-size: 0.85rem;">
            <tr style="background: var(--zinc100);">
                <th style="text-align: left; padding: 0.5rem; border: 1px solid var(--zinc200);">Method</th>
                <th style="text-align: left; padding: 0.5rem; border: 1px solid var(--zinc200);">Description</th>
            </tr>
            <tr>
                <td style="padding: 0.5rem; border: 1px solid var(--zinc200); font-family: monospace;">Simpute(exclude=[])</td>
                <td style="padding: 0.5rem; border: 1px solid var(--zinc200);">Initialize imputer, optionally exclude columns</td>
            </tr>
            <tr>
                <td style="padding: 0.5rem; border: 1px solid var(--zinc200); font-family: monospace;">fit_transform(df)</td>
                <td style="padding: 0.5rem; border: 1px solid var(--zinc200);">Profile columns, select models, impute missing values</td>
            </tr>
            <tr>
                <td style="padding: 0.5rem; border: 1px solid var(--zinc200); font-family: monospace;">getmodelselection()</td>
                <td style="padding: 0.5rem; border: 1px solid var(--zinc200);">Returns dict of column → selected model</td>
            </tr>
            </table>""",
            unsafe_allow_html=True,
        )

    st.markdown('<p class="card-eyebrow">Run Imputation</p>', unsafe_allow_html=True)
    with st.container(border=True):
        uploaded = st.file_uploader("Upload CSV", type=["csv"], key="simpute_upload")
        if uploaded:
            df = pd.read_csv(uploaded)
            nulls_before = int(df.isna().sum().sum())
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Rows", f"{len(df):,}")
            with c2:
                st.metric("Columns", f"{len(df.columns):,}")
            with c3:
                st.metric("Missing Values", f"{nulls_before:,}")

            # Show null counts per column
            null_counts = df.isna().sum()
            null_cols = null_counts[null_counts > 0]
            if len(null_cols) > 0:
                null_df = pd.DataFrame({
                    "Column": null_cols.index,
                    "Missing Values": null_cols.values,
                    "Missing %": (null_cols.values / len(df) * 100).round(1),
                }).sort_values("Missing Values", ascending=False)
                try:
                    import plotly.express as px
                    from Utils.theme import PALETTE
                    fig = px.bar(
                        null_df, x="Column", y="Missing Values",
                        color="Missing %",
                        color_continuous_scale=[PALETTE["zinc400"], PALETTE["amber"], PALETTE["navy"]],
                        text_auto=",",
                        hover_data={"Missing %": ":.1f"},
                    )
                    fig.update_layout(
                        title="Missing Values per Column",
                        xaxis_tickangle=-35,
                        height=350,
                    )
                    fig.update_traces(textposition="outside")
                    st.plotly_chart(fig, width='stretch')
                except Exception:
                    st.dataframe(null_df, width='stretch')
            else:
                st.success("No missing values detected in the uploaded dataset.")

            if st.button("Run Smart Imputation", type="primary", width='stretch'):
                with st.spinner("Imputing..."):
                    try:
                        from simpute import Simpute
                        imputer = Simpute(exclude=[])
                        cleaned = imputer.fit_transform(df)
                    except Exception as exc:
                        st.error(f"Simpute failed: {exc}")
                        cleaned = None

                if cleaned is not None:
                    nulls_after = int(cleaned.isna().sum().sum())
                    st.success(f"Imputation complete. Remaining nulls: {nulls_after:,}")

                    before = pd.DataFrame({
                        "Metric": ["Rows", "Columns", "Missing Values"],
                        "Before": [len(df), len(df.columns), nulls_before],
                        "After": [len(cleaned), len(cleaned.columns), nulls_after],
                    })
                    st.dataframe(before, width='stretch')

                    try:
                        import plotly.express as px
                        from Utils.theme import PALETTE
                        mdf = pd.DataFrame({
                            "Stage": ["Before", "After"],
                            "Missing": [nulls_before, nulls_after],
                        })
                        fig = px.bar(mdf, x="Stage", y="Missing", color="Stage",
                                     color_discrete_sequence=[PALETTE["amber"], PALETTE["navy"]],
                                     text_auto=",")
                        fig.update_layout(title="Missing Values Before vs After Imputation")
                        st.plotly_chart(fig, width='stretch')
                    except Exception:
                        pass

                    buf = io.BytesIO()
                    cleaned.to_csv(buf, index=False)
                    st.download_button(
                        "Download Cleaned CSV",
                        buf.getvalue(),
                        "cleaned.csv",
                        "text/csv",
                        width='stretch',
                    )