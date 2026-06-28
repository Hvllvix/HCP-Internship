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
        '<p class="prose-block">Upload a CSV with missing values. Simpute selects the optimal '
        "imputation strategy per column (KNN, LightGBM, or statistical) and returns a clean dataset.</p>",
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
        st.markdown('<p class="section-heading">Simpute Repository Structure</p>', unsafe_allow_html=True)
        try:
            from pathlib import Path
            simpute_tree = Path("Others/simpute_tree.txt").read_text(encoding="utf-8")
            st.markdown('<div class="tree-section">', unsafe_allow_html=True)
            st.code(simpute_tree, language=None)
            st.markdown("</div>", unsafe_allow_html=True)
        except Exception:
            st.info("Simpute tree unavailable (Others/simpute_tree.txt missing)")

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
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Rows", f"{len(df):,}")
            with c2:
                st.metric("Columns", f"{len(df.columns):,}")
            nulls_before = int(df.isna().sum().sum())
            st.metric("Missing Values", f"{nulls_before:,}")

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