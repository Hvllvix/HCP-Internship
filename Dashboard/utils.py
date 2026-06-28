"""Dashboard helpers: data audit, layout, diagrams, boot."""
import sys

import streamlit as st

from data_loader import (
    GEOJSON_REGIONS,
    ROOT,
    build_code_maps,
    build_geo_id_map,
    build_label_maps,
    build_region_name_map,
    load_all_lgbm,
    load_deps_encdm,
    load_encdm,
    load_geojson,
    load_raw_encdm,
    load_raw_rgph,
    load_rgph,
)
from hypernet import get_hypernet_engine
from plots import CHART_H

PKG_DEPS = [
    ("pandas", ">=2.0", "Tabular data"),
    ("numpy", ">=1.24", "Numerics"),
    ("pyarrow", ">=14.0", "Parquet I/O"),
    ("pyreadstat", ">=1.2", "SPSS .sav reader"),
    ("scikit-learn", ">=1.3", "Scaling and metrics"),
    ("lightgbm", ">=4.0", "Gradient boosting"),
    ("joblib", ">=1.3", "Model serialization"),
    ("streamlit", ">=1.28", "Dashboard UI"),
    ("plotly", ">=5.18", "Interactive charts"),
    ("torch", ">=2.0", "Hypernetwork inference"),
]

MERMAID_INGEST = """\
flowchart LR
  RAW["Raw HCP .sav\\nENCDM + RGPH"] --> AUDIT["Schema Audit\\nNull Profiling"]
  AUDIT --> IMP["Adaptive Imputation\\nKNN / LightGBM"]
  IMP --> SCALE["Standardization\\nJoblib Scalers"]
  SCALE --> CLEAN["Clean Parquet\\nDashboard Ready"]
"""

MERMAID_INFER = """\
flowchart TD
  IN["Household Form Input"] --> ENC["Label Encoding\\nSurvey Code Maps"]
  ENC --> SCL["Feature Scaling\\nStandardScaler"]
  SCL --> LGBM["LightGBM Inference\\nPauvre + Vulnérable"]
  ENC --> EMB["ENCDM Embeddings\\nCategorical + Numerical"]
  CTX["RGPH Strata Context\\nRegion x Milieu"] --> REMB["RGPH Embeddings\\nLayerNorm"]
  REMB --> HYP["Hypernetwork MLP\\nWeight Generator"]
  HYP --> WTS["Dynamic Weights\\nTarget Network"]
  EMB --> TN["Target Forward Pass"]
  WTS --> TN
  TN --> OUT["Sigmoid Probabilities"]
  LGBM --> CMP["Dual Comparison"]
  OUT --> CMP
"""


class Loader:
    @staticmethod
    def encdm():
        return load_encdm()

    @staticmethod
    def rgph():
        return load_rgph()

    @staticmethod
    def raw_encdm():
        return load_raw_encdm()

    @staticmethod
    def raw_rgph():
        return load_raw_rgph()

    @staticmethod
    def geojson():
        return load_geojson()

    @staticmethod
    def labels():
        return build_label_maps()

    @staticmethod
    def codes():
        return build_code_maps()

    @staticmethod
    def regions():
        return build_region_name_map()

    @staticmethod
    def deps():
        return load_deps_encdm()

    @staticmethod
    def models():
        return load_all_lgbm()

    @staticmethod
    def geoidmap(geojson):
        return build_geo_id_map(geojson, GEOJSON_REGIONS)

    @staticmethod
    def tree_text():
        path = ROOT / "Others" / "tree.txt"
        if path.exists():
            return path.read_text(encoding="utf-8")
        return "Run Scripts/tree.py to generate Others/tree.txt"


def _audit_frame(label, df):
    if df is None:
        return {"label": label, "rows": 0, "cols": 0, "total_nulls": 0, "cols_with_nulls": 0, "top_null_cols": {}}
    percol = df.isnull().sum()
    miss = percol[percol > 0].sort_values(ascending=False)
    return {
        "label": label,
        "rows": len(df),
        "cols": len(df.columns),
        "total_nulls": int(percol.sum()),
        "cols_with_nulls": int((percol > 0).sum()),
        "top_null_cols": miss.head(5).astype(int).to_dict(),
    }


@st.cache_data(show_spinner=False)
def audit_nulls():
    raw_e, raw_r = load_raw_encdm(), load_raw_rgph()
    clean_e, clean_r = load_encdm(), load_rgph()
    return {
        "encdm_raw": _audit_frame("ENCDM Raw", raw_e),
        "rgph_raw": _audit_frame("RGPH Raw", raw_r),
        "encdm_clean": _audit_frame("ENCDM Clean", clean_e),
        "rgph_clean": _audit_frame("RGPH Clean", clean_r),
    }


def print_audit(report):
    print("=" * 60, file=sys.stderr)
    print("DATA INTEGRITY AUDIT", file=sys.stderr)
    for r in report.values():
        print(
            f"{r['label']}: rows={r['rows']:,} cols={r['cols']} "
            f"total_nulls={r['total_nulls']:,} cols_with_nulls={r['cols_with_nulls']}",
            file=sys.stderr,
        )
        if r["top_null_cols"]:
            print(f"  top null cols: {r['top_null_cols']}", file=sys.stderr)
    print("=" * 60, file=sys.stderr)


@st.cache_resource
def boot():
    report = audit_nulls()
    encdm = Loader.encdm()
    rgph = Loader.rgph()
    geojson = Loader.geojson()
    labels = Loader.labels()
    codes = Loader.codes()
    regions = Loader.regions()
    deps = Loader.deps()
    bundles = Loader.models()
    geoidmap = Loader.geoidmap(geojson)
    get_hypernet_engine(rgph)
    return encdm, rgph, geojson, labels, codes, regions, deps, bundles, geoidmap


@st.cache_data(show_spinner="Loading raw survey files...")
def boot_raw():
    return Loader.raw_encdm(), Loader.raw_rgph()


def plot_block(title, desc, fig, h=CHART_H):
    st.markdown('<div class="stretch-card">', unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown(f'<p class="section-heading">{title}</p>', unsafe_allow_html=True)
        if desc:
            st.markdown(f'<p class="plot-desc">{desc}</p>', unsafe_allow_html=True)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True, height=h)
    st.markdown("</div>", unsafe_allow_html=True)


def plot_row(items, h=CHART_H):
    st.markdown('<div class="stretch-row">', unsafe_allow_html=True)
    cols = st.columns(len(items))
    for col, (title, desc, fig) in zip(cols, items):
        with col:
            plot_block(title, desc, fig, h=h)
    st.markdown("</div>", unsafe_allow_html=True)


def safe_plot_row(items, h=CHART_H, ctx="Regional Analytics"):
    """Render plot row with per-panel error isolation."""
    safe = []
    for title, desc, factory in items:
        try:
            fig = factory() if callable(factory) else factory
            safe.append((title, desc, fig))
        except Exception as exc:
            safe.append((title, desc, None))
            st.error(f"{ctx} | {title} failed: {exc}")
    plot_row(safe, h=h)


def parse_map_click(event, geoidmap):
    if event is None or not getattr(event, "selection", None):
        return None
    sel = event.selection
    pts = sel.get("points", []) if isinstance(sel, dict) else getattr(sel, "points", [])
    for pt in pts:
        raw = pt.get("customdata") or pt.get("location")
        try:
            cid = int(raw[0] if isinstance(raw, (list, tuple)) else raw)
            if cid in geoidmap:
                return int(geoidmap[cid])
        except (TypeError, ValueError, IndexError):
            continue
    return None


def render_mermaid(syntax):
    st.markdown(f"```mermaid\n{syntax.strip()}\n```")


def render_tree():
    st.markdown('<div class="tree-section">', unsafe_allow_html=True)
    st.code(Loader.tree_text(), language=None)
    st.markdown("</div>", unsafe_allow_html=True)


def dep_cards(deps, limit=8):
    items = []
    for target, pred in deps.items():
        if isinstance(pred, list) and pred:
            items.append((target, pred[:4]))
    if not items:
        st.caption("No dependency graph loaded.")
        return
    rows = [items[i:i + 4] for i in range(0, min(len(items), limit), 4)]
    for row in rows:
        cols = st.columns(len(row))
        for col, (target, preds) in zip(cols, row):
            with col:
                st.markdown(
                    f'<div class="dep-card">'
                    f'<div class="dep-icon">IMP</div>'
                    f'<div class="dep-name">{target}</div>'
                    f'<div class="dep-meta">{", ".join(str(p) for p in preds)}</div>'
                    f"</div>",
                    unsafe_allow_html=True,
                )


def pkg_cards():
    rows = [PKG_DEPS[i:i + 4] for i in range(0, len(PKG_DEPS), 4)]
    for row in rows:
        cols = st.columns(len(row))
        for col, (name, ver, role) in zip(cols, row):
            with col:
                st.markdown(
                    f'<div class="dep-card">'
                    f'<div class="dep-icon">PKG</div>'
                    f'<div class="dep-name">{name}</div>'
                    f'<div class="dep-meta">{ver} · {role}</div>'
                    f"</div>",
                    unsafe_allow_html=True,
                )
