"""
Global feature translation dictionary for HCP datasets.
Maps raw column codes to human-readable labels across plots, tables, and forms.
"""
from .data_loader import (
    ENCDM_COLUMN_LABELS, RGPH_COLUMN_LABELS,
    ENCDM_LABELS, RGPH_LABELS,
    load_encdm_map, load_rgph_map, get_column_label,
)

REGION_NAMES = {
    1.0: "Tanger-Tetouan-Al Hoceima", 2.0: "Oriental", 3.0: "Fes-Meknes",
    4.0: "Rabat-Sale-Kenitra", 5.0: "Beni Mellal-Khenifra", 6.0: "Casablanca-Settat",
    7.0: "Marrakech-Safi", 8.0: "Draa-Tafilalet", 9.0: "Souss-Massa",
    10.0: "Guelmim-Oued Noun", 11.0: "Laayoune-Sakia El Hamra", 12.0: "Dakhla-Oued Ed Dahab",
}

REGION_NAME_TO_CODE = {v: k for k, v in REGION_NAMES.items()}

# GeoJSON `properties.region` labels (Morocco-Regions.geojson) -> ENCDM region codes
GEOJSON_TO_CODE = {
    "Tanger-Tetouan-Hoceima": 1.0,
    "Oriental": 2.0,
    "Fes-Meknes": 3.0,
    "Rabat-Sale-Kenitra": 4.0,
    "Beni Mellal-Khenifra": 5.0,
    "Casablanca-Settat": 6.0,
    "Marrakech-Safi": 7.0,
    "Daraa-Tafilelt": 8.0,
    "Souss Massa": 9.0,
    "Guelmim-Oued Noun": 10.0,
    "Laayoune-Saguia Hamra": 11.0,
    "Dakhla-Oued Eddahab": 12.0,
}

REGION_ORDER = [REGION_NAMES[float(i)] for i in range(1, 13)]

DEFAULT_REGION_CODE = 10.0
DEFAULT_REGION_NAME = "Guelmim-Oued Noun"


def translate_column(col_name, dataset="encdm"):
    """Return professional label for a database column."""
    return get_column_label(col_name, dataset=dataset)


def translate_value(col_name, code, dataset="encdm"):
    """Decode a numeric or string code to a human-readable label."""
    if code is None or (isinstance(code, float) and np_isnan(code)):
        return "Unknown"
    labels = ENCDM_LABELS if dataset == "encdm" else RGPH_LABELS
    if col_name in labels and code in labels[col_name]:
        return labels[col_name][code]
    if col_name in labels:
        try:
            return labels[col_name].get(float(code), str(code))
        except (TypeError, ValueError):
            pass
    return str(code)


def np_isnan(v):
    try:
        import math
        return math.isnan(float(v))
    except (TypeError, ValueError):
        return False


def translate_series(series, col_name, dataset="encdm"):
    """Vectorized decode for a pandas Series."""
    import pandas as pd
    labels = ENCDM_LABELS if dataset == "encdm" else RGPH_LABELS
    mapping = labels.get(col_name, {})
    return series.map(lambda x: mapping.get(float(x), mapping.get(x, str(x))) if pd.notna(x) else "Unknown")


def decode_dataframe(df, dataset="encdm", columns=None):
    """Return a copy with selected columns decoded to labels."""
    import pandas as pd
    out = df.copy()
    cols = columns or list(out.columns)
    for col in cols:
        if col in out.columns:
            out[col] = translate_series(out[col], col, dataset=dataset)
    return out


def rename_columns_readable(df, dataset="encdm"):
    """Rename all columns to human-readable headers."""
    mapping = ENCDM_COLUMN_LABELS if dataset == "encdm" else RGPH_COLUMN_LABELS
    return df.rename(columns={c: mapping.get(c, c) for c in df.columns})


def build_label_maps():
    """Load inverted JSON maps (code -> label) for both datasets."""
    encdm_raw = load_encdm_map() or {}
    rgph_raw = load_rgph_map() or {}
    encdm = {
        col: {float(code): label for label, code in vals.items() if code is not None}
        for col, vals in encdm_raw.items()
    }
    rgph = {
        col: {float(code): label for label, code in vals.items() if code is not None}
        for col, vals in rgph_raw.items()
    }
    return encdm, rgph
