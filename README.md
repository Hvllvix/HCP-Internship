## Executive Summary

Morocco’s poverty and vulnerability dynamics are **spatially heterogeneous** and structurally linked to household demographics, labor-market position, and living-conditions infrastructure. Public policy design therefore benefits from analytical systems that can (i) *characterize* socioeconomic stratification and (ii) *simulate* how household-level covariates map to poverty and vulnerability risk.

This repository implements a **machine learning and socioeconomic analytics dashboard** that fuses two official sources produced by the *Haut-Commissariat au Plan (HCP)*:

- **ENCDM (2019–2020)**: the National Survey on Household Consumption and Expenditure, providing high-granularity consumption/household profiles and survey weights for population-level inference.
- **RGPH (2014)**: the General Census of Population and Housing, providing near-universal coverage of housing quality and amenities across Morocco’s administrative regions.

The project operationalizes a **dual-engine predictive framework**—a calibrated gradient-boosted baseline (LightGBM) and a geography-conditioned PyTorch Hypernetwork—and exposes the full pipeline through an interactive Streamlit application for **data integrity auditing**, **spatial analytics**, and an **end-to-end predictive sandbox**.

A companion open-source library, **[Simpute](https://pypi.org/project/simpute/)** (*Smart Imputation*), generalizes the adaptive per-column imputation logic originally built for this internship’s preprocessing phase.

> **Research intent**: build a reproducible, auditable workflow that connects *microdata preprocessing* → *model inference* → *interactive decision-facing visualization* under explicit assumptions about scaling, imputation, and weighting (e.g., `coef_ménage`, `coef_indiv`).

---

## Repository Architecture

### Directory Tree

```text
HCP-Internship/
├── .streamlit/
│   └── config.toml
├── Assets/
│   ├── Dependencies/
│   │   ├── DependenciesENCDM.json
│   │   └── DependenciesRGPH.json
│   ├── Maps/
│   │   ├── MapENCDM.json
│   │   ├── MapRGPH.json
│   │   └── Morocco-Regions.geojson
│   ├── Plots/
│   │   ├── Hypernetwork Training Loss.png
│   │   ├── Model Feature Importance.png
│   │   ├── Poverty Rate ENCDM vs RGPH Predicted.png
│   │   └── ... (additional static figures)
│   └── Types/
│       ├── TypesENCDM.json
│       └── TypesRGPH.json
├── Dashboard/
│   ├── app.py                    # Streamlit UI (3 sections)
│   ├── data_loader.py            # Parquet, mappings, models, weights
│   ├── plots.py                  # Plotly analytics
│   ├── sandbox.py                # Dual LightGBM + Hypernet inference
│   ├── hypernet.py               # PyTorch hypernetwork engine
│   ├── network.py                # Imputation dependency graphs
│   └── theme.py                  # UI theme and layout CSS
├── Data/
│   ├── Processed/
│   │   ├── CleanENCDM.parquet
│   │   └── CleanRGPH.parquet
│   └── Raw/
│       ├── ENCDM.sav
│       └── RGPH.sav
├── Models/
│   ├── Classifier/
│   │   ├── ENCDM_LGBM_Pauvre.joblib
│   │   ├── ENCDM_LGBM_Vulnérable.joblib
│   │   ├── Transfer_LGBM_Pauvre.joblib
│   │   ├── Transfer_LGBM_Vulnérable.joblib
│   │   └── Hypernet.pt
│   ├── Imputers/
│   │   ├── ENCDM/                # KNN-based imputers
│   │   └── RGPH/                 # LightGBM-based imputers
│   └── Scalers/
│       ├── ENCDM/                # StandardScaler objects
│       └── RGPH/
├── Notebooks/
│   ├── Pre-processing.ipynb
│   ├── Analysis.ipynb
│   └── Modeling.ipynb
├── .gitignore
├── requirements.txt
└── README.md
```

### Purpose of Main Directories

- **`Assets/`**
  - **`Maps/`**: JSON codebooks (`MapENCDM.json`, `MapRGPH.json`) and the Morocco regions GeoJSON used for choropleth mapping.
  - **`Dependencies/`**: imputation dependency schemas rendered as interactive network graphs in the dashboard.
  - **`Types/`**: type dictionaries for consistent casting and labeling across the pipeline.
  - **`Plots/`**: static figures used for reporting and narrative context.

- **`Dashboard/`**
  - **`app.py`**: Streamlit application with three sections — Data Integrity, Regional Analytics, Predictive Engine.
  - **`data_loader.py`**: loads Parquet datasets, mappings, model bundles, scalers, and inverse-scaled household weights.
  - **`plots.py`**: Plotly charts for missingness, regional profiles, model benchmarks, and EDA.
  - **`sandbox.py`**: encodes form inputs, applies scalers, and runs dual LightGBM + Hypernetwork inference.
  - **`hypernet.py`**: loads `Hypernet.pt`, builds RGPH strata context, and performs geography-conditioned prediction.
  - **`network.py`**: interactive ENCDM/RGPH imputation dependency graphs.
  - **`theme.py`**: premium light UI theme (navy/amber palette).

- **`Data/`**
  - **`Raw/`**: original HCP deliverables in SPSS format (`.sav`).
  - **`Processed/`**: cleaned Parquet files consumed by the dashboard and notebooks.

- **`Models/`**
  - **`Classifier/`**: trained LightGBM classifiers and the PyTorch hypernetwork checkpoint.
  - **`Scalers/`**: persisted `StandardScaler` objects for ENCDM and RGPH features.
  - **`Imputers/`**: dataset-specific imputers (KNN for ENCDM, LightGBM for RGPH).

- **`Notebooks/`**
  - **`Pre-processing.ipynb`**: ingestion, null profiling, imputation, and standardization.
  - **`Analysis.ipynb`**: descriptive analysis and socioeconomic stratification.
  - **`Modeling.ipynb`**: LightGBM training, transfer models, and hypernetwork experiments.

---

## Companion Project: Simpute (Smart Imputation)

The preprocessing phase of this internship exposed a recurring challenge: ENCDM and RGPH columns differ in type, cardinality, missingness, and cross-column signal. A single global imputation strategy (mean, median, MICE) is insufficient.

**Simpute** (`pip install simpute`) extracts the core idea into a standalone, sklearn-compatible library:

- **Profiles each column** (type, missingness ratio, cardinality, distribution shape).
- **Routes each target** to an adaptive backend (LightGBM, CatBoost, KNN, Bayesian Ridge, etc.).
- **Imputes sequentially** so earlier fills inform later columns.
- **Warns** when missingness exceeds 70% on a column.

In this repository, the internship pipeline uses persisted KNN (ENCDM) and LightGBM (RGPH) imputers with explicit dependency graphs under `Assets/Dependencies/`. Simpute generalizes that approach for any tabular dataset.

- **PyPI**: https://pypi.org/project/simpute/
- **GitHub**: https://github.com/Hvllvix/Simpute

---

## Data Pipeline & Processing

### Data Sources: ENCDM vs RGPH

| Dimension | ENCDM (Survey) | RGPH (Census) |
|---|---:|---:|
| **Institutional source** | HCP | HCP |
| **Raw storage** | `Data/Raw/ENCDM.sav` | `Data/Raw/RGPH.sav` |
| **Processed storage** | `Data/Processed/CleanENCDM.parquet` | `Data/Processed/CleanRGPH.parquet` |
| **Primary analytic role** | Poverty/vulnerability labels and household demographics | Housing quality, amenities, and infrastructure |
| **Weighting** | `coef_indiv`, `coef_ménage` (inverse-scaled for rates in dashboard) | `PDS` where relevant |
| **Dashboard usage** | Regional poverty stats, EDA, LightGBM features | Amenity access, hypernetwork strata context |

### Cleaning, Standardization, and Imputation

- **Ingestion**: raw `.sav` files are read via `pyreadstat` in notebooks and `Dashboard/data_loader.py`; cleaned data is persisted to Parquet.
- **Imputation**: KNN imputers (ENCDM) and LightGBM imputers (RGPH) under `Models/Imputers/`, guided by dependency JSON in `Assets/Dependencies/`.
- **Scaling**: feature-specific `StandardScaler` objects under `Models/Scalers/`; applied on-the-fly during sandbox inference.
- **Labeling**: human-readable mappings centralized in `Assets/Maps/` and `Dashboard/data_loader.py`.
- **Weighting**: household-weighted poverty rates use inverse-scaled `coef_ménage` because the clean Parquet stores scaled survey weights.

---

## Machine Learning Architecture

### Dual-Engine Predictive Strategy

| Component | Purpose | Artifact | Dashboard usage |
|---|---|---|---|
| **LightGBM (baseline)** | Strong tabular classifiers for poverty and vulnerability | `Models/Classifier/ENCDM_LGBM_*.joblib` | **Active** — live probabilistic inference |
| **Hypernetwork** | Geography-conditioned weight generation from RGPH strata | `Models/Classifier/Hypernet.pt` | **Active** — dual inference alongside LightGBM |
| **Transfer LightGBM** | Cross-dataset transfer variants | `Models/Classifier/Transfer_LGBM_*.joblib` | Trained artifacts; available for extension |

### LightGBM Classifiers

Predict **`Pauvre`** (poverty) and **`Vulnérable`** (vulnerability). The sandbox:

1. Encodes categorical UI inputs to numeric survey codes.
2. Applies persisted ENCDM scalers (`Age_CM`, `Taille_ménage`, etc.).
3. Returns calibrated probabilities from bundled models and thresholds.
4. Renders feature-importance contributions for interpretability.

### PyTorch Hypernetwork (`hypernet.py`)

The hypernetwork is the most structurally complex model in the repository:

1. **RGPH strata context** — for each Region × Milieu pair, census records are aggregated into a representative housing-quality embedding (modal categoricals + mean numerics).
2. **Hypernetwork MLP** — reads the RGPH embedding and **generates all weights** of a small feed-forward target network.
3. **ENCDM embedding** — household survey categoricals and numerics are embedded separately.
4. **Dynamic forward pass** — the generated weights act on ENCDM features to produce poverty and vulnerability logits.

The optional **rural transfer** counterfactual swaps the RGPH stratum to rural while holding household characteristics fixed, illustrating spatial sensitivity of the deep model.

---

## Dashboard & UI Features

The Streamlit app (`Dashboard/app.py`) has three sidebar sections:

### 1) Data Integrity

- Platform overview with weighted national poverty rate.
- **Simpute** companion-project narrative and repository tree.
- **Raw missingness** plots (pre-imputation `.sav` files).
- Weighted poverty/vulnerability breakdowns, national education and employment structure.
- ENCDM/RGPH data previews and imputation dependency summary.

### 2) Regional Analytics

- **Clickable Morocco choropleth** (`Assets/Maps/Morocco-Regions.geojson`) — select a region on the map to drive all downstream panels.
- Regional KPI strip (poverty rate, households, urban share, average age and household size).
- Three localized plots per selection: age distribution, education structure, RGPH amenity access.
- Comparative panels: urban/rural poverty, regional ranking, employment mix, milieu split, gender representation, housing room counts.
- Interactive ENCDM/RGPH imputation dependency networks with node-focus filtering.

### 3) Predictive Engine

- Model benchmarks: feature importances, ROC curves, hypernetwork training loss.
- **Dual-inference sandbox**: configure household characteristics and run LightGBM + Hypernetwork in parallel.
- Side-by-side probability comparison and LGBM feature contribution chart.
- Rural transfer stress-test for hypernetwork spatial counterfactuals.

---

## Installation & Setup

### 1) Clone the repository

```bash
git clone https://github.com/Hvllvix/HCP-Internship.git
cd HCP-Internship
```

### 2) Create and activate a virtual environment

```bash
python -m venv .venv
```

**Windows (PowerShell)**

```bash
.\.venv\Scripts\Activate.ps1
```

**macOS/Linux**

```bash
source .venv/bin/activate
```

### 3) Install dependencies

Requires **Python 3.10+**.

```bash
pip install -r requirements.txt
```

> **PyTorch note**: on CPU-only machines, install a CPU build of PyTorch consistent with your Python version if CUDA errors occur. `requirements.txt` lists `torch>=2.0`.

### 4) Launch the dashboard

```bash
streamlit run Dashboard/app.py
```

---

## Reproducibility Notes

**Notebook-first (methodological traceability)**

- `Notebooks/Pre-processing.ipynb` → ingestion, null profiling, imputation, standardization
- `Notebooks/Analysis.ipynb` → descriptive analysis and stratification
- `Notebooks/Modeling.ipynb` → LightGBM training and hypernetwork experiments

**Dashboard-first (deployment and communication)**

- `Dashboard/app.py` → UI composition and narrative
- `Dashboard/data_loader.py` → data, mappings, weights, and artifact loading
- `Dashboard/sandbox.py` → encode → scale → dual inference pipeline
- `Dashboard/hypernet.py` → strata context and dynamic weight generation

---

## License and Data Governance

This repository contains references to official HCP data sources and includes local copies under `Data/Raw/`. If you intend to redistribute or publish derivative datasets or model artifacts, ensure compliance with applicable **HCP data usage and confidentiality constraints**, and apply appropriate anonymization and governance controls where required.
