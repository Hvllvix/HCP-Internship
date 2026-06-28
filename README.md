# HCP Morocco Socioeconomic Intelligence Platform

**Repository:** [github.com/Hvllvix/HCP-Internship](https://github.com/Hvllvix/HCP-Internship)

Interactive Streamlit dashboard fusing official **Haut-Commissariat au Plan (HCP)** microdata: ENCDM household consumption (2019-2020) and RGPH census (2014). Supports data integrity auditing, regional spatial analytics, and dual-model poverty/vulnerability inference (LightGBM + PyTorch Hypernetwork).

Companion imputation library: **[Simpute](https://pypi.org/project/simpute/)** (Smart Imputation).

---

## Quick Start

```bash
git clone https://github.com/Hvllvix/HCP-Internship.git
cd HCP-Internship
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
streamlit run Dashboard/app.py
```

Requires **Python 3.10+**. Place raw `.sav` files under `Data/Raw/` and processed parquet under `Data/Processed/` (see notebooks for generation).

Regenerate the repository tree shown in the Overview tab:

```bash
python Scripts/tree.py
```

---

## Dashboard Sections

| Section | Purpose |
|---|---|
| **Overview** | Dataset metadata, dimensional profiles, raw missingness (zero-miss features omitted), Simpute narrative, dynamic repo tree (`Others/tree.txt`), imputation dependency summary, clean data previews |
| **Regional Analytics** | Clickable Morocco choropleth with instant cross-filtering, regional KPIs, national structure plots, localized demographic/amenity panels, imputation network graphs |
| **Predictive Engine** | Mermaid architecture diagrams, model benchmarks, dual-inference sandbox with log-scaled comparison charts, OOD warnings, session-persisted results |

### Map Interactivity

Region selection uses `st.session_state.sel_reg` with full-page `st.rerun()` on choropleth click. A sync selectbox provides keyboard-accessible region switching. Map geo scope is clipped to Morocco with transparent outer terrain.

### Inference Sandbox

1. Encode categorical UI inputs to survey codes
2. Apply persisted ENCDM `StandardScaler` objects
3. Run LightGBM classifiers (`Pauvre`, `Vulnérable`)
4. Run geography-conditioned Hypernetwork (RGPH strata context)
5. Display log-scaled probability comparison, feature contributions, encoded diagnostics

Probabilities are **policy research estimates**, not individual ground-truth labels.

---

## Repository Layout

```text
Internship-HCP/
├── Dashboard/
│   ├── app.py           # Streamlit UI (Overview · Regional · Predictive)
│   ├── utils.py         # Audit, layout, diagrams, boot helpers
│   ├── data_loader.py   # Parquet, mappings, models, weights
│   ├── plots.py         # Plotly analytics
│   ├── sandbox.py       # Dual inference pipeline
│   ├── hypernet.py      # PyTorch hypernetwork engine
│   ├── network.py       # Imputation dependency graphs
│   └── theme.py         # Vercel-inspired light UI theme
├── Assets/
│   ├── Dependencies/    # Imputation schemas (JSON)
│   ├── Maps/            # GeoJSON + codebooks
│   ├── Mermaid/         # Architecture diagrams (.mmd)
│   └── Plots/           # Static report figures
│       ├── Analysis/
│       ├── Modeling/
│       └── Pre-processing/
├── Data/
│   ├── Raw/             # ENCDM.sav · RGPH.sav
│   └── Processed/       # CleanENCDM.parquet · CleanRGPH.parquet
├── Models/
│   ├── Classifier/      # LightGBM + Hypernet.pt
│   ├── Scalers/         # StandardScaler joblib files
│   └── Imputers/        # KNN (ENCDM) · LightGBM (RGPH)
├── Notebooks/
│   ├── Pre-processing.ipynb
│   ├── Analysis.ipynb
│   └── Modeling.ipynb
├── Scripts/
│   └── tree.py          # Writes Others/tree.txt
├── Others/
│   └── tree.txt         # Generated directory tree (+ Simpute subtree)
└── requirements.txt
```

---

## Data Pipeline

| Stage | ENCDM | RGPH |
|---|---|---|
| Raw | `Data/Raw/ENCDM.sav` | `Data/Raw/RGPH.sav` |
| Clean | `Data/Processed/CleanENCDM.parquet` | `Data/Processed/CleanRGPH.parquet` |
| Imputation | KNN + dependency graph | LightGBM + dependency graph |
| Weights | `coef_ménage`, `coef_indiv` | `PDS` |
| Role | Poverty labels, demographics | Housing quality, amenities, hypernet context |

Household-weighted rates inverse-scale `coef_ménage` because clean parquet stores scaled survey weights.

---

## Machine Learning Stack

| Model | Artifact | Role |
|---|---|---|
| LightGBM | `Models/Classifier/ENCDM_LGBM_*.joblib` | Tabular baseline for Pauvre / Vulnérable |
| Transfer LGBM | `Models/Classifier/Transfer_LGBM_*.joblib` | Cross-dataset transfer (notebook artifact) |
| Hypernetwork | `Models/Classifier/Hypernet.pt` | RGPH context generates target-network weights per stratum |

Hypernetwork flow: RGPH Region x Milieu strata embed into a weight generator MLP; ENCDM household features pass through the dynamically parameterized target network. Optional **rural transfer** counterfactual swaps RGPH milieu while holding household traits fixed.

---

## Dependencies

```text
pandas>=2.0
numpy>=1.24
pyarrow>=14.0
pyreadstat>=1.2
scikit-learn>=1.3
lightgbm>=4.0
joblib>=1.3
matplotlib>=3.7
streamlit>=1.28
plotly>=5.18
torch>=2.0
```

Install a CPU-only PyTorch build if CUDA is unavailable.

---

## Reproducibility

**Notebooks (methodology)**

- `Notebooks/Pre-processing.ipynb` - ingestion, null profiling, imputation, scaling
- `Notebooks/Analysis.ipynb` - descriptive regional/socioeconomic analysis
- `Notebooks/Modeling.ipynb` - LightGBM training, transfer models, hypernetwork

**Dashboard (deployment)**

- `Dashboard/app.py` - UI composition
- `Dashboard/sandbox.py` - encode, scale, dual inference
- `Dashboard/hypernet.py` - checkpoint load, strata context, prediction

---

## Simpute Companion Project

Simpute generalizes the internship's adaptive imputation logic into a standalone sklearn-compatible package:

- Profiles each column (type, missingness, cardinality, distribution)
- Routes to LightGBM, CatBoost, KNN, or Bayesian Ridge per column
- Sequential multi-pass imputation with guard-test validation

- PyPI: https://pypi.org/project/simpute/
- GitHub: https://github.com/Hvllvix/Simpute

---

## Data Governance

This repository references official HCP data under `Data/Raw/`. Ensure compliance with HCP confidentiality and redistribution constraints before publishing derivative datasets or model artifacts.
