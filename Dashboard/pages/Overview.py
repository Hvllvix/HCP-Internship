import pandas as pd
import pathlib
import streamlit as st
from utils.chrome import section, body, kpi_row, footer, card
from utils.data import load_data, load_dependencies, raw_null_stats
from utils.plots import overview_raw_nulls_plot

encdm, rgph, mapencdm, maprgph, poorrate = load_data()
nullstats = None
try :
    nullstats = raw_null_stats()
except Exception :
    pass
depsencdm, depsrgph = load_dependencies()
theme = st.session_state.theme

st.markdown('<h1 class="hero-title">Morocco Poverty & Census Intelligence</h1>', unsafe_allow_html = True)
st.markdown('<p class="hero-sub">ENCDM × RGPH · HCP Guelmim Internship</p>', unsafe_allow_html = True)

kpi_row(
    [f'{len(encdm):,}', f'{len(rgph):,}', f'{poorrate:.1f}%', '12'],
    ['ENCDM individuals', 'RGPH households', 'Weighted poverty rate', 'Regions'],
)

section('Project Introduction')
body("""
This internship project links two official HCP datasets to study poverty patterns at both survey and census scale.
<strong>ENCDM</strong> (Enquête Nationale sur la Consommation et les Dépenses des Ménages) provides individual-level poverty
labels on ~16k respondents. <strong>RGPH</strong> (Recensement Général de la Population et de l'Habitat) covers ~730k
households with housing and amenity indicators nationwide.
""")
body("""
The goal is not to merge both datasets into one table, but to bridge them through shared strata (region × milieu),
train poverty classifiers on ENCDM, and explore how census-scale housing structure relates to deprivation patterns
observed in the consumption survey.
""")

col1, col2 = st.columns(2)
with col1 :
    with card() :
        section('Workflow')
        st.markdown("""
        <div style="display:flex;align-items:center;gap:0.35rem;flex-wrap:wrap;margin:0.25rem 0 1rem 0;">
            <div class="pipeline-step">Raw .sav</div><span class="pipeline-arrow">→</span>
            <div class="pipeline-step">Pre-processing</div><span class="pipeline-arrow">→</span>
            <div class="pipeline-step">Analysis</div><span class="pipeline-arrow">→</span>
            <div class="pipeline-step">Modeling</div><span class="pipeline-arrow">→</span>
            <div class="pipeline-step">Dashboard</div>
        </div>
        """, unsafe_allow_html = True)
        body("""
        <strong>Pre-processing</strong> imputes missing values (KNN for ENCDM, LightGBM for RGPH), scales numerics,
        and exports clean parquet files plus maps and dependency graphs.
        <strong>Modeling</strong> trains LightGBM classifiers, a cross-dataset transfer model, and a hypernetwork prototype.
        """)

with col2 :
    with card() :
        section('Repository Layout')
        st.markdown("""<pre class="tree">Internship-HCP/
├── Data/Raw/          # original .sav files
├── Data/Processed/    # CleanENCDM & CleanRGPH parquet
├── Notebooks/
│   ├── Pre-processing.ipynb
│   ├── Analysis.ipynb
│   └── Modeling.ipynb
├── Models/
│   ├── Imputers/      # KNN (ENCDM) + LGBM (RGPH)
│   ├── Scalers/
│   └── Classifier/    # poverty models + Hypernet.pt
├── Assets/Maps/       # categorical label maps
└── Dashboard/         # this app</pre>""", unsafe_allow_html = True)

section('Raw Data & Missing Values')
if nullstats :
    body(f"""
    Before imputation, the raw SPSS files contain substantial missingness:
    <strong>{nullstats['encdm_nulls']:,}</strong> null cells across ENCDM ({nullstats['encdm_null_cols']} columns affected)
    and <strong>{nullstats['rgph_nulls']:,}</strong> across RGPH ({nullstats['rgph_null_cols']} columns affected).
    """)
    st.image(overview_raw_nulls_plot(theme), width = 'stretch')
else :
    body('Raw .sav files not found locally — skipping null audit. Processed parquet is still loaded.')

hero = pathlib.Path(__file__).resolve().parent.parent.parent / 'Assets' / 'Plots' / 'Socioeconomic Status by Region.png'
if hero.exists() :
    section('National Poverty Overview')
    st.image(str(hero), width = 'stretch')

section('Maps & Dependency Graphs')
body("""
Categorical columns are stored as integer codes in the parquet files. <strong>MapENCDM.json</strong> and
<strong>MapRGPH.json</strong> translate codes to French labels for plots and the dashboard UI.
A critical detail: the same region name can map to different integers between datasets (e.g. Guelmim is code 2 in ENCDM but 9 in RGPH).
""")
mcol1, mcol2 = st.columns(2)
with mcol1 :
    with card() :
        st.markdown('**Map coverage**')
        mapdf = pd.DataFrame({
            'Dataset': ['ENCDM', 'RGPH'],
            'Mapped columns': [len(mapencdm), len(maprgph)],
            'Example fields': ['Région_12, Milieu, Pauvre', 'REG, MIL, TYPE.LOG'],
        })
        st.dataframe(mapdf, width = 'stretch', hide_index = True)
        regionencdm = mapencdm['Région_12'].get(2, '?')
        regionrgph = maprgph['REG'].get(9, '?')
        st.caption(f'Region bridge example: ENCDM code 2 = {regionencdm}, RGPH code 9 = {regionrgph}')

with mcol2 :
    with card() :
        st.markdown('**Imputation dependencies**')
        depdf = pd.DataFrame({
            'Dataset': ['ENCDM', 'RGPH'],
            'Imputed targets': [len(depsencdm), len(depsrgph)],
            'Engine': ['KNN', 'LightGBM'],
            'Graph file': ['DependenciesENCDM.json', 'DependenciesRGPH.json'],
        })
        st.dataframe(depdf, width = 'stretch', hide_index = True)
        sample = next(iter(depsencdm))
        st.caption(f'Example: {sample} is predicted from {", ".join(depsencdm[sample][:4])}...')

section('Dataset Preview')
tab1, tab2 = st.tabs(['ENCDM, consumption survey', 'RGPH, general census'])
previewencdm = encdm.head(8).copy()
previewrgph = rgph.head(8).copy()
for col in ['Région_12', 'Milieu', 'Pauvre', 'Sexe_CM'] :
    if col in previewencdm.columns :
        previewencdm[col] = previewencdm[col].map(mapencdm.get(col, {}))
for col in ['REG', 'MIL', 'ELEC', 'NET'] :
    if col in previewrgph.columns :
        previewrgph[col] = previewrgph[col].map(maprgph.get(col, {}))
with tab1 :
    body('Individual-level survey with poverty labels, education, profession, and survey weights (coef_indiv).')
    st.dataframe(previewencdm, width = 'stretch', hide_index = True)
with tab2 :
    body('Household-level census with housing quality, amenities, and sampling weight (PDS).')
    st.dataframe(previewrgph, width = 'stretch', hide_index = True)

section('Key Technical Decisions')
dcol1, dcol2, dcol3 = st.columns(3)
with dcol1 :
    with card() :
        st.markdown('**Region bridge**')
        body('ENCDM and RGPH use different integer codes for the same region names. Linkage is done by label, not by code.')
with dcol2 :
    with card() :
        st.markdown('**Survey weights**')
        body('All aggregates use coef_indiv (ENCDM) or PDS (RGPH) so national totals are representative.')
with dcol3 :
    with card() :
        st.markdown('**Strata matching**')
        body('Cross-dataset models pair observations by region × milieu during training, not by merging rows.')

footer()
