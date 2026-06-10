import json, joblib, pathlib
import pandas as pd
import streamlit as st
from .i18n import PLOT_DESCRIPTIONS

PROJECTROOT = pathlib.Path(__file__).resolve().parent.parent.parent
DATAPATH = PROJECTROOT / 'Data' / 'Processed'
MAPPATH = PROJECTROOT / 'Assets' / 'Maps'
PLOTSPATH = PROJECTROOT / 'Assets' / 'Plots'
MODELPATH = PROJECTROOT / 'Models' / 'Classifier'
METRICSPATH = pathlib.Path(__file__).resolve().parent / 'metrics.json'
LOGOPATH = pathlib.Path(__file__).resolve().parent.parent / 'assets' / 'hcp_logo.png'

SHAREDFEATURES = ['Milieu', 'Région_12', 'Taille_ménage', 'Taille_agregée']
FULLFEATURES = ['Milieu', 'Région_12', 'Taille_agregée', 'Sexe_CM', 'Niveau_scolaire_agreg_CM', 'Situation_profession_agreg_CM', 'Taille_ménage', 'Age_CM']

@st.cache_data(show_spinner = False)
def load_maps():
    with open(MAPPATH / 'MapENCDM.json', encoding = 'utf-8') as file :
        raw = json.load(file)
    mapencdm = {column : {code : label for label, code in values.items() if code is not None} for column, values in raw.items()}
    with open(MAPPATH / 'MapRGPH.json', encoding = 'utf-8') as file :
        raw = json.load(file)
    maprgph = {column : {code : label for label, code in values.items() if code is not None} for column, values in raw.items()}
    return mapencdm, maprgph

@st.cache_data(show_spinner = False)
def load_data():
    encdm = pd.read_parquet(DATAPATH / 'CleanENCDM.parquet')
    rgph = pd.read_parquet(DATAPATH / 'CleanRGPH.parquet')
    mapencdm, maprgph = load_maps()
    for scaler in pathlib.Path(PROJECTROOT / 'Models' / 'Scalers' / 'ENCDM').glob('SS_*.joblib') :
        column = scaler.stem.removeprefix('SS_')
        if column in encdm.columns :
            encdm[column] = joblib.load(scaler).inverse_transform(pd.DataFrame(encdm[column])).ravel()
    poorrate = encdm.loc[encdm['Pauvre'] == 1, 'coef_indiv'].sum() / encdm['coef_indiv'].sum() * 100
    return encdm, rgph, mapencdm, maprgph, poorrate

@st.cache_resource(show_spinner = False)
def load_models():
    models = {}
    for name in ['ENCDM_LGBM_Pauvre', 'ENCDM_LGBM_Vulnérable', 'Transfer_LGBM_Pauvre', 'Transfer_LGBM_Vulnérable'] :
        path = MODELPATH / f'{name}.joblib'
        if path.exists() : models[name] = joblib.load(path)
    hyperpath = MODELPATH / 'Hypernet.pt'
    models['hyperexists'] = hyperpath.exists()
    return models

def load_metrics():
    with open(METRICSPATH, encoding = 'utf-8') as file :
        return json.load(file)

def plot_manifest():
  plots = []
  for path in sorted(PLOTSPATH.glob('*.png')) :
      meta = PLOT_DESCRIPTIONS.get(path.name, {'en': path.stem, 'fr': path.stem, 'category': 'Poverty'})
      plots.append({'file': path.name, 'path': path, 'category': meta['category'], 'en': meta['en'], 'fr': meta['fr']})
  return plots

def plot_path(filename):
    return PLOTSPATH / filename

def map_options(mapdict, lang = 'en'):
    return {label if lang == 'fr' else label : code for code, label in sorted(mapdict.items(), key = lambda item : item[0])}

def predict(encdmfeatures, useadvanced):
    models = load_models()
    results = {}
    for target in ['Pauvre', 'Vulnérable'] :
        if useadvanced :
            bundle = models[f'ENCDM_LGBM_{target}']
            row = pd.DataFrame([{f : encdmfeatures[f] for f in bundle['features']}])
            prob = bundle['model'].predict_proba(row.astype('float32'))[0, 1]
        else :
            bundle = models[f'Transfer_LGBM_{target}']
            row = pd.DataFrame([{f : encdmfeatures[f] for f in SHAREDFEATURES}])
            raw = bundle['model'].predict_proba(row.astype('float32'))[0, 1]
            prob = bundle['calibrator'].transform([raw])[0] if 'calibrator' in bundle else raw
        results[target] = float(prob)
    return results
