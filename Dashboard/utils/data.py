import json, joblib, pathlib
import pandas as pd
import streamlit as st

PROJECTROOT = pathlib.Path(__file__).resolve().parent.parent.parent
DATAPATH = PROJECTROOT / 'Data' / 'Processed'
MAPPATH = PROJECTROOT / 'Assets' / 'Maps'
MODELPATH = PROJECTROOT / 'Models' / 'Classifier'
IMPUTERENCDM = PROJECTROOT / 'Models' / 'Imputers' / 'ENCDM'
IMPUTERRGPH = PROJECTROOT / 'Models' / 'Imputers' / 'RGPH'
DEPSPATH = PROJECTROOT / 'Assets' / 'Dependencies'
METRICSPATH = pathlib.Path(__file__).resolve().parent / 'metrics.json'
LOGOPATH = pathlib.Path(__file__).resolve().parent.parent / 'assets' / 'hcp_logo.png'
PLOTDIR = PROJECTROOT / 'Assets' / 'Plots'

SHAREDFEATURES = ['Milieu', 'Région_12', 'Taille_ménage', 'Taille_agregée']

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
def load_classifiers():
    models = {}
    for name in ['ENCDM_LGBM_Pauvre', 'ENCDM_LGBM_Vulnérable', 'Transfer_LGBM_Pauvre', 'Transfer_LGBM_Vulnérable'] :
        path = MODELPATH / f'{name}.joblib'
        if path.exists() : models[name] = joblib.load(path)
    models['hyperexists'] = (MODELPATH / 'Hypernet.pt').exists()
    return models

@st.cache_resource(show_spinner = False)
def load_imputers():
    encdm = {p.stem.removeprefix('KNN_') : joblib.load(p) for p in IMPUTERENCDM.glob('KNN_*.joblib')}
    rgph = {p.stem.removeprefix('LGBM_') : joblib.load(p) for p in IMPUTERRGPH.glob('LGBM_*.joblib')}
    return encdm, rgph

def load_dependencies():
    with open(DEPSPATH / 'DependenciesENCDM.json', encoding = 'utf-8') as file :
        depsencdm = json.load(file)
    with open(DEPSPATH / 'DependenciesRGPH.json', encoding = 'utf-8') as file :
        depsrgph = json.load(file)
    return depsencdm, depsrgph

def load_metrics():
    with open(METRICSPATH, encoding = 'utf-8') as file :
        return json.load(file)

@st.cache_data(show_spinner = False)
def raw_null_stats():
    import pyreadstat
    rawencdm, _ = pyreadstat.read_sav(str(PROJECTROOT / 'Data' / 'Raw' / 'ENCDM.sav'))
    rawrgph, _ = pyreadstat.read_sav(str(PROJECTROOT / 'Data' / 'Raw' / 'RGPH.sav'))
    encdmtop = rawencdm.isnull().sum().sort_values(ascending = False)
    encdmtop = encdmtop[encdmtop > 0].head(8)
    rgphnulls = rawrgph.isnull().sum()
    rgphnulls = rgphnulls[rgphnulls > 0]
    return {
        'encdm_rows': len(rawencdm),
        'rgph_rows': len(rawrgph),
        'encdm_cols': len(rawencdm.columns),
        'rgph_cols': len(rawrgph.columns),
        'encdm_nulls': int(rawencdm.isnull().sum().sum()),
        'rgph_nulls': int(rawrgph.isnull().sum().sum()),
        'encdm_null_cols': int((rawencdm.isnull().sum() > 0).sum()),
        'rgph_null_cols': int((rawrgph.isnull().sum() > 0).sum()),
        'encdm_top': encdmtop,
        'rgph_top': rgphnulls.sort_values(ascending = False).head(8),
    }

@st.cache_resource(show_spinner = False)
def load_hypernetwork():
    path = MODELPATH / 'Hypernet.pt'
    if not path.exists() :
        return None
    import torch
    checkpoint = torch.load(path, map_location = 'cpu', weights_only = False)
    checkpoint['size_mb'] = round(path.stat().st_size / 1e6, 2)
    checkpoint['path'] = str(path)
    return checkpoint

def map_options(mapdict):
    return {label : code for code, label in sorted(mapdict.items(), key = lambda item : item[0])}

def predict(encdmfeatures, useadvanced):
    models = load_classifiers()
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

def impute_encdm(target, features, mapencdm):
    model = load_imputers()[0][target]
    deps, _ = load_dependencies()
    row = pd.DataFrame([features])
    pred = model.predict(row[deps[target]].astype('float32'))[0]
    label = mapencdm[target].get(int(pred), str(pred))
    return int(pred), label

def impute_rgph(target, features, maprgph):
    model = load_imputers()[1][target]
    _, deps = load_dependencies()
    row = pd.DataFrame([features])
    pred = model.predict(row[deps[target]].astype('float32'))[0]
    label = maprgph[target].get(int(pred), str(pred))
    return int(pred), label

def interpret_prediction(results, regionlabel, milieulabel):
    poor, vuln = results['Pauvre'], results['Vulnérable']
    lines = []
    if poor >= 0.15 :
        lines.append(f'High poverty risk for a household in **{regionlabel}** ({milieulabel}). Structural features alone suggest elevated deprivation.')
    elif poor >= 0.05 :
        lines.append(f'Moderate poverty probability in **{regionlabel}**. Context sits above the national average but below high-risk strata.')
    else :
        lines.append(f'Low poverty probability in **{regionlabel}** ({milieulabel}) given the selected profile.')
    if vuln >= 0.2 :
        lines.append('Vulnerability score is high. This profile is close to the poverty threshold and sensitive to shocks.')
    elif vuln >= 0.1 :
        lines.append('Some vulnerability detected. Not poor yet, but exposed to income instability.')
    else :
        lines.append('Vulnerability remains limited relative to national patterns.')
    return ' '.join(lines)
