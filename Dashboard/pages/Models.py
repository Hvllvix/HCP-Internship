import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from utils.chrome import section, body, progress_bar, footer, card
from utils.data import load_data, load_metrics, map_options, predict, interpret_prediction, PLOTDIR
from utils.palette import PALETTE
from utils.theme import plotly_template

lang = st.session_state.lang
encdm, rgph, mapencdm, maprgph, poorrate = load_data()
metrics = load_metrics()
theme = st.session_state.theme
p = PALETTE[theme]
ptemplate = plotly_template(theme)

head = {'en': ('Model', 'LightGBM classifiers · live inference'), 'fr': ('Modèle', 'Classifieurs LightGBM · inférence')}
st.markdown(f'<h1 class="hero-title">{head[lang][0]}</h1>', unsafe_allow_html = True)
st.markdown(f'<p class="hero-sub">{head[lang][1]}</p>', unsafe_allow_html = True)

section('Poverty Classifiers' if lang == 'en' else 'Classifieurs de pauvreté', 'LightGBM')
testm = metrics['lgbm']['test']['Pauvre']
testv = metrics['lgbm']['test']['Vulnérable']
mcol1, mcol2, mcol3, mcol4 = st.columns(4)
for col, key in zip([mcol1, mcol2, mcol3, mcol4], ['AUC', 'F1', 'Precision', 'Recall']) :
    col.metric(f"Pauvre {key}", f"{testm[key]:.4f}")
vcol1, vcol2, vcol3, vcol4 = st.columns(4)
for col, key in zip([vcol1, vcol2, vcol3, vcol4], ['AUC', 'F1', 'Precision', 'Recall']) :
    col.metric(f"Vulnérable {key}", f"{testv[key]:.4f}")

pcol1, pcol2 = st.columns(2)
featpng = PLOTDIR / 'Model Feature Importance.png'
stratapng = PLOTDIR / 'Poverty Rate ENCDM vs RGPH Predicted.png'
with pcol1 :
    if featpng.exists() : st.image(str(featpng), width = 'stretch')
with pcol2 :
    if stratapng.exists() : st.image(str(stratapng), width = 'stretch')
    body(
        'Full ENCDM LightGBM uses 8 features (AUC ~0.78). Transfer model uses 4 shared structural features with isotonic calibration.'
        if lang == 'en' else
        'LightGBM complet : 8 variables (AUC ~0,78). Modèle de transfert : 4 variables partagées, calibré isotoniquement.',
    )

st.divider()
section('Predict' if lang == 'en' else 'Prédiction')

milieuopts = map_options(mapencdm['Milieu'])
regionopts = map_options(mapencdm['Région_12'])
sizeopts = map_options(mapencdm['Taille_agregée'])
sexopts = map_options(mapencdm['Sexe_CM'])
eduopts = map_options(mapencdm['Niveau_scolaire_agreg_CM'])
profopts = map_options(mapencdm['Situation_profession_agreg_CM'])

showadvanced = st.toggle('Show advanced features' if lang == 'en' else 'Afficher les variables avancées', value = False)
useadvanced = False
sexlabel = list(sexopts.keys())[0]
edulabel = list(eduopts.keys())[0]
proflabel = list(profopts.keys())[0]
age = int(encdm['Age_CM'].median())

if showadvanced :
    useadvanced = st.checkbox('Use full ENCDM model (8 features)' if lang == 'en' else 'Modèle ENCDM complet (8 variables)', value = True)
    a1, a2, a3, a4 = st.columns(4)
    with a1 : sexlabel = st.selectbox('Sex' if lang == 'en' else 'Sexe', list(sexopts.keys()))
    with a2 : edulabel = st.selectbox('Education' if lang == 'en' else 'Éducation', list(eduopts.keys()))
    with a3 : proflabel = st.selectbox('Profession', list(profopts.keys()))
    with a4 : age = st.slider('Age', int(encdm['Age_CM'].min()), int(encdm['Age_CM'].max()), int(encdm['Age_CM'].median()))

with st.form('predict_form') :
    r1c1, r1c2, r1c3, r1c4 = st.columns(4)
    with r1c1 : milieulabel = st.selectbox('Milieu', list(milieuopts.keys()))
    with r1c2 : regionlabel = st.selectbox('Region' if lang == 'en' else 'Région', list(regionopts.keys()))
    with r1c3 : taille = st.slider('Household size' if lang == 'en' else 'Taille ménage', int(encdm['Taille_ménage'].min()), int(encdm['Taille_ménage'].max()), int(encdm['Taille_ménage'].median()))
    with r1c4 : sizelabel = st.selectbox('Size category' if lang == 'en' else 'Taille agrégée', list(sizeopts.keys()))
    submitted = st.form_submit_button('Run prediction' if lang == 'en' else 'Lancer la prédiction', width = 'stretch', type = 'primary')

if submitted :
    features = {
        'Milieu': milieuopts[milieulabel],
        'Région_12': regionopts[regionlabel],
        'Taille_ménage': taille,
        'Taille_agregée': sizeopts[sizelabel],
        'Sexe_CM': sexopts[sexlabel],
        'Niveau_scolaire_agreg_CM': eduopts[edulabel],
        'Situation_profession_agreg_CM': profopts[proflabel],
        'Age_CM': age,
    }
    results = predict(features, useadvanced)
    mode = 'Full ENCDM (8)' if useadvanced else 'Transfer (4)'

    st.markdown(f"""
    <div class="result-card">
        <div class="section-label">{'Result' if lang == 'en' else 'Résultat'}</div>
        <h3 style="margin:0;color:var(--accent);">P(poor) = {results['Pauvre']:.1%} · P(vulnerable) = {results['Vulnérable']:.1%}</h3>
        <p class="muted" style="margin:0.4rem 0 0 0;">{mode} · {regionlabel}</p>
    </div>
    """, unsafe_allow_html = True)

    rcol1, rcol2 = st.columns(2)
    with rcol1 :
        progress_bar('Poverty' if lang == 'en' else 'Pauvreté', results['Pauvre'], 'poverty', theme)
        st.markdown('<div style="height:0.75rem"></div>', unsafe_allow_html = True)
        progress_bar('Vulnerability' if lang == 'en' else 'Vulnérabilité', results['Vulnérable'], 'vulnerable', theme)
        st.markdown(f'<p class="body-text">{interpret_prediction(results, regionlabel, milieulabel).replace("**", "")}</p>', unsafe_allow_html = True)
    with rcol2 :
        barfig = go.Figure(go.Bar(
            x = [results['Pauvre'], results['Vulnérable']],
            y = ['Pauvre', 'Vulnérable'],
            orientation = 'h',
            marker = dict(color = [p['poverty'], p['vulnerable']]),
            text = [f"{results['Pauvre']:.1%}", f"{results['Vulnérable']:.1%}"],
            textposition = 'outside',
        ))
        barfig.update_layout(**ptemplate, title = dict(text = 'Probabilities', font = dict(size = 14)), showlegend = False)
        st.plotly_chart(barfig, width = 'stretch')

st.caption('Indicative model output, not official HCP statistics.' if lang == 'en' else 'Résultat indicatif, pas une statistique officielle du HCP.')
footer()
