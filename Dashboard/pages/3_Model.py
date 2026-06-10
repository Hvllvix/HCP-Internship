import streamlit as st
from utils.chrome import render_header, render_footer, plot_image, prob_bar, t
from utils.data import load_data, load_metrics, plot_path, map_options, predict

encdm, _ = render_header()

st.title(t('nav_model'))
metrics = load_metrics()
lang = st.session_state.lang

target = st.radio(t('target'), ['Pauvre', 'Vulnérable'], format_func = lambda x : t('pauvre') if x == 'Pauvre' else t('vulnerable'), horizontal = True)
m = metrics['lgbm']['test'][target]

st.subheader(t('metrics_title'))
mcols = st.columns(4)
for col, key in zip(mcols, ['AUC', 'F1', 'Precision', 'Recall']) :
    with col :
        st.metric(t(key.lower()) if key != 'AUC' else t('auc'), f"{m[key]:.4f}")

pcol1, pcol2 = st.columns(2)
with pcol1 :
    plot_image(plot_path('Model Feature Importance.png'))
with pcol2 :
    plot_image(plot_path('Poverty Rate ENCDM vs RGPH Predicted.png'))

st.divider()
st.subheader(t('predict'))

_, _, mapencdm, _, _ = load_data()
milieuopts = map_options(mapencdm['Milieu'], lang)
regionopts = map_options(mapencdm['Région_12'], lang)
sizeopts = map_options(mapencdm['Taille_agregée'], lang)

st.markdown(f'**{t("simplified")}**')
c1, c2, c3, c4 = st.columns(4)
with c1 :
    milieulabel = st.selectbox(t('milieu'), list(milieuopts.keys()), key = 'milieu')
with c2 :
    regionlabel = st.selectbox(t('region'), list(regionopts.keys()), key = 'region')
with c3 :
    taille = st.slider(t('household_size'), int(encdm['Taille_ménage'].min()), int(encdm['Taille_ménage'].max()), int(encdm['Taille_ménage'].median()))
with c4 :
    sizelabel = st.selectbox(t('size_category'), list(sizeopts.keys()), key = 'size')

sexopts = map_options(mapencdm['Sexe_CM'], lang)
eduopts = map_options(mapencdm['Niveau_scolaire_agreg_CM'], lang)
profopts = map_options(mapencdm['Situation_profession_agreg_CM'], lang)
useadvanced = False
with st.expander(t('advanced')) :
    useadvanced = st.checkbox(t('full_model'), value = False)
    a1, a2, a3, a4 = st.columns(4)
    with a1 :
        sexlabel = st.selectbox(t('sex'), list(sexopts.keys()))
    with a2 :
        edulabel = st.selectbox(t('education'), list(eduopts.keys()))
    with a3 :
        proflabel = st.selectbox(t('profession'), list(profopts.keys()))
    with a4 :
        age = st.slider(t('age'), int(encdm['Age_CM'].min()), int(encdm['Age_CM'].max()), int(encdm['Age_CM'].median()))
if st.button(t('predict_btn'), type = 'primary') :
    features = {
        'Milieu': milieuopts[milieulabel],
        'Région_12': regionopts[regionlabel],
        'Taille_ménage': taille,
        'Taille_agregée': sizeopts[sizelabel],
        'Sexe_CM': map_options(mapencdm['Sexe_CM'], lang)[sexlabel],
        'Niveau_scolaire_agreg_CM': map_options(mapencdm['Niveau_scolaire_agreg_CM'], lang)[edulabel],
        'Situation_profession_agreg_CM': map_options(mapencdm['Situation_profession_agreg_CM'], lang)[proflabel],
        'Age_CM': age,
    }
    results = predict(features, useadvanced)
    prob_bar(t('prob_poor'), results['Pauvre'], 'poverty')
    prob_bar(t('prob_vuln'), results['Vulnérable'], 'vulnerable')

st.caption(t('disclaimer'))
render_footer()
