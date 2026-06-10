import streamlit as st
from utils.chrome import render_header, render_footer, kpi_row, plot_image, t
from utils.data import load_data, plot_path

encdm, rgph = render_header()

st.title(t('nav_overview'))

_, _, _, mapencdm, poorrate = load_data()
nregions = len(mapencdm['Région_12'])

kpi_row(
    [f'{len(encdm):,}', f'{len(rgph):,}', f'{poorrate:.1f}%', str(nregions)],
    [t('individuals'), t('households'), t('poor_wtd'), t('regions')],
)

st.subheader(t('pipeline'))
st.markdown(f'<div class="pipeline">{t("pipeline_steps")}</div>', unsafe_allow_html = True)

col1, col2 = st.columns(2)
with col1 :
    st.markdown(f'<div class="dataset-card"><h4>{t("encdm_title")}</h4><p>{t("encdm_desc")}</p></div>', unsafe_allow_html = True)
with col2 :
    st.markdown(f'<div class="dataset-card"><h4>{t("rgph_title")}</h4><p>{t("rgph_desc")}</p></div>', unsafe_allow_html = True)

st.subheader(t('hero_caption'))
plot_image(plot_path('Socioeconomic Status by Region.png'))

render_footer()
