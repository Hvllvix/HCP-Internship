import pandas as pd
import streamlit as st
from utils.chrome import render_header, render_footer, plot_image, t
from utils.data import load_models, load_metrics, plot_path

render_header()

st.title(t('nav_hyper'))
metrics = load_metrics()
models = load_models()

st.subheader(t('hyper_arch'))
st.markdown(f'<div class="arch-box">{t("hyper_flow")}</div>', unsafe_allow_html = True)

st.subheader(t('hyper_loss'))
plot_image(plot_path('Hypernetwork Training Loss.png'))

st.subheader(t('hyper_metrics'))
rows = []
for target in ['Pauvre', 'Vulnérable'] :
    rows.append({
        t('target'): t('pauvre') if target == 'Pauvre' else t('vulnerable'),
        f"{t('split_val')} {t('auc')}": f"{metrics['hypernetwork']['val'][target]['AUC']:.4f}",
        f"{t('split_val')} {t('f1')}": f"{metrics['hypernetwork']['val'][target]['F1']:.4f}",
        f"{t('split_test')} {t('auc')}": f"{metrics['hypernetwork']['test'][target]['AUC']:.4f}",
        f"{t('split_test')} {t('f1')}": f"{metrics['hypernetwork']['test'][target]['F1']:.4f}",
    })
st.dataframe(pd.DataFrame(rows), use_container_width = True, hide_index = True)

st.subheader(t('hyper_how'))
st.markdown(f'- {t("hyper_b1")}')
st.markdown(f'- {t("hyper_b2")}')
st.markdown(f'- {t("hyper_b3")}')

if models.get('hyperexists') :
    st.success('Hypernet.pt')
else :
    st.warning('Hypernet.pt not found — run Modeling.ipynb cells 9–12.')

render_footer()
