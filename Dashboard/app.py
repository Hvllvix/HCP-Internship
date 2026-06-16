import streamlit as st
from utils.chrome import init_session, apply_theme, sidebar_controls
from utils.data import LOGOPATH
from utils.i18n import t

st.set_page_config(page_title = 'HCP Poverty Dashboard', layout = 'wide', initial_sidebar_state = 'expanded')

init_session()
apply_theme()

with st.sidebar :
    if LOGOPATH.exists() :
        st.image(str(LOGOPATH), width = 'stretch')
    lang = st.session_state.lang
    st.markdown(
        f'<div class="sidebar-brand"><h3>{t("brand", lang)}</h3>'
        f'<span class="muted">{t("brand_sub", lang)}</span></div>',
        unsafe_allow_html = True,
    )
    sidebar_controls()

pages = [
    st.Page('pages/Overview.py', title = 'Overview', default = True),
    st.Page('pages/Explorer.py', title = 'Explorer'),
    st.Page('pages/Models.py', title = 'Model'),
    st.Page('pages/Hypernetwork.py', title = 'Hypernetwork'),
]

st.navigation(pages).run()
