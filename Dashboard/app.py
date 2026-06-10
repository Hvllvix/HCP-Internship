import streamlit as st
from utils.chrome import init_session, apply_theme, render_sidebar

st.set_page_config(page_title = 'HCP Poverty Dashboard', page_icon = '📊', layout = 'wide', initial_sidebar_state = 'expanded')

init_session()
render_sidebar()
apply_theme()

pages = [
    st.Page('pages/1_Overview.py', title = 'Overview', icon = '📊', default = True),
    st.Page('pages/2_Explorer.py', title = 'Explorer', icon = '🔍'),
    st.Page('pages/3_Model.py', title = 'Model', icon = '🧠'),
    st.Page('pages/4_Hypernetwork.py', title = 'Hypernetwork', icon = '⚡'),
]

st.navigation(pages).run()
