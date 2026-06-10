import streamlit as st
from .i18n import STRINGS, CATEGORY_KEYS
from .theme import css, THEMES
from .data import LOGOPATH, load_data

def init_session():
    if 'lang' not in st.session_state : st.session_state.lang = 'en'
    if 'theme' not in st.session_state : st.session_state.theme = 'dark'

def t(key, **kwargs):
    lang = st.session_state.get('lang', 'en')
    text = STRINGS.get(key, {}).get(lang, key)
    return text.format(**kwargs) if kwargs else text

def apply_theme():
    st.markdown(css(st.session_state.theme), unsafe_allow_html = True)

def render_sidebar():
    init_session()
    with st.sidebar :
        if LOGOPATH.exists() :
            st.image(str(LOGOPATH), use_container_width = True)
        st.markdown(f"### {t('app_title')}")
        st.divider()
        cols = st.columns(2)
        with cols[0] :
            if st.button('EN', use_container_width = True, type = 'primary' if st.session_state.lang == 'en' else 'secondary') :
                st.session_state.lang = 'en'
        with cols[1] :
            if st.button('FR', use_container_width = True, type = 'primary' if st.session_state.lang == 'fr' else 'secondary') :
                st.session_state.lang = 'fr'
        st.caption(t('lang'))
        cols = st.columns(2)
        with cols[0] :
            if st.button(t('dark'), use_container_width = True, type = 'primary' if st.session_state.theme == 'dark' else 'secondary') :
                st.session_state.theme = 'dark'
        with cols[1] :
            if st.button(t('light'), use_container_width = True, type = 'primary' if st.session_state.theme == 'light' else 'secondary') :
                st.session_state.theme = 'light'
        st.caption(t('theme'))

def render_header():
    with st.spinner(t('loading')) :
        encdm, rgph, _, _, _ = load_data()
    st.markdown(
        f'<div class="header-strip">{t("app_title")} &nbsp;·&nbsp; ENCDM · {len(encdm):,} {t("individuals")} &nbsp;·&nbsp; RGPH · {len(rgph):,} {t("households")}</div>',
        unsafe_allow_html = True,
    )
    return encdm, rgph

def render_footer():
    st.markdown(f'<div class="app-footer">{t("footer")}</div>', unsafe_allow_html = True)

def kpi_row(values, labels):
    cols = st.columns(len(values))
    for col, value, label in zip(cols, values, labels) :
        with col :
            st.markdown(f'<div class="kpi-card"><div class="kpi-value">{value}</div><div class="kpi-label">{label}</div></div>', unsafe_allow_html = True)

def plot_image(path, useframe = True):
    frameclass = 'plot-frame' if st.session_state.theme == 'light' else ''
    if useframe and frameclass :
        st.markdown(f'<div class="{frameclass}">', unsafe_allow_html = True)
    st.image(str(path), use_container_width = True)
    if useframe and frameclass :
        st.markdown('</div>', unsafe_allow_html = True)

def category_label(category):
    return t(CATEGORY_KEYS.get(category, 'all'))

def prob_bar(label, prob, colorkey):
    color = THEMES[st.session_state.theme][colorkey]
    width = int(min(max(prob, 0), 1) * 100)
    st.markdown(f'**{label}** = {prob:.2%}')
    st.markdown(
        f'<div style="background:#444C56;border-radius:4px;height:12px;width:100%;max-width:400px;">'
        f'<div style="background:{color};width:{width}%;height:12px;border-radius:4px;"></div></div>',
        unsafe_allow_html = True,
    )
