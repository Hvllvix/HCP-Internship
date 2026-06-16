import streamlit as st
from contextlib import contextmanager
from .theme import css
from .palette import PALETTE
from .i18n import t

def init_session():
    if 'theme' not in st.session_state : st.session_state.theme = 'dark'
    if 'lang' not in st.session_state : st.session_state.lang = 'en'

def apply_theme():
    st.markdown(css(st.session_state.theme), unsafe_allow_html = True)

def sidebar_controls():
    lang = st.session_state.lang
    st.markdown(f'<p class="section-label">{t("lang", lang)}</p>', unsafe_allow_html = True)
    c1, c2 = st.columns(2)
    if c1.button('EN', width = 'stretch', type = 'primary' if lang == 'en' else 'secondary') :
        st.session_state.lang = 'en'
        st.rerun()
    if c2.button('FR', width = 'stretch', type = 'primary' if lang == 'fr' else 'secondary') :
        st.session_state.lang = 'fr'
        st.rerun()

    isdark = st.toggle('Dark mode' if lang == 'en' else 'Mode sombre', value = st.session_state.theme == 'dark', key = 'theme_toggle')
    if isdark != (st.session_state.theme == 'dark') :
        st.session_state.theme = 'dark' if isdark else 'light'
        st.rerun()

@contextmanager
def card():
    with st.container(border = True) :
        yield

def kpi_row(values, labels):
    cols = st.columns(len(values))
    for col, value, label in zip(cols, values, labels) :
        with col :
            st.markdown(
                f'<div style="text-align:center;padding:0.25rem 0;"><div class="kpi-value">{value}</div><div class="kpi-label">{label}</div></div>',
                unsafe_allow_html = True,
            )

def section(title, subtitle = ''):
    if subtitle :
        st.markdown(f'<p class="section-label">{subtitle}</p>', unsafe_allow_html = True)
    st.markdown(f'## {title}')

def body(text):
    st.markdown(f'<p class="body-text">{text}</p>', unsafe_allow_html = True)

def footer():
    lang = st.session_state.lang
    st.markdown(f"""
    <div class="app-footer">
        <strong>{t('footer_name', lang)}</strong> · {t('footer_sub', lang)}<br>
        <span class="muted">{t('footer_note', lang)}</span>
    </div>
    """, unsafe_allow_html = True)

def progress_bar(label, value, colorkey, theme):
    p = PALETTE[theme]
    color = p[colorkey]
    width = int(min(max(value, 0), 1) * 100)
    st.markdown(f'<div style="margin-bottom:0.35rem;font-size:0.88rem;color:var(--text);">{label} <b>{value:.1%}</b></div>', unsafe_allow_html = True)
    st.markdown(f'<div class="progress-bg"><div class="progress-fill" style="width:{width}%;background:{color};"></div></div>', unsafe_allow_html = True)

def png_frame(theme):
    if theme == 'light' :
        return 'border:1px solid var(--border);border-radius:12px;padding:0.35rem;background:#1a1d24;'
    return ''
