import streamlit as st
from utils.chrome import render_header, render_footer, plot_image, t, category_label
from utils.data import load_data, plot_manifest, map_options
from utils.i18n import CATEGORY_KEYS

render_header()

st.title(t('nav_explorer'))

_, _, mapencdm, _, _ = load_data()
lang = st.session_state.lang
plots = plot_manifest()
categories = ['All'] + list(CATEGORY_KEYS.keys())
regionlabels = {t('all'): None, **{label : code for label, code in map_options(mapencdm['Région_12'], lang).items()}}
milieulabels = {t('all'): None, t('urban'): 0, t('rural'): 1}

st.subheader(t('explorer_filters'))
fcol1, fcol2, fcol3 = st.columns(3)
with fcol1 :
    category = st.selectbox(t('category'), categories, format_func = lambda c : t('all') if c == 'All' else category_label(c))
with fcol2 :
    regionlabel = st.selectbox(t('region'), list(regionlabels.keys()))
with fcol3 :
    milieulabel = st.selectbox(t('milieu'), list(milieulabels.keys()))

filtered = [p for p in plots if category == 'All' or p['category'] == category]

st.subheader(t('plot_grid'))
if not filtered :
    st.info(t('no_plots'))
else :
    thumbcols = st.columns(3)
    if 'selected_plot' not in st.session_state : st.session_state.selected_plot = filtered[0]['file']
    for index, plot in enumerate(filtered) :
        with thumbcols[index % 3] :
            if st.button(plot['file'].removesuffix('.png'), key = f'thumb_{plot["file"]}', use_container_width = True) :
                st.session_state.selected_plot = plot['file']
            st.caption(plot[lang])

    selected = next((p for p in filtered if p['file'] == st.session_state.selected_plot), filtered[0])
    st.subheader(t('plot_preview'))
    plot_image(selected['path'])
    st.caption(f"**{selected['file']}** — {selected[lang]}")
    if regionlabels[regionlabel] is not None :
        regionname = next(label for code, label in mapencdm['Région_12'].items() if code == regionlabels[regionlabel])
        st.caption(t('region_context', region = regionname))

render_footer()
