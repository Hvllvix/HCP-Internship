"""
Grid-based layout components. Swiss institutional style.
"""
import streamlit as st


def page_header(title, subtitle):
    st.markdown(f"""
    <div class="page-header">
        <h1>{title}</h1>
        <div class="subtitle">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)


def kpi_strip(metrics):
    cells = "".join([f"""
    <div class="kpi-cell">
        <div class="kpi-label">{m[0]}</div>
        <div class="kpi-value">{m[1]}</div>
        <div class="kpi-note">{m[2]}</div>
    </div>
    """ for m in metrics])
    st.markdown(f'<div class="kpi-strip">{cells}</div>', unsafe_allow_html=True)


def section_label(text):
    st.markdown(f'<div class="section-label">{text}</div>', unsafe_allow_html=True)


def section_title(text):
    st.markdown(f'<div class="section-title">{text}</div>', unsafe_allow_html=True)


def section_desc(text):
    st.markdown(f'<div class="section-desc">{text}</div>', unsafe_allow_html=True)


def card(title, body_html):
    st.markdown(f"""
    <div class="card">
        <div class="card-title">{title}</div>
        <div class="card-body">{body_html}</div>
    </div>
    """, unsafe_allow_html=True)


def plot_card(fig, title, description=""):
    if fig is None:
        st.markdown(f"""
        <div class="plot-card">
            <div class="plot-card-title">{title}</div>
            <div class="plot-card-desc">{description}</div>
            <p style="color:var(--ink-muted); font-size:0.85rem;">Visualization not available for current selection.</p>
        </div>
        """, unsafe_allow_html=True)
        return
    st.markdown(f"""
    <div class="plot-card">
        <div class="plot-card-title">{title}</div>
        {f'<div class="plot-card-desc">{description}</div>' if description else ''}
    </div>
    """, unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True)


def render_grid(items, cols=2):
    for i in range(0, len(items), cols):
        columns = st.columns(cols)
        for j in range(cols):
            idx = i + j
            if idx < len(items):
                with columns[j]:
                    item = items[idx]
                    if len(item) == 2:
                        fig, desc = item
                        title = ""
                    else:
                        title, fig, desc = item
                    if callable(fig):
                        fig = fig()
                    if title:
                        plot_card(fig, title, desc)
                    else:
                        if fig is not None:
                            st.plotly_chart(fig, use_container_width=True)


def metric_strip(metrics):
    cells = "".join([f"""
    <div class="metric-cell">
        <div class="metric-label">{m[0]}</div>
        <div class="metric-value">{m[1]}</div>
    </div>
    """ for m in metrics])
    st.markdown(f'<div class="metric-strip">{cells}</div>', unsafe_allow_html=True)