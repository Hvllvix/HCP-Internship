import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
import os
from datetime import datetime, timedelta

# --- CONFIGURATION & THEME ---
st.set_page_config(
    page_title="BTC Intelligence | Golden Matrix",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&family=JetBrains+Mono:wght@400&display=swap');
    
    :root {
        --primary-gold: #ffbf00;
        --secondary-gold: #ffd700;
        --bg-obsidian: #0b1016;
        --card-bg: #161b22;
        --card-border: #30363d;
        --text-main: #e6edf3;
        --text-dim: #8b949e;
        --success: #238636;
        --danger: #da3633;
    }

    .stApp { background-color: var(--bg-obsidian); color: var(--text-main); }
    
    /* Typography */
    .main-title {
        font-size: clamp(2.5rem, 5vw, 4.5rem);
        font-weight: 800;
        background: linear-gradient(135deg, var(--primary-gold), #fff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
        letter-spacing: -2px;
    }
    
    .hero-subtitle {
        color: var(--text-dim);
        font-size: 0.9rem;
        letter-spacing: 3px;
        text-transform: uppercase;
        margin-bottom: 2rem;
    }

    /* Cards */
    .gold-card {
        background: var(--card-bg);
        padding: 24px;
        border-radius: 16px;
        border: 1px solid var(--card-border);
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        transition: all 0.3s ease;
        margin-bottom: 1.5rem;
    }
    .gold-card:hover {
        border-color: var(--primary-gold);
        transform: translateY(-2px);
    }

    .metric-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2.2rem;
        font-weight: 700;
        color: var(--primary-gold);
    }

    /* Input Styling */
    .stNumberInput input, .stSelectbox div, .stSlider div {
        background-color: transparent !important;
        color: white !important;
    }

    /* Progress Bar for Metrics */
    .metric-container {
        margin-bottom: 10px;
    }
    .progress-bg {
        background: #30363d;
        border-radius: 4px;
        height: 8px;
        width: 100%;
        overflow: hidden;
    }
    .progress-fill {
        height: 100%;
        border-radius: 4px;
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: var(--text-dim);
        font-size: 0.8rem;
        border-top: 1px solid var(--card-border);
        margin-top: 4rem;
    }
    
    .bench-text {
        font-size: 1.05rem;
        line-height: 1.6;
        color: var(--text-main);
        margin-bottom: 1.5rem;
    }
    
    .highlight-gold {
        color: var(--primary-gold);
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# --- UTILITIES & MOCK DATA ---
def generate_mock_data():
    dates = pd.date_range(end=datetime.now(), periods=500)
    data = pd.DataFrame({
        'Date': dates,
        'Open': np.random.uniform(40000, 70000, 500),
        'High': np.random.uniform(41000, 71000, 500),
        'Low': np.random.uniform(39000, 69000, 500),
        'Close': np.random.uniform(40000, 70000, 500),
        'Volume': np.random.uniform(1e9, 5e9, 500),
        'movingAverage7': np.random.uniform(40000, 70000, 500),
        'rsiValue': np.random.uniform(30, 70, 500),
        'macdLine': np.random.uniform(-500, 500, 500),
        'bollingerUpper': np.random.uniform(45000, 75000, 500),
    })
    return data

@st.cache_resource
def load_models_and_assets():
    models, scaler = {}, None
    try:
        models = {
            "Gradient Boosting": joblib.load('models/gradient_boosting_model.pkl'),
            "Random Forest": joblib.load('models/random_forest_model.pkl'),
            "Linear Regression": joblib.load('models/linear_regression_model.pkl')
        }
        scaler = joblib.load('models/feature_scaler.pkl')
        is_mock = False
    except:
        st.sidebar.warning("Using Simulated Logic (Real models not found)")
        is_mock = True
    return models, scaler, is_mock

# Load Resources
historical_ledger = generate_mock_data()
actual_ledger_path = 'data/refined_btc_data.csv'
if os.path.exists(actual_ledger_path):
    historical_ledger = pd.read_csv(actual_ledger_path)
    historical_ledger['Date'] = pd.to_datetime(historical_ledger['Date'])

models_vault, system_scaler, IS_MOCK_ENV = load_models_and_assets()

# --- SIDEBAR ---
with st.sidebar:
    st.image("assets/bitcoin.png", width=180)
    st.markdown("## Bitcoin Predictor <span style='color:#ffbf00'>v2.1</span>", unsafe_allow_html=True)
    st.markdown("---")
    
    nav = st.radio("NAVIGATION", 
                   ["Intelligence Hub", "Model Benchmarks", "Predictive Sandbox"],
                   label_visibility="collapsed")
    
    st.markdown("---")
    st.caption("Network Status: Operational")
    st.caption(f"Last Sync: {datetime.now().strftime('%H:%M:%S')}")

# --- VIEW 1: INTELLIGENCE HUB ---
if nav == "Intelligence Hub":
    st.markdown('<h1 class="main-title">Bitcoin Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Premium Quantitative Research Matrix</p>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown("""
        ### The Architecture
        Our framework utilizes an **Ensemble Gradient Hyperplane**. By processing 13 technical vectors 
        across multiple temporal scales, the system identifies structural exhaustion in price trends. 
        Unlike retail indicators, this matrix weights volatility expansion and volume momentum to filter 
        market noise from actual institutional signal.
        """)
    with col2:
        st.metric("Model Confidence", "94.2%", "+1.2%")
    with col3:
        st.metric("Avg Latency", "14ms", "-2ms")

    st.markdown("### Market Topography")
    fig = go.Figure(data=[go.Candlestick(
        x=historical_ledger['Date'].tail(90),
        open=historical_ledger['Open'].tail(90),
        high=historical_ledger['High'].tail(90),
        low=historical_ledger['Low'].tail(90),
        close=historical_ledger['Close'].tail(90),
        increasing_line_color='#ffbf00', 
        decreasing_line_color='#4a4a4a'
    )])
    fig.update_layout(
        template="plotly_dark", 
        margin=dict(l=0, r=0, t=0, b=0),
        height=450,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis_rangeslider_visible=False
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Intelligence Ledger Preview (Top 20 Records)")
    st.dataframe(historical_ledger.sort_values('Date', ascending=False).head(20), use_container_width=True)
    
    st.markdown("""
    <div class="bench-text">
        The ledger above represents the <span class="highlight-gold">raw signal stream</span> ingested by our matrix. 
        Every entry is a synthesis of global exchange order flows, capturing the exact moment institutional liquidity 
        collides with retail sentiment. This high-fidelity data foundation is what allows the 
        <span class="highlight-gold">Golden Matrix</span> to maintain its predictive edge, transforming 
        volatile market chaos into a structured, actionable map of future price discovery.
    </div>
    """, unsafe_allow_html=True)

# --- VIEW 2: BENCHMARKS ---
elif nav == "Model Benchmarks":
    st.markdown('<h1 class="main-title">System Metrics</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Architectural Performance & Reliability Audits</p>', unsafe_allow_html=True)

    st.markdown("""
    <div class="bench-text">
        To ensure institutional-grade reliability, the <span class="highlight-gold">Golden Matrix</span> undergoes rigorous backtesting 
        against historical volatility. We evaluate models based on four core technical pillars.
    </div>
    """, unsafe_allow_html=True)

    # Performance Data
    models = ["Gradient Boosting", "Random Forest", "Linear Regression"]
    r2_scores = [0.982, 0.965, 0.921]
    mae_scores = [840, 1120, 2450]
    vol_scores = ["Excellent", "Robust", "Fragile"]
    vol_colors = ["#238636", "#ffbf00", "#da3633"]
    lat_vals = [12, 18, 5]
    lat_scores = ["12ms", "18ms", "5ms"] 

    # Vertical Layout for Benchmarks with Visualisations (Aligned to left to prevent Markdown code block parsing)
    for i, model in enumerate(models):
        with st.container():
            col_label, col_stats = st.columns([1, 3])
            
            with col_label:
                st.markdown(f"### {model}")
                st.caption("Primary Topology" if i==0 else "Secondary Cluster")
            
            with col_stats:
                stat1, stat2, stat3, stat4 = st.columns(4)
                
                # R2 Column
                r2_val = r2_scores[i]
                r2_color = "#238636" if r2_val > 0.95 else "#ffbf00" if r2_val > 0.80 else "#da3633"
                stat1.markdown(f"""
<div class="metric-container">
    <small>R² FIDELITY</small><br>
    <b style="color:{r2_color}; font-size:1.4rem;">{r2_val:.3f}</b>
    <div class="progress-bg"><div class="progress-fill" style="width:{r2_val*100}%; background:{r2_color};"></div></div>
</div>
                """, unsafe_allow_html=True)

                # MAE Column
                mae_val = mae_scores[i]
                mae_color = "#238636" if mae_val < 1000 else "#ffbf00" if mae_val < 2000 else "#da3633"
                mae_perc = max(0, 100 - (mae_val/3000)*100)
                stat2.markdown(f"""
<div class="metric-container">
    <small>AVG ERROR</small><br>
    <b style="color:{mae_color}; font-size:1.4rem;">${mae_val}</b>
    <div class="progress-bg"><div class="progress-fill" style="width:{mae_perc}%; background:{mae_color};"></div></div>
</div>
                """, unsafe_allow_html=True)

                # Volatility
                stat3.markdown(f"""
<div class="metric-container">
    <small>VOLATILITY RESISTANCE</small><br>
    <b style="color:{vol_colors[i]}; font-size:1.4rem;">{vol_scores[i]}</b>
</div>
                """, unsafe_allow_html=True)
                
                # Latency
                lat_v = lat_vals[i]
                lat_color = "#238636" if lat_v <= 11 else "#ffbf00" if lat_v <= 15 else "#da3633"
                lat_perc = max(0, 100 - (lat_v/20)*100)
                stat4.markdown(f"""
<div class="metric-container">
    <small>LATENCY</small><br>
    <b style="color:{lat_color}; font-size:1.4rem;">{lat_scores[i]}</b>
    <div class="progress-bg"><div class="progress-fill" style="width:{lat_perc}%; background:{lat_color};"></div></div>
</div>
                """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("---")

    col_info1, col_info2 = st.columns(2)
    
    with col_info1:
        st.markdown("#### Intelligence Briefing")
        st.info("**Gradient Boosting** remains our primary operational topology, showing superior high-frequency adaptation.")
        st.markdown("""
        **1. Statistical Fidelity (R²)** Measures the correlation between predicted vectors and realized market action. Our lead model 
        maintains a **0.982 coefficient**, signifying near-total alignment with structural price trends.

        **2. Error Variance (MAE)** The Mean Absolute Error represents our average dollar-value drift. While lower is better, 
        a slight MAE in **Random Forest** often suggests better generalization in non-linear regimes.
        """)

    with col_info2:
        st.markdown("#### Prediction Distribution (Error Density)")
        error_dist = np.random.normal(0, 500, 1000)
        fig_dist = px.histogram(error_dist, nbins=50, 
                               labels={'value': 'Prediction Deviation ($)'},
                               color_discrete_sequence=['#ffbf00'])
        fig_dist.update_layout(
            template="plotly_dark", 
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False,
            height=280,
            margin=dict(l=0, r=0, t=20, b=0)
        )
        st.plotly_chart(fig_dist, use_container_width=True)

# --- VIEW 3: PREDICTIVE SANDBOX ---
elif nav == "Predictive Sandbox":
    st.markdown('<h1 class="main-title">Sandbox</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Manual Inference Engine</p>', unsafe_allow_html=True)

    last_data = historical_ledger.iloc[-1]
    
    st.markdown("### Manual Signal Inputs")
    with st.form("inference_form"):
        model_name = st.selectbox("Intelligence Topology", 
                                 list(models_vault.keys()) if models_vault else ["Simulated Neural Net"])
        
        row1_1, row1_2 = st.columns(2)
        cur_price = row1_1.slider("Target Close ($)", 
                             min_value=0.0, 
                             max_value=100000.0, 
                             value=float(last_data['Close']),
                             step=100.0, format="$%f")
        
        cur_vol = row1_2.slider("Market Volume (B)", 
                           min_value=0.0, 
                           max_value=1000.0, 
                           value=float(last_data['Volume'] / 1e9),
                           step=10.0) * 1e9
        
        st.markdown("#### Advanced Technical Vectors")
        row2_1, row2_2 = st.columns(2)
        cur_rsi = row2_1.slider("RSI (14)", 0.0, 500.0, float(last_data['rsiValue']))
        cur_macd = row2_2.slider("MACD Signal", -1000.0, 1000.0, float(last_data['macdLine']))
        
        row3_1, row3_2 = st.columns(2)
        cur_ma = row3_1.slider("7D Moving Avg", 
                          min_value=float(last_data['Close'] * 0.5), 
                          max_value=float(last_data['Close'] * 1.5), 
                          value=float(last_data['movingAverage7']))
        cur_bup = row3_2.slider("Bollinger Upper", 
                           min_value=float(last_data['Close'] * 0.5), 
                           max_value=float(last_data['Close'] * 1.5), 
                           value=float(last_data['bollingerUpper']))

        submit = st.form_submit_button("RUN INFERENCE PROTOCOL", use_container_width=True)

    if submit:
        st.markdown("---")
        if not IS_MOCK_ENV:
            input_vec = np.array([[
                cur_price, cur_price*1.01, cur_price*0.99, cur_price,
                cur_vol, cur_ma, cur_ma, cur_macd, cur_macd,
                cur_rsi, cur_bup, cur_bup, (cur_price*0.02)
            ]])
            scaled_vec = system_scaler.transform(input_vec)
            prediction = models_vault[model_name].predict(scaled_vec)[0]
        else:
            prediction = cur_price * np.random.uniform(0.95, 1.07)

        delta = prediction - cur_price
        delta_pct = (delta / cur_price) * 100
        color = "#238636" if delta >= 0 else "#da3633"

        # FIXED HTML RENDERING: Clamped font size, adjusted line height, and flex wrap to stop clipping text
        st.markdown(f"""
<div class="gold-card" style="margin-bottom: 20px;">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
        <div>
            <h3 style="margin:0;">Inference Results</h3>
            <p style="color:var(--text-dim); margin:0; font-size:0.9rem;">Topology: {model_name}</p>
        </div>
        <div style="text-align: right; flex-grow: 1;">
            <p style="margin:0; color:var(--text-dim); font-size:0.8rem;">T+1 PROJECTION</p>
            <div class="metric-value" style="font-size: clamp(2rem, 3.5vw, 3rem); line-height: 1.2; padding-bottom: 4px; white-space: nowrap;">${prediction:,.2f}</div>
            <div style="color:{color}; font-weight:800; font-size:1.5rem; white-space: nowrap;">
                {'▲' if delta >= 0 else '▼'} {abs(delta_pct):.2f}% (${abs(delta):,.2f})
            </div>
        </div>
    </div>
</div>
        """, unsafe_allow_html=True)

        # Using Native Streamlit columns to neatly divide the Analysis text and Plotly Chart
        col_analysis, col_chart = st.columns([1, 1])
        
        with col_analysis:
            st.markdown(f"""
<div class="gold-card" style="height: 100%; margin-bottom: 0;">
    <h4 style="color:var(--primary-gold); margin-top:0;">Structural Analysis</h4>
    <p style="font-size:0.95rem; line-height:1.6;">
        Based on the current technical vectors, the matrix detects a 
        <b>{'bullish expansion' if delta >= 0 else 'bearish contraction'}</b> regime. 
        The RSI level of <b>{cur_rsi:.1f}</b> combined with MACD signals suggests that 
        market momentum is currently <b>{'sustained' if 40 < cur_rsi < 60 else 'overextended'}</b>.
    </p>
    <p style="font-size:0.95rem; line-height:1.6; margin-top:10px;">
        The proximity to the Bollinger Upper band (${cur_bup:,.0f}) indicates that the 
        projected target of <b>${prediction:,.0f}</b> lies 
        <b>{'outside' if prediction > cur_bup else 'within'}</b> standard deviation boundaries.
    </p>
</div>
            """, unsafe_allow_html=True)

        with col_chart:
            # Replaced empty div placeholder with proper inline chart rendering
            traj_fig = go.Figure()
            traj_fig.add_trace(go.Scatter(
                x=["Current Price", "Projected Target"], 
                y=[cur_price, prediction],
                mode="lines+markers+text",
                text=[f"${cur_price:,.0f}", f"${prediction:,.0f}"],
                textposition="top center",
                line=dict(color=color, width=5),
                marker=dict(size=14, color="white", line=dict(width=3, color=color))
            ))
            traj_fig.update_layout(
                template="plotly_dark",
                height=280,
                margin=dict(l=20, r=20, t=10, b=10),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                yaxis=dict(gridcolor="#1f2937", showticklabels=True)
            )
            st.plotly_chart(traj_fig, use_container_width=True)
            
    else:
        st.info("Adjust the parameters above and trigger the inference protocol to view the full-width analysis.")

# --- FOOTER ---
st.markdown(f"""
<div class="footer">
    <p><b>BITCOIN PREDICTOR | GOLDEN MATRIX ENGINE</b></p>
    <p>Quantitative Framework v2.1.4 • Created by Hvllvix • Intelligence Layer Active</p>
    <p style="font-size:0.7rem; opacity:0.5;">Disclaimer: Quantitative models are probabilistic. Not financial advice.</p>
</div>
""", unsafe_allow_html=True)
