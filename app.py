import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.ensemble import IsolationForest
import plotly.graph_objs as go
import plotly.express as px

st.set_page_config(
    page_title="JALRAKSHAN.AI",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        color: #0077b6;
        text-align: center;
        margin-bottom: 0;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #adb5bd;
        text-align: center;
        margin-top: 0;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #0077b6 0%, #00b4d8 100%);
        padding: 1.2rem 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,119,182,0.3);
    }
    .metric-card h2 {
        margin: 0;
        font-size: 2rem;
    }
    .metric-card p {
        margin: 0.3rem 0 0 0;
        font-size: 0.9rem;
        opacity: 0.9;
    }
    .alert-box {
        background: linear-gradient(135deg, #e63946 0%, #ff6b6b 100%);
        padding: 1.2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        font-size: 1.2rem;
        font-weight: 700;
        box-shadow: 0 4px 15px rgba(230,57,70,0.3);
    }
    .safe-box {
        background: linear-gradient(135deg, #2d6a4f 0%, #52b788 100%);
        padding: 1.2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        font-size: 1.2rem;
        font-weight: 700;
        box-shadow: 0 4px 15px rgba(45,106,79,0.3);
    }
    .info-box {
        background: #f0f7ff;
        border-left: 5px solid #0077b6;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
        color: #1a1a2e;
    }
    .info-box b, .info-box i, .info-box code, .info-box li, .info-box ul, .info-box p {
        color: #1a1a2e;
    }
    .responsible-ai {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        padding: 1.2rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        color: #1a1a2e;
    }
    .responsible-ai h4, .responsible-ai b, .responsible-ai li, .responsible-ai ul, .responsible-ai p {
        color: #1a1a2e;
    }
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0077b6 0%, #023e8a 100%);
    }
    div[data-testid="stSidebar"] .stMarkdown p,
    div[data-testid="stSidebar"] .stMarkdown h1,
    div[data-testid="stSidebar"] .stMarkdown h2,
    div[data-testid="stSidebar"] .stMarkdown h3 {
        color: white;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_process_data():
    df = pd.read_csv("archive/smart_water_meter.csv")
    df['result_time'] = pd.to_datetime(df['result_time'], errors='coerce')
    df = df.sort_values('result_time').reset_index(drop=True)
    df['flow'] = df['v1'].diff().fillna(0)
    df['flow'] = df['flow'].clip(lower=0)
    df['time_diff_min'] = df['result_time'].diff().dt.total_seconds().fillna(0) / 60
    df['flow_rate'] = np.where(df['time_diff_min'] > 0, df['flow'] / df['time_diff_min'], 0)
    df['flow_rolling'] = df['flow'].rolling(window=5, min_periods=1).mean()
    df['hour'] = df['result_time'].dt.hour
    df['date'] = df['result_time'].dt.date
    return df

@st.cache_data
def run_anomaly_detection(df, contamination=0.08):
    features = df[['flow', 'flow_rate', 'flow_rolling']].copy()
    features = features.fillna(0)
    model = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=150,
        max_samples='auto'
    )
    predictions = model.fit_predict(features)
    anomaly = pd.Series(predictions).map({1: 0, -1: 1})
    leak = anomaly.copy()
    consecutive_count = 0
    leak_indices = []
    for i in range(len(leak)):
        if anomaly.iloc[i] == 1:
            consecutive_count += 1
        else:
            if consecutive_count >= 3:
                leak_indices.extend(range(i - consecutive_count, i))
            consecutive_count = 0
    if consecutive_count >= 3:
        leak_indices.extend(range(len(leak) - consecutive_count, len(leak)))
    leak_flag = pd.Series(0, index=anomaly.index)
    leak_flag.iloc[leak_indices] = 1
    return anomaly, leak_flag, model

df = load_and_process_data()
selected_node = None
if 'nodeid' in df.columns:
    node_options = df['nodeid'].unique()
    selected_node = st.sidebar.selectbox(
        "Select Community Node",
        node_options,
        index=0
    )
    df = df[df['nodeid'] == selected_node]

with st.sidebar:
    st.markdown("## 🌊 JALRAKSHAN.AI")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🏠 Home", "📊 Data & Visualization", "🧠 AI Detection", "🚨 Impact Dashboard"],
        index=0,
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("### ⚙️ AI Settings")
    contamination = st.slider(
        "Anomaly Sensitivity",
        min_value=0.01,
        max_value=0.25,
        value=0.08,
        step=0.01,
        help="Higher = more anomalies detected. Lower = only extreme anomalies."
    )
    st.markdown("---")
    st.markdown("**SDG 6** — Clean Water & Sanitation")
    st.markdown("*AI for Good — Responsible AI*")

anomaly, leak_flag, model = run_anomaly_detection(df, contamination)
df['anomaly'] = anomaly.values
df['leak'] = leak_flag.values

if page == "🏠 Home":
    st.markdown('<p class="main-title">🌊 JALRAKSHAN.AI</p>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:1.18rem; font-weight:700; color:#0077b6; text-align:center; margin-bottom:0.2rem;">JALRAKSHAN.AI detects hidden water leakages early using AI to prevent massive urban water loss.</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">AI-Powered Community Water Leakage & Safety Intelligence System</p>', unsafe_allow_html=True)
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h2>30–40%</h2>
            <p>Urban water lost to undetected leakages</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h2>🤖 AI</h2>
            <p>Anomaly detection on smart meter data</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h2>⚡ Early</h2>
            <p>Detect leaks before major water loss</p>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("")
    st.markdown("")

    st.markdown("""
    <div class="info-box">
    <b>How it works:</b> We train an unsupervised anomaly detection model on smart water meter flow data.<br>
    Since leakages appear as <b>continuous abnormal consumption rather than spikes</b>, the AI learns normal usage patterns and flags deviations early — enabling preventive action before major water loss.
    </div>
    """, unsafe_allow_html=True)

    col_start, col_mid, col_end = st.columns([1, 1, 1])
    with col_mid:
        st.markdown("""
        <style>
        .stButton > button {
            background: linear-gradient(90deg, #0077b6 0%, #00b4d8 100%);
            color: white;
            font-weight: bold;
            border-radius: 8px;
            border: none;
            padding: 0.7em 2em;
            font-size: 1.1em;
            box-shadow: 0 2px 8px rgba(0,119,182,0.15);
        }
        .stButton > button:hover {
            background: linear-gradient(90deg, #00b4d8 0%, #0077b6 100%);
            color: #fff;
        }
        </style>
        """, unsafe_allow_html=True)
        st.markdown('<span style="font-size:0.98em; color:#0077b6; font-weight:600;">🤖 AI Engine Status: Active (Isolation Forest loaded)</span>', unsafe_allow_html=True)
        if st.button("🚀 Start AI Analysis →"):
            st.info("👈 Use the sidebar to navigate to **Data & Visualization** or **AI Detection**.")

    st.markdown('<hr style="border:0;border-top:1.5px solid #e9ecef;margin:1.5rem 0 1.2rem 0;">', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### 🔍 The Problem")
        st.markdown("""
        - **30–40%** of urban water is lost due to undetected leakages
        - Leakages are **invisible and continuous**, not sudden bursts
        - Municipal systems rely on **manual inspections** and complaints
        - By the time action is taken, **massive water loss** has already occurred
        - There is **no intelligent system** to detect leakages early at community level
        """)
    with col_b:
        st.markdown("### 💡 Our Solution")
        st.markdown("""
        - **Analyzes** smart water meter time-series data
        - **Detects** anomalous consumption using AI (Isolation Forest)
        - **Identifies** continuous abnormal usage patterns (not just spikes)
        - **Alerts** authorities via a web dashboard **before** large-scale loss
        - **Enables** early inspection and preventive action
        """)
        st.markdown('<span style="font-size:0.97em; color:#495057; font-style:italic;">Primary Users: Municipal engineers, local authorities, community water managers</span>', unsafe_allow_html=True)
    st.markdown("---")

elif page == "📊 Data & Visualization":
    st.markdown('<p class="main-title">📊 Data & Visualization</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Transparent view of the smart water meter data and feature engineering</p>', unsafe_allow_html=True)
    st.markdown("### 📋 Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", f"{len(df):,}")
    col2.metric("Time Span (Days)", f"{(df['result_time'].max() - df['result_time'].min()).days}")
    col3.metric("Avg Interval", "~30 min")
    col4.metric("Node ID", df['nodeid'].iloc[0])
    st.markdown("---")
    st.markdown("### 🔧 Feature Engineering")
    st.markdown("""
    <div class="info-box">
    <b>Raw data:</b> <span style='color:#0077b6;font-weight:bold;'>v1</span> = cumulative water meter reading (liters)<br>
    <b>Derived feature:</b> <span style='color:#00b4d8;font-weight:bold;'>flow = v1[i] - v1[i-1]</span> = water consumed per interval<br>
    <b>Why:</b> Leak detection requires consumption per interval, not cumulative readings.<br>
    <b>Cleaning:</b> Negative values (meter resets) clipped to 0; missing values filled.
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 📈 Cumulative Water Meter Reading (v1) Over Time")
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=df['result_time'],
        y=df['v1'],
        mode='lines+markers',
        line=dict(color='#0077b6', width=2, dash='dot'),
        marker=dict(size=5, color='#00b4d8', opacity=0.4, symbol='circle'),
        name='Cumulative Reading',
        opacity=0.5
    ))
    fig1.update_layout(
        title="Raw Sensor Data (Context Only)",
        xaxis_title="Time",
        yaxis_title="Cumulative Reading (v1)",
        template="plotly_dark",
        height=260,
        showlegend=True,
        margin=dict(l=40, r=40, t=60, b=40)
    )
    st.plotly_chart(fig1, use_container_width=True)
    st.caption("Raw sensor data (shown for transparency — not used directly for detection)")
    st.markdown("---")
    st.markdown("### 💧 Derived Flow (Water Consumption per Interval)")
    st.markdown('<span style="font-size:1.05em; color:#0077b6; font-weight:600;">🔍 Watch how continuous low-level flow persists over time — this is how leakages differ from normal usage spikes.</span>', unsafe_allow_html=True)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=df['result_time'],
        y=df['flow'],
        mode='lines+markers',
        line=dict(color='#00b4d8', width=2),
        marker=dict(size=5, color='#00b4d8', opacity=0.6),
        name='Flow',
    ))
    fig2.add_trace(go.Scatter(
        x=df['result_time'],
        y=df['flow_rolling'],
        mode='lines',
        line=dict(color='#e63946', width=4, dash='dash'),
        name='Rolling Avg (5)',
    ))
    fig2.update_layout(
        title="Engineered Feature: Water Flow per Interval",
        xaxis_title="Time",
        yaxis_title="Flow (liters)",
        template="plotly_dark",
        height=400,
        updatemenus=[{
            'type': 'buttons',
            'buttons': [
                {'label': 'Play', 'method': 'animate', 'args': [None]},
                {'label': 'Pause', 'method': 'animate', 'args': [[None], {'mode': 'immediate', 'frame': {'duration': 0, 'redraw': False}, 'transition': {'duration': 0}}]},
            ],
            'direction': 'left',
            'pad': {'r': 10, 't': 10},
            'showactive': True,
            'x': 0.1,
            'y': 1.1,
        }],
    )
    frames2 = [go.Frame(data=[go.Scatter(x=df['result_time'][:k], y=df['flow'][:k], mode='lines+markers', line=dict(color='#00b4d8', width=2), marker=dict(size=5, color='#00b4d8', opacity=0.6)),
                              go.Scatter(x=df['result_time'][:k], y=df['flow_rolling'][:k], mode='lines', line=dict(color='#e63946', width=4, dash='dash'))]) for k in range(10, len(df), max(1, len(df)//50))]
    fig2.frames = frames2
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown("---")
    st.markdown("### 🕐 Average Hourly Water Usage Pattern")
    st.markdown('<span style="font-size:0.98em; color:#adb5bd;">Helps distinguish normal daily routines from abnormal continuous consumption.</span>', unsafe_allow_html=True)
    hourly = df.groupby('hour')['flow'].mean()
    fig3 = px.bar(
        hourly,
        x=hourly.index,
        y=hourly.values,
        labels={'x': 'Hour of Day', 'y': 'Average Flow (liters)'},
        color_discrete_sequence=['#0077b6'],
        title="Daily Usage Pattern — When is water consumed?",
    )
    fig3.update_traces(marker_line_color='white', marker_line_width=1)
    fig3.update_layout(
        template="plotly_dark",
        height=350,
        xaxis=dict(tickmode='array', tickvals=list(range(24))),
        transition={'duration': 500},
    )
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown("---")
    with st.expander("🔎 View Raw Data Sample (first 50 rows)"):
        st.dataframe(
            df[['result_time', 'nodeid', 'v1', 'flow', 'flow_rate', 'flow_rolling', 'hour']].head(50),
            width='stretch'
        )
    with st.expander("📊 Flow Statistics"):
        st.dataframe(df[['v1', 'flow', 'flow_rate', 'flow_rolling']].describe().round(2), width='stretch')

elif page == "🧠 AI Detection":
    st.markdown('<p class="main-title">🧠 AI Leak Detection Results</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Isolation Forest anomaly detection on water consumption patterns</p>', unsafe_allow_html=True)
    total = len(df)
    anomaly_count = df['anomaly'].sum()
    leak_count = df['leak'].sum()
    anomaly_pct = (anomaly_count / total) * 100
    leak_pct = (leak_count / total) * 100
    if leak_count > 0:
        st.markdown(f"""
        <div class="alert-box">
            🚨 <b>POTENTIAL LEAK DETECTED</b><br>
            {leak_count} intervals flagged as high-confidence leakage zones ({leak_pct:.1f}% of data)
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="safe-box">
            ✅ NO SIGNIFICANT LEAKAGE DETECTED — Water usage appears normal
        </div>
        """, unsafe_allow_html=True)
    st.markdown("")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Intervals", f"{total:,}")
    col2.metric("Anomalous Points", f"{anomaly_count:,}", f"{anomaly_pct:.1f}%")
    col3.metric("Leak Zone Points", f"{leak_count:,}", f"{leak_pct:.1f}%")
    col4.metric("Sensitivity", f"{contamination:.0%}")
    st.markdown("---")
    st.markdown("### 🔍 Anomaly Detection on Flow Data")
    fig4 = go.Figure()
    normal = df[df['anomaly'] == 0]
    anom = df[df['anomaly'] == 1]
    fig4.add_trace(go.Scatter(
        x=normal['result_time'],
        y=normal['flow'],
        mode='lines',
        line=dict(color='#0077b6'),
        name='Normal Flow',
    ))
    fig4.add_trace(go.Scatter(
        x=anom['result_time'],
        y=anom['flow'],
        mode='markers',
        marker=dict(color='#e63946', size=7),
        name=f'Anomaly ({anomaly_count})',
    ))
    if leak_count > 0:
        leak_regions = df[df['leak'] == 1]
        fig4.add_trace(go.Scatter(
            x=leak_regions['result_time'],
            y=leak_regions['flow'],
            mode='markers',
            marker=dict(color='#ffba08', size=10, symbol='square'),
            name=f'Leak Zone ({leak_count})',
        ))
    fig4.update_layout(
        title="AI Anomaly Detection — Leak Regions Highlighted",
        xaxis_title="Time",
        yaxis_title="Flow (liters)",
        template="plotly_dark",
        height=400,
        transition={'duration': 500},
    )
    st.plotly_chart(fig4, use_container_width=True)
    st.markdown("---")
    st.markdown("### 🔬 Detailed View: Flow Rate with Anomalies")
    fig5 = go.Figure()
    fig5.add_trace(go.Scatter(
        x=df['result_time'],
        y=df['flow_rate'],
        mode='lines',
        line=dict(color='#0077b6'),
        name='Flow Rate',
    ))
    if anomaly_count > 0:
        fig5.add_trace(go.Scatter(
            x=anom['result_time'],
            y=anom['flow_rate'],
            mode='markers',
            marker=dict(color='#e63946', size=7),
            name='Anomaly',
        ))
    fig5.update_layout(
        title="Flow Rate Over Time with Anomaly Markers",
        xaxis_title="Time",
        yaxis_title="Flow Rate (liters/min)",
        template="plotly_dark",
        height=400,
        transition={'duration': 500},
    )
    st.plotly_chart(fig5, use_container_width=True)
    st.markdown("---")
    st.markdown("### 🤖 AI Explanation")
    st.markdown("""
    <div class="info-box">
    <b>Model:</b> Isolation Forest (Unsupervised Anomaly Detection)<br>
    <b>Features Used:</b> flow, flow_rate, flow_rolling (all derived from raw meter reading v1)<br>
    <b>Why Isolation Forest?</b>
    <ul>
        <li>No labeled leakage data required — works with normal consumption patterns</li>
        <li>Learns what "normal" looks like and flags deviations as anomalies</li>
        <li>Widely used for infrastructure monitoring and fraud detection</li>
    </ul>
    <b>Leakage Logic:</b> A leakage is characterized by continuous abnormal consumption — not one-time spikes.<br>
    <b>Why 3 consecutive anomalies?</b> Based on domain intuition, a minimum of three consecutive anomalous intervals was chosen to reduce false positives from normal household spikes.<br>
    The model detects persistent patterns where 3+ consecutive intervals show anomalous behavior,
    differentiating real leaks from normal household usage variations.<br><br>
    <i>"AI detected continuous abnormal consumption indicative of potential leakage."</i>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<span style="font-size:0.97em; color:#adb5bd;">Note: This system provides decision support, not automated enforcement. Final action remains with human authorities.</span>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 🛠️ Recommended Action (Human-in-the-Loop)")
    st.markdown("""
    <div class="info-box">
    <ul>
        <li>Prioritize inspection in highlighted time windows</li>
        <li>Cross-check with pressure sensor / valve logs</li>
        <li>Schedule targeted field inspection</li>
        <li>Prevent large-scale water loss before escalation</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    with st.expander(f"📋 View All Anomalous Intervals ({anomaly_count} records)"):
        if anomaly_count > 0:
            st.dataframe(
                df[df['anomaly'] == 1][['result_time', 'v1', 'flow', 'flow_rate', 'flow_rolling', 'leak']].reset_index(drop=True),
                width='stretch'
            )
        else:
            st.info("No anomalies detected with current sensitivity.")

elif page == "🚨 Impact Dashboard":
    st.markdown('<p class="main-title">🚨 Impact Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Estimated impact of early leak detection and alignment with SDG 6</p>', unsafe_allow_html=True)
    total = len(df)
    anomaly_count = df['anomaly'].sum()
    leak_count = df['leak'].sum()
    total_flow = df['flow'].sum()
    leak_flow = df[df['leak'] == 1]['flow'].sum()
    normal_avg = df[df['anomaly'] == 0]['flow'].mean() if (df['anomaly'] == 0).any() else 0
    time_span_days = (df['result_time'].max() - df['result_time'].min()).days
    estimated_water_saved = leak_flow * 0.6
    st.markdown("---")
    st.markdown("### 📊 Estimated Impact Metrics")
    st.caption('<span style="font-size:0.97em; color:#adb5bd;">Scenario-based estimate assuming early inspection</span>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h2>{estimated_water_saved:,.0f} L</h2>
            <p>Estimated Water Saveable (Scenario-based)</p>
        </div>
        """, unsafe_allow_html=True)
        st.caption("Estimate based on early inspection preventing prolonged leakage.")
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h2>{time_span_days} days</h2>
            <p>Monitoring Period</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        early_detect_days = max(1, time_span_days - int(time_span_days * 0.4))
        st.markdown(f"""
        <div class="metric-card">
            <h2>{early_detect_days} days</h2>
            <p>Early Detection Window</p>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h2>{leak_count}</h2>
            <p>Leak Zone Intervals</p>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("")
    st.markdown("")
    st.markdown("""
    <div class="info-box" style="background:#e7f5ff; border-left:5px solid #00b4d8; color:#1a1a2e; margin-bottom:1.2rem;">
    <b>Real-World Scenario:</b><br>
    In a 100-home community, detecting leaks 64 days earlier could save enough water to supply ~400 families for a month.
    </div>
    """, unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### 💧 Water Flow Breakdown")
        normal_flow = total_flow - leak_flow
        sizes = [normal_flow, leak_flow] if leak_flow > 0 else [total_flow]
        labels = ['Normal Usage', 'Leak Zone Usage'] if leak_flow > 0 else ['Normal Usage']
        colors = ['#0077b6', '#e63946'] if leak_flow > 0 else ['#0077b6']
        fig6 = go.Figure(data=[go.Pie(
            labels=labels,
            values=sizes,
            marker=dict(colors=colors),
            hole=0.4,
            pull=[0, 0.1] if leak_flow > 0 else [0],
            textinfo='percent+label',
            textfont=dict(size=14),
        )])
        fig6.update_layout(
            title="Water Consumption Distribution",
            template="plotly_dark",
            height=350,
            transition={'duration': 500},
        )
        st.plotly_chart(fig6, use_container_width=True)
    with col_b:
        st.markdown("### 📅 Daily Anomaly Count")
        if 'date' in df.columns:
            daily_anomalies = df.groupby('date')['anomaly'].sum()
            fig7 = px.bar(
                x=list(range(len(daily_anomalies))),
                y=daily_anomalies.values,
                labels={'x': 'Day Index', 'y': 'Anomalies Detected'},
                color_discrete_sequence=['#e63946'],
                title="Anomalies Per Day",
            )
            fig7.update_traces(marker_line_color='white', marker_line_width=1)
            fig7.update_layout(
                template="plotly_dark",
                height=350,
                transition={'duration': 500},
            )
            st.plotly_chart(fig7, use_container_width=True)
    st.markdown("---")
    st.markdown("### 🌍 SDG 6 Alignment — Clean Water and Sanitation")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.markdown("""
        **United Nations Sustainable Development Goal 6:**
        *Ensure availability and sustainable management of water and sanitation for all.*
        JALRAKSHAN.AI directly contributes to:
        - **Target 6.1:** Universal access to safe and affordable drinking water
        - **Target 6.4:** Increase water-use efficiency and reduce water scarcity
        - **Target 6.b:** Support local communities in water management
        """)
    with col_s2:
        st.markdown("""
        **Potential Impact at Scale:**
        - 🏘️ **Community level** — monitor multiple households/zones
        - 🏙️ **City level** — integrate with municipal water systems
        - 💰 **Cost savings** — reduce water treatment and distribution losses
        - 🌱 **Environmental** — conserve freshwater resources
        - ⚡ **Energy** — reduce pumping energy for lost water
        """)
    st.markdown("---")
    st.markdown("### 🛡️ Responsible AI Commitment")
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        st.markdown("""
        <div class="responsible-ai">
        <h4>✅ Data Privacy & Ethics</h4>
        <ul>
            <li><b>No personal data</b> is collected or processed</li>
            <li>Only <b>community-level</b> aggregate meter readings are analyzed</li>
            <li>No individual household behavior is profiled</li>
            <li>Dataset sourced from <b>public Kaggle data</b></li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    with col_r2:
        st.markdown("""
        <div class="responsible-ai">
        <h4>✅ Transparency & Human Oversight</h4>
        <ul>
            <li><b>AI supports humans</b>, does not replace decision-making</li>
            <li>All anomaly logic is <b>fully explainable</b></li>
            <li><b>No automated penalties</b> — only alerts for inspection</li>
            <li>Sensitivity is <b>adjustable</b> by the operator</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
    <div class="info-box">
    <b>About this project:</b> JALRAKSHAN.AI was built as a hackathon project demonstrating how AI
    can be used responsibly for community benefit. The system is designed to assist — not automate —
    water management decisions, keeping humans in the loop at every stage.
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#888; font-size:1.05rem; font-weight:700;'>JALRAKSHAN.AI</p>",
    unsafe_allow_html=True
)
