"""
F1 Live Prediction Dashboard
==============================
Streamlit app that shows real-time predictions from the pipeline.

Requirements:
    pip install streamlit plotly pandas requests joblib

Run:
    streamlit run f1_dashboard.py
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import joblib
import time
import os
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

# Import helpers from the pipeline
import sys
sys.path.insert(0, os.path.dirname(__file__))
from f1_realtime_pipeline import (
    get_latest_session_key, get_drivers, get_laps,
    get_stints, get_weather, get_positions,
    build_features, predict_next_lap, load_model,
    FEATURE_COLS, openf1_get
)

# ─── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="F1 Live Predictor",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── STYLING ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=DM+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Mono', monospace;
    }
    h1, h2, h3 {
        font-family: 'Orbitron', monospace !important;
        letter-spacing: 2px;
    }
    .metric-card {
        background: #0d0d0d;
        border: 1px solid #e10600;
        border-radius: 8px;
        padding: 16px 20px;
        text-align: center;
    }
    .metric-value {
        font-family: 'Orbitron', monospace;
        font-size: 28px;
        font-weight: 700;
        color: #e10600;
    }
    .metric-label {
        font-size: 11px;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-top: 4px;
    }
    .driver-row {
        font-family: 'DM Mono', monospace;
        font-size: 13px;
    }
    .stDataFrame { font-family: 'DM Mono', monospace; }
    div[data-testid="stMetricValue"] {
        font-family: 'Orbitron', monospace;
        color: #e10600;
    }
    .status-live {
        display: inline-block;
        background: #e10600;
        color: white;
        font-size: 11px;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 3px;
        letter-spacing: 2px;
        animation: pulse 1.5s infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50%       { opacity: 0.5; }
    }
</style>
""", unsafe_allow_html=True)

# ─── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# 🏎️ F1 Predictor")
    st.markdown("---")

    mode = st.radio("Mode", ["🔴 Live Race", "📼 Replay Past Race"])

    if "Live" in mode:
        session_key_input = st.number_input(
            "Session Key (0 = latest)", min_value=0, value=0
        )
        session_key = int(session_key_input) if session_key_input > 0 else None
        auto_refresh = st.toggle("Auto-refresh (15s)", value=True)
        refresh_rate = 15
    else:
        col1, col2 = st.columns(2)
        with col1:
            replay_year  = st.selectbox("Season", [2024, 2023], index=0)
        with col2:
            replay_round = st.number_input("Round", min_value=1, max_value=24, value=1)
        replay_lap = st.slider("Reveal up to lap", min_value=1, max_value=70, value=10)
        auto_refresh = False

    st.markdown("---")
    st.markdown("**Model status**")
    model_exists = os.path.exists("f1_lap_model.pkl")
    if model_exists:
        st.success("Model loaded ✓")
    else:
        st.error("No model found. Run `--train` first.")

    st.markdown("---")
    st.caption("Data: OpenF1 API · Model: GradientBoosting")

# ─── LOAD MODEL ────────────────────────────────────────────────────────────────
@st.cache_resource
def cached_load_model():
    return load_model()

model = cached_load_model()

# ─── DATA FETCHING ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=14)   # cache 14s for live mode
def fetch_session_data(key: int):
    laps    = get_laps(key)
    stints  = get_stints(key)
    weather = get_weather(key)
    drivers = get_drivers(key)
    pos     = get_positions(key)
    return laps, stints, weather, drivers, pos


def get_session_key_for_replay(year: int, round_num: int) -> int | None:
    sessions = openf1_get("sessions", {"year": year, "session_type": "Race"})
    if not sessions:
        return None
    sessions.sort(key=lambda x: x.get("date_start", ""))
    if round_num > len(sessions):
        return None
    return sessions[round_num - 1]["session_key"]

# ─── MAIN DASHBOARD ────────────────────────────────────────────────────────────
st.markdown("# 🏎️ F1 LIVE PREDICTION DASHBOARD")

if model is None:
    st.error("⚠️ No trained model found. Run `python f1_realtime_pipeline.py --train` first.")
    st.stop()

# Resolve session key
if "Live" in mode:
    if session_key is None:
        with st.spinner("Finding latest session..."):
            session_key = get_latest_session_key()
    if session_key is None:
        st.warning("No live session found. Try Replay mode or provide a session key.")
        st.stop()
    st.markdown(f'<span class="status-live">LIVE</span>  Session key: `{session_key}`', unsafe_allow_html=True)
else:
    with st.spinner("Looking up session..."):
        session_key = get_session_key_for_replay(replay_year, replay_round)
    if session_key is None:
        st.error("Session not found.")
        st.stop()
    st.markdown(f"📼 **Replay:** {replay_year} Round {replay_round} — session `{session_key}`")

# Fetch data
with st.spinner("Fetching data from OpenF1..."):
    laps, stints, weather, drivers_df, positions_df = fetch_session_data(session_key)

if laps.empty:
    st.warning("No lap data yet. The session may not have started.")
    st.stop()

# Apply lap filter in replay mode
features = build_features(laps, stints, weather)
if features.empty:
    st.warning("Could not build features from session data.")
    st.stop()

if "Replay" in mode:
    features = features[features["lap_number"] <= replay_lap]

# ─── TOP METRICS ───────────────────────────────────────────────────────────────
current_lap  = int(features["lap_number"].max())
num_drivers  = features["driver_number"].nunique()
avg_lap      = features["lap_duration"].mean()
fastest_lap  = features["lap_duration"].min()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Current lap",    current_lap)
col2.metric("Active drivers", num_drivers)
col3.metric("Avg lap time",   f"{avg_lap:.3f}s")
col4.metric("Fastest lap",    f"{fastest_lap:.3f}s")

st.markdown("---")

# ─── DRIVER TABLE ──────────────────────────────────────────────────────────────
driver_info = {}
if not drivers_df.empty:
    for _, row in drivers_df.iterrows():
        driver_info[row["driver_number"]] = {
            "acronym": row.get("name_acronym", str(row["driver_number"])),
            "team":    row.get("team_name", "Unknown")
        }

# Latest position data
latest_positions = {}
if not positions_df.empty:
    pos_sorted = positions_df.sort_values("date")
    for drv, grp in pos_sorted.groupby("driver_number"):
        latest_positions[drv] = int(grp.iloc[-1]["position"])

rows = []
for drv_num, grp in features.groupby("driver_number"):
    grp     = grp.sort_values("lap_number")
    last    = grp.iloc[-1]
    pred    = predict_next_lap(model, last)
    delta   = pred - float(last["lap_duration"])
    info    = driver_info.get(drv_num, {"acronym": str(drv_num), "team": "Unknown"})
    pos     = latest_positions.get(drv_num, "—")
    compound = last.get("compound", "UNK")

    rows.append({
        "Pos":          pos,
        "Driver":       info["acronym"],
        "Team":         info["team"],
        "Lap":          int(last["lap_number"]),
        "Last lap (s)": round(float(last["lap_duration"]), 3),
        "Pred next (s)":round(pred, 3),
        "Δ (s)":        round(delta, 3),
        "Tyre":         compound,
        "Tyre age":     int(last.get("tyre_age", 0)),
    })

table_df = pd.DataFrame(rows)
if "Pos" in table_df.columns:
    try:
        table_df = table_df.sort_values("Pos")
    except:
        pass

col_left, col_right = st.columns([1.4, 1])

with col_left:
    st.subheader("Driver predictions")
    st.dataframe(
        table_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Δ (s)": st.column_config.NumberColumn(format="%.3f"),
            "Last lap (s)": st.column_config.NumberColumn(format="%.3f"),
            "Pred next (s)": st.column_config.NumberColumn(format="%.3f"),
        }
    )

with col_right:
    st.subheader("Lap time trend")
    if not table_df.empty:
        # Top 5 drivers by fewest laps behind (most data)
        top_drivers = features.groupby("driver_number")["lap_number"].max().nlargest(5).index.tolist()
        fig = go.Figure()
        colors = ["#e10600", "#0090ff", "#00d2be", "#ff8700", "#ffffff"]
        for i, drv in enumerate(top_drivers):
            drv_laps = features[features["driver_number"] == drv].sort_values("lap_number")
            acronym  = driver_info.get(drv, {}).get("acronym", str(drv))
            fig.add_trace(go.Scatter(
                x=drv_laps["lap_number"],
                y=drv_laps["lap_duration"],
                mode="lines+markers",
                name=acronym,
                line=dict(color=colors[i % len(colors)], width=2),
                marker=dict(size=4)
            ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#aaa", size=11),
            legend=dict(bgcolor="rgba(0,0,0,0.5)", bordercolor="#333"),
            margin=dict(l=10, r=10, t=10, b=30),
            xaxis=dict(title="Lap", gridcolor="#222"),
            yaxis=dict(title="Lap time (s)", gridcolor="#222"),
            height=320
        )
        st.plotly_chart(fig, use_container_width=True)

# ─── TYRE STRATEGY PLOT ────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Tyre strategy overview")

if not stints.empty:
    merged_stints = stints.copy()
    merged_stints["acronym"] = merged_stints["driver_number"].map(
        lambda d: driver_info.get(d, {}).get("acronym", str(d))
    )
    compound_colors = {
        "SOFT": "#e10600", "MEDIUM": "#ffd700",
        "HARD": "#ffffff", "INTERMEDIATE": "#39b54a",
        "WET":  "#0067ff", "UNKNOWN": "#888"
    }
    fig2 = go.Figure()
    for _, stint_row in merged_stints.iterrows():
        compound = stint_row.get("compound", "UNKNOWN")
        fig2.add_trace(go.Bar(
            x=[stint_row.get("lap_end", 0) - stint_row.get("lap_start", 0)],
            y=[stint_row["acronym"]],
            base=[stint_row.get("lap_start", 0)],
            orientation="h",
            marker_color=compound_colors.get(compound, "#888"),
            name=compound,
            text=compound[0],
            textposition="inside",
            showlegend=False,
            hovertemplate=f"{stint_row['acronym']} · {compound} · Laps {stint_row.get('lap_start',0)}–{stint_row.get('lap_end',0)}<extra></extra>"
        ))
    fig2.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#aaa", size=11),
        barmode="overlay",
        xaxis=dict(title="Lap number", gridcolor="#222"),
        yaxis=dict(title="Driver"),
        margin=dict(l=10, r=10, t=10, b=30),
        height=max(250, len(merged_stints["acronym"].unique()) * 28)
    )
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("Stint / tyre data not yet available for this session.")

# ─── WEATHER STRIP ─────────────────────────────────────────────────────────────
if not weather.empty:
    st.markdown("---")
    st.subheader("Session weather")
    wc1, wc2, wc3, wc4, wc5 = st.columns(5)
    wc1.metric("Air temp",   f"{weather['air_temperature'].median():.1f} °C")
    wc2.metric("Track temp", f"{weather['track_temperature'].median():.1f} °C")
    wc3.metric("Humidity",   f"{weather['humidity'].median():.0f} %")
    wc4.metric("Wind",       f"{weather['wind_speed'].median():.1f} m/s")
    rain_flag = "🌧️ Yes" if weather["rainfall"].max() > 0 else "☀️ No"
    wc5.metric("Rain",       rain_flag)

# ─── AUTO REFRESH ──────────────────────────────────────────────────────────────
if auto_refresh and "Live" in mode:
    time.sleep(1)   # let page render
    st.rerun()
