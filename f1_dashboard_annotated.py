"""
F1 Live Prediction Dashboard
==============================
Streamlit app that shows real-time predictions from the pipeline.

HOW IT WORKS — OVERVIEW
------------------------
This file is a Streamlit web app that wraps the pipeline from
f1_realtime_pipeline.py in an interactive browser UI.

Streamlit works by re-running this entire script from top to bottom
every time the user interacts with the app (changes a slider, clicks
a button, etc.). This makes it easy to build reactive UIs without
writing JavaScript — but it means you need to be careful about what
runs on every re-run vs what should only run once:
  - @st.cache_resource: run once per session (model loading)
  - @st.cache_data(ttl=14): run once per 14 seconds (API calls)

Two modes:
  - Live Race:       polls OpenF1 every 15s for the current session
  - Replay Past Race: shows a completed race up to a lap you choose
                      (useful for development when no race is live)

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

# Import helpers from the pipeline — the dashboard is purely a UI layer,
# all data fetching and ML logic lives in f1_realtime_pipeline.py
import sys
sys.path.insert(0, os.path.dirname(__file__))
from f1_realtime_pipeline import (
    get_latest_session_key, get_drivers, get_laps,
    get_stints, get_weather, get_positions,
    build_features, predict_next_lap, load_model,
    FEATURE_COLS, openf1_get
)

# ─── PAGE CONFIG ───────────────────────────────────────────────────────────────
# Must be the first Streamlit call in the script.
# layout="wide" uses the full browser width (better for a multi-column dashboard).
st.set_page_config(
    page_title="F1 Live Predictor",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── STYLING ───────────────────────────────────────────────────────────────────
# Custom CSS injected into the Streamlit page using st.markdown with
# unsafe_allow_html=True. This gives us F1-themed fonts and a dark aesthetic.
#
# Orbitron: a geometric display font used for headings (matches F1 broadcast style)
# DM Mono:  a clean monospace font for data tables (lap times, driver numbers)
#
# .status-live: the pulsing red "LIVE" badge shown in live mode.
# The CSS animation uses opacity keyframes to create a breathing effect.
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
        color: #e10600;  /* F1 official red */
    }
    .metric-label {
        font-size: 11px;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-top: 4px;
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
# The sidebar holds all controls. Streamlit renders these as interactive
# widgets — every time the user changes one, the whole script re-runs
# with the new values.
with st.sidebar:
    st.markdown("# 🏎️ F1 Predictor")
    st.markdown("---")

    mode = st.radio("Mode", ["🔴 Live Race", "📼 Replay Past Race"])

    if "Live" in mode:
        session_key_input = st.number_input(
            "Session Key (0 = latest)", min_value=0, value=0
        )
        # 0 means "fetch the latest session automatically"
        session_key = int(session_key_input) if session_key_input > 0 else None
        auto_refresh = st.toggle("Auto-refresh (15s)", value=True)
        refresh_rate = 15
    else:
        col1, col2 = st.columns(2)
        with col1:
            replay_year  = st.selectbox("Season", [2024, 2023], index=0)
        with col2:
            replay_round = st.number_input("Round", min_value=1, max_value=24, value=1)
        # Lap slider: only show data up to this lap number, simulating live reveal
        replay_lap = st.slider("Reveal up to lap", min_value=1, max_value=70, value=10)
        auto_refresh = False

    st.markdown("---")
    st.markdown("**Model status**")
    # Check whether the trained model file exists on disk
    model_exists = os.path.exists("f1_lap_model.pkl")
    if model_exists:
        st.success("Model loaded ✓")
    else:
        st.error("No model found. Run `--train` first.")

    st.markdown("---")
    st.caption("Data: OpenF1 API · Model: GradientBoosting")

# ─── LOAD MODEL ────────────────────────────────────────────────────────────────
# @st.cache_resource keeps the model in memory for the whole session.
# Without caching, load_model() would be called on every user interaction —
# reading from disk each time is slow and unnecessary.
@st.cache_resource
def cached_load_model():
    return load_model()

model = cached_load_model()

# ─── DATA FETCHING ─────────────────────────────────────────────────────────────
# @st.cache_data(ttl=14) caches the API responses for 14 seconds.
# This means in live mode, data refreshes every 14s even if the user
# interacts with the app more frequently — prevents hammering the API.
@st.cache_data(ttl=14)
def fetch_session_data(key: int):
    laps    = get_laps(key)
    stints  = get_stints(key)
    weather = get_weather(key)
    drivers = get_drivers(key)
    pos     = get_positions(key)
    return laps, stints, weather, drivers, pos


def get_session_key_for_replay(year: int, round_num: int) -> int | None:
    """
    Look up the session_key for a specific year + round number.
    Sessions are sorted by date and indexed by round (1-indexed).
    """
    sessions = openf1_get("sessions", {"year": year, "session_type": "Race"})
    if not sessions:
        return None
    sessions.sort(key=lambda x: x.get("date_start", ""))
    if round_num > len(sessions):
        return None
    return sessions[round_num - 1]["session_key"]

# ─── MAIN DASHBOARD ────────────────────────────────────────────────────────────
st.markdown("# 🏎️ F1 LIVE PREDICTION DASHBOARD")

# Guard: if no model is loaded, stop rendering here rather than
# showing confusing errors later in the script.
if model is None:
    st.error("⚠️ No trained model found. Run `python f1_realtime_pipeline.py --train` first.")
    st.stop()

# Resolve session key based on mode
if "Live" in mode:
    if session_key is None:
        with st.spinner("Finding latest session..."):
            session_key = get_latest_session_key()
    if session_key is None:
        st.warning("No live session found. Try Replay mode or provide a session key.")
        st.stop()
    # The pulsing LIVE badge is injected as raw HTML
    st.markdown(f'<span class="status-live">LIVE</span>  Session key: `{session_key}`', unsafe_allow_html=True)
else:
    with st.spinner("Looking up session..."):
        session_key = get_session_key_for_replay(replay_year, replay_round)
    if session_key is None:
        st.error("Session not found.")
        st.stop()
    st.markdown(f"📼 **Replay:** {replay_year} Round {replay_round} — session `{session_key}`")

# Fetch all data for this session
with st.spinner("Fetching data from OpenF1..."):
    laps, stints, weather, drivers_df, positions_df = fetch_session_data(session_key)

if laps.empty:
    st.warning("No lap data yet. The session may not have started.")
    st.stop()

# Build features from the three data sources
features = build_features(laps, stints, weather)
if features.empty:
    st.warning("Could not build features from session data.")
    st.stop()

# In replay mode, filter to only show laps up to the selected lap number
if "Replay" in mode:
    features = features[features["lap_number"] <= replay_lap]

# ─── TOP METRICS ───────────────────────────────────────────────────────────────
# Four summary stats shown at the top — quick at-a-glance race status.
# st.metric() shows a value with an optional delta (change from previous value).
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
# Build a lookup dict {driver_number: {acronym, team}} for display.
# Driver identity is not a model feature but we need it for the UI.
driver_info = {}
if not drivers_df.empty:
    for _, row in drivers_df.iterrows():
        driver_info[row["driver_number"]] = {
            "acronym": row.get("name_acronym", str(row["driver_number"])),
            "team":    row.get("team_name", "Unknown")
        }

# Get the latest known position for each driver
latest_positions = {}
if not positions_df.empty:
    pos_sorted = positions_df.sort_values("date")
    for drv, grp in pos_sorted.groupby("driver_number"):
        latest_positions[drv] = int(grp.iloc[-1]["position"])

# Build one row per driver: last lap time, predicted next lap, and the delta
rows = []
for drv_num, grp in features.groupby("driver_number"):
    grp     = grp.sort_values("lap_number")
    last    = grp.iloc[-1]                             # most recent lap
    pred    = predict_next_lap(model, last)            # model inference
    delta   = pred - float(last["lap_duration"])       # predicted change
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
        "Δ (s)":        round(delta, 3),   # positive = predicted to be slower
        "Tyre":         compound,
        "Tyre age":     int(last.get("tyre_age", 0)),
    })

table_df = pd.DataFrame(rows)
if "Pos" in table_df.columns:
    try:
        table_df = table_df.sort_values("Pos")   # sort by race position
    except:
        pass

col_left, col_right = st.columns([1.4, 1])

with col_left:
    st.subheader("Driver predictions")
    # column_config controls how Streamlit formats individual columns
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
        # Show lap time history for the 5 drivers with the most laps completed
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
        # Transparent backgrounds so the plot blends with the dark dashboard theme
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
# A horizontal bar chart (Gantt-style) showing each driver's stint history.
# Each bar represents one stint: its x-position is the lap range, its colour
# is the compound. This is the standard way F1 broadcasts show strategy.
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
        # base = lap start, x = lap span; this creates a horizontal bar
        fig2.add_trace(go.Bar(
            x=[stint_row.get("lap_end", 0) - stint_row.get("lap_start", 0)],
            y=[stint_row["acronym"]],
            base=[stint_row.get("lap_start", 0)],
            orientation="h",
            marker_color=compound_colors.get(compound, "#888"),
            name=compound,
            text=compound[0],       # single letter label (S/M/H) inside the bar
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
# Five weather metrics shown as st.metric() cards.
# Track temperature is the most strategically important: above ~45°C
# tyre degradation accelerates significantly, which can force earlier pit stops.
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
# st.rerun() causes Streamlit to re-run the entire script from the top,
# which re-fetches data and re-renders everything. Combined with the
# @st.cache_data(ttl=14) on fetch_session_data, this effectively polls
# OpenF1 every 15 seconds without needing a manual page refresh.
#
# Note: time.sleep(1) before rerun gives the page a moment to fully render
# before the next fetch begins — prevents a flickering effect.
if auto_refresh and "Live" in mode:
    time.sleep(1)
    st.rerun()
