"""
F1 Podium Predictor — Streamlit Dashboard
===========================================
Interactive pre-race dashboard showing predicted podium finishers,
driver probabilities, feature importance, and driver detail cards.

Run:
    streamlit run f1_podium_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import requests
import time
import plotly.graph_objects as go
import plotly.express as px

from f1_podium_model import FEATURE_COLS, predict_race, MODEL_PATH, ENCODER_PATH
from f1_podium_features import jolpica_get, get_qualifying_results

# ── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="F1 Podium Predictor",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=DM+Mono&display=swap');
    h1,h2,h3 { font-family:'Orbitron',monospace!important; letter-spacing:1.5px; }
    .podium-p1 { border-top: 3px solid #BA7517 !important; }
    .podium-p2 { border-top: 3px solid #888780 !important; }
    .podium-p3 { border-top: 3px solid #993C1D !important; }
    div[data-testid="stMetricValue"] { font-family:'Orbitron',monospace; }
</style>
""", unsafe_allow_html=True)

# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# 🏆 Podium Predictor")
    st.markdown("---")
    season = st.selectbox("Season", [2024, 2023], index=0)
    round_num = st.number_input("Round number", min_value=1, max_value=24, value=1)
    threshold = st.slider("Podium probability threshold", 0.1, 0.9, 0.35, 0.05,
                           help="Lowers = more drivers flagged as podium candidates")
    st.markdown("---")
    if os.path.exists(MODEL_PATH):
        st.success("Model loaded ✓")
    else:
        st.error("No model found — run f1_podium_model.py first")
    st.caption("Sources: Jolpica · FastF1 · XGBoost")

# ── LOAD MODEL ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model_cached():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)

model = load_model_cached()
if model is None:
    st.error("No trained model found. Run `python f1_podium_model.py` first.")
    st.stop()

# ── FETCH RACE DATA ────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def fetch_race_info(season: int, round_num: int):
    """Fetch race name, circuit, grid, and qualifying data from Jolpica."""
    data = jolpica_get(f"{season}/{round_num}/qualifying")
    races = data.get("RaceTable", {}).get("Races", [])
    if not races:
        return None, pd.DataFrame()

    race = races[0]
    race_name = race["raceName"]
    rows = []
    for res in race.get("QualifyingResults", []):
        q_time = res.get("Q3", res.get("Q2", res.get("Q1", None)))
        rows.append({
            "driver_id":   res["Driver"]["driverId"],
            "driver_code": res["Driver"].get("code", "???"),
            "driver_name": f"{res['Driver']['givenName']} {res['Driver']['familyName']}",
            "constructor": res["Constructor"]["constructorId"],
            "team_name":   res["Constructor"]["name"],
            "quali_pos":   int(res["position"]),
            "q_time":      q_time,
        })

    return race_name, pd.DataFrame(rows)


@st.cache_data(ttl=3600)
def fetch_historical_features(season: int, driver_ids: list) -> pd.DataFrame:
    """
    Pull the last 5 races of results to compute rolling form features
    for each driver entering the current race weekend.
    """
    all_rows = []
    for prev_round in range(max(1, 1), round_num):
        data = jolpica_get(f"{season}/{prev_round}/results")
        races_data = data.get("RaceTable", {}).get("Races", [])
        for race in races_data:
            for res in race.get("Results", []):
                drv = res["Driver"]["driverId"]
                if drv not in driver_ids:
                    continue
                all_rows.append({
                    "driver_id":   drv,
                    "round":       int(race["round"]),
                    "position":    int(res["position"]) if res.get("position","").isdigit() else 99,
                    "points":      float(res.get("points", 0)),
                    "status":      res.get("status", ""),
                    "on_podium":   1 if res.get("position","").isdigit() and int(res["position"]) <= 3 else 0,
                    "grid":        int(res.get("grid", 10)),
                })
    return pd.DataFrame(all_rows)


def build_prediction_features(quali_df: pd.DataFrame, history: pd.DataFrame,
                               season: int) -> pd.DataFrame:
    """
    Assemble the feature row for each driver based on their
    qualifying result and recent historical form.
    """
    # Parse qualifying times
    def parse_q_time(t):
        if not t or not isinstance(t, str):
            return None
        try:
            m, s = t.split(":")
            return float(m) * 60 + float(s)
        except Exception:
            return None

    quali_df = quali_df.copy()
    quali_df["q_sec"] = quali_df["q_time"].apply(parse_q_time)
    pole_time = quali_df["q_sec"].min()
    quali_df["quali_gap"] = (quali_df["q_sec"] - pole_time).fillna(3.0)

    # Rolling form per driver
    form_rows = {}
    for drv_id in quali_df["driver_id"]:
        drv_hist = history[history["driver_id"] == drv_id].sort_values("round").tail(5)
        if len(drv_hist) == 0:
            form_rows[drv_id] = {
                "rolling_podium_rate": 0.15,
                "rolling_win_rate":    0.05,
                "rolling_points_avg":  8.0,
                "rolling_dnf_rate":    0.1,
                "rolling_grid_avg":    8.0,
            }
        else:
            form_rows[drv_id] = {
                "rolling_podium_rate": drv_hist["on_podium"].mean(),
                "rolling_win_rate":    (drv_hist["position"] == 1).mean(),
                "rolling_points_avg":  drv_hist["points"].mean(),
                "rolling_dnf_rate":    drv_hist["status"].str.contains("Retired|Accident|Engine", na=False).mean(),
                "rolling_grid_avg":    drv_hist["grid"].mean(),
            }

    form_df = pd.DataFrame(form_rows).T.reset_index().rename(columns={"index": "driver_id"})
    feat_df = quali_df.merge(form_df, on="driver_id", how="left")

    # Car rank proxy from constructor (static lookup for common teams)
    CAR_RANK = {
        "red_bull": 1, "ferrari": 2, "mclaren": 3, "mercedes": 4,
        "aston_martin": 5, "alpine": 6, "williams": 7, "rb": 8,
        "kick_sauber": 9, "haas": 10,
    }
    feat_df["car_rank"]   = feat_df["constructor"].map(CAR_RANK).fillna(6)
    feat_df["car_points"] = (11 - feat_df["car_rank"]) * 50   # rough proxy

    # Style features — use neutral defaults (FastF1 telemetry requires loading session)
    feat_df["tyre_deg_slope"]   = 0.07
    feat_df["avg_throttle_pct"] = 75.0
    feat_df["overtake_count"]   = 2.0

    return feat_df

# ── MAIN UI ───────────────────────────────────────────────────────────────────
st.markdown("# 🏆 F1 Podium Predictor")

with st.spinner("Fetching race and qualifying data..."):
    race_name, quali_df = fetch_race_info(season, round_num)

if race_name is None or quali_df.empty:
    st.warning("No qualifying data found for this round. Try a different season/round.")
    st.stop()

st.markdown(f"### {race_name} — {season} Season, Round {round_num}")
st.markdown("---")

driver_ids = quali_df["driver_id"].tolist()
with st.spinner("Loading historical form..."):
    history = fetch_historical_features(season, driver_ids)

feat_df  = build_prediction_features(quali_df, history, season)
results  = predict_race(model, feat_df)

# ── PREDICTED PODIUM ──────────────────────────────────────────────────────────
st.subheader("Predicted podium")
top3 = results.head(3)
cols = st.columns(3)
MEDAL_COLORS = ["#BA7517","#888780","#993C1D"]
MEDAL_LABELS = ["P1 — Winner","P2 — Second","P3 — Third"]

for i, (col, (_, row)) in enumerate(zip(cols, top3.iterrows())):
    with col:
        prob = row["podium_prob"]
        st.markdown(f"""
        <div style="border:0.5px solid var(--secondary-background-color);
                    border-top:3px solid {MEDAL_COLORS[i]};
                    border-radius:10px;padding:16px;text-align:center;">
          <div style="font-size:11px;font-weight:600;color:{MEDAL_COLORS[i]};margin-bottom:4px">{MEDAL_LABELS[i]}</div>
          <div style="font-size:20px;font-weight:700;margin-bottom:2px">{row['driver_code']}</div>
          <div style="font-size:13px;color:gray;margin-bottom:12px">{row['team_name']}</div>
          <div style="font-size:28px;font-weight:700;color:{MEDAL_COLORS[i]}">{prob*100:.0f}%</div>
          <div style="font-size:11px;color:gray">confidence</div>
        </div>""", unsafe_allow_html=True)

st.markdown("---")

# ── FULL FIELD TABLE + PROBABILITY CHART ─────────────────────────────────────
col_left, col_right = st.columns([1.2, 1])

with col_left:
    st.subheader("All drivers")
    display_df = results[["driver_code","driver_name","team_name","quali_pos",
                           "quali_gap","rolling_podium_rate","car_rank","podium_prob"]].copy()
    display_df.columns = ["Code","Driver","Team","Grid","Quali gap (s)",
                           "Podium rate","Car rank","P(podium)"]
    display_df["Quali gap (s)"]  = display_df["Quali gap (s)"].round(3)
    display_df["Podium rate"]    = display_df["Podium rate"].apply(lambda x: f"{x*100:.0f}%")
    display_df["P(podium)"]      = display_df["P(podium)"].apply(lambda x: f"{x*100:.0f}%")
    st.dataframe(display_df, use_container_width=True, hide_index=True)

with col_right:
    st.subheader("Probability chart")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=results["driver_code"],
        x=results["podium_prob"] * 100,
        orientation="h",
        marker_color=["#7F77DD" if p >= threshold else "#D3D1C7"
                      for p in results["podium_prob"]],
        text=[f"{p*100:.0f}%" for p in results["podium_prob"]],
        textposition="outside",
    ))
    fig.update_layout(
        xaxis_title="Podium probability (%)",
        yaxis=dict(autorange="reversed"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12),
        margin=dict(l=10,r=40,t=10,b=30),
        height=420,
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

# ── FEATURE IMPORTANCE ────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("What the model is using to decide")

if hasattr(model, "feature_importances_"):
    imp = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=True)
    fig2 = go.Figure(go.Bar(
        y=imp.index,
        x=imp.values * 100,
        orientation="h",
        marker_color="#7F77DD",
    ))
    fig2.update_layout(
        xaxis_title="Importance (%)",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12),
        margin=dict(l=10,r=20,t=10,b=30),
        height=350,
    )
    st.plotly_chart(fig2, use_container_width=True)

# ── DRIVER DETAIL ─────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Driver deep-dive")
selected = st.selectbox("Select driver", results["driver_name"].tolist())
drv_row = results[results["driver_name"] == selected].iloc[0]

d1, d2, d3, d4, d5, d6 = st.columns(6)
d1.metric("Podium prob",    f"{drv_row['podium_prob']*100:.0f}%")
d2.metric("Grid position",  f"P{int(drv_row['quali_pos'])}")
d3.metric("Quali gap",      f"+{drv_row['quali_gap']:.3f}s")
d4.metric("Podium rate",    f"{drv_row['rolling_podium_rate']*100:.0f}%")
d5.metric("Car rank",       f"#{int(drv_row['car_rank'])}")
d6.metric("Avg points",     f"{drv_row['rolling_points_avg']:.1f}")

# Radar chart of driver profile
cats = ["Quali pace","Podium form","Car quality","Win rate","Reliability"]
vals = [
    max(0, 1 - drv_row["quali_gap"] / 3),
    drv_row["rolling_podium_rate"],
    1 - (drv_row["car_rank"] - 1) / 9,
    drv_row["rolling_win_rate"],
    1 - drv_row["rolling_dnf_rate"],
]
fig3 = go.Figure(go.Scatterpolar(
    r=vals + [vals[0]],
    theta=cats + [cats[0]],
    fill="toself",
    fillcolor="rgba(127,119,221,0.2)",
    line_color="#7F77DD",
))
fig3.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0,1])),
    paper_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=40,r=40,t=30,b=30),
    height=300,
    showlegend=False,
)
st.plotly_chart(fig3, use_container_width=True)
