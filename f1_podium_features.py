"""
F1 Podium Predictor — Feature Engineering Pipeline
====================================================
Pulls data from three sources and merges them into one flat
feature matrix: one row per driver per race.

Sources:
  - Jolpica API  → historical results, grid positions, points
  - FastF1       → telemetry, lap times, driving style proxies
  - Car specs    → constructor ranking, quali pace, reliability

Run:
    python f1_podium_features.py
    → writes  f1_podium_features.csv
"""

import requests
import pandas as pd
import numpy as np
import fastf1
import time
import warnings
warnings.filterwarnings("ignore")

import os
os.makedirs("f1_cache", exist_ok=True)
fastf1.Cache.enable_cache("f1_cache")

JOLPICA_BASE = "https://api.jolpi.ca/ergast/f1"
SEASONS      = list(range(2018, 2025))   # 7 seasons of training data
ROLLING_N    = 5                          # laps of rolling form window


# ── JOLPICA HELPERS ────────────────────────────────────────────────────────────

def jolpica_get(path: str, retries=3) -> dict:
    """Fetch one page of Jolpica JSON. Returns the MRData dict."""
    url = f"{JOLPICA_BASE}/{path}.json?limit=100"
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            return r.json().get("MRData", {})
        except Exception as e:
            if attempt == retries - 1:
                print(f"  [WARN] Jolpica {path} failed: {e}")
                return {}
            time.sleep(2 ** attempt)
    return {}


def get_race_results(season: int) -> pd.DataFrame:
    """
    Fetch all race results for a season from Jolpica.
    Returns one row per (race, driver) with finishing position,
    grid position, status, and points.
    """
    data = jolpica_get(f"{season}/results")
    races = data.get("RaceTable", {}).get("Races", [])
    rows = []
    for race in races:
        round_num = int(race["round"])
        race_name = race["raceName"]
        circuit   = race["Circuit"]["circuitId"]
        for res in race.get("Results", []):
            rows.append({
                "season":      season,
                "round":       round_num,
                "race_name":   race_name,
                "circuit_id":  circuit,
                "driver_id":   res["Driver"]["driverId"],
                "driver_code": res["Driver"].get("code", "UNK"),
                "constructor": res["Constructor"]["constructorId"],
                "grid":        int(res.get("grid", 0)),
                "position":    int(res["position"]) if res.get("position", "\\R").isdigit() else 99,
                "points":      float(res.get("points", 0)),
                "status":      res.get("status", "Unknown"),
                "on_podium":   int(res["position"]) <= 3 if res.get("position", "").isdigit() else 0,
            })
    return pd.DataFrame(rows)


def get_qualifying_results(season: int) -> pd.DataFrame:
    """
    Fetch qualifying positions and Q3 times.
    Used to compute gap-to-pole — the single strongest predictor.
    """
    data = jolpica_get(f"{season}/qualifying")
    races = data.get("RaceTable", {}).get("Races", [])
    rows = []
    for race in races:
        round_num = int(race["round"])
        for res in race.get("QualifyingResults", []):
            q3 = res.get("Q3", res.get("Q2", res.get("Q1", None)))
            rows.append({
                "season":    season,
                "round":     round_num,
                "driver_id": res["Driver"]["driverId"],
                "quali_pos": int(res["position"]),
                "q_time":    q3,
            })
    return pd.DataFrame(rows)


def get_constructor_standings(season: int) -> pd.DataFrame:
    """
    Fetch end-of-season constructor standings.
    Position 1 = best car, position 10 = worst car.
    Used as a proxy for overall car performance level.
    """
    data = jolpica_get(f"{season}/constructorStandings")
    standings = data.get("StandingsTable", {}).get("StandingsLists", [])
    rows = []
    for s in standings:
        for entry in s.get("ConstructorStandings", []):
            rows.append({
                "season":      season,
                "constructor": entry["Constructor"]["constructorId"],
                "car_rank":    int(entry["position"]),
                "car_points":  float(entry.get("points", 0)),
            })
    return pd.DataFrame(rows)


# ── HISTORICAL FORM FEATURES ───────────────────────────────────────────────────

def add_rolling_form(results: pd.DataFrame) -> pd.DataFrame:
    """
    For each driver/race row, compute rolling statistics over the
    previous ROLLING_N races. These capture recent form rather than
    career averages — a driver coming off 5 podiums in a row is very
    different from one who averaged 5 podiums over 5 years.

    Features added:
      rolling_podium_rate  — fraction of last N races with podium finish
      rolling_win_rate     — fraction of last N races with race win
      rolling_points_avg   — average points scored per race (last N)
      rolling_dnf_rate     — fraction of last N races with a DNF
      rolling_grid_avg     — average qualifying position (last N races)
    """
    results = results.sort_values(["driver_id", "season", "round"]).copy()

    def rolling_stat(grp, col, func, n=ROLLING_N):
        return grp[col].shift(1).rolling(n, min_periods=1).agg(func)

    g = results.groupby("driver_id")

    results["rolling_podium_rate"] = g.apply(
        lambda x: rolling_stat(x, "on_podium", "mean")
    ).reset_index(level=0, drop=True)

    results["rolling_win_rate"] = g.apply(
        lambda x: rolling_stat(x, "position", lambda s: (s == 1).mean())
    ).reset_index(level=0, drop=True)

    results["rolling_points_avg"] = g.apply(
        lambda x: rolling_stat(x, "points", "mean")
    ).reset_index(level=0, drop=True)

    # ── FIX: convert status string to numeric 0/1 flag before rolling ──
    results["is_dnf"] = results["status"].str.contains(
        "Retired|Accident|Engine|Gearbox|Collision|Mechanical|Hydraulics|Suspension",
        na=False
    ).astype(float)

    results["rolling_dnf_rate"] = g.apply(
        lambda x: rolling_stat(x, "is_dnf", "mean")
    ).reset_index(level=0, drop=True)
    # ───────────────────────────────────────────────────────────────────

    results["rolling_grid_avg"] = g.apply(
        lambda x: rolling_stat(x, "grid", "mean")
    ).reset_index(level=0, drop=True)

    return results


# ── FASTF1 DRIVING STYLE FEATURES ─────────────────────────────────────────────

def extract_driving_style(season: int, round_num: int) -> pd.DataFrame:
    """
    Load FastF1 telemetry for a race and derive per-driver style proxies.

    Driving style features:
      tyre_deg_slope   — how fast lap times increase per lap of tyre age
      avg_throttle_pct — mean throttle application across the race
      overtake_count   — positions gained from grid to finish
    """
    try:
        session = fastf1.get_session(season, round_num, "R")
        session.load(telemetry=True, weather=False)
        laps = session.laps

        rows = []
        for driver in laps["DriverNumber"].unique():
            drv_laps = laps[laps["DriverNumber"] == driver].copy()
            drv_laps = drv_laps[drv_laps["LapTime"].notna()]
            drv_laps["lap_sec"] = drv_laps["LapTime"].dt.total_seconds()
            drv_laps = drv_laps[drv_laps["lap_sec"] > 60]

            # Tyre degradation slope
            deg_slope = 0.0
            if len(drv_laps) > 5 and "TyreLife" in drv_laps.columns:
                valid = drv_laps[["TyreLife", "lap_sec"]].dropna()
                if len(valid) > 3:
                    deg_slope = float(np.polyfit(valid["TyreLife"], valid["lap_sec"], 1)[0])

            # Throttle percentage from telemetry
            avg_throttle = 0.0
            try:
                sample_laps = drv_laps.head(10)
                throttle_vals = []
                for _, lap_row in sample_laps.iterrows():
                    tel = lap_row.get_telemetry()
                    if tel is not None and "Throttle" in tel.columns:
                        throttle_vals.append(tel["Throttle"].mean())
                if throttle_vals:
                    avg_throttle = float(np.mean(throttle_vals))
            except Exception:
                pass

            # Positions gained
            info = session.get_driver(driver)
            grid_pos   = drv_laps.iloc[0].get("GridPosition", 10) if len(drv_laps) > 0 else 10
            finish_pos = drv_laps.iloc[-1].get("Position", grid_pos) if len(drv_laps) > 0 else grid_pos
            overtakes  = max(0, int(grid_pos) - int(finish_pos)) if grid_pos and finish_pos else 0

            rows.append({
                "driver_number":    driver,
                "driver_id":        info.get("DriverId", str(driver)) if info else str(driver),
                "tyre_deg_slope":   round(deg_slope, 4),
                "avg_throttle_pct": round(avg_throttle, 2),
                "overtake_count":   overtakes,
            })

        return pd.DataFrame(rows)

    except Exception as e:
        print(f"  [WARN] FastF1 {season} R{round_num}: {e}")
        return pd.DataFrame()


# ── QUALI GAP FEATURE ──────────────────────────────────────────────────────────

def add_quali_gap(df: pd.DataFrame, quali: pd.DataFrame) -> pd.DataFrame:
    """
    Compute each driver's qualifying gap to pole position in seconds.
    Drivers who failed to set a Q3 time get a penalty gap of 3.0s.
    """
    def parse_time(t):
        if pd.isna(t) or not isinstance(t, str):
            return None
        try:
            parts = t.split(":")
            if len(parts) == 2:
                return float(parts[0]) * 60 + float(parts[1])
            return float(t)
        except Exception:
            return None

    quali = quali.copy()
    quali["q_sec"] = quali["q_time"].apply(parse_time)

    pole_times = (
        quali[quali["quali_pos"] == 1]
        .set_index(["season", "round"])["q_sec"]
        .rename("pole_time")
    )
    quali = quali.join(pole_times, on=["season", "round"])
    quali["quali_gap"] = quali["q_sec"] - quali["pole_time"]
    quali["quali_gap"] = quali["quali_gap"].fillna(3.0)

    return df.merge(
        quali[["season", "round", "driver_id", "quali_pos", "quali_gap"]],
        on=["season", "round", "driver_id"],
        how="left"
    )


# ── MAIN BUILD FUNCTION ────────────────────────────────────────────────────────

def build_feature_dataset(seasons=SEASONS, use_telemetry=True) -> pd.DataFrame:
    """
    Orchestrate the full feature build across all seasons.
    Returns a DataFrame ready for model training.
    """
    all_results = []
    all_quali   = []
    all_constr  = []

    print("Fetching Jolpica data...")
    for season in seasons:
        print(f"  Season {season}...")
        all_results.append(get_race_results(season))
        all_quali.append(get_qualifying_results(season))
        all_constr.append(get_constructor_standings(season))
        time.sleep(0.3)

    results = pd.concat(all_results, ignore_index=True)
    quali   = pd.concat(all_quali,   ignore_index=True)
    constr  = pd.concat(all_constr,  ignore_index=True)

    print("Adding rolling form features...")
    results = add_rolling_form(results)

    print("Merging constructor standings...")
    results = results.merge(
        constr[["season", "constructor", "car_rank", "car_points"]],
        on=["season", "constructor"],
        how="left"
    )

    print("Adding qualifying gap features...")
    results = add_quali_gap(results, quali)

    if use_telemetry:
        print("Extracting FastF1 driving style (slow — disable with use_telemetry=False)...")
        style_frames = []
        for season in seasons[-2:]:
            races = results[results["season"] == season]["round"].unique()
            for round_num in races[:5]:
                print(f"  FastF1 {season} R{round_num}...")
                style = extract_driving_style(season, int(round_num))
                if not style.empty:
                    style["season"] = season
                    style["round"]  = round_num
                    style_frames.append(style)
                time.sleep(0.5)

        if style_frames:
            style_df = pd.concat(style_frames, ignore_index=True)
            results = results.merge(
                style_df[["season", "round", "driver_id", "tyre_deg_slope",
                           "avg_throttle_pct", "overtake_count"]],
                on=["season", "round", "driver_id"],
                how="left"
            )

    # Fill missing style features with neutral defaults
    for col in ["tyre_deg_slope", "avg_throttle_pct", "overtake_count"]:
        if col not in results.columns:
            results[col] = 0.0
        results[col] = results[col].fillna(0.0)

    # Fill other missing values
    results["quali_gap"]  = results["quali_gap"].fillna(3.0)
    results["quali_pos"]  = results["quali_pos"].fillna(10.0)
    results["car_rank"]   = results["car_rank"].fillna(5.0)
    results["car_points"] = results["car_points"].fillna(100.0)

    print(f"\nDone. Dataset shape: {results.shape}")
    print(f"Podium rate in dataset: {results['on_podium'].mean()*100:.1f}%")

    return results


if __name__ == "__main__":
    df = build_feature_dataset(seasons=SEASONS, use_telemetry=False)
    df.to_csv("f1_podium_features.csv", index=False)
    print("Saved to f1_podium_features.csv")
