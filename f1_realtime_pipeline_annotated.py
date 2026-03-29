"""
F1 Real-Time Prediction Pipeline
=================================
Uses OpenF1 API to poll live race data each lap,
then feeds features into a trained ML model to
predict lap times and race positions in real-time.

HOW IT WORKS — OVERVIEW
------------------------
This pipeline has three modes:

  1. TRAIN  — pull historical race data from OpenF1, build features,
              and train a GradientBoosting regression model to predict
              lap times. The model is saved to disk as f1_lap_model.pkl.

  2. LIVE   — during an actual race, poll OpenF1 every 15 seconds,
              build features from the latest laps, and print predictions
              for every driver's next lap time.

  3. REPLAY — simulate live predictions on a completed past race,
              revealing one lap at a time so you can see how the model
              would have performed in real-time.

The target variable is lap_duration (seconds) — this is a REGRESSION
problem, unlike the pit stop classifier in the notebook. Here we're
predicting a continuous number (how fast the next lap will be) rather
than a yes/no outcome.

Requirements:
    pip install requests pandas scikit-learn fastf1 schedule joblib

Usage:
    1. Train the model first:   python f1_realtime_pipeline.py --train
    2. Run during a live race:  python f1_realtime_pipeline.py --live
    3. Replay a past session:   python f1_realtime_pipeline.py --replay --year 2024 --round 5
"""

import requests
import pandas as pd
import numpy as np
import time
import argparse
import joblib
import os
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

# ─── CONFIG ────────────────────────────────────────────────────────────────────
#
# Central settings — edit these to change behaviour without touching the code.
#
# POLL_INTERVAL: how often (seconds) to hit the OpenF1 API during a live race.
#   15s is a reasonable balance — F1 lap times are ~80–100s, so 15s gives
#   several updates per lap without hammering the API.
#
# HISTORY_SEASONS: which seasons to train on. More seasons = more data,
#   but also more API calls at training time. Start with one season to
#   test your setup, then expand.
#
# TYRE_MAP: converts compound names to integers so the model can use them
#   as a numeric feature. UNKNOWN defaults to 2 (HARD) as a safe fallback.
#
OPENF1_BASE      = "https://api.openf1.org/v1"
POLL_INTERVAL    = 15        # seconds between live polls
MODEL_PATH       = "f1_lap_model.pkl"
ENCODER_PATH     = "f1_encoders.pkl"
HISTORY_SEASONS  = [2023, 2024]   # seasons to train on

TYRE_MAP = {"SOFT": 0, "MEDIUM": 1, "HARD": 2,
            "INTERMEDIATE": 3, "WET": 4, "UNKNOWN": 2}

# ─── OPENF1 HELPERS ────────────────────────────────────────────────────────────
#
# These functions are thin wrappers around the OpenF1 REST API.
# OpenF1 is free, requires no API key, and returns JSON arrays.
# Each endpoint maps to a specific type of race data.
#
def openf1_get(endpoint: str, params: dict = {}, retries: int = 3) -> list:
    """
    Hit the OpenF1 API and return JSON as a list of dicts.

    Uses exponential backoff on failure (waits 1s, 2s, 4s between retries).
    Returns an empty list if all retries fail — callers must handle this.
    """
    url = f"{OPENF1_BASE}/{endpoint}"
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            if attempt == retries - 1:
                print(f"  [WARN] OpenF1 {endpoint} failed: {e}")
                return []
            time.sleep(2 ** attempt)   # 1s, 2s, 4s
    return []


def get_latest_session_key() -> int | None:
    """
    Return the session_key for the most recent F1 race session.

    session_key is OpenF1's unique ID for a session — we use it
    in every subsequent API call to scope data to that session.
    """
    data = openf1_get("sessions", {"session_type": "Race"})
    if not data:
        return None
    # Sort by date descending, return most recent
    data.sort(key=lambda x: x.get("date_start", ""), reverse=True)
    key = data[0].get("session_key")
    name = data[0].get("meeting_name", "Unknown")
    print(f"  Latest session: {name} (key={key})")
    return key


def get_drivers(session_key: int) -> pd.DataFrame:
    """
    Fetch driver info (acronym + team) for a session.

    Used for display only — the model doesn't use driver identity
    as a feature (we want it to generalise to new drivers/teams).
    """
    data = openf1_get("drivers", {"session_key": session_key})
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)[["driver_number", "name_acronym", "team_name"]]
    return df.drop_duplicates("driver_number")


def get_laps(session_key: int, driver_number: int | None = None) -> pd.DataFrame:
    """
    Fetch lap-by-lap timing data.

    Key columns:
    - lap_duration:      total lap time in seconds (this is our target variable)
    - duration_sector_X: split times for each of the three sectors
    - i1_speed/i2_speed/st_speed: speed trap readings at three points on circuit
    - is_pit_out_lap:    True if the driver was leaving the pits on this lap
                         (pit-out laps are slower and should be treated separately)
    """
    params = {"session_key": session_key}
    if driver_number:
        params["driver_number"] = driver_number
    data = openf1_get("laps", params)
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    keep = ["driver_number", "lap_number", "lap_duration",
            "duration_sector_1", "duration_sector_2", "duration_sector_3",
            "i1_speed", "i2_speed", "st_speed",
            "segments_sector_1", "segments_sector_2", "segments_sector_3",
            "is_pit_out_lap"]
    return df[[c for c in keep if c in df.columns]]


def get_stints(session_key: int) -> pd.DataFrame:
    """
    Fetch tyre stint data — which compound each driver is on and when
    they started their current stint.

    A 'stint' is one continuous run on a set of tyres (from pit-out
    to the next pit stop). This lets us calculate tyre_age (how many
    laps the current set has been driven), which is a key predictive feature.
    """
    data = openf1_get("stints", {"session_key": session_key})
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    keep = ["driver_number", "lap_start", "lap_end", "compound", "tyre_age_at_start"]
    return df[[c for c in keep if c in df.columns]]


def get_weather(session_key: int) -> pd.DataFrame:
    """
    Fetch weather readings — temperature, rain, wind.

    Track temperature is particularly important: hotter tracks degrade
    tyres faster, which slows lap times. Rain is a binary flag that
    typically causes large lap time increases.

    We sample weather at session level (using medians) rather than
    lap level, since weather changes slowly relative to lap frequency.
    """
    data = openf1_get("weather", {"session_key": session_key})
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    keep = ["date", "air_temperature", "track_temperature",
            "rainfall", "wind_speed", "humidity"]
    return df[[c for c in keep if c in df.columns]]


def get_positions(session_key: int) -> pd.DataFrame:
    """
    Fetch real-time position data (race order, updated throughout).

    Used by the dashboard for display — not currently a model feature,
    though gap-to-leader (derived from positions) would be a valuable
    addition to improve strategy predictions.
    """
    data = openf1_get("position", {"session_key": session_key})
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    keep = ["driver_number", "date", "position"]
    return df[[c for c in keep if c in df.columns]]

# ─── FEATURE ENGINEERING ───────────────────────────────────────────────────────
#
# Feature engineering is where raw API data becomes model inputs.
# We merge three separate data sources (laps, stints, weather) into
# one flat DataFrame where each row is a single (driver, lap) pair.
#
# Feature categories:
#   - Tyre features:    what compound, how many laps on it
#   - Weather features: temperature, rain, wind
#   - Race state:       lap number, pit-out flag, estimated fuel load
#   - Speed traps:      average speed at three points on the circuit
#
def build_features(laps: pd.DataFrame,
                   stints: pd.DataFrame,
                   weather: pd.DataFrame) -> pd.DataFrame:
    """
    Merge laps + stints + weather into a flat feature matrix.
    One row per (driver, lap).
    """
    if laps.empty:
        return pd.DataFrame()

    df = laps.copy()

    # --- Tyre features ---
    # For each lap, look up which stint the driver was in to get
    # compound and tyre_age. This requires a range join: find the stint
    # where lap_start <= current_lap <= lap_end.
    if not stints.empty:
        def get_tyre_info(row):
            mask = (
                (stints["driver_number"] == row["driver_number"]) &
                (stints["lap_start"] <= row["lap_number"]) &
                (stints["lap_end"]   >= row["lap_number"])
            )
            match = stints[mask]
            if match.empty:
                return pd.Series({"compound": "UNKNOWN", "tyre_age": 0})
            stint = match.iloc[0]
            # tyre_age = laps since this stint started + any pre-existing age
            # (tyres can be used sets carried over from qualifying)
            age = row["lap_number"] - stint["lap_start"] + stint.get("tyre_age_at_start", 0)
            return pd.Series({"compound": stint["compound"], "tyre_age": age})

        tyre_info = df.apply(get_tyre_info, axis=1)
        df = pd.concat([df, tyre_info], axis=1)
    else:
        df["compound"] = "UNKNOWN"
        df["tyre_age"]  = 0

    # Convert compound string to integer so the model can use it
    df["tyre_code"] = df["compound"].map(TYRE_MAP).fillna(2).astype(int)

    # --- Weather features ---
    # Use session medians rather than per-lap values — this smooths out
    # sensor noise and avoids timestamp alignment issues between endpoints.
    if not weather.empty:
        df["air_temp"]   = weather["air_temperature"].median()
        df["track_temp"] = weather["track_temperature"].median()
        df["rain"]       = int(weather["rainfall"].max() > 0)
        df["wind"]       = weather["wind_speed"].median()
    else:
        # Sensible defaults for a dry circuit (Bahrain-like conditions)
        df["air_temp"]   = 25.0
        df["track_temp"] = 35.0
        df["rain"]       = 0
        df["wind"]       = 5.0

    # --- Derived features ---
    # is_pit_out_lap: pit-out laps are always slower (cold tyres, unsafe speed
    # through pit lane). Flagging them lets the model learn to ignore this
    # as a degradation signal.
    df["is_pit_out_lap"] = df.get("is_pit_out_lap", pd.Series(False, index=df.index)).fillna(False).astype(int)

    # fuel_load_est: F1 cars start with ~110kg of fuel and burn ~1.8kg/lap.
    # Heavier cars are slower — this gives the model a proxy for fuel weight
    # without needing telemetry data.
    df["fuel_load_est"]  = (70 - df["lap_number"] * 1.8).clip(0, 70)

    # --- Speed trap average ---
    # Speed at three points on circuit (intermediate 1, intermediate 2,
    # speed trap at the finish line). Average them as a proxy for
    # how much downforce/setup the car has that day.
    speed_cols = [c for c in ["i1_speed", "i2_speed", "st_speed"] if c in df.columns]
    if speed_cols:
        df["avg_speed"] = df[speed_cols].mean(axis=1)
    else:
        df["avg_speed"] = 0

    # --- Sanity filters ---
    # Drop laps with no recorded lap time (safety car, red flag, etc.)
    # and anything under 60s (physically impossible — must be a data error).
    df = df.dropna(subset=["lap_duration"])
    df = df[df["lap_duration"] > 60]

    return df


# FEATURE_COLS defines the exact list of columns fed to the model.
# The order matters — it must be consistent between training and inference.
# Adding a new feature here requires re-training the model.
FEATURE_COLS = [
    "lap_number",       # Race progress proxy (fuel load decreases, tyres age)
    "tyre_code",        # Compound (Soft/Medium/Hard/etc.)
    "tyre_age",         # Laps on current tyre set (key degradation signal)
    "air_temp",         # Ambient temperature
    "track_temp",       # Track surface temperature (affects tyre grip directly)
    "rain",             # Binary rain flag (large impact on lap times)
    "wind",             # Wind speed (affects aerodynamic efficiency)
    "fuel_load_est",    # Estimated remaining fuel (heavier = slower)
    "avg_speed",        # Speed trap average (downforce/setup proxy)
    "is_pit_out_lap"    # Flag for slow pit exit laps (avoid distorting predictions)
]


def get_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series | None]:
    """
    Extract (X, y) from the feature DataFrame.

    X = the input features (FEATURE_COLS)
    y = the target variable (lap_duration in seconds)

    For inference (no ground truth), y will be None.
    Missing feature columns are filled with 0 — a safe fallback
    that won't crash the model, though adding the real feature is better.
    """
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    for c in missing:
        df[c] = 0
    X = df[FEATURE_COLS].fillna(0)
    y = df["lap_duration"] if "lap_duration" in df.columns else None
    return X, y

# ─── TRAINING ──────────────────────────────────────────────────────────────────
#
# Training has two stages:
#   1. build_training_data() — pull historical sessions from OpenF1,
#      build feature matrices, and concatenate into one large DataFrame.
#   2. train_model() — fit a GradientBoostingRegressor on that data
#      and save the model to disk.
#
# Why GradientBoosting over Random Forest?
# Both are tree ensemble methods, but GradientBoosting trains trees
# sequentially — each tree corrects the errors of the previous one.
# This often achieves lower error on structured tabular data, at the
# cost of slower training. For a dataset this size (~thousands of laps),
# the extra training time is negligible.
#
def build_training_data(seasons: list[int]) -> pd.DataFrame:
    """
    Pull historical session data from OpenF1 and build a training dataset.

    We limit to 5 races per season (the :5 slice) to keep initial training
    fast. Remove that slice to train on the full season — expect ~10x longer
    fetch time but better generalisation across circuits.
    """
    print(f"\n[TRAIN] Fetching historical sessions for seasons: {seasons}")
    all_frames = []

    for year in seasons:
        sessions = openf1_get("sessions", {"year": year, "session_type": "Race"})
        print(f"  Season {year}: found {len(sessions)} race sessions")

        for sess in sessions[:5]:   # limit to 5 races per season for demo speed
            key  = sess["session_key"]
            name = sess.get("meeting_name", "Unknown")
            print(f"    Loading: {name} (key={key})")

            laps    = get_laps(key)
            stints  = get_stints(key)
            weather = get_weather(key)

            if laps.empty:
                print(f"      [SKIP] No lap data")
                continue

            features = build_features(laps, stints, weather)
            if not features.empty:
                features["session_key"]   = key
                features["meeting_name"]  = name
                all_frames.append(features)
            time.sleep(0.5)   # be a good API citizen — don't hammer the server

    if not all_frames:
        print("[ERROR] No training data collected.")
        return pd.DataFrame()

    combined = pd.concat(all_frames, ignore_index=True)
    print(f"\n[TRAIN] Total rows collected: {len(combined)}")
    return combined


def train_model(seasons: list[int] = HISTORY_SEASONS):
    """
    Train a lap time prediction model and save to disk.

    Model choice — GradientBoostingRegressor:
    - n_estimators=200:  200 sequential trees (more = better fit, slower)
    - max_depth=5:       each tree can have up to 5 levels of splits
    - learning_rate=0.05: small steps to avoid overfitting (shrinkage)
    - subsample=0.8:     each tree trains on 80% of data (stochastic GB)

    The MAE (Mean Absolute Error) reported at the end is in seconds.
    A good result is typically 0.3–0.8s — within the margin of real
    strategy decisions (1–2s gap to the car ahead in a pit stop).
    """
    df = build_training_data(seasons)
    if df.empty:
        print("[ERROR] Cannot train — no data.")
        return

    X, y = get_feature_matrix(df)

    # Random split here is acceptable for a regression model
    # (unlike the pit stop classifier where we split by season).
    # Lap times don't have the same within-race correlation issue.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("\n[TRAIN] Fitting GradientBoostingRegressor...")
    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae   = mean_absolute_error(y_test, preds)
    print(f"[TRAIN] Test MAE: {mae:.3f}s  ({mae*1000:.0f}ms per lap)")
    # A MAE of 0.5s means on average the model is within half a second
    # of the actual lap time — competitive with professional timing software.

    # Feature importance shows which inputs drove the model most.
    # Expect tyre_age and fuel_load_est to dominate.
    importance = pd.Series(model.feature_importances_, index=FEATURE_COLS)
    print("\n[TRAIN] Feature importances:")
    print(importance.sort_values(ascending=False).to_string())

    joblib.dump(model, MODEL_PATH)
    print(f"\n[TRAIN] Model saved to: {MODEL_PATH}")

# ─── LIVE PREDICTION ───────────────────────────────────────────────────────────

def load_model():
    """Load the trained model from disk."""
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] No model found at {MODEL_PATH}. Run --train first.")
        return None
    return joblib.load(MODEL_PATH)


def predict_next_lap(model, last_lap_row: pd.Series) -> float:
    """
    Predict the next lap time given the last observed lap's features.

    We advance the state forward by one lap:
    - lap_number +1:      we're predicting the next lap
    - tyre_age +1:        tyres are one lap older
    - fuel_load_est -1.8: approximately 1.8kg of fuel burned

    All other features (weather, compound) are assumed constant —
    a simplification that holds unless there's a pit stop or weather change.
    This is where the model would benefit most from real-time pit stop
    detection logic (e.g. from the classifier in the notebook).
    """
    next_lap = last_lap_row.copy()
    next_lap["lap_number"] += 1
    next_lap["tyre_age"]   += 1
    next_lap["fuel_load_est"] = max(0, next_lap["fuel_load_est"] - 1.8)

    X = pd.DataFrame([next_lap[FEATURE_COLS].fillna(0)])
    return float(model.predict(X)[0])


def print_live_table(predictions: dict, drivers: pd.DataFrame, lap: int):
    """Pretty-print current predictions to terminal."""
    print(f"\n{'─'*65}")
    print(f"  LAP {lap} PREDICTIONS  —  {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'─'*65}")
    print(f"  {'DRV':<6} {'TEAM':<22} {'LAST LAP':>10} {'PRED NEXT':>10}")
    print(f"  {'─'*60}")

    for drv_num, info in sorted(predictions.items(), key=lambda x: x[1].get("position", 99)):
        acronym = info.get("acronym", str(drv_num))
        team    = info.get("team", "")[:20]
        last    = f"{info['last_lap']:.3f}s" if info.get("last_lap") else "  —  "
        pred    = f"{info['predicted']:.3f}s" if info.get("predicted") else "  —  "
        print(f"  {acronym:<6} {team:<22} {last:>10} {pred:>10}")

    print(f"{'─'*65}")


def run_live(session_key: int | None = None):
    """
    Main live polling loop.

    Architecture:
    - known_laps tracks the last seen lap per driver to avoid
      reprocessing the same lap multiple times in one poll cycle.
    - predictions is a dict of {driver_number: prediction_info}
      that persists across loop iterations, so drivers who haven't
      completed a new lap still show their last prediction.
    - The loop runs until Ctrl+C — in production you'd wrap this
      in a process manager (e.g. supervisord) to handle crashes.
    """
    model = load_model()
    if model is None:
        return

    if session_key is None:
        print("\n[LIVE] Finding latest session...")
        session_key = get_latest_session_key()
        if session_key is None:
            print("[ERROR] Could not find a live session.")
            return

    print(f"\n[LIVE] Starting live prediction loop for session {session_key}")
    print(f"       Polling every {POLL_INTERVAL}s  |  Ctrl+C to stop\n")

    known_laps    = {}   # {driver_number: last lap number we processed}
    predictions   = {}
    drivers_df    = get_drivers(session_key)

    driver_info = {}
    if not drivers_df.empty:
        for _, row in drivers_df.iterrows():
            driver_info[row["driver_number"]] = {
                "acronym": row.get("name_acronym", str(row["driver_number"])),
                "team":    row.get("team_name", "")
            }

    try:
        while True:
            laps    = get_laps(session_key)
            stints  = get_stints(session_key)
            weather = get_weather(session_key)

            if laps.empty:
                print("[LIVE] Waiting for lap data...")
                time.sleep(POLL_INTERVAL)
                continue

            features = build_features(laps, stints, weather)
            if features.empty:
                time.sleep(POLL_INTERVAL)
                continue

            max_lap = int(features["lap_number"].max())

            for drv_num, grp in features.groupby("driver_number"):
                grp = grp.sort_values("lap_number")
                latest_lap = int(grp["lap_number"].max())

                # Skip if we've already processed this lap for this driver
                if known_laps.get(drv_num) == latest_lap:
                    continue
                known_laps[drv_num] = latest_lap

                last_row  = grp[grp["lap_number"] == latest_lap].iloc[0]
                pred_time = predict_next_lap(model, last_row)

                predictions[drv_num] = {
                    **driver_info.get(drv_num, {"acronym": str(drv_num), "team": ""}),
                    "last_lap":  float(last_row["lap_duration"]),
                    "predicted": pred_time,
                    "lap":       latest_lap
                }

            if predictions:
                print_live_table(predictions, drivers_df, max_lap)

            time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        print("\n[LIVE] Stopped.")

# ─── REPLAY MODE ───────────────────────────────────────────────────────────────
#
# Replay is useful for development and evaluation — it lets you test
# the model against a completed race where you know the actual outcomes.
# The errors list accumulates |predicted - actual| for every driver/lap
# so you get an end-of-race MAE summary.
#
def run_replay(year: int, round_num: int):
    """
    Simulate live predictions on a past race, one lap at a time.

    Reveals data lap-by-lap with a short delay to mimic real-time pacing.
    At the end, prints the mean prediction error across the whole race.
    """
    model = load_model()
    if model is None:
        return

    print(f"\n[REPLAY] Fetching sessions for {year} round {round_num}...")
    sessions = openf1_get("sessions", {"year": year, "session_type": "Race"})
    if not sessions:
        print("[ERROR] No sessions found.")
        return

    sessions.sort(key=lambda x: x.get("date_start", ""))
    if round_num > len(sessions):
        print(f"[ERROR] Only {len(sessions)} races in {year}.")
        return

    sess     = sessions[round_num - 1]
    key      = sess["session_key"]
    name     = sess.get("meeting_name", "Unknown")
    print(f"[REPLAY] Replaying: {name} ({year}) session_key={key}\n")

    laps    = get_laps(key)
    stints  = get_stints(key)
    weather = get_weather(key)
    drivers_df = get_drivers(key)

    driver_info = {}
    if not drivers_df.empty:
        for _, row in drivers_df.iterrows():
            driver_info[row["driver_number"]] = {
                "acronym": row.get("name_acronym", str(row["driver_number"])),
                "team":    row.get("team_name", "")
            }

    features = build_features(laps, stints, weather)
    if features.empty:
        print("[ERROR] No features built.")
        return

    max_lap = int(features["lap_number"].max())
    errors  = []

    # Simulate live: reveal up to lap N, predict lap N+1
    for lap_n in range(2, max_lap + 1):
        lap_data    = features[features["lap_number"] <= lap_n]
        predictions = {}

        for drv_num, grp in lap_data.groupby("driver_number"):
            grp      = grp.sort_values("lap_number")
            last_row = grp.iloc[-1]

            pred = predict_next_lap(model, last_row)
            predictions[drv_num] = {
                **driver_info.get(drv_num, {"acronym": str(drv_num), "team": ""}),
                "last_lap":  float(last_row["lap_duration"]),
                "predicted": pred,
            }

            # If we have the actual next lap, measure the error
            next_lap_data = features[
                (features["driver_number"] == drv_num) &
                (features["lap_number"]    == lap_n + 1)
            ]
            if not next_lap_data.empty:
                actual = float(next_lap_data.iloc[0]["lap_duration"])
                errors.append(abs(pred - actual))

        print_live_table(predictions, drivers_df, lap_n)
        time.sleep(0.8)   # simulate real-time pace (0.8s per lap ~= watchable speed)

    if errors:
        print(f"\n[REPLAY] Mean prediction error across all drivers/laps: {np.mean(errors):.3f}s")

# ─── CLI ───────────────────────────────────────────────────────────────────────
#
# Three entrypoints accessed via command-line flags:
#   --train          Build dataset from OpenF1 + train + save model
#   --live           Start live polling loop for current session
#   --replay         Simulate live on a past race
#
# Typical workflow:
#   python f1_realtime_pipeline.py --train
#   python f1_realtime_pipeline.py --replay --year 2024 --round 3
#   python f1_realtime_pipeline.py --live
#
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="F1 Real-Time ML Pipeline")
    parser.add_argument("--train",  action="store_true", help="Train the model on historical data")
    parser.add_argument("--live",   action="store_true", help="Run live predictions on current race")
    parser.add_argument("--replay", action="store_true", help="Replay a past race")
    parser.add_argument("--year",   type=int, default=2024, help="Season year for replay")
    parser.add_argument("--round",  type=int, default=1,    help="Round number for replay (1-indexed)")
    parser.add_argument("--session", type=int, default=None, help="Specific session_key for live mode")

    args = parser.parse_args()

    if args.train:
        train_model()
    elif args.live:
        run_live(session_key=args.session)
    elif args.replay:
        run_replay(year=args.year, round_num=args.round)
    else:
        parser.print_help()
