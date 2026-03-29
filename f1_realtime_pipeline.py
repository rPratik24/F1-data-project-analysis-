"""
F1 Real-Time Prediction Pipeline
=================================
Uses OpenF1 API to poll live race data each lap,
then feeds features into a trained ML model to
predict lap times and race positions in real-time.

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
OPENF1_BASE      = "https://api.openf1.org/v1"
POLL_INTERVAL    = 15        # seconds between live polls
MODEL_PATH       = "f1_lap_model.pkl"
ENCODER_PATH     = "f1_encoders.pkl"
HISTORY_SEASONS  = [2023, 2024]   # seasons to train on

TYRE_MAP = {"SOFT": 0, "MEDIUM": 1, "HARD": 2,
            "INTERMEDIATE": 3, "WET": 4, "UNKNOWN": 2}

# ─── OPENF1 HELPERS ────────────────────────────────────────────────────────────
def openf1_get(endpoint: str, params: dict = {}, retries: int = 3) -> list:
    """Hit the OpenF1 API and return JSON as a list of dicts."""
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
            time.sleep(2 ** attempt)
    return []


def get_latest_session_key() -> int | None:
    """Return the session_key for the most recent F1 session."""
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
    """Fetch driver info for a session."""
    data = openf1_get("drivers", {"session_key": session_key})
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)[["driver_number", "name_acronym", "team_name"]]
    return df.drop_duplicates("driver_number")


def get_laps(session_key: int, driver_number: int | None = None) -> pd.DataFrame:
    """Fetch lap data. Optionally filter by driver."""
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
    """Fetch tyre stint data (compound + age)."""
    data = openf1_get("stints", {"session_key": session_key})
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    keep = ["driver_number", "lap_start", "lap_end", "compound", "tyre_age_at_start"]
    return df[[c for c in keep if c in df.columns]]


def get_weather(session_key: int) -> pd.DataFrame:
    """Fetch weather readings for a session."""
    data = openf1_get("weather", {"session_key": session_key})
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    keep = ["date", "air_temperature", "track_temperature",
            "rainfall", "wind_speed", "humidity"]
    return df[[c for c in keep if c in df.columns]]


def get_positions(session_key: int) -> pd.DataFrame:
    """Fetch real-time position data."""
    data = openf1_get("position", {"session_key": session_key})
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    keep = ["driver_number", "date", "position"]
    return df[[c for c in keep if c in df.columns]]

# ─── FEATURE ENGINEERING ───────────────────────────────────────────────────────
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
            age = row["lap_number"] - stint["lap_start"] + stint.get("tyre_age_at_start", 0)
            return pd.Series({"compound": stint["compound"], "tyre_age": age})

        tyre_info = df.apply(get_tyre_info, axis=1)
        df = pd.concat([df, tyre_info], axis=1)
    else:
        df["compound"] = "UNKNOWN"
        df["tyre_age"]  = 0

    df["tyre_code"] = df["compound"].map(TYRE_MAP).fillna(2).astype(int)

    # --- Weather features ---
    if not weather.empty:
        # Use median values as session-level weather summary
        df["air_temp"]   = weather["air_temperature"].median()
        df["track_temp"] = weather["track_temperature"].median()
        df["rain"]       = int(weather["rainfall"].max() > 0)
        df["wind"]       = weather["wind_speed"].median()
    else:
        df["air_temp"]   = 25.0
        df["track_temp"] = 35.0
        df["rain"]       = 0
        df["wind"]       = 5.0

    # --- Derived features ---
    df["is_pit_out_lap"] = df.get("is_pit_out_lap", pd.Series(False, index=df.index)).fillna(False).astype(int)
    df["fuel_load_est"]  = (70 - df["lap_number"] * 1.8).clip(0, 70)   # rough 1.8 kg/lap estimate

    # --- Speed trap average ---
    speed_cols = [c for c in ["i1_speed", "i2_speed", "st_speed"] if c in df.columns]
    if speed_cols:
        df["avg_speed"] = df[speed_cols].mean(axis=1)
    else:
        df["avg_speed"] = 0

    # --- Drop laps with no recorded lap time ---
    df = df.dropna(subset=["lap_duration"])
    df = df[df["lap_duration"] > 60]   # sanity filter: no lap under 1 min

    return df


FEATURE_COLS = [
    "lap_number", "tyre_code", "tyre_age",
    "air_temp", "track_temp", "rain", "wind",
    "fuel_load_est", "avg_speed", "is_pit_out_lap"
]


def get_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series | None]:
    """Return (X, y) where y is lap_duration (None for inference)."""
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    for c in missing:
        df[c] = 0
    X = df[FEATURE_COLS].fillna(0)
    y = df["lap_duration"] if "lap_duration" in df.columns else None
    return X, y

# ─── TRAINING ──────────────────────────────────────────────────────────────────
def build_training_data(seasons: list[int]) -> pd.DataFrame:
    """
    Pull historical session data from OpenF1 for given seasons
    and build a training dataset.
    """
    print(f"\n[TRAIN] Fetching historical sessions for seasons: {seasons}")
    all_frames = []

    for year in seasons:
        sessions = openf1_get("sessions", {"year": year, "session_type": "Race"})
        print(f"  Season {year}: found {len(sessions)} race sessions")

        for sess in sessions[:5]:   # limit to 5 races per season to keep it quick for demo
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
            time.sleep(0.5)   # respect rate limit

    if not all_frames:
        print("[ERROR] No training data collected.")
        return pd.DataFrame()

    combined = pd.concat(all_frames, ignore_index=True)
    print(f"\n[TRAIN] Total rows collected: {len(combined)}")
    return combined


def train_model(seasons: list[int] = HISTORY_SEASONS):
    """Train a lap time prediction model and save to disk."""
    df = build_training_data(seasons)
    if df.empty:
        print("[ERROR] Cannot train — no data.")
        return

    X, y = get_feature_matrix(df)

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

    # Feature importance
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
    """Predict next lap time given the last observed lap's features."""
    next_lap = last_lap_row.copy()
    next_lap["lap_number"] += 1
    next_lap["tyre_age"]   += 1
    next_lap["fuel_load_est"] = max(0, next_lap["fuel_load_est"] - 1.8)

    X = pd.DataFrame([next_lap[FEATURE_COLS].fillna(0)])
    return float(model.predict(X)[0])


def print_live_table(predictions: dict, drivers: pd.DataFrame, lap: int):
    """Pretty-print the current predictions to terminal."""
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
    Polls OpenF1 every POLL_INTERVAL seconds and prints predictions.
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

    known_laps    = {}   # driver_number -> max lap seen
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

                # Only update if we have a new lap
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
def run_replay(year: int, round_num: int):
    """
    Simulate live predictions on a past race.
    Reveals one lap at a time with a short delay.
    """
    model = load_model()
    if model is None:
        return

    print(f"\n[REPLAY] Fetching sessions for {year} round {round_num}...")
    sessions = openf1_get("sessions", {"year": year, "session_type": "Race"})
    if not sessions:
        print("[ERROR] No sessions found.")
        return

    # Pick by round number (1-indexed)
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

    for lap_n in range(2, max_lap + 1):
        lap_data    = features[features["lap_number"] <= lap_n]
        predictions = {}

        for drv_num, grp in lap_data.groupby("driver_number"):
            grp      = grp.sort_values("lap_number")
            last_row = grp.iloc[-1]

            # Only predict if we have a next lap to evaluate against
            next_lap_data = features[
                (features["driver_number"] == drv_num) &
                (features["lap_number"]    == lap_n + 1)
            ]

            pred = predict_next_lap(model, last_row)
            predictions[drv_num] = {
                **driver_info.get(drv_num, {"acronym": str(drv_num), "team": ""}),
                "last_lap":  float(last_row["lap_duration"]),
                "predicted": pred,
            }

            if not next_lap_data.empty:
                actual = float(next_lap_data.iloc[0]["lap_duration"])
                errors.append(abs(pred - actual))

        print_live_table(predictions, drivers_df, lap_n)
        time.sleep(0.8)   # simulate real-time pace

    if errors:
        print(f"\n[REPLAY] Mean prediction error across all drivers/laps: {np.mean(errors):.3f}s")

# ─── CLI ───────────────────────────────────────────────────────────────────────
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
