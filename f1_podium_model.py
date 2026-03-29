"""
F1 Podium Predictor — Model Training
======================================
Trains an XGBoost binary classifier to predict podium finishes.
Handles class imbalance, evaluates properly, and saves the model.

Usage:
    python f1_podium_model.py
    → reads   f1_podium_features.csv
    → writes  f1_podium_model.pkl + f1_podium_encoders.pkl
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (classification_report, roc_auc_score,
                             precision_recall_curve, average_precision_score)
from sklearn.calibration import CalibratedClassifierCV
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    XGB_AVAILABLE = False
    print("[INFO] xgboost not installed — using GradientBoostingClassifier instead.")
    print("       Install with: pip install xgboost")

MODEL_PATH   = "f1_podium_model.pkl"
ENCODER_PATH = "f1_podium_encoders.pkl"

FEATURE_COLS = [
    # Qualifying / pre-race pace
    "quali_gap",           # seconds behind pole — strongest single predictor
    "quali_pos",           # grid position on race day
    # Historical form
    "rolling_podium_rate", # podium rate over last 5 races
    "rolling_win_rate",    # win rate over last 5 races
    "rolling_points_avg",  # average points over last 5 races
    "rolling_dnf_rate",    # reliability — DNF rate over last 5 races
    "rolling_grid_avg",    # average qualifying position (form proxy)
    # Car performance
    "car_rank",            # constructor championship position (1=best)
    "car_points",          # constructor points (continuous car quality signal)
    # Driving style (from FastF1 telemetry)
    "tyre_deg_slope",      # lap time increase per tyre age lap (aggression proxy)
    "avg_throttle_pct",    # average throttle application
    "overtake_count",      # positions gained from grid to finish
]
TARGET = "on_podium"


def load_and_prepare(csv_path="f1_podium_features.csv"):
    """
    Load features CSV and prepare for training.

    Key preparation steps:
    1. Drop rows with missing target or key features
    2. Encode categorical columns (constructor, circuit)
    3. Split by season — 2018-2022 train, 2023-2024 test
       (never split randomly for time-series race data)
    """
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df):,} rows from {csv_path}")
    print(f"Podium rate: {df[TARGET].mean()*100:.1f}%  "
          f"({df[TARGET].sum()} podiums / {len(df)} driver-races)")

    # Encode constructor as integer
    enc = LabelEncoder()
    df["constructor_enc"] = enc.fit_transform(df["constructor"].fillna("unknown"))

    # Encode circuit if present
    circuit_enc = LabelEncoder()
    if "circuit_id" in df.columns:
        df["circuit_enc"] = circuit_enc.fit_transform(df["circuit_id"].fillna("unknown"))

    # Save encoders for inference
    joblib.dump({"constructor": enc, "circuit": circuit_enc}, ENCODER_PATH)

    # Drop rows missing critical features
    df = df.dropna(subset=[TARGET] + ["quali_gap", "car_rank"])
    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0)

    # Train on 2018-2022, test on 2023-2024
    train = df[df["season"] <= 2022]
    test  = df[df["season"] >= 2023]

    print(f"\nTrain: {len(train):,} rows ({train['season'].min()}–{train['season'].max()})")
    print(f"Test:  {len(test):,} rows  ({test['season'].min()}–{test['season'].max()})")

    return train, test


def train_model(train: pd.DataFrame) -> object:
    """
    Train an XGBoost (or GradientBoosting) classifier.

    Why XGBoost over Random Forest?
    - Builds trees sequentially, each correcting the previous one's errors
    - Better calibrated probabilities out of the box (important for ranking)
    - scale_pos_weight handles class imbalance natively

    Key hyperparameters:
    - n_estimators=300:      300 boosting rounds
    - max_depth=4:           shallow trees prevent overfitting on small dataset
    - learning_rate=0.05:    small steps for stability
    - scale_pos_weight:      ratio of negatives to positives — fixes imbalance
      e.g. if 15% podiums → weight = 85/15 ≈ 5.7
    - subsample=0.8:         use 80% of rows per tree (stochastic boosting)
    - colsample_bytree=0.8:  use 80% of features per tree (regularisation)
    """
    X = train[FEATURE_COLS]
    y = train[TARGET]

    neg, pos = (y == 0).sum(), (y == 1).sum()
    scale_weight = neg / pos
    print(f"\nClass imbalance ratio: {scale_weight:.1f}x  "
          f"(neg={neg:,} / pos={pos:,})")

    if XGB_AVAILABLE:
        model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_weight,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )
    else:
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        )

    model.fit(X, y)
    print("Model trained.")
    return model


def evaluate(model, test: pd.DataFrame):
    """
    Evaluate the model on the held-out test seasons.

    Metrics used:
    - ROC-AUC:         area under receiver operating characteristic curve
                       1.0 = perfect, 0.5 = random. Target: >0.85
    - Average precision: area under precision-recall curve
                       better than ROC-AUC for imbalanced data
    - Top-3 accuracy:  for each race, do the 3 highest-probability drivers
                       match the 3 actual podium finishers?
                       This is the real-world metric that matters.
    - Precision/Recall at threshold 0.35 (lower than default 0.5 because
                       podium events are rare — we want higher recall)

    NOTE: Never use raw accuracy here. A model predicting "no podium" for
    everyone would be ~85% accurate but completely useless.
    """
    X_test = test[FEATURE_COLS]
    y_test = test[TARGET]
    probs  = model.predict_proba(X_test)[:, 1]

    print("\n── Evaluation ──────────────────────────────────")
    print(f"ROC-AUC:           {roc_auc_score(y_test, probs):.3f}")
    print(f"Average precision: {average_precision_score(y_test, probs):.3f}")

    # Top-3 accuracy per race (the metric that matters most)
    test = test.copy()
    test["prob"] = probs
    correct_races = 0
    total_races   = 0
    for (season, round_num), race_df in test.groupby(["season", "round"]):
        if len(race_df) < 3:
            continue
        predicted_top3 = set(race_df.nlargest(3, "prob")["driver_id"])
        actual_top3    = set(race_df[race_df[TARGET] == 1]["driver_id"])
        # Count as correct if at least 2 of 3 predicted match actual podium
        overlap = len(predicted_top3 & actual_top3)
        if overlap >= 2:
            correct_races += 1
        total_races += 1

    top3_acc = correct_races / total_races if total_races > 0 else 0
    print(f"Top-3 accuracy:    {top3_acc*100:.1f}%  "
          f"(≥2 of 3 correct in {correct_races}/{total_races} races)")

    # Classification report at threshold 0.35
    y_pred = (probs >= 0.35).astype(int)
    print("\nClassification report (threshold=0.35):")
    print(classification_report(y_test, y_pred, target_names=["No podium", "Podium"]))

    return test


def plot_feature_importance(model, top_n=10):
    """Plot feature importance — which features drove predictions most."""
    if hasattr(model, "feature_importances_"):
        imp = pd.Series(model.feature_importances_, index=FEATURE_COLS)
        imp = imp.sort_values(ascending=True).tail(top_n)
        fig, ax = plt.subplots(figsize=(8, 4))
        colors = ["#7F77DD"] * len(imp)
        imp.plot(kind="barh", ax=ax, color=colors)
        ax.set_xlabel("Importance")
        ax.set_title("Feature importance — F1 podium model")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        plt.savefig("f1_feature_importance.png", dpi=120)
        plt.show()
        print("\nFeature importance (top features):")
        print(imp.sort_values(ascending=False).to_string())


def predict_race(model, race_features: pd.DataFrame) -> pd.DataFrame:
    """
    Predict podium probabilities for all drivers in an upcoming race.

    Takes a DataFrame with one row per driver (pre-filled with features)
    and returns it sorted by predicted podium probability descending.

    The top 3 rows are the model's predicted podium finishers.
    """
    race_features = race_features.copy()
    missing = [c for c in FEATURE_COLS if c not in race_features.columns]
    for c in missing:
        race_features[c] = 0.0

    X = race_features[FEATURE_COLS].fillna(0)
    race_features["podium_prob"] = model.predict_proba(X)[:, 1]
    return race_features.sort_values("podium_prob", ascending=False)


if __name__ == "__main__":
    train, test = load_and_prepare()
    model       = train_model(train)
    test_result = evaluate(model, test)
    plot_feature_importance(model)
    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")
