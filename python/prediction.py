"""
prediction.py
-------------
Machine Learning module for Smart City Traffic & Accident Analytics.

MODEL 1 — Traffic Congestion Prediction (Regression)
    Target  : Congestion_Level_%  (continuous, 0–100)
    Algorithm: Random Forest Regressor (best balance of accuracy vs. interpretability)
    Baseline : Linear Regression

MODEL 2 — Accident Risk Classification (Binary Classification)
    Target  : Accident_Flag (0 = no accident, 1 = accident)
    Algorithm: Gradient Boosting Classifier
    Evaluation: Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix

Run standalone:
    python python/prediction.py
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    classification_report, confusion_matrix, roc_auc_score, ConfusionMatrixDisplay
)
import joblib

from utils import get_logger, set_style, save_fig, PROCESSED_DATA

logger = get_logger("ml")
set_style()

MODEL_DIR = Path(__file__).parent.parent / "data" / "processed"


# ─────────────────────────────────────────────────────────────────────────────
# SHARED UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def load_processed() -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DATA / "traffic_processed.csv")


def encode_features(df: pd.DataFrame, cat_cols: list) -> pd.DataFrame:
    """Label-encode selected categorical columns in-place (copy)."""
    df = df.copy()
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 1 — CONGESTION PREDICTION (Regression)
# ─────────────────────────────────────────────────────────────────────────────

def model1_congestion_prediction(df: pd.DataFrame):
    """
    Predict Congestion_Level_% from operational and environmental features.

    Features chosen:
        - Vehicle_Count          (direct driver of congestion)
        - Avg_Speed_kmph         (inverse relationship with congestion)
        - Hour                   (time-of-day pattern)
        - Weekend_Flag           (weekend vs weekday pattern)
        - Weather_Risk_Flag      (bad weather slows traffic)
        - Poor_Road_Flag         (road quality reduces throughput)
        - Faulty_Signal_Flag     (disrupts flow)
        - Public_Transport_Count (transit shifts private vehicle demand)
        - City (label-encoded)
        - Road_Name (label-encoded)

    ⚠ Data Leakage Avoidance:
        - Accident_Flag / Severity are excluded (they are outcomes of congestion,
          not causes, and using them would leak the target relationship)
    """
    logger.info("=== MODEL 1: Congestion Prediction ===")

    features = [
        "Vehicle_Count","Avg_Speed_kmph","Hour","Weekend_Flag",
        "Weather_Risk_Flag","Poor_Road_Flag","Faulty_Signal_Flag",
        "Public_Transport_Count","City","Road_Name"
    ]
    target = "Congestion_Level_%"

    sub = df[features + [target]].dropna()
    sub = encode_features(sub, ["City", "Road_Name"])

    X = sub[features]
    y = sub[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ── Baseline: Linear Regression ──────────────────────────────────────────
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    lr_mae  = mean_absolute_error(y_test, lr_pred)
    lr_r2   = r2_score(y_test, lr_pred)
    logger.info("Linear Regression → MAE: %.2f  R²: %.4f", lr_mae, lr_r2)

    # ── Main Model: Random Forest ─────────────────────────────────────────────
    rf = RandomForestRegressor(n_estimators=80, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_mae  = mean_absolute_error(y_test, rf_pred)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
    rf_r2   = r2_score(y_test, rf_pred)
    logger.info("Random Forest  MAE: %.2f  RMSE: %.2f  R2: %.4f", rf_mae, rf_rmse, rf_r2)

    # Cross-validation (R²) — 3 folds for speed on 247K rows
    cv_scores = cross_val_score(rf, X, y, cv=3, scoring="r2", n_jobs=-1)
    logger.info("CV R² scores: %s  Mean: %.4f", cv_scores.round(3), cv_scores.mean())

    # ── Feature Importance Plot ──────────────────────────────────────────────
    fi = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    fi.plot(kind="barh", ax=ax, color="#3b82d4", edgecolor="white")
    ax.set_title("Random Forest — Feature Importance (Congestion Prediction)")
    ax.set_xlabel("Importance Score")
    save_fig(fig, "ml_01_congestion_feature_importance")

    # ── Actual vs Predicted Plot ──────────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    ax2.scatter(y_test, rf_pred, alpha=0.1, s=5, color="#7c5cd8")
    ax2.plot([0, 100], [0, 100], "r--", linewidth=1.5, label="Perfect Prediction")
    ax2.set_xlabel("Actual Congestion (%)")
    ax2.set_ylabel("Predicted Congestion (%)")
    ax2.set_title("Congestion Prediction: Actual vs Predicted")
    ax2.legend()
    save_fig(fig2, "ml_02_congestion_actual_vs_pred")

    # ── Save model ────────────────────────────────────────────────────────────
    joblib.dump(rf, MODEL_DIR / "model_congestion_rf.pkl")
    logger.info("Model saved: model_congestion_rf.pkl")

    return {
        "model": "Random Forest Regressor",
        "features": features,
        "baseline_mae": round(lr_mae, 2),
        "baseline_r2": round(lr_r2, 4),
        "rf_mae": round(rf_mae, 2),
        "rf_rmse": round(rf_rmse, 2),
        "rf_r2": round(rf_r2, 4),
        "cv_r2_mean": round(cv_scores.mean(), 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 2 — ACCIDENT RISK CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def model2_accident_risk(df: pd.DataFrame):
    """
    Predict whether an accident is likely (Accident_Flag = 1).

    Features chosen:
        - Vehicle_Count          (more vehicles → more collision exposure)
        - Avg_Speed_kmph         (higher speed → more severe outcomes)
        - Congestion_Level_%     (bumper-to-bumper increases rear-end risk)
        - Hour                   (time-of-day pattern)
        - Weekend_Flag
        - Weather_Risk_Flag      (adverse weather → higher risk)
        - Poor_Road_Flag
        - Faulty_Signal_Flag
        - Low_Visibility_Flag
        - Traffic_Risk_Score     (composite risk indicator)
        - City (label-encoded)
        - Road_Name (label-encoded)

    ⚠ Data Leakage Avoidance:
        - Accident_Type, Accident_Severity, Severity_Score, Emergency_Response_Min
          are ALL excluded — they are observed AFTER the accident, not before.
    """
    logger.info("=== MODEL 2: Accident Risk Classification ===")

    features = [
        "Vehicle_Count","Avg_Speed_kmph","Congestion_Level_%","Hour","Weekend_Flag",
        "Weather_Risk_Flag","Poor_Road_Flag","Faulty_Signal_Flag",
        "Low_Visibility_Flag","Traffic_Risk_Score","City","Road_Name"
    ]
    target = "Accident_Flag"

    sub = df[features + [target]].dropna()
    sub = encode_features(sub, ["City", "Road_Name"])

    X = sub[features].values
    y = sub[target].values

    # ── Class balance check ──────────────────────────────────────────────────
    accident_rate = y.mean() * 100
    logger.info("Accident rate in dataset: %.1f%%", accident_rate)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale (GBM doesn't need it but helps comparability)
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # ── Gradient Boosting Classifier ─────────────────────────────────────────
    gb = GradientBoostingClassifier(
        n_estimators=80, learning_rate=0.10, max_depth=4, random_state=42
    )
    gb.fit(X_train_sc, y_train)
    y_pred = gb.predict(X_test_sc)
    y_prob = gb.predict_proba(X_test_sc)[:, 1]

    print("\n=== Accident Risk Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=["No Accident","Accident"]))
    roc = roc_auc_score(y_test, y_prob)
    logger.info("ROC-AUC: %.4f", roc)

    # Cross-validation — 3 folds for speed
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    cv_f1 = cross_val_score(gb, X_train_sc, y_train, cv=skf, scoring="f1", n_jobs=-1)
    logger.info("CV F1: %s  Mean: %.4f", cv_f1.round(3), cv_f1.mean())

    # ── Confusion Matrix ──────────────────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(cm, display_labels=["No Accident","Accident"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix — Accident Risk Classifier")
    save_fig(fig, "ml_03_accident_confusion_matrix")

    # ── Feature Importance ────────────────────────────────────────────────────
    fi = pd.Series(gb.feature_importances_, index=features).sort_values(ascending=True)
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    fi.plot(kind="barh", ax=ax2, color="#ef4444", edgecolor="white")
    ax2.set_title("GBM — Feature Importance (Accident Risk Classification)")
    ax2.set_xlabel("Importance Score")
    save_fig(fig2, "ml_04_accident_feature_importance")

    # ── Save model + scaler ───────────────────────────────────────────────────
    joblib.dump(gb,     MODEL_DIR / "model_accident_gb.pkl")
    joblib.dump(scaler, MODEL_DIR / "scaler_accident.pkl")
    logger.info("Model saved: model_accident_gb.pkl")

    return {
        "model": "Gradient Boosting Classifier",
        "features": features,
        "roc_auc": round(roc, 4),
        "cv_f1_mean": round(cv_f1.mean(), 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = load_processed()
    logger.info("Loaded: %d rows × %d cols", *df.shape)

    results_m1 = model1_congestion_prediction(df)
    results_m2 = model2_accident_risk(df)

    print("\n" + "="*50)
    print("MODEL 1 — Congestion Prediction")
    print("="*50)
    for k, v in results_m1.items():
        if k != "features":
            print(f"  {k:25s}: {v}")

    print("\n" + "="*50)
    print("MODEL 2 — Accident Risk Classification")
    print("="*50)
    for k, v in results_m2.items():
        if k != "features":
            print(f"  {k:25s}: {v}")
