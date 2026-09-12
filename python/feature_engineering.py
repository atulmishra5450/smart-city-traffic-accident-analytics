"""
feature_engineering.py
-----------------------
Creates analytically meaningful features from the cleaned dataset.
Every feature is explained in-line. Run after data_cleaning.py.

Run standalone:
    python python/feature_engineering.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

from utils import get_logger, save_processed

logger = get_logger("features")

# ─────────────────────────────────────────────────────────────────────────────
# 1. TRAFFIC CATEGORY
# ─────────────────────────────────────────────────────────────────────────────

def add_traffic_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Segment Vehicle_Count into human-readable traffic buckets.
    Thresholds derived from dataset quartiles:
        Q1 ~565, Median ~740, Q3 ~1055
    """
    bins   = [0, 400, 700, 1000, float("inf")]
    labels = ["Low", "Moderate", "High", "Very High"]
    df["Traffic_Category"] = pd.cut(
        df["Vehicle_Count"], bins=bins, labels=labels, right=True
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. CONGESTION CATEGORY
# ─────────────────────────────────────────────────────────────────────────────

def add_congestion_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map the continuous Congestion_Level_% to four named tiers.
    Standard traffic-engineering thresholds:
        0–25  = Free Flow
        25–50 = Moderate
        50–75 = Heavy
        75+   = Gridlock
    """
    bins   = [0, 25, 50, 75, 100]
    labels = ["Free Flow", "Moderate", "Heavy", "Gridlock"]
    df["Congestion_Category"] = pd.cut(
        df["Congestion_Level_%"], bins=bins, labels=labels,
        right=True, include_lowest=True
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. SPEED CATEGORY
# ─────────────────────────────────────────────────────────────────────────────

def add_speed_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify average speed into named tiers.
    Urban speed-limit norms (India): typical city limit 40–60 km/h.
    """
    bins   = [0, 20, 40, 60, float("inf")]
    labels = ["Very Slow", "Slow", "Normal", "Fast"]
    df["Speed_Category"] = pd.cut(
        df["Avg_Speed_kmph"], bins=bins, labels=labels, right=True
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4. SEVERE ACCIDENT FLAG
# ─────────────────────────────────────────────────────────────────────────────

def add_severe_accident_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Binary flag: 1 if Accident_Severity is 'Severe' or 'Fatal', else 0.
    Useful for emergency resource planning — these cases need intensive response.
    """
    df["Severe_Accident_Flag"] = df["Accident_Severity"].apply(
        lambda x: 1 if str(x) in ("Severe", "Fatal") else 0
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 5. WEEKEND FLAG (numeric)
# ─────────────────────────────────────────────────────────────────────────────

def add_weekend_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Numeric version of Weekday_Weekend (1 = Weekend, 0 = Weekday).
    Required for ML models that consume numeric features.
    """
    df["Weekend_Flag"] = (df["Day_Name"].isin(["Saturday", "Sunday"])).astype(int)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 6. POOR ROAD FLAG
# ─────────────────────────────────────────────────────────────────────────────

def add_poor_road_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    1 if Road_Condition is 'Poor' or 'Under Construction', else 0.
    These conditions reduce traction and driver reaction time.
    """
    df["Poor_Road_Flag"] = df["Road_Condition"].isin(
        ["Poor", "Under Construction"]
    ).astype(int)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 7. FAULTY SIGNAL FLAG
# ─────────────────────────────────────────────────────────────────────────────

def add_faulty_signal_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    1 if Traffic_Signal_Status is 'Faulty', else 0.
    Faulty signals remove orderly traffic flow → higher collision probability.
    """
    df["Faulty_Signal_Flag"] = (df["Traffic_Signal_Status"] == "Faulty").astype(int)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 8. LOW VISIBILITY FLAG
# ─────────────────────────────────────────────────────────────────────────────

def add_low_visibility_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    1 if Visibility_km ≤ 3 km, else 0.
    Below 3 km significantly reduces stopping distance and driver sight lines.
    """
    df["Low_Visibility_Flag"] = (df["Visibility_km"] <= 3.0).astype(int)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 9. WEATHER RISK FLAG
# ─────────────────────────────────────────────────────────────────────────────

HIGH_RISK_WEATHER = {"Fog", "Heavy Rain", "Storm"}

def add_weather_risk_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    1 if Weather is Fog, Heavy Rain, or Storm, else 0.
    These conditions are statistically associated with higher accident rates.
    """
    df["Weather_Risk_Flag"] = df["Weather"].isin(HIGH_RISK_WEATHER).astype(int)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 10. EMERGENCY RESPONSE CATEGORY
# ─────────────────────────────────────────────────────────────────────────────

def add_emergency_response_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify Emergency_Response_Min into service-level tiers.
    Benchmarks based on urban EMS guidelines (India target ≤ 15 min):
        0        = No accident
        0–10     = Excellent
        10–20    = Acceptable
        20–30    = Delayed
        30+      = Critical
    """
    def _cat(x):
        if x == 0:
            return "No Accident"
        elif x <= 10:
            return "Excellent"
        elif x <= 20:
            return "Acceptable"
        elif x <= 30:
            return "Delayed"
        else:
            return "Critical"

    df["Emergency_Response_Category"] = df["Emergency_Response_Min"].apply(_cat)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 11. TRAFFIC RISK SCORE
# ─────────────────────────────────────────────────────────────────────────────

def add_traffic_risk_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Composite Traffic_Risk_Score (0–10 scale) combining five independent risk factors:

    Component               Weight  Rationale
    ──────────────────────  ──────  ─────────────────────────────────────────────
    Congestion (normalised)   0.25  High congestion → increased rear-end risk
    Low Visibility Flag       0.20  Reduces reaction time
    Poor Road Flag            0.20  Reduces vehicle control
    Weather Risk Flag         0.20  Adverse weather → higher accident probability
    Faulty Signal Flag        0.15  Loss of right-of-way enforcement

    Score = 10 × weighted_sum
    """
    cong_norm = df["Congestion_Level_%"] / 100.0  # normalise to [0, 1]

    df["Traffic_Risk_Score"] = (
        0.25 * cong_norm
        + 0.20 * df["Low_Visibility_Flag"]
        + 0.20 * df["Poor_Road_Flag"]
        + 0.20 * df["Weather_Risk_Flag"]
        + 0.15 * df["Faulty_Signal_Flag"]
    ) * 10

    df["Traffic_Risk_Score"] = df["Traffic_Risk_Score"].round(2)
    logger.info(
        "Traffic_Risk_Score range: %.2f – %.2f",
        df["Traffic_Risk_Score"].min(),
        df["Traffic_Risk_Score"].max(),
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering steps and return the enriched DataFrame."""
    df = add_traffic_category(df)
    df = add_congestion_category(df)
    df = add_speed_category(df)
    df = add_severe_accident_flag(df)
    df = add_weekend_flag(df)
    df = add_poor_road_flag(df)
    df = add_faulty_signal_flag(df)
    df = add_low_visibility_flag(df)
    df = add_weather_risk_flag(df)
    df = add_emergency_response_category(df)
    df = add_traffic_risk_score(df)
    logger.info("Feature engineering complete. Total columns: %d", len(df.columns))
    return df


if __name__ == "__main__":
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    from utils import CLEANED_DATA
    df = pd.read_csv(CLEANED_DATA / "traffic_cleaned.csv")
    df = engineer_features(df)
    save_processed(df)
    print("\n✓ Feature engineering complete.")
    print(df[["Traffic_Category","Congestion_Category","Speed_Category",
              "Traffic_Risk_Score","Accident_Flag"]].head(10))
