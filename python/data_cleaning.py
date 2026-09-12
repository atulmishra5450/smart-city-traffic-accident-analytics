"""
data_cleaning.py
----------------
Complete, reproducible data-cleaning pipeline for the Smart City Traffic &
Accident Analytics project.

Steps
-----
1.  Remove duplicate rows
2.  Handle missing values (column-by-column with business logic)
3.  Normalise inconsistent text categories
4.  Fix invalid Avg_Speed_kmph values
5.  Fix invalid Congestion_Level_% values
6.  Treat Temperature_C outliers (IQR method + domain rules)
7.  Parse Date & Time → datetime features
8.  Create Accident_Flag
9.  Create Severity_Score

Run standalone:
    python python/data_cleaning.py
"""

import re
import pandas as pd
import numpy as np
from pathlib import Path

from utils import get_logger, load_raw, save_cleaned, set_style

logger = get_logger("cleaning")

# ─────────────────────────────────────────────────────────────────────────────
# 1. REMOVE DUPLICATES
# ─────────────────────────────────────────────────────────────────────────────

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop fully-duplicate rows.
    Record_ID should be unique; we also check for duplicate Record_IDs
    which would indicate data-pipeline double-loads.
    """
    before = len(df)
    df = df.drop_duplicates()
    dup_ids = df.duplicated(subset=["Record_ID"]).sum()
    if dup_ids:
        # Keep first occurrence of each Record_ID
        df = df.drop_duplicates(subset=["Record_ID"], keep="first")
    after = len(df)
    logger.info("Duplicates removed: %d rows dropped (kept %d)", before - after, after)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. HANDLE MISSING VALUES
# ─────────────────────────────────────────────────────────────────────────────

def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Column-by-column missing-value treatment with explicit business logic.

    - Vehicle_Count       → median imputation (numeric, right-skewed)
    - Avg_Speed_kmph      → median imputation per City (local patterns matter)
    - Congestion_Level_%  → median imputation
    - Weather             → mode imputation (most common condition)
    - Temperature_C       → median imputation per Month (seasonal temperatures)
    - Visibility_km       → median imputation per Weather condition
    - Vehicle_Type        → mode imputation
    - Accident_Type       → 'None' where Accident_Severity is NaN (no accident rows)
    - Accident_Severity   → left as NaN then used to create Accident_Flag
    - Emergency_Response_Min → 0 where no accident (Accident_Flag = 0), else median
    - Road_Condition      → mode imputation
    - Traffic_Signal_Status → mode imputation
    - Road_Closure        → 'No' (conservative default)
    - Public_Transport_Count → median imputation
    - Latitude / Longitude → median per Area (geographically coherent)
    - Data_Source         → 'Unknown'
    """
    # --- Vehicle_Count ---
    vc_med = df["Vehicle_Count"].median()
    df["Vehicle_Count"] = df["Vehicle_Count"].fillna(vc_med)

    # --- Avg_Speed_kmph (median per City) ---
    df["Avg_Speed_kmph"] = df.groupby("City")["Avg_Speed_kmph"].transform(
        lambda x: x.fillna(x.median())
    )
    # Fallback for any remaining NaN (city with all NaN — unlikely but safe)
    df["Avg_Speed_kmph"] = df["Avg_Speed_kmph"].fillna(df["Avg_Speed_kmph"].median())

    # --- Congestion_Level_% ---
    df["Congestion_Level_%"] = df["Congestion_Level_%"].fillna(
        df["Congestion_Level_%"].median()
    )

    # --- Weather ---
    df["Weather"] = df["Weather"].fillna(df["Weather"].mode()[0])

    # --- Temperature_C (median per Month — requires Date to be parsed first) ---
    # Parse date temporarily so we can group by month
    df["_month_tmp"] = pd.to_datetime(df["Date"], errors="coerce").dt.month
    df["Temperature_C"] = df.groupby("_month_tmp")["Temperature_C"].transform(
        lambda x: x.fillna(x.median())
    )
    df["Temperature_C"] = df["Temperature_C"].fillna(df["Temperature_C"].median())
    df.drop(columns=["_month_tmp"], inplace=True)

    # --- Visibility_km (median per Weather) ---
    df["Visibility_km"] = df.groupby("Weather")["Visibility_km"].transform(
        lambda x: x.fillna(x.median())
    )
    df["Visibility_km"] = df["Visibility_km"].fillna(df["Visibility_km"].median())

    # --- Vehicle_Type ---
    df["Vehicle_Type"] = df["Vehicle_Type"].fillna(df["Vehicle_Type"].mode()[0])

    # --- Accident_Type / Accident_Severity ---
    # NaN in Accident_Severity means no accident occurred.
    # Accident_Type is also NaN in these rows → set to 'None'
    df["Accident_Type"] = df["Accident_Type"].fillna("None")
    # Accident_Severity NaN stays as NaN here; handled in Accident_Flag step

    # --- Emergency_Response_Min ---
    # Rows without an accident have 0 response time (no dispatch needed)
    # Rows with an accident use median
    accident_mask = df["Accident_Severity"].notna()
    df.loc[~accident_mask, "Emergency_Response_Min"] = 0
    erm_med = df.loc[accident_mask, "Emergency_Response_Min"].median()
    df["Emergency_Response_Min"] = df["Emergency_Response_Min"].fillna(erm_med)

    # --- Road_Condition ---
    df["Road_Condition"] = df["Road_Condition"].fillna(df["Road_Condition"].mode()[0])

    # --- Traffic_Signal_Status ---
    df["Traffic_Signal_Status"] = df["Traffic_Signal_Status"].fillna(
        df["Traffic_Signal_Status"].mode()[0]
    )

    # --- Road_Closure ---
    df["Road_Closure"] = df["Road_Closure"].fillna("No")

    # --- Public_Transport_Count ---
    df["Public_Transport_Count"] = df["Public_Transport_Count"].fillna(
        df["Public_Transport_Count"].median()
    )

    # --- Latitude / Longitude (median per Area) ---
    for col in ["Latitude", "Longitude"]:
        df[col] = df.groupby("Area")[col].transform(lambda x: x.fillna(x.median()))
        df[col] = df[col].fillna(df[col].median())

    # --- Data_Source ---
    df["Data_Source"] = df["Data_Source"].fillna("Unknown")

    missing_after = df.isnull().sum().sum()
    logger.info("Missing values after imputation: %d", missing_after)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. NORMALISE TEXT CATEGORIES
# ─────────────────────────────────────────────────────────────────────────────

WEATHER_MAP = {
    "clear": "Clear",
    "cloudy": "Cloudy",
    "fog": "Fog",
    "rain": "Rain",
    "heavy rain": "Heavy Rain",
    "heavy_rain": "Heavy Rain",
    "storm": "Storm",
}

ROAD_MAP = {
    "good": "Good",
    "fair": "Fair",
    "poor": "Poor",
    "under_construction": "Under Construction",
    "under construction": "Under Construction",
}

ACCIDENT_TYPE_MAP = {
    "head-on": "Head-on",
    "rear-end": "Rear-end",
    "side_collision": "Side Collision",
    "side collision": "Side Collision",
    "skidding": "Skidding",
    "pedestrian": "Pedestrian",
    "hit_&_run": "Hit & Run",
    "hit & run": "Hit & Run",
    "none": "None",
}


def _normalise_col(series: pd.Series, mapping: dict) -> pd.Series:
    """Lowercase-strip → apply mapping → return standardised series."""
    return series.str.strip().str.lower().map(
        lambda x: mapping.get(x, x.title()) if isinstance(x, str) else x
    )


def normalise_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise inconsistent text values in categorical columns."""
    df["Weather"]       = _normalise_col(df["Weather"], WEATHER_MAP)
    df["Road_Condition"] = _normalise_col(df["Road_Condition"], ROAD_MAP)
    df["Accident_Type"] = _normalise_col(df["Accident_Type"], ACCIDENT_TYPE_MAP)
    # Ensure title-case on remaining categoricals
    for col in ["Day_Type", "Vehicle_Type", "Accident_Severity",
                "Traffic_Signal_Status", "Road_Closure"]:
        df[col] = df[col].str.strip().str.title()
    logger.info("Text categories normalised.")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4. FIX INVALID SPEED VALUES
# ─────────────────────────────────────────────────────────────────────────────

SPEED_MIN = 1    # km/h — a vehicle must be moving (standing = not a traffic reading)
SPEED_MAX = 150  # km/h — urban/suburban city roads; nothing realistic above this

def fix_speed(df: pd.DataFrame) -> pd.DataFrame:
    """
    Business rule:
    - Speed ≤ 0 or > 150 km/h is physically unrealistic on city roads.
    - Replace with the city-level median speed computed from valid rows only.
    """
    valid_mask = df["Avg_Speed_kmph"].between(SPEED_MIN, SPEED_MAX)
    invalid_count = (~valid_mask).sum()

    city_median = (
        df.loc[valid_mask]
        .groupby("City")["Avg_Speed_kmph"]
        .median()
    )
    df.loc[~valid_mask, "Avg_Speed_kmph"] = df.loc[~valid_mask, "City"].map(city_median)
    logger.info("Speed outliers fixed: %d values replaced.", invalid_count)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 5. FIX INVALID CONGESTION VALUES
# ─────────────────────────────────────────────────────────────────────────────

def fix_congestion(df: pd.DataFrame) -> pd.DataFrame:
    """
    Congestion is a percentage → must be in [0, 100].
    Values < 0 or > 100 are replaced with the overall median.
    """
    valid_mask = df["Congestion_Level_%"].between(0, 100)
    invalid_count = (~valid_mask).sum()
    med = df.loc[valid_mask, "Congestion_Level_%"].median()
    df.loc[~valid_mask, "Congestion_Level_%"] = med
    logger.info("Congestion outliers fixed: %d values replaced with median %.2f.", invalid_count, med)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 6. TREAT TEMPERATURE OUTLIERS
# ─────────────────────────────────────────────────────────────────────────────

def fix_temperature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Use IQR method:  lower = Q1 − 1.5×IQR,  upper = Q3 + 1.5×IQR
    Additionally apply domain rules for Indian cities:
        realistic range  →  −5 °C  to  50 °C
    Values outside IQR fence AND outside the domain range are replaced
    with the monthly median (seasonality preserved).
    """
    q1 = df["Temperature_C"].quantile(0.25)
    q3 = df["Temperature_C"].quantile(0.75)
    iqr = q3 - q1
    lower_iqr = q1 - 1.5 * iqr
    upper_iqr = q3 + 1.5 * iqr

    domain_low, domain_high = -5, 50  # realistic Indian city range

    lower = max(lower_iqr, domain_low)
    upper = min(upper_iqr, domain_high)

    outlier_mask = ~df["Temperature_C"].between(lower, upper)
    count = outlier_mask.sum()

    df["_month_tmp"] = pd.to_datetime(df["Date"], errors="coerce").dt.month
    monthly_med = df.groupby("_month_tmp")["Temperature_C"].transform("median")
    df.loc[outlier_mask, "Temperature_C"] = monthly_med[outlier_mask]
    df.drop(columns=["_month_tmp"], inplace=True)

    logger.info(
        "Temperature outliers: %d replaced. IQR bounds [%.1f, %.1f], domain [%.1f, %.1f].",
        count, lower_iqr, upper_iqr, domain_low, domain_high,
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 7. PARSE DATE / TIME → FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def parse_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a proper Datetime column and extract analytical time features.
    """
    df["Datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"], errors="coerce")

    df["Year"]       = df["Datetime"].dt.year
    df["Month"]      = df["Datetime"].dt.month
    df["Month_Name"] = df["Datetime"].dt.strftime("%b")
    df["Quarter"]    = df["Datetime"].dt.quarter
    df["Week"]       = df["Datetime"].dt.isocalendar().week.astype(int)
    df["Day"]        = df["Datetime"].dt.day
    df["Day_Name"]   = df["Datetime"].dt.day_name()
    df["Hour"]       = df["Datetime"].dt.hour

    # Peak hours: 07–10 (morning) and 17–20 (evening) — standard urban rush hours
    df["Peak_Hour_Flag"] = df["Hour"].apply(
        lambda h: 1 if (7 <= h <= 10) or (17 <= h <= 20) else 0
    )

    # Weekend flag (consistent with Day_Type column — added for numeric ML use)
    df["Weekday_Weekend"] = df["Day_Name"].apply(
        lambda d: "Weekend" if d in ("Saturday", "Sunday") else "Weekday"
    )

    logger.info("Datetime features created.")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 8. ACCIDENT FLAG
# ─────────────────────────────────────────────────────────────────────────────

def create_accident_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Accident_Flag = 1 if an accident occurred (Accident_Severity is not None/NaN),
                  = 0 otherwise.
    This binary flag is the ML classification target and simplifies many aggregations.
    """
    df["Accident_Flag"] = df["Accident_Severity"].apply(
        lambda x: 0 if (pd.isna(x) or str(x).strip().lower() in ("none", "nan", "")) else 1
    )
    logger.info(
        "Accident_Flag created: %d accidents (%.1f%%)",
        df["Accident_Flag"].sum(),
        df["Accident_Flag"].mean() * 100,
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 9. SEVERITY SCORE
# ─────────────────────────────────────────────────────────────────────────────

SEVERITY_MAP = {
    "None":     0,
    "Minor":    1,
    "Moderate": 2,
    "Severe":   3,
    "Fatal":    4,
}

def create_severity_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert ordinal Accident_Severity into a numeric Severity_Score (0–4).

    Why useful:
    - Enables correlation analysis (e.g. does poor weather → higher score?)
    - Can be used as a regression target
    - Allows weighted accident KPIs rather than simple counts
    """
    df["Severity_Score"] = df["Accident_Severity"].map(SEVERITY_MAP).fillna(0).astype(int)
    logger.info("Severity_Score created (0=None … 4=Fatal).")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def clean_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Run every cleaning step in the correct order and return the cleaned DataFrame."""
    df = remove_duplicates(df)
    df = handle_missing(df)
    df = normalise_categories(df)
    df = fix_speed(df)
    df = fix_congestion(df)
    df = fix_temperature(df)
    df = parse_datetime(df)
    df = create_accident_flag(df)
    df = create_severity_score(df)
    return df


if __name__ == "__main__":
    raw = load_raw()
    cleaned = clean_pipeline(raw)
    save_cleaned(cleaned)
    print("\n✓ Cleaning complete.")
    print(cleaned.shape)
    print(cleaned.dtypes)
    print(cleaned.isnull().sum())
