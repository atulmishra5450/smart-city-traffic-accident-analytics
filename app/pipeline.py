"""
pipeline.py
-----------
Real-time / fresh-data pipeline for Smart City Traffic & Accident Analytics.

Architecture:
    OpenWeatherMap API  (weather)
    +  TomTom Traffic API (traffic)
           ↓
    Python validation & transformation
           ↓
    SQL Server (SmartCityTrafficDB)
           ↓
    Power BI auto-refresh

Usage:
    1. Copy .env.example to .env and fill in your API keys.
    2. python app/pipeline.py

Environment variables (see .env.example):
    OPENWEATHER_API_KEY
    TOMTOM_API_KEY
    SQL_SERVER
    SQL_DATABASE
    SQL_USERNAME
    SQL_PASSWORD
"""

import os
import sys
import time
import json
import logging
import requests
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv(Path(__file__).parent.parent / ".env")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

OW_API_KEY  = os.getenv("OPENWEATHER_API_KEY", "")
TT_API_KEY  = os.getenv("TOMTOM_API_KEY", "")
SQL_SERVER  = os.getenv("SQL_SERVER",   "localhost")
SQL_DB      = os.getenv("SQL_DATABASE", "SmartCityTrafficDB")
SQL_USER    = os.getenv("SQL_USERNAME", "")
SQL_PASS    = os.getenv("SQL_PASSWORD", "")

# Cities + representative bounding boxes / coords
CITIES = {
    "Lucknow":    (26.847, 80.947),
    "Kanpur":     (26.449, 80.331),
    "Delhi":      (28.613, 77.209),
    "Mumbai":     (19.076, 72.877),
    "Bengaluru":  (12.971, 77.594),
    "Hyderabad":  (17.385, 78.486),
    "Pune":       (18.520, 73.856),
    "Jaipur":     (26.912, 75.787),
    "Noida":      (28.535, 77.391),
    "Varanasi":   (25.317, 83.013),
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("pipeline")


# ─────────────────────────────────────────────────────────────────────────────
# 1. FETCH WEATHER DATA (OpenWeatherMap)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_weather(city: str, lat: float, lon: float) -> dict:
    """
    Fetch current weather for a city from OpenWeatherMap.
    Returns a dict with standardised weather fields.
    API docs: https://openweathermap.org/current
    Replace with your real API key in .env
    """
    if not OW_API_KEY:
        logger.warning("OPENWEATHER_API_KEY not set — using mock data for %s", city)
        return _mock_weather(city)

    url = (
        f"https://api.openweathermap.org/data/2.5/weather"
        f"?lat={lat}&lon={lon}&appid={OW_API_KEY}&units=metric"
    )
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return {
            "city":           city,
            "weather":        _map_weather(data["weather"][0]["main"]),
            "temperature_c":  round(data["main"]["temp"], 1),
            "visibility_km":  round(data.get("visibility", 10000) / 1000, 1),
            "timestamp":      datetime.now(timezone.utc),
        }
    except Exception as e:
        logger.error("Weather API error for %s: %s", city, e)
        return _mock_weather(city)


def _map_weather(raw: str) -> str:
    """Map OpenWeatherMap condition string to project categories."""
    mapping = {
        "Clear":       "Clear",
        "Clouds":      "Cloudy",
        "Fog":         "Fog",
        "Mist":        "Fog",
        "Haze":        "Fog",
        "Rain":        "Rain",
        "Drizzle":     "Rain",
        "Thunderstorm":"Storm",
        "Snow":        "Fog",  # rare for Indian cities but safe default
    }
    return mapping.get(raw, "Cloudy")


def _mock_weather(city: str) -> dict:
    """Return mock weather data when API key is unavailable."""
    return {
        "city":           city,
        "weather":        "Clear",
        "temperature_c":  27.0,
        "visibility_km":  8.0,
        "timestamp":      datetime.now(timezone.utc),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. FETCH TRAFFIC FLOW DATA (TomTom)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_traffic_flow(city: str, lat: float, lon: float) -> dict:
    """
    Fetch real-time traffic flow from TomTom Traffic API.
    API docs: https://developer.tomtom.com/traffic-api/documentation/traffic-flow
    """
    if not TT_API_KEY:
        logger.warning("TOMTOM_API_KEY not set — using mock data for %s", city)
        return _mock_traffic(city)

    url = (
        f"https://api.tomtom.com/traffic/services/4/flowSegmentData/relative0/10/json"
        f"?point={lat},{lon}&key={TT_API_KEY}"
    )
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json().get("flowSegmentData", {})
        current_speed  = data.get("currentSpeed", 40)
        free_flow_speed = data.get("freeFlowSpeed", 60)
        congestion = max(0, min(100, round((1 - current_speed / max(free_flow_speed, 1)) * 100, 1)))
        return {
            "city":             city,
            "avg_speed_kmph":   round(current_speed, 1),
            "free_flow_speed":  round(free_flow_speed, 1),
            "congestion_pct":   congestion,
            "timestamp":        datetime.now(timezone.utc),
        }
    except Exception as e:
        logger.error("Traffic API error for %s: %s", city, e)
        return _mock_traffic(city)


def _mock_traffic(city: str) -> dict:
    return {
        "city":             city,
        "avg_speed_kmph":   42.0,
        "free_flow_speed":  60.0,
        "congestion_pct":   35.0,
        "timestamp":        datetime.now(timezone.utc),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. VALIDATE FRESH DATA
# ─────────────────────────────────────────────────────────────────────────────

def validate_record(rec: dict) -> bool:
    """
    Basic validation before inserting into the database.
    Returns True if record passes all checks.
    """
    checks = [
        1 <= rec.get("avg_speed_kmph", 0) <= 150,
        0 <= rec.get("congestion_pct", -1) <= 100,
        -5 <= rec.get("temperature_c", -99) <= 50,
        0 < rec.get("visibility_km", 0) <= 20,
    ]
    if not all(checks):
        logger.warning("Validation failed for record: %s", rec)
        return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# 4. WRITE TO DATABASE (SQL Server via pyodbc)
# ─────────────────────────────────────────────────────────────────────────────

def write_to_db(records: list[dict]):
    """
    Insert validated fresh records into Fact_Traffic staging table.
    Requires pyodbc + SQL Server ODBC driver.
    """
    try:
        import pyodbc
    except ImportError:
        logger.warning("pyodbc not installed — saving to CSV instead.")
        _write_to_csv(records)
        return

    if not SQL_USER:
        logger.warning("SQL credentials not set — saving to CSV instead.")
        _write_to_csv(records)
        return

    conn_str = (
        f"DRIVER={{ODBC Driver 17 for SQL Server}};"
        f"SERVER={SQL_SERVER};DATABASE={SQL_DB};"
        f"UID={SQL_USER};PWD={SQL_PASS}"
    )
    try:
        conn = pyodbc.connect(conn_str)
        cur  = conn.cursor()
        for rec in records:
            cur.execute("""
                INSERT INTO dbo.Stg_FreshTraffic
                    (City, Avg_Speed_kmph, Congestion_Pct, Weather,
                     Temperature_C, Visibility_km, Timestamp)
                VALUES (?,?,?,?,?,?,?)
            """,
            rec["city"], rec["avg_speed_kmph"], rec["congestion_pct"],
            rec["weather"], rec["temperature_c"], rec["visibility_km"],
            rec["timestamp"])
        conn.commit()
        conn.close()
        logger.info("Inserted %d fresh records into SQL.", len(records))
    except Exception as e:
        logger.error("DB insert failed: %s", e)
        _write_to_csv(records)


def _write_to_csv(records: list[dict]):
    """Fallback: append fresh data to a CSV file."""
    out = Path(__file__).parent.parent / "data" / "processed" / "fresh_data.csv"
    df = pd.DataFrame(records)
    df.to_csv(out, mode="a", header=not out.exists(), index=False)
    logger.info("Appended %d records to %s", len(records), out)


# ─────────────────────────────────────────────────────────────────────────────
# 5. MAIN PIPELINE LOOP
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(interval_minutes: int = 15):
    """
    Continuously fetch weather + traffic data for all cities,
    validate, and write to the database.

    interval_minutes: time between each collection cycle
    """
    logger.info("Pipeline started. Refresh every %d minutes.", interval_minutes)
    while True:
        batch = []
        for city, (lat, lon) in CITIES.items():
            weather_data  = fetch_weather(city, lat, lon)
            traffic_data  = fetch_traffic_flow(city, lat, lon)

            record = {
                "city":            city,
                "avg_speed_kmph":  traffic_data["avg_speed_kmph"],
                "congestion_pct":  traffic_data["congestion_pct"],
                "weather":         weather_data["weather"],
                "temperature_c":   weather_data["temperature_c"],
                "visibility_km":   weather_data["visibility_km"],
                "timestamp":       datetime.now(timezone.utc).isoformat(),
            }

            if validate_record(record):
                batch.append(record)
            else:
                logger.warning("Skipping invalid record for %s", city)

        if batch:
            write_to_db(batch)
            logger.info("Cycle complete — %d cities processed.", len(batch))

        time.sleep(interval_minutes * 60)


if __name__ == "__main__":
    run_pipeline(interval_minutes=15)
