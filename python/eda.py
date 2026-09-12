"""
eda.py
------
Exploratory Data Analysis for Smart City Traffic & Accident Analytics.
Generates 20 publication-quality visualisations saved to reports/.

Run standalone:
    python python/eda.py
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from utils import get_logger, set_style, save_fig, PROCESSED_DATA

logger = get_logger("eda")
set_style()


def load_data() -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DATA / "traffic_processed.csv")


# ─────────────────────────────────────────────────────────────────────────────
# CHART 1 – Traffic Volume by City
# ─────────────────────────────────────────────────────────────────────────────
def plot_traffic_by_city(df):
    city_vol = df.groupby("City")["Vehicle_Count"].sum().sort_values(ascending=True) / 1e6
    fig, ax = plt.subplots(figsize=(9, 5))
    city_vol.plot(kind="barh", ax=ax, color="#3b82d4", edgecolor="white")
    ax.set_xlabel("Total Vehicles (Millions)")
    ax.set_title("Total Traffic Volume by City")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}M"))
    save_fig(fig, "01_traffic_by_city")
    logger.info("Chart 1 saved.")


# ─────────────────────────────────────────────────────────────────────────────
# CHART 2 – Hourly Traffic Volume
# ─────────────────────────────────────────────────────────────────────────────
def plot_traffic_by_hour(df):
    hourly = df.groupby("Hour")["Vehicle_Count"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(hourly["Hour"], hourly["Vehicle_Count"], marker="o", color="#3b82d4", linewidth=2)
    ax.fill_between(hourly["Hour"], hourly["Vehicle_Count"], alpha=0.15, color="#3b82d4")
    ax.set_xticks(range(0, 24))
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Avg Vehicle Count")
    ax.set_title("Average Traffic Volume by Hour of Day")
    save_fig(fig, "02_traffic_by_hour")
    logger.info("Chart 2 saved.")


# ─────────────────────────────────────────────────────────────────────────────
# CHART 3 – Traffic Volume by Day Name
# ─────────────────────────────────────────────────────────────────────────────
def plot_traffic_by_day(df):
    order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    day_vol = df.groupby("Day_Name")["Vehicle_Count"].mean().reindex(order)
    fig, ax = plt.subplots(figsize=(9, 4))
    colors = ["#5c9de4" if d not in ("Saturday","Sunday") else "#f59e0b" for d in order]
    day_vol.plot(kind="bar", ax=ax, color=colors, edgecolor="white", rot=30)
    ax.set_ylabel("Avg Vehicle Count")
    ax.set_title("Average Traffic Volume by Day of Week")
    save_fig(fig, "03_traffic_by_day")
    logger.info("Chart 3 saved.")


# ─────────────────────────────────────────────────────────────────────────────
# CHART 4 – Monthly Traffic Trend
# ─────────────────────────────────────────────────────────────────────────────
def plot_monthly_traffic(df):
    month_order = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]
    df["Month_Name"] = pd.Categorical(df["Month_Name"], categories=month_order, ordered=True)
    monthly = df.groupby("Month_Name", observed=True)["Vehicle_Count"].sum() / 1e6
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(monthly.index, monthly.values, color="#3b82d4", edgecolor="white")
    ax.set_xlabel("Month")
    ax.set_ylabel("Total Vehicles (Millions)")
    ax.set_title("Monthly Traffic Volume Trend (2025)")
    save_fig(fig, "04_monthly_traffic")
    logger.info("Chart 4 saved.")


# ─────────────────────────────────────────────────────────────────────────────
# CHART 5 – Average Speed by City
# ─────────────────────────────────────────────────────────────────────────────
def plot_speed_by_city(df):
    speed = df.groupby("City")["Avg_Speed_kmph"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(9, 5))
    speed.plot(kind="barh", ax=ax, color="#22c55e", edgecolor="white")
    ax.set_xlabel("Average Speed (km/h)")
    ax.set_title("Average Vehicle Speed by City")
    save_fig(fig, "05_speed_by_city")
    logger.info("Chart 5 saved.")


# ─────────────────────────────────────────────────────────────────────────────
# CHART 6 – Average Congestion by City
# ─────────────────────────────────────────────────────────────────────────────
def plot_congestion_by_city(df):
    cong = df.groupby("City")["Congestion_Level_%"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(cong.index, cong.values, color="#f97316", edgecolor="white")
    ax.set_ylabel("Avg Congestion Level (%)")
    ax.set_title("Average Congestion Level by City")
    ax.set_xticklabels(cong.index, rotation=30, ha="right")
    save_fig(fig, "06_congestion_by_city")
    logger.info("Chart 6 saved.")


# ─────────────────────────────────────────────────────────────────────────────
# CHART 7 – Congestion by Road
# ─────────────────────────────────────────────────────────────────────────────
def plot_congestion_by_road(df):
    road_cong = df.groupby("Road_Name")["Congestion_Level_%"].mean().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    road_cong.plot(kind="barh", ax=ax, color="#f97316", edgecolor="white")
    ax.set_xlabel("Avg Congestion Level (%)")
    ax.set_title("Average Congestion Level by Road Type")
    save_fig(fig, "07_congestion_by_road")
    logger.info("Chart 7 saved.")


# ─────────────────────────────────────────────────────────────────────────────
# CHART 8 – Accident Count by City
# ─────────────────────────────────────────────────────────────────────────────
def plot_accidents_by_city(df):
    acc = df[df["Accident_Flag"] == 1].groupby("City").size().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    acc.plot(kind="barh", ax=ax, color="#ef4444", edgecolor="white")
    ax.set_xlabel("Number of Accidents")
    ax.set_title("Total Accident Count by City")
    save_fig(fig, "08_accidents_by_city")
    logger.info("Chart 8 saved.")


# ─────────────────────────────────────────────────────────────────────────────
# CHART 9 – Accident Count by Road
# ─────────────────────────────────────────────────────────────────────────────
def plot_accidents_by_road(df):
    acc = df[df["Accident_Flag"] == 1].groupby("Road_Name").size().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    acc.plot(kind="bar", ax=ax, color="#ef4444", edgecolor="white", rot=30)
    ax.set_ylabel("Number of Accidents")
    ax.set_title("Total Accident Count by Road Type")
    save_fig(fig, "09_accidents_by_road")
    logger.info("Chart 9 saved.")


# ─────────────────────────────────────────────────────────────────────────────
# CHART 10 – Accident Type Distribution
# ─────────────────────────────────────────────────────────────────────────────
def plot_accident_type(df):
    acc = df[df["Accident_Flag"] == 1]
    type_dist = acc["Accident_Type"].value_counts()
    fig, ax = plt.subplots(figsize=(8, 5))
    type_dist.plot(kind="bar", ax=ax, color="#6366f1", edgecolor="white", rot=35)
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Accident Types")
    save_fig(fig, "10_accident_type_dist")
    logger.info("Chart 10 saved.")


# ─────────────────────────────────────────────────────────────────────────────
# CHART 11 – Accident Severity Distribution
# ─────────────────────────────────────────────────────────────────────────────
def plot_accident_severity(df):
    acc = df[df["Accident_Flag"] == 1]
    sev = acc["Accident_Severity"].value_counts().reindex(["Minor","Moderate","Severe","Fatal"])
    colors = ["#86efac","#fbbf24","#f97316","#ef4444"]
    fig, ax = plt.subplots(figsize=(7, 4))
    sev.plot(kind="bar", ax=ax, color=colors, edgecolor="white", rot=0)
    ax.set_ylabel("Count")
    ax.set_title("Accident Severity Distribution")
    save_fig(fig, "11_accident_severity_dist")
    logger.info("Chart 11 saved.")


# ─────────────────────────────────────────────────────────────────────────────
# CHART 12 – Vehicle Type vs Accidents
# ─────────────────────────────────────────────────────────────────────────────
def plot_vehicle_vs_accidents(df):
    veh = df.groupby("Vehicle_Type")["Accident_Flag"].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    veh.plot(kind="bar", ax=ax, color="#8b5cf6", edgecolor="white", rot=30)
    ax.set_ylabel("Accident Count")
    ax.set_title("Accidents by Vehicle Type")
    save_fig(fig, "12_vehicle_vs_accidents")
    logger.info("Chart 12 saved.")


# ─────────────────────────────────────────────────────────────────────────────
# CHART 13 – Weather vs Accidents
# ─────────────────────────────────────────────────────────────────────────────
def plot_weather_vs_accidents(df):
    weather_acc = (
        df.groupby("Weather")
        .agg(total=("Accident_Flag", "count"), accidents=("Accident_Flag", "sum"))
        .assign(rate=lambda x: x["accidents"] / x["total"] * 100)
        .sort_values("rate", ascending=False)
    )
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(weather_acc.index, weather_acc["rate"], color="#f97316", edgecolor="white")
    ax.set_ylabel("Accident Rate (%)")
    ax.set_title("Accident Rate by Weather Condition")
    ax.set_xticklabels(weather_acc.index, rotation=30, ha="right")
    save_fig(fig, "13_weather_vs_accidents")
    logger.info("Chart 13 saved.")


# ─────────────────────────────────────────────────────────────────────────────
# CHART 14 – Weather vs Congestion
# ─────────────────────────────────────────────────────────────────────────────
def plot_weather_vs_congestion(df):
    wc = df.groupby("Weather")["Congestion_Level_%"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(wc.index, wc.values, color="#3b82d4", edgecolor="white")
    ax.set_ylabel("Avg Congestion Level (%)")
    ax.set_title("Average Congestion by Weather Condition")
    ax.set_xticklabels(wc.index, rotation=30, ha="right")
    save_fig(fig, "14_weather_vs_congestion")
    logger.info("Chart 14 saved.")


# ─────────────────────────────────────────────────────────────────────────────
# CHART 15 – Visibility vs Accident Severity
# ─────────────────────────────────────────────────────────────────────────────
def plot_visibility_vs_accidents(df):
    acc = df[df["Accident_Flag"] == 1]
    fig, ax = plt.subplots(figsize=(8, 4))
    for sev, color in zip(["Minor","Moderate","Severe","Fatal"],
                           ["#86efac","#fbbf24","#f97316","#ef4444"]):
        sub = acc[acc["Accident_Severity"] == sev]["Visibility_km"]
        ax.hist(sub, bins=20, alpha=0.65, label=sev, color=color, edgecolor="white")
    ax.set_xlabel("Visibility (km)")
    ax.set_ylabel("Count")
    ax.set_title("Visibility Distribution by Accident Severity")
    ax.legend(title="Severity")
    save_fig(fig, "15_visibility_vs_severity")
    logger.info("Chart 15 saved.")


# ─────────────────────────────────────────────────────────────────────────────
# CHART 16 – Road Condition vs Accidents
# ─────────────────────────────────────────────────────────────────────────────
def plot_road_condition_vs_accidents(df):
    rc = (
        df.groupby("Road_Condition")
        .agg(total=("Accident_Flag","count"), accidents=("Accident_Flag","sum"))
        .assign(rate=lambda x: x["accidents"]/x["total"]*100)
        .sort_values("rate", ascending=False)
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(rc.index, rc["rate"], color=["#ef4444","#f97316","#fbbf24","#86efac"][:len(rc)],
           edgecolor="white")
    ax.set_ylabel("Accident Rate (%)")
    ax.set_title("Accident Rate by Road Condition")
    ax.set_xticklabels(rc.index, rotation=20, ha="right")
    save_fig(fig, "16_road_condition_vs_accidents")
    logger.info("Chart 16 saved.")


# ─────────────────────────────────────────────────────────────────────────────
# CHART 17 – Traffic Signal Status vs Accidents
# ─────────────────────────────────────────────────────────────────────────────
def plot_signal_vs_accidents(df):
    sig = (
        df.groupby("Traffic_Signal_Status")
        .agg(total=("Accident_Flag","count"), accidents=("Accident_Flag","sum"))
        .assign(rate=lambda x: x["accidents"]/x["total"]*100)
        .sort_values("rate", ascending=False)
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(sig.index, sig["rate"], color=["#ef4444","#fbbf24","#86efac"][:len(sig)],
           edgecolor="white")
    ax.set_ylabel("Accident Rate (%)")
    ax.set_title("Accident Rate by Traffic Signal Status")
    save_fig(fig, "17_signal_vs_accidents")
    logger.info("Chart 17 saved.")


# ─────────────────────────────────────────────────────────────────────────────
# CHART 18 – Emergency Response Time by City
# ─────────────────────────────────────────────────────────────────────────────
def plot_emergency_by_city(df):
    acc = df[df["Accident_Flag"] == 1]
    ert = acc.groupby("City")["Emergency_Response_Min"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    ert.plot(kind="bar", ax=ax, color="#7c5cd8", edgecolor="white", rot=30)
    ax.axhline(15, color="red", linestyle="--", linewidth=1.2, label="15-min target")
    ax.set_ylabel("Avg Response Time (min)")
    ax.set_title("Average Emergency Response Time by City")
    ax.legend()
    save_fig(fig, "18_emergency_by_city")
    logger.info("Chart 18 saved.")


# ─────────────────────────────────────────────────────────────────────────────
# CHART 19 – Congestion vs Emergency Response
# ─────────────────────────────────────────────────────────────────────────────
def plot_congestion_vs_response(df):
    acc = df[df["Accident_Flag"] == 1]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(acc["Congestion_Level_%"], acc["Emergency_Response_Min"],
               alpha=0.15, s=6, color="#7c5cd8")
    # Trend line
    z = np.polyfit(acc["Congestion_Level_%"].dropna(),
                   acc["Emergency_Response_Min"].dropna(), 1)
    p = np.poly1d(z)
    xs = np.linspace(acc["Congestion_Level_%"].min(), acc["Congestion_Level_%"].max(), 200)
    ax.plot(xs, p(xs), color="#ef4444", linewidth=2, label="Trend")
    ax.set_xlabel("Congestion Level (%)")
    ax.set_ylabel("Emergency Response Time (min)")
    ax.set_title("Congestion vs Emergency Response Time")
    ax.legend()
    save_fig(fig, "19_congestion_vs_response")
    logger.info("Chart 19 saved.")


# ─────────────────────────────────────────────────────────────────────────────
# CHART 20 – Geographic Accident Hotspots
# ─────────────────────────────────────────────────────────────────────────────
def plot_geo_hotspots(df):
    acc = df[df["Accident_Flag"] == 1]
    no_acc = df[df["Accident_Flag"] == 0].sample(min(20000, len(df[df["Accident_Flag"]==0])),
                                                   random_state=42)
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(no_acc["Longitude"], no_acc["Latitude"],
               alpha=0.05, s=2, color="#3b82d4", label="No Accident")
    ax.scatter(acc["Longitude"], acc["Latitude"],
               alpha=0.15, s=4, color="#ef4444", label="Accident")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Geographic Distribution of Traffic Events & Accident Hotspots")
    ax.legend(markerscale=4)
    save_fig(fig, "20_geo_accident_hotspots")
    logger.info("Chart 20 saved.")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def run_eda():
    df = load_data()
    logger.info("Loaded processed data: %d rows × %d cols", *df.shape)

    plot_traffic_by_city(df)
    plot_traffic_by_hour(df)
    plot_traffic_by_day(df)
    plot_monthly_traffic(df)
    plot_speed_by_city(df)
    plot_congestion_by_city(df)
    plot_congestion_by_road(df)
    plot_accidents_by_city(df)
    plot_accidents_by_road(df)
    plot_accident_type(df)
    plot_accident_severity(df)
    plot_vehicle_vs_accidents(df)
    plot_weather_vs_accidents(df)
    plot_weather_vs_congestion(df)
    plot_visibility_vs_accidents(df)
    plot_road_condition_vs_accidents(df)
    plot_signal_vs_accidents(df)
    plot_emergency_by_city(df)
    plot_congestion_vs_response(df)
    plot_geo_hotspots(df)

    logger.info("EDA complete — 20 charts saved to reports/")


if __name__ == "__main__":
    run_eda()
