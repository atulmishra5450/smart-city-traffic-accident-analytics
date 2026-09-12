"""
geospatial.py
-------------
Geospatial analysis for Smart City Traffic & Accident Analytics.
Creates interactive maps and static heatmaps showing:
  1. Accident hotspots
  2. Congestion hotspots
  3. High-risk location clusters

Requirements:
    pip install folium

Output:
    reports/map_accident_hotspots.html
    reports/map_congestion_hotspots.html
    reports/map_high_risk.html

Run standalone:
    python python/geospatial.py
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from utils import get_logger, set_style, save_fig, PROCESSED_DATA

logger = get_logger("geo")
set_style()

REPORTS = Path(__file__).parent.parent / "reports"
REPORTS.mkdir(exist_ok=True)


def load_data():
    return pd.read_csv(PROCESSED_DATA / "traffic_processed.csv")


# ─────────────────────────────────────────────────────────────────────────────
# 1. ACCIDENT HOTSPOT MAP (Interactive — Folium)
# ─────────────────────────────────────────────────────────────────────────────

def create_accident_hotspot_map(df: pd.DataFrame) -> str:
    """
    Creates an interactive folium map of accident locations.
    Color-coded by severity:
        Minor = green, Moderate = orange, Severe = red, Fatal = darkred

    City authorities can use this map to:
    - Identify clustering of accidents in specific neighbourhoods
    - Prioritise road safety interventions at dense clusters
    - Correlate hotspots with road geometry and signal locations

    Returns path to the saved HTML map.
    """
    try:
        import folium
        from folium.plugins import HeatMap
    except ImportError:
        logger.warning("folium not installed. Run: pip install folium")
        return ""

    acc = df[df["Accident_Flag"] == 1].dropna(subset=["Latitude","Longitude"])
    acc = acc.sample(min(10000, len(acc)), random_state=42)

    center = [acc["Latitude"].mean(), acc["Longitude"].mean()]
    m = folium.Map(location=center, zoom_start=10, tiles="CartoDB positron")

    severity_colors = {
        "Minor":    "green",
        "Moderate": "orange",
        "Severe":   "red",
        "Fatal":    "darkred",
    }

    for _, row in acc.iterrows():
        color = severity_colors.get(str(row.get("Accident_Severity","")), "blue")
        folium.CircleMarker(
            location=[row["Latitude"], row["Longitude"]],
            radius=4,
            color=color,
            fill=True,
            fill_opacity=0.55,
            tooltip=(f"City: {row['City']} | Road: {row['Road_Name']} "
                     f"| Severity: {row.get('Accident_Severity','?')} "
                     f"| Risk: {row.get('Traffic_Risk_Score','?')}")
        ).add_to(m)

    # Heatmap layer
    heat_data = acc[["Latitude","Longitude"]].values.tolist()
    HeatMap(heat_data, radius=12, blur=8, gradient={0.4:"blue",0.65:"orange",1:"red"}).add_to(m)

    # Legend
    legend_html = """
    <div style="position:fixed;bottom:50px;left:50px;z-index:1000;
                background:#fff;padding:10px;border:1px solid #ccc;font-size:12px;">
    <b>Accident Severity</b><br>
    <span style="color:green">&#9679;</span> Minor<br>
    <span style="color:orange">&#9679;</span> Moderate<br>
    <span style="color:red">&#9679;</span> Severe<br>
    <span style="color:darkred">&#9679;</span> Fatal<br>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    out = REPORTS / "map_accident_hotspots.html"
    m.save(str(out))
    logger.info("Accident hotspot map saved: %s", out)
    return str(out)


# ─────────────────────────────────────────────────────────────────────────────
# 2. CONGESTION HOTSPOT MAP (Interactive — Folium)
# ─────────────────────────────────────────────────────────────────────────────

def create_congestion_hotspot_map(df: pd.DataFrame) -> str:
    """
    Interactive map showing congestion intensity across the dataset.
    Colour intensity corresponds to Congestion_Level_%.
    """
    try:
        import folium
        from folium.plugins import HeatMap
    except ImportError:
        logger.warning("folium not installed.")
        return ""

    high_cong = df[df["Congestion_Level_%"] >= 60].dropna(subset=["Latitude","Longitude"])
    high_cong = high_cong.sample(min(8000, len(high_cong)), random_state=42)

    center = [df["Latitude"].median(), df["Longitude"].median()]
    m = folium.Map(location=center, zoom_start=10, tiles="CartoDB dark_matter")

    # Weight heatmap by congestion level
    heat_data = [
        [row["Latitude"], row["Longitude"], row["Congestion_Level_%"] / 100]
        for _, row in high_cong.iterrows()
    ]
    HeatMap(heat_data, radius=14, blur=10,
            gradient={0.4:"blue",0.65:"yellow",1:"red"}).add_to(m)

    out = REPORTS / "map_congestion_hotspots.html"
    m.save(str(out))
    logger.info("Congestion hotspot map saved: %s", out)
    return str(out)


# ─────────────────────────────────────────────────────────────────────────────
# 3. HIGH-RISK LOCATION MAP
# ─────────────────────────────────────────────────────────────────────────────

def create_risk_map(df: pd.DataFrame) -> str:
    """
    Map showing locations with Traffic_Risk_Score >= 6 (Very High Risk).
    These are the highest-priority locations for city authority intervention.

    How city authorities use this:
    - Identify which specific intersections / road segments need immediate attention
    - Allocate patrol officers or mobile speed cameras to these locations
    - Cross-reference with infrastructure maintenance schedules
    """
    try:
        import folium
    except ImportError:
        logger.warning("folium not installed.")
        return ""

    high_risk = df[df["Traffic_Risk_Score"] >= 6].dropna(subset=["Latitude","Longitude"])
    high_risk = high_risk.sample(min(5000, len(high_risk)), random_state=42)

    center = [df["Latitude"].median(), df["Longitude"].median()]
    m = folium.Map(location=center, zoom_start=10, tiles="CartoDB positron")

    for _, row in high_risk.iterrows():
        folium.CircleMarker(
            location=[row["Latitude"], row["Longitude"]],
            radius=5,
            color="#ef4444",
            fill=True,
            fill_opacity=0.5,
            tooltip=(f"Risk Score: {row['Traffic_Risk_Score']:.1f} "
                     f"| City: {row['City']} | Area: {row['Area']}")
        ).add_to(m)

    out = REPORTS / "map_high_risk_locations.html"
    m.save(str(out))
    logger.info("High-risk map saved: %s", out)
    return str(out)


# ─────────────────────────────────────────────────────────────────────────────
# 4. STATIC HOTSPOT SCATTER (Matplotlib — for reports/PDF)
# ─────────────────────────────────────────────────────────────────────────────

def create_static_hotspot_map(df: pd.DataFrame):
    """
    Publication-quality static map showing accident density by severity.
    Uses lat/lon scatter — city boundaries visible from point clusters.
    """
    acc = df[df["Accident_Flag"] == 1].dropna(subset=["Latitude","Longitude"])
    no_acc = df[df["Accident_Flag"] == 0].dropna(subset=["Latitude","Longitude"])
    no_acc = no_acc.sample(min(20000, len(no_acc)), random_state=42)

    fig, ax = plt.subplots(figsize=(12, 9))

    # Background traffic readings
    ax.scatter(no_acc["Longitude"], no_acc["Latitude"],
               alpha=0.03, s=2, color="#94a3b8", label="No Accident", zorder=1)

    # Accidents by severity
    colors = {"Minor":"#86efac","Moderate":"#fbbf24","Severe":"#f97316","Fatal":"#ef4444"}
    for sev, color in colors.items():
        sub = acc[acc["Accident_Severity"] == sev]
        ax.scatter(sub["Longitude"], sub["Latitude"],
                   alpha=0.2, s=5, color=color, label=sev, zorder=2)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Geographic Accident Hotspot Map (247K Records, 10 Cities)")
    ax.legend(title="Severity", markerscale=4, framealpha=0.9)
    save_fig(fig, "geo_static_hotspot")
    logger.info("Static hotspot map saved.")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = load_data()
    logger.info("Loaded: %d rows", len(df))

    create_static_hotspot_map(df)
    create_accident_hotspot_map(df)
    create_congestion_hotspot_map(df)
    create_risk_map(df)

    logger.info(
        "\nGeospatial analysis complete.\n"
        "Open the HTML maps in a browser for interactive exploration:\n"
        "  reports/map_accident_hotspots.html\n"
        "  reports/map_congestion_hotspots.html\n"
        "  reports/map_high_risk_locations.html\n\n"
        "HOW CITY AUTHORITIES USE THESE MAPS:\n"
        "  - Accident hotspot map: identify clusters -> priority road safety investments\n"
        "  - Congestion map: identify gridlock zones -> signal timing adjustments\n"
        "  - Risk map: pre-empt incidents -> ambulance pre-positioning, patrol deployment"
    )
