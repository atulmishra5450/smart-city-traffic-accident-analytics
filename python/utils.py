"""
utils.py
--------
Shared utility functions used across the Smart City Traffic & Accident Analytics project.
"""

import os
import logging
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# ── Logging ──────────────────────────────────────────────────────────────────

def get_logger(name: str = "smart_city") -> logging.Logger:
    """Return a consistently configured logger."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  [%(levelname)s]  %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(name)


# ── Path helpers ──────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent          # project root
RAW_DATA    = ROOT / "data" / "raw"
CLEANED_DATA = ROOT / "data" / "cleaned"
PROCESSED_DATA = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"

def ensure_dirs():
    """Create all project data directories if they do not exist."""
    for d in [RAW_DATA, CLEANED_DATA, PROCESSED_DATA, REPORTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


# ── DataFrame helpers ─────────────────────────────────────────────────────────

def load_raw(filename: str = "smart_city_traffic_accident_messy_350k.csv") -> pd.DataFrame:
    """Load the raw dataset from data/raw/."""
    path = RAW_DATA / filename
    df = pd.read_csv(path)
    get_logger().info("Loaded raw data: %s rows × %s cols", *df.shape)
    return df


def save_cleaned(df: pd.DataFrame, filename: str = "traffic_cleaned.csv") -> Path:
    """Save the cleaned dataset to data/cleaned/."""
    path = CLEANED_DATA / filename
    df.to_csv(path, index=False)
    get_logger().info("Saved cleaned data → %s", path)
    return path


def save_processed(df: pd.DataFrame, filename: str = "traffic_processed.csv") -> Path:
    """Save the feature-engineered dataset to data/processed/."""
    path = PROCESSED_DATA / filename
    df.to_csv(path, index=False)
    get_logger().info("Saved processed data → %s", path)
    return path


# ── Plotting helpers ──────────────────────────────────────────────────────────

PALETTE = "Blues_d"
ACCENT  = "#3b82d4"
FIG_DPI = 120

def set_style():
    """Apply a consistent, clean plot style across all visualisations."""
    sns.set_theme(style="whitegrid", palette="Blues_d", font_scale=1.05)
    plt.rcParams.update({
        "figure.dpi": FIG_DPI,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
    })


def save_fig(fig: plt.Figure, name: str, folder: str = "reports") -> Path:
    """Save a matplotlib figure to reports/ as a PNG."""
    out = ROOT / folder / f"{name}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out
