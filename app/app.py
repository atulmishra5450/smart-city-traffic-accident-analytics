"""
app.py
------
Streamlit Analytics Application — Smart City Traffic & Accident Analytics
Loads data from data/processed/traffic_processed.csv.
If processed files are missing (e.g. on Streamlit Cloud first run),
the cleaning + feature-engineering pipeline runs automatically.

Run:
    streamlit run app/app.py
"""

import sys
import subprocess
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import folium
from streamlit_folium import st_folium
import joblib

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
ROOT           = Path(__file__).parent.parent
RAW_CSV        = ROOT / "data" / "smart_city_traffic_accident_messy_350k.csv"
PROCESSED_PATH = ROOT / "data" / "processed" / "traffic_processed.csv"
MODEL_RF_PATH  = ROOT / "data" / "processed" / "model_congestion_rf.pkl"
MODEL_GB_PATH  = ROOT / "data" / "processed" / "model_accident_gb.pkl"
SCALER_PATH    = ROOT / "data" / "processed" / "scaler_accident.pkl"


def _run_pipeline():
    """
    Run cleaning + feature-engineering when processed files are absent.
    Called automatically on Streamlit Cloud (or any fresh clone).
    """
    # Ensure output directories exist
    (ROOT / "data" / "cleaned").mkdir(parents=True, exist_ok=True)
    (ROOT / "data" / "processed").mkdir(parents=True, exist_ok=True)

    python_exe = sys.executable

    with st.spinner("First-time setup: cleaning raw data (this takes ~30 seconds)..."):
        result = subprocess.run(
            [python_exe, str(ROOT / "python" / "data_cleaning.py")],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            st.error("data_cleaning.py failed:\n" + result.stderr[-2000:])
            st.stop()

    with st.spinner("Building engineered features (~15 seconds)..."):
        result = subprocess.run(
            [python_exe, str(ROOT / "python" / "feature_engineering.py")],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            st.error("feature_engineering.py failed:\n" + result.stderr[-2000:])
            st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart City Traffic Analytics",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stSidebar"] { background: #f7f8fa; }
    .kpi-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 1rem 1rem 0.7rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .kpi-label { font-size: 0.72rem; color: #57606a; margin-bottom: 4px; text-transform: uppercase; letter-spacing: 0.04em; }
    .kpi-value { font-size: 1.55rem; font-weight: 700; color: #1f2328; line-height: 1.2; }
    .kpi-sub   { font-size: 0.7rem; color: #57606a; margin-top: 3px; }
    .kpi-card.red   .kpi-value { color: #dc2626; }
    .kpi-card.orange .kpi-value { color: #ea580c; }
    .kpi-card.green  .kpi-value { color: #16a34a; }
    .kpi-card.blue   .kpi-value { color: #2563eb; }
    .section-title {
        font-size: 1.05rem; font-weight: 700; color: #1f2328;
        border-left: 4px solid #3b82d4; padding-left: 0.6rem;
        margin: 1.2rem 0 0.7rem;
    }
    .insight-box {
        background: #f0f6ff; border-left: 4px solid #3b82d4;
        border-radius: 0 8px 8px 0; padding: 0.6rem 0.9rem;
        margin: 0.4rem 0; font-size: 0.84rem;
    }
    .insight-box.warn { background: #fffbeb; border-color: #f59e0b; }
    .insight-box.good { background: #f0fdf4; border-color: #22c55e; }
</style>
""", unsafe_allow_html=True)


def kpi(label, value, sub="", color=""):
    cls = f"kpi-card {color}".strip()
    return f"""<div class="{cls}">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub">{sub}</div>
    </div>"""


def section(title):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)


def insight(text, kind=""):
    st.markdown(f'<div class="insight-box {kind}">{text}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA (cached — loads once, stays in memory)
# Auto-runs pipeline if processed files are missing (Streamlit Cloud support)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading Smart City data...")
def load_data():
    if not PROCESSED_PATH.exists():
        if not RAW_CSV.exists():
            st.error(
                "Raw dataset not found. Expected at:\n"
                f"`{RAW_CSV}`\n\n"
                "Please ensure `data/smart_city_traffic_accident_messy_350k.csv` "
                "is present in the repository."
            )
            st.stop()
        # Auto-run the pipeline on first launch
        _run_pipeline()

    df = pd.read_csv(PROCESSED_PATH)
    # Ensure correct types
    df["Accident_Flag"]    = df["Accident_Flag"].astype(int)
    df["Severity_Score"]   = df["Severity_Score"].astype(int)
    df["Weekend_Flag"]     = df["Weekend_Flag"].astype(int)
    df["Peak_Hour_Flag"]   = df["Peak_Hour_Flag"].astype(int)
    df["Hour"]             = df["Hour"].astype(int)
    df["Month"]            = df["Month"].astype(int)
    return df


df_full = load_data()

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR FILTERS
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.image("https://via.placeholder.com/250x60/3b82d4/ffffff?text=Smart+City+Analytics",
                 use_container_width=True)
st.sidebar.markdown("### Filters")

cities   = ["All Cities"] + sorted(df_full["City"].unique().tolist())
weathers = ["All Weather"] + sorted(df_full["Weather"].unique().tolist())
vtypes   = ["All Vehicles"] + sorted(df_full["Vehicle_Type"].unique().tolist())

sel_city    = st.sidebar.selectbox("City",         cities)
sel_weather = st.sidebar.selectbox("Weather",      weathers)
sel_vtype   = st.sidebar.selectbox("Vehicle Type", vtypes)
sel_months  = st.sidebar.slider("Month Range", 1, 12, (1, 12), format="Month %d")
sel_acconly = st.sidebar.checkbox("Show Accidents Only", value=False)

# Apply filters
mask = pd.Series(True, index=df_full.index)
if sel_city    != "All Cities":   mask &= df_full["City"]         == sel_city
if sel_weather != "All Weather":  mask &= df_full["Weather"]      == sel_weather
if sel_vtype   != "All Vehicles": mask &= df_full["Vehicle_Type"] == sel_vtype
mask &= df_full["Month"].between(sel_months[0], sel_months[1])
if sel_acconly: mask &= df_full["Accident_Flag"] == 1

df = df_full[mask].copy()

st.sidebar.markdown("---")
st.sidebar.metric("Records", f"{len(df):,}")
st.sidebar.metric("Accidents", f"{int(df['Accident_Flag'].sum()):,}")
st.sidebar.markdown("---")
st.sidebar.markdown("**Dataset:** 247,204 records | 10 cities | 2025")

# Guard against empty filter
if len(df) == 0:
    st.warning("No records match your filters. Adjust the filters in the sidebar.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.title("🚦 Smart City Traffic & Accident Analytics")
st.caption(f"Showing **{len(df):,}** of **{len(df_full):,}** records  |  "
           f"City: **{sel_city}**  |  Weather: **{sel_weather}**  |  "
           f"Months: **{sel_months[0]}–{sel_months[1]}**")

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview",
    "🚗 Traffic Intelligence",
    "🚨 Accident Intelligence",
    "🗺️ Hotspot Map",
    "🤖 ML Prediction",
    "📋 Data Explorer",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — EXECUTIVE OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:

    # KPI Row 1
    total_v  = int(df["Vehicle_Count"].sum())
    total_a  = int(df["Accident_Flag"].sum())
    avg_sp   = float(df["Avg_Speed_kmph"].mean())
    avg_cg   = float(df["Congestion_Level_%"].mean())
    acc_rt   = total_a / len(df) * 100
    avg_em   = float(df.loc[df["Accident_Flag"]==1, "Emergency_Response_Min"].mean()) if total_a > 0 else 0
    fatal_n  = int((df["Accident_Severity"] == "Fatal").sum())
    risk_avg = float(df["Traffic_Risk_Score"].mean())

    c1,c2,c3,c4 = st.columns(4)
    c1.markdown(kpi("Total Vehicles",  f"{total_v/1e6:.2f}M",   "all readings",      "blue"),   unsafe_allow_html=True)
    c2.markdown(kpi("Total Accidents", f"{total_a:,}",           f"{acc_rt:.1f}% rate","red"),   unsafe_allow_html=True)
    c3.markdown(kpi("Avg Speed",       f"{avg_sp:.1f} km/h",     "urban average"),              unsafe_allow_html=True)
    c4.markdown(kpi("Avg Congestion",  f"{avg_cg:.1f}%",         "0–100 scale"),                unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c5,c6,c7,c8 = st.columns(4)
    c5.markdown(kpi("Accident Rate",   f"{acc_rt:.1f}%",         "of all records",    "orange"), unsafe_allow_html=True)
    c6.markdown(kpi("Avg EMS Response",f"{avg_em:.1f} min",      "target: 15 min",    "orange" if avg_em > 15 else "green"), unsafe_allow_html=True)
    c7.markdown(kpi("Fatal Accidents", f"{fatal_n:,}",            f"{fatal_n/total_a*100:.1f}% of accidents" if total_a > 0 else "", "red"), unsafe_allow_html=True)
    c8.markdown(kpi("Avg Risk Score",  f"{risk_avg:.2f}/10",      "composite index"),            unsafe_allow_html=True)

    st.markdown("---")

    # Charts row 1
    col_a, col_b = st.columns(2)

    with col_a:
        section("Monthly Traffic Volume")
        mon_order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        mdata = df.groupby("Month_Name")["Vehicle_Count"].sum() / 1e6
        mdata = mdata.reindex([m for m in mon_order if m in mdata.index])
        fig, ax = plt.subplots(figsize=(6, 3))
        bars = ax.bar(mdata.index, mdata.values, color="#3b82d4", edgecolor="white", linewidth=0.5)
        ax.set_ylabel("Vehicles (Millions)", fontsize=9)
        ax.set_title("Monthly Traffic Volume Trend", fontsize=10, fontweight="bold")
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_b:
        section("Accidents by City")
        city_a = df.groupby("City")["Accident_Flag"].sum().sort_values(ascending=False)
        fig2, ax2 = plt.subplots(figsize=(6, 3))
        colors_bar = ["#ef4444" if i == 0 else "#f97316" if i < 3 else "#3b82d4"
                      for i in range(len(city_a))]
        ax2.bar(city_a.index, city_a.values, color=colors_bar, edgecolor="white", linewidth=0.5)
        ax2.set_ylabel("Accidents", fontsize=9)
        ax2.set_title("Total Accidents by City", fontsize=10, fontweight="bold")
        ax2.tick_params(axis="x", rotation=30, labelsize=8)
        ax2.spines[["top","right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

    # Charts row 2
    col_c, col_d = st.columns(2)

    with col_c:
        section("Monthly Accident Trend")
        macc = df.groupby("Month_Name")["Accident_Flag"].sum()
        macc = macc.reindex([m for m in mon_order if m in macc.index])
        fig3, ax3 = plt.subplots(figsize=(6, 3))
        ax3.plot(macc.index, macc.values, color="#ef4444", linewidth=2.5, marker="o", ms=5)
        ax3.fill_between(range(len(macc)), macc.values, alpha=0.12, color="#ef4444")
        ax3.set_xticks(range(len(macc)))
        ax3.set_xticklabels(macc.index, rotation=45, fontsize=8)
        ax3.set_ylabel("Accidents", fontsize=9)
        ax3.set_title("Monthly Accident Trend", fontsize=10, fontweight="bold")
        ax3.spines[["top","right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()

    with col_d:
        section("City Congestion Comparison")
        city_cong = df.groupby("City")["Congestion_Level_%"].mean().sort_values(ascending=False)
        fig4, ax4 = plt.subplots(figsize=(6, 3))
        clrs = ["#ef4444" if v > 45 else "#f97316" if v > 40 else "#3b82d4"
                for v in city_cong.values]
        ax4.bar(city_cong.index, city_cong.values, color=clrs, edgecolor="white", linewidth=0.5)
        ax4.axhline(city_cong.mean(), color="#1f2328", linestyle="--", linewidth=1,
                    label=f"Avg {city_cong.mean():.1f}%")
        ax4.set_ylabel("Avg Congestion (%)", fontsize=9)
        ax4.set_title("Average Congestion by City", fontsize=10, fontweight="bold")
        ax4.tick_params(axis="x", rotation=30, labelsize=8)
        ax4.legend(fontsize=8)
        ax4.spines[["top","right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig4)
        plt.close()

    # Insights
    st.markdown("---")
    section("Key Insights")
    insight(f"<strong>Vehicle-Congestion Correlation r=0.81 (p&lt;0.001)</strong> — More vehicles directly drive congestion. Reducing peak-hour load by 20% would cut congestion by ~16%.", "good")
    insight(f"<strong>Average Emergency Response: {avg_em:.1f} min</strong> — {'Above' if avg_em > 15 else 'Within'} the 15-minute urban EMS target. Pre-positioning ambulances at high-risk locations during peak hours is the highest-impact intervention.", "warn" if avg_em > 15 else "good")
    insight(f"<strong>Fatal accidents: {fatal_n:,} ({fatal_n/total_a*100:.1f}% of accidents)</strong> — Severe + Fatal together represent the highest-priority emergency response cases.", "")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — TRAFFIC INTELLIGENCE
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    col_a, col_b = st.columns(2)

    with col_a:
        section("Hourly Traffic Pattern")
        hourly = df.groupby("Hour")["Vehicle_Count"].mean()
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(hourly.index, hourly.values, color="#3b82d4", linewidth=2.5, marker="o", ms=4)
        ax.fill_between(hourly.index, hourly.values, alpha=0.12, color="#3b82d4")
        # Shade peak hours
        for h_start, h_end in [(7, 10), (17, 20)]:
            ax.axvspan(h_start, h_end, alpha=0.12, color="#f59e0b", label="Peak hours" if h_start==7 else "")
        ax.set_xlabel("Hour of Day", fontsize=9)
        ax.set_ylabel("Avg Vehicle Count", fontsize=9)
        ax.set_title("Average Traffic by Hour of Day", fontsize=10, fontweight="bold")
        ax.set_xticks(range(0, 24, 2))
        ax.legend(fontsize=8)
        ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_b:
        section("Average Speed by City")
        spd = df.groupby("City")["Avg_Speed_kmph"].mean().sort_values(ascending=True)
        fig2, ax2 = plt.subplots(figsize=(6, 3))
        ax2.barh(spd.index, spd.values, color="#22c55e", edgecolor="white", linewidth=0.5)
        ax2.axvline(spd.mean(), color="#1f2328", linestyle="--", linewidth=1, label=f"Avg {spd.mean():.1f}")
        ax2.set_xlabel("Avg Speed (km/h)", fontsize=9)
        ax2.set_title("Average Speed by City", fontsize=10, fontweight="bold")
        ax2.legend(fontsize=8)
        ax2.spines[["top","right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

    col_c, col_d = st.columns(2)

    with col_c:
        section("Road-wise Congestion")
        road_cong = df.groupby("Road_Name")["Congestion_Level_%"].mean().sort_values(ascending=True)
        fig3, ax3 = plt.subplots(figsize=(6, 3))
        ax3.barh(road_cong.index, road_cong.values, color="#f97316", edgecolor="white", linewidth=0.5)
        ax3.set_xlabel("Avg Congestion (%)", fontsize=9)
        ax3.set_title("Average Congestion by Road Type", fontsize=10, fontweight="bold")
        ax3.spines[["top","right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()

    with col_d:
        section("Vehicle Type Distribution")
        vt = df["Vehicle_Type"].value_counts()
        colors_pie = ["#3b82d4","#22c55e","#f97316","#ef4444","#8b5cf6","#f59e0b","#06b6d4"]
        fig4, ax4 = plt.subplots(figsize=(6, 3))
        wedges, texts, autotexts = ax4.pie(
            vt.values, labels=vt.index, autopct="%1.1f%%",
            startangle=90, colors=colors_pie[:len(vt)],
            pctdistance=0.82, textprops={"fontsize": 8}
        )
        ax4.set_title("Vehicle Type Distribution", fontsize=10, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig4)
        plt.close()

    col_e, col_f = st.columns(2)

    with col_e:
        section("Weekday vs Weekend Traffic")
        wd = df.groupby("Weekend_Flag")["Vehicle_Count"].mean()
        wd.index = ["Weekday", "Weekend"]
        fig5, ax5 = plt.subplots(figsize=(5, 3))
        ax5.bar(wd.index, wd.values, color=["#3b82d4","#f59e0b"], edgecolor="white",
                linewidth=0.5, width=0.45)
        for i, (lbl, val) in enumerate(wd.items()):
            ax5.text(i, val + 5, f"{val:.0f}", ha="center", fontsize=9, fontweight="bold")
        ax5.set_ylabel("Avg Vehicles", fontsize=9)
        ax5.set_title("Weekday vs Weekend Traffic", fontsize=10, fontweight="bold")
        ax5.spines[["top","right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig5)
        plt.close()

    with col_f:
        section("Peak vs Off-Peak Traffic")
        pk = df.groupby("Peak_Hour_Flag")["Vehicle_Count"].agg(["mean","sum"])
        pk.index = ["Off-Peak", "Peak"]
        fig6, ax6 = plt.subplots(figsize=(5, 3))
        ax6.bar(pk.index, pk["mean"], color=["#94a3b8","#ef4444"], edgecolor="white",
                linewidth=0.5, width=0.45)
        ax6.set_ylabel("Avg Vehicles", fontsize=9)
        ax6.set_title("Peak vs Off-Peak Avg Traffic", fontsize=10, fontweight="bold")
        ax6.spines[["top","right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig6)
        plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ACCIDENT INTELLIGENCE
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    acc_df = df[df["Accident_Flag"] == 1].copy()

    if len(acc_df) == 0:
        st.warning("No accident records match the current filters.")
    else:
        # Top KPIs
        c1,c2,c3,c4 = st.columns(4)
        c1.markdown(kpi("Total Accidents",  f"{len(acc_df):,}",  "", "red"),    unsafe_allow_html=True)
        c2.markdown(kpi("Fatal Accidents",  f"{int((acc_df['Accident_Severity']=='Fatal').sum()):,}", f"{(acc_df['Accident_Severity']=='Fatal').mean()*100:.1f}%","red"), unsafe_allow_html=True)
        c3.markdown(kpi("Severe+Fatal",     f"{int((acc_df['Accident_Severity'].isin(['Severe','Fatal'])).sum()):,}", "high priority","orange"), unsafe_allow_html=True)
        c4.markdown(kpi("Avg Response Time",f"{acc_df['Emergency_Response_Min'].mean():.1f} min", "target 15 min", "orange" if acc_df['Emergency_Response_Min'].mean()>15 else "green"), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col_a, col_b = st.columns(2)

        with col_a:
            section("Accident Severity Distribution")
            sev = acc_df["Accident_Severity"].value_counts().reindex(["Minor","Moderate","Severe","Fatal"])
            sev = sev.fillna(0)
            fig, ax = plt.subplots(figsize=(6, 3))
            colors_sev = ["#86efac","#fbbf24","#f97316","#ef4444"]
            bars = ax.bar(sev.index, sev.values, color=colors_sev, edgecolor="white", linewidth=0.5)
            for bar, val in zip(bars, sev.values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                        f"{int(val):,}", ha="center", fontsize=8, fontweight="bold")
            ax.set_ylabel("Count", fontsize=9)
            ax.set_title("Accident Severity Distribution", fontsize=10, fontweight="bold")
            ax.spines[["top","right"]].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col_b:
            section("Accident Type Distribution")
            at = acc_df["Accident_Type"].value_counts().head(8)
            at = at[at.index != "None"]
            fig2, ax2 = plt.subplots(figsize=(6, 3))
            ax2.barh(at.index[::-1], at.values[::-1], color="#6366f1", edgecolor="white", linewidth=0.5)
            ax2.set_xlabel("Count", fontsize=9)
            ax2.set_title("Accident Types", fontsize=10, fontweight="bold")
            ax2.spines[["top","right"]].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()

        col_c, col_d = st.columns(2)

        with col_c:
            section("Accidents by Vehicle Type")
            veh_acc = acc_df["Vehicle_Type"].value_counts()
            fig3, ax3 = plt.subplots(figsize=(6, 3))
            ax3.bar(veh_acc.index, veh_acc.values, color="#8b5cf6", edgecolor="white", linewidth=0.5)
            ax3.set_ylabel("Accidents", fontsize=9)
            ax3.set_title("Accidents by Vehicle Type", fontsize=10, fontweight="bold")
            ax3.tick_params(axis="x", rotation=30, labelsize=8)
            ax3.spines[["top","right"]].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig3)
            plt.close()

        with col_d:
            section("Weather vs Accident Rate")
            w_acc = df.groupby("Weather").agg(
                total=("Accident_Flag","count"), acc=("Accident_Flag","sum")
            ).assign(rate=lambda x: x["acc"]/x["total"]*100).sort_values("rate", ascending=False)
            fig4, ax4 = plt.subplots(figsize=(6, 3))
            ax4.bar(w_acc.index, w_acc["rate"], color="#f97316", edgecolor="white", linewidth=0.5)
            ax4.axhline(w_acc["rate"].mean(), color="#1f2328", linestyle="--",
                        linewidth=1, label=f"Avg {w_acc['rate'].mean():.1f}%")
            ax4.set_ylabel("Accident Rate (%)", fontsize=9)
            ax4.set_title("Accident Rate by Weather", fontsize=10, fontweight="bold")
            ax4.tick_params(axis="x", rotation=25, labelsize=8)
            ax4.legend(fontsize=8)
            ax4.spines[["top","right"]].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig4)
            plt.close()

        col_e, col_f = st.columns(2)

        with col_e:
            section("Road Condition vs Accident Rate")
            rc = df.groupby("Road_Condition").agg(
                total=("Accident_Flag","count"), acc=("Accident_Flag","sum")
            ).assign(rate=lambda x: x["acc"]/x["total"]*100).sort_values("rate", ascending=False)
            fig5, ax5 = plt.subplots(figsize=(6, 3))
            cmap = {"Good":"#86efac","Fair":"#fbbf24","Poor":"#f97316","Under Construction":"#ef4444"}
            bar_colors = [cmap.get(i,"#3b82d4") for i in rc.index]
            ax5.bar(rc.index, rc["rate"], color=bar_colors, edgecolor="white", linewidth=0.5)
            ax5.set_ylabel("Accident Rate (%)", fontsize=9)
            ax5.set_title("Accident Rate by Road Condition", fontsize=10, fontweight="bold")
            ax5.tick_params(axis="x", rotation=20, labelsize=8)
            ax5.spines[["top","right"]].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig5)
            plt.close()

        with col_f:
            section("Emergency Response by City")
            ert = acc_df.groupby("City")["Emergency_Response_Min"].mean().sort_values(ascending=False)
            fig6, ax6 = plt.subplots(figsize=(6, 3))
            clr6 = ["#ef4444" if v > 18 else "#f97316" if v > 15 else "#22c55e"
                    for v in ert.values]
            ax6.bar(ert.index, ert.values, color=clr6, edgecolor="white", linewidth=0.5)
            ax6.axhline(15, color="#1f2328", linestyle="--", linewidth=1.2, label="15-min target")
            ax6.set_ylabel("Avg Response (min)", fontsize=9)
            ax6.set_title("Avg Emergency Response by City", fontsize=10, fontweight="bold")
            ax6.tick_params(axis="x", rotation=30, labelsize=8)
            ax6.legend(fontsize=8)
            ax6.spines[["top","right"]].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig6)
            plt.close()

        insight("<strong>Peak hour accidents are statistically more severe</strong> (Welch t-test p=0.046). Peak mean severity: 1.274 vs off-peak: 1.264. Deploy enhanced monitoring during 07:00–10:00 and 17:00–20:00.", "warn")
        insight("<strong>Fatal + Severe = 26.1% of all accidents</strong> (43,029 incidents). These require ambulance + trauma coordination. Use the High-Risk map to pre-position resources.", "")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — HOTSPOT MAP
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    section("Accident Hotspot Map")
    st.caption("Colour by severity: Green=Minor  Orange=Moderate  Red=Severe  DarkRed=Fatal  |  Max 3,000 points plotted")

    map_col, info_col = st.columns([3, 1])

    with info_col:
        st.markdown("**Map Controls**")
        map_type = st.radio("Map type", ["Accident Hotspots", "Risk Score", "Congestion"], index=0)
        max_pts = st.slider("Max points", 500, 5000, 2000, step=500)
        show_heat = st.checkbox("Show heatmap layer", value=True)

    with map_col:
        acc_map_df = df[df["Accident_Flag"] == 1][
            ["Latitude","Longitude","Accident_Severity","City","Road_Name",
             "Traffic_Risk_Score","Congestion_Level_%"]
        ].dropna()

        if len(acc_map_df) == 0:
            st.warning("No accident records with coordinates in current filter.")
        else:
            acc_map_df = acc_map_df.sample(min(max_pts, len(acc_map_df)), random_state=42)
            center = [acc_map_df["Latitude"].mean(), acc_map_df["Longitude"].mean()]
            m = folium.Map(location=center, zoom_start=10, tiles="CartoDB positron")

            if show_heat:
                from folium.plugins import HeatMap
                heat_data = acc_map_df[["Latitude","Longitude"]].values.tolist()
                HeatMap(heat_data, radius=10, blur=8,
                        gradient={0.4:"blue", 0.65:"orange", 1:"red"}).add_to(m)

            sev_colors = {"Minor":"green","Moderate":"orange","Severe":"red","Fatal":"darkred"}

            for _, row in acc_map_df.iterrows():
                color = sev_colors.get(str(row.get("Accident_Severity","")), "blue")
                folium.CircleMarker(
                    location=[row["Latitude"], row["Longitude"]],
                    radius=4, color=color, fill=True, fill_opacity=0.65,
                    tooltip=(f"<b>{row['City']}</b> | {row['Road_Name']}<br>"
                             f"Severity: <b>{row.get('Accident_Severity','?')}</b><br>"
                             f"Risk Score: {row.get('Traffic_Risk_Score','?')}")
                ).add_to(m)

            st_folium(m, width=750, height=480, returned_objects=[])

    insight("<strong>Open full interactive maps</strong> in browser: <code>reports/map_accident_hotspots.html</code>  |  <code>reports/map_congestion_hotspots.html</code>  |  <code>reports/map_high_risk_locations.html</code>", "")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — ML PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    pred_col, result_col = st.columns([1, 1])

    with pred_col:
        section("Congestion Level Predictor")
        st.caption("Random Forest model — trained on 247,204 records  |  CV R² = 0.654")

        vehicle_count    = st.slider("Vehicle Count",           100, 2000, 800, step=50)
        avg_speed_input  = st.slider("Average Speed (km/h)",    10,  130,  44,  step=1)
        hour_input       = st.slider("Hour of Day",             0,   23,   9)
        public_tr        = st.slider("Public Transport Count",  0,   256,  94,  step=5)
        weather_risk     = st.checkbox("High-Risk Weather  (Fog / Storm / Heavy Rain)", value=False)
        poor_road        = st.checkbox("Poor Road Condition  (Poor / Under Construction)", value=False)
        faulty_signal    = st.checkbox("Faulty Traffic Signal", value=False)
        weekend_flag     = st.checkbox("Weekend", value=False)

        predict_btn = st.button("Predict Congestion Level", type="primary", use_container_width=True)

    with result_col:
        section("Prediction Result")

        if predict_btn:
            if not MODEL_RF_PATH.exists():
                st.error("Model not found. Run: `python python/prediction.py`")
            else:
                model_rf = joblib.load(MODEL_RF_PATH)
                X_in = np.array([[vehicle_count, avg_speed_input, hour_input,
                                  int(weekend_flag), int(weather_risk), int(poor_road),
                                  int(faulty_signal), public_tr, 0, 0]])
                pred_val = float(np.clip(model_rf.predict(X_in)[0], 0, 100))

                if pred_val < 25:
                    level, color, emoji = "Free Flow",  "#16a34a", "🟢"
                elif pred_val < 50:
                    level, color, emoji = "Moderate",   "#ca8a04", "🟡"
                elif pred_val < 75:
                    level, color, emoji = "Heavy",      "#ea580c", "🟠"
                else:
                    level, color, emoji = "Gridlock",   "#dc2626", "🔴"

                st.markdown(f"""
                <div style="text-align:center;padding:2rem 1rem;background:#f7f8fa;
                            border-radius:12px;border:2px solid {color};margin-top:1rem">
                    <div style="font-size:3rem">{emoji}</div>
                    <div style="font-size:2.5rem;font-weight:800;color:{color}">{pred_val:.1f}%</div>
                    <div style="font-size:1.1rem;font-weight:600;color:{color}">{level}</div>
                    <div style="font-size:0.8rem;color:#57606a;margin-top:0.5rem">Predicted Congestion Level</div>
                </div>
                """, unsafe_allow_html=True)
                st.progress(int(pred_val))

                # Breakdown
                st.markdown("**Input Summary**")
                st.dataframe(pd.DataFrame({
                    "Feature": ["Vehicle Count","Avg Speed","Hour","Weekend",
                                "Weather Risk","Poor Road","Faulty Signal","Public Transport"],
                    "Value":   [vehicle_count, f"{avg_speed_input} km/h", f"{hour_input}:00",
                                "Yes" if weekend_flag else "No",
                                "Yes" if weather_risk else "No",
                                "Yes" if poor_road else "No",
                                "Yes" if faulty_signal else "No",
                                public_tr]
                }), use_container_width=True, hide_index=True)
        else:
            st.info("Set parameters on the left and click **Predict Congestion Level**")
            st.markdown("""
            **How the model works:**
            - Trained on 197,763 records (80% split)
            - 10 input features (no accident data — no leakage)
            - Random Forest with 80 trees, max depth 10
            - Cross-validated R² = **0.654** (3-fold)
            - MAE = **9.52%** congestion points
            """)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 — DATA EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════
with tab6:
    section("Dataset Explorer")

    col_info, col_shape = st.columns(2)
    with col_info:
        st.metric("Filtered Records",  f"{len(df):,}")
        st.metric("Total Records",     f"{len(df_full):,}")
    with col_shape:
        st.metric("Columns",           f"{df.shape[1]}")
        st.metric("Accident Records",  f"{int(df['Accident_Flag'].sum()):,}")

    st.markdown("---")

    # Column selector
    default_cols = ["City","Area","Road_Name","Hour","Day_Name","Vehicle_Count",
                    "Congestion_Level_%","Avg_Speed_kmph","Weather","Accident_Flag",
                    "Accident_Severity","Emergency_Response_Min","Traffic_Risk_Score"]
    default_cols = [c for c in default_cols if c in df.columns]

    cols_show = st.multiselect(
        "Select columns to display (showing first 500 rows)",
        options=df.columns.tolist(),
        default=default_cols
    )

    if cols_show:
        st.dataframe(df[cols_show].head(500), use_container_width=True)

    st.markdown("---")
    section("Quick Statistics")
    num_cols = st.multiselect(
        "Select numeric columns for stats",
        options=df.select_dtypes(include="number").columns.tolist(),
        default=["Vehicle_Count","Congestion_Level_%","Avg_Speed_kmph",
                 "Emergency_Response_Min","Traffic_Risk_Score"]
    )
    if num_cols:
        st.dataframe(df[num_cols].describe().round(2), use_container_width=True)

    # Download button
    st.markdown("---")
    csv_out = df[cols_show].to_csv(index=False).encode("utf-8") if cols_show else b""
    st.download_button(
        label="Download Filtered Data as CSV",
        data=csv_out,
        file_name="smart_city_filtered.csv",
        mime="text/csv",
        use_container_width=True,
    )
