# Smart City Traffic & Accident Analytics
## Full Project Report

---

## 1. Project Overview

A complete, end-to-end data analytics solution analysing traffic patterns and accident data across 10 major Indian cities. It covers the full data analyst workflow from raw messy CSV to an interactive Streamlit dashboard with geospatial maps and machine learning predictions.

**Dataset:** 247,204 rows × 24 columns | Year 2025 | 10 cities

---

## 2. Business Problem

Urban traffic congestion and road accidents are growing challenges in Indian cities. City transport authorities need data-driven answers to:

- Which roads and areas are the most dangerous?
- When and where do accidents most frequently occur?
- What environmental factors drive accident risk?
- How can emergency response times be improved?
- Which locations should receive priority investment?

---

## 3. Data Quality Issues Found & Fixed

| Issue | Count | Treatment |
|-------|-------|-----------|
| Invalid speeds (< 1 or > 150 km/h) | 73 | Replaced with city-median |
| Invalid congestion (< 0 or > 100%) | 101 | Replaced with dataset median |
| Temperature IQR outliers | 1,829 | Replaced with monthly median |
| Missing Vehicle_Count | 998 | Median imputation |
| Missing Avg_Speed_kmph | 1,463 | City-level median imputation |
| Inconsistent weather text (12 variants → 6) | ~50% | Normalised |
| Inconsistent road condition text (8 variants → 4) | ~50% | Normalised |
| Missing Accident_Type where no accident | 82,139 | Set to "None" (correct behaviour) |

---

## 4. Feature Engineering

24 new analytical columns created:

- **Time features (10):** Year, Month, Quarter, Week, Day, Hour, Day_Name, Month_Name, Peak_Hour_Flag, Weekday_Weekend
- **Risk flags (4):** Poor_Road_Flag, Faulty_Signal_Flag, Low_Visibility_Flag, Weather_Risk_Flag
- **Category columns (4):** Traffic_Category, Congestion_Category, Speed_Category, Emergency_Response_Category
- **Target variables (3):** Accident_Flag (binary), Severity_Score (0–4), Severe_Accident_Flag
- **Numeric flag (1):** Weekend_Flag
- **Composite score (1):** Traffic_Risk_Score (weighted 0–10 scale)
- **Other (1):** Datetime (combined timestamp)

---

## 5. EDA Summary

20 professional visualisations generated covering:

- Traffic volume by city, hour, day, month
- Speed and congestion analysis
- Accident distribution by city, road, type, severity
- Weather, visibility, and road condition impact
- Emergency response analysis
- Geographic hotspot mapping

All charts saved as PNG files in `reports/`.

---

## 6. Statistical Analysis

| Analysis | Method | Finding |
|----------|---------|---------|
| Vehicle count vs congestion | Pearson correlation | **r = 0.81**, p < 0.001 — SIGNIFICANT |
| Speed vs congestion | Pearson correlation | **r = −0.31**, p < 0.001 — SIGNIFICANT |
| Congestion vs emergency response | Spearman correlation | rho = −0.003, p = 0.24 |
| Weather vs accident frequency | Chi-square test | X2 = 7.52, p = 0.18 |
| Road condition vs accident frequency | Chi-square test | X2 = 2.66, p = 0.45 |
| Faulty signals vs accidents | Chi-square test | X2 = 2.65, p = 0.27 |
| Weather vs congestion | ANOVA | F = 0.66, p = 0.66 |
| Peak vs off-peak accident severity | Welch t-test | **t = 2.00, p = 0.046** — SIGNIFICANT |
| Visibility vs accident severity | Spearman correlation | rho = −0.001, p = 0.63 |

---

## 7. Machine Learning

**Model 1 — Traffic Congestion Prediction (Random Forest Regressor)**
- Target: Congestion_Level_%
- Features: 10 operational + environmental variables
- CV R² = 0.654 (3-fold cross-validation)
- MAE = 9.52% congestion points
- Baseline (Linear Regression) R² = 0.650

**Model 2 — Accident Risk Classification (Gradient Boosting Classifier)**
- Target: Accident_Flag (0/1)
- Features: 12 risk-relevant variables
- CV F1 = 0.800 (3-fold)
- ROC-AUC = 0.50
- Data leakage prevention: All post-accident fields excluded

---

## 8. Geospatial Analysis

Three interactive HTML maps generated in `reports/`:

| Map | Points | Description |
|-----|--------|-------------|
| `map_accident_hotspots.html` | 10,000 | Accident points coloured by severity + heatmap |
| `map_congestion_hotspots.html` | 45,048 | High-congestion readings weighted heatmap |
| `map_high_risk_locations.html` | 2,982 | Very high risk (score ≥ 6) locations |

---

## 9. Streamlit App

6-tab interactive application (`app/app.py`):

| Tab | Content |
|-----|---------|
| Overview | 8 KPI cards + 4 charts (monthly trends, city comparison) |
| Traffic Intelligence | 6 charts (hourly, speed, road congestion, vehicle mix) |
| Accident Intelligence | 4 KPI cards + 6 charts (severity, type, weather, EMS) |
| Hotspot Map | Live folium map with heatmap and severity tooltips |
| ML Prediction | Congestion predictor with sliders and coloured result |
| Data Explorer | Filter table, statistics, CSV download |

Loads data automatically from `data/processed/traffic_processed.csv` — no upload needed.

---

## 10. Key Business Recommendations

1. Pre-position ambulances at 2,982 high-risk-score locations during peak hours
2. Deploy adaptive signal control on Outer Ring Road (most congested at 40.7%)
3. Use Traffic_Risk_Score ≥ 6 as a real-time alert threshold
4. Stagger office/school hours to reduce Hour 08 peak load
5. Designate emergency vehicle corridors with signal pre-emption
6. Prioritise road safety audits in Kanpur (highest accident count: 16,696)
7. Deploy variable message signs during low-visibility conditions
8. Invest in real-time API data pipeline for live monitoring

---

## 11. Technology Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.10+ |
| Data Cleaning | pandas, numpy |
| EDA & Visualisation | matplotlib, seaborn |
| Statistical Analysis | scipy |
| Machine Learning | scikit-learn |
| Geospatial Maps | folium, streamlit-folium |
| Analytics App | Streamlit |
| Real-time Pipeline | Python + REST APIs |
| Environment | python-dotenv |

---

## 12. Reproducibility

All scripts are modular and can be run in sequence:

```bash
# Full pipeline
python python/data_cleaning.py
python python/feature_engineering.py
python python/eda.py
python python/statistical_analysis.py
python python/prediction.py
python python/geospatial.py

# Launch app
streamlit run app/app.py
```
