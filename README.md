# Smart City Traffic & Accident Analytics

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-green)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)](https://scikit-learn.org)
[![folium](https://img.shields.io/badge/Folium-Maps-red)](https://python-visualization.github.io/folium)

---

## Project Overview

A complete, portfolio-grade data analytics project analysing traffic patterns and road accidents across **10 major Indian cities** using **247,204 records** from 2025. The project demonstrates the full data analyst workflow:

```
Raw messy CSV  →  Data Cleaning  →  Feature Engineering  →  EDA (20 charts)
      →  Statistical Analysis  →  Machine Learning  →  Geospatial Maps
      →  Interactive Streamlit App  →  Business Recommendations
```

---

## Business Problem

Urban traffic congestion and road accidents are critical challenges for Indian cities. This project answers:

- Which cities and roads are the most dangerous?
- What are peak traffic hours and congestion hotspots?
- How do weather, road conditions, and signal status affect accidents?
- Which cities have the slowest emergency response?
- Can congestion and accident risk be predicted before they occur?

---

## Dataset

| Property | Value |
|----------|-------|
| File | `data/raw/smart_city_traffic_accident_messy_350k.csv` |
| Records | 247,204 rows |
| Columns | 24 raw → 48 after feature engineering |
| Cities | Bengaluru, Delhi, Hyderabad, Jaipur, Kanpur, Lucknow, Mumbai, Noida, Pune, Varanasi |
| Period | January–December 2025 |

### Raw Columns
`Record_ID`, `City`, `Area`, `Road_Name`, `Date`, `Time`, `Day_Type`, `Vehicle_Count`,
`Avg_Speed_kmph`, `Congestion_Level_%`, `Weather`, `Temperature_C`, `Visibility_km`,
`Vehicle_Type`, `Accident_Type`, `Accident_Severity`, `Emergency_Response_Min`,
`Road_Condition`, `Traffic_Signal_Status`, `Road_Closure`, `Public_Transport_Count`,
`Latitude`, `Longitude`, `Data_Source`

---

## Data Quality Issues Fixed

| Issue | Count | Treatment |
|-------|-------|-----------|
| Invalid speed (< 1 or > 150 km/h) | 73 | City-level median |
| Invalid congestion (< 0 or > 100%) | 101 | Dataset median |
| Temperature IQR outliers | 1,829 | Monthly median |
| Missing Vehicle_Count | 998 | Median imputation |
| Missing Avg_Speed_kmph | 1,463 | City-level median |
| Weather text variants (12 → 6) | ~50% rows | Normalised |
| Road condition variants (8 → 4) | ~50% rows | Normalised |
| Missing accident fields (no accident rows) | 82,139–82,455 | Correctly set to "None"/0 |
| Duplicate rows | 0 | None found |

---

## Technology Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.10+ |
| Data Cleaning | pandas, numpy |
| EDA & Visualisation | matplotlib, seaborn |
| Statistical Analysis | scipy |
| Machine Learning | scikit-learn |
| Geospatial Maps | folium, streamlit-folium |
| Analytics App | Streamlit |
| Real-time Pipeline | Python + REST APIs (OpenWeatherMap, TomTom) |
| Environment | python-dotenv |

---

## Folder Structure

```
Smart City Traffic & Accident Analytics/
│
├── data/
│   ├── raw/                    ← original messy CSV (do not modify)
│   ├── cleaned/                ← traffic_cleaned.csv
│   └── processed/              ← traffic_processed.csv + ML models (.pkl)
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_eda.ipynb
│   └── 04_ml_model.ipynb
│
├── python/
│   ├── utils.py                ← shared helpers & path constants
│   ├── data_cleaning.py        ← 9-step cleaning pipeline
│   ├── feature_engineering.py  ← 24 engineered features
│   ├── eda.py                  ← 20 EDA visualisations
│   ├── statistical_analysis.py ← 9 formal statistical tests
│   ├── prediction.py           ← 2 ML models (RF + GBM)
│   └── geospatial.py           ← 3 interactive maps + static chart
│
├── app/
│   ├── app.py                  ← Streamlit analytics app (6 tabs)
│   └── pipeline.py             ← real-time data pipeline (API)
│
├── reports/
│   ├── 01–20_*.png             ← 20 EDA charts
│   ├── ml_0*.png               ← 4 ML charts
│   ├── map_*.html              ← 3 interactive geospatial maps
│   ├── geo_static_hotspot.png  ← static hotspot chart
│   ├── FULL_PROJECT_REPORT.html← complete HTML report
│   ├── data_dictionary.md
│   ├── business_insights.md
│   ├── project_report.md
│   └── resume_and_interview_prep.md
│
├── requirements.txt
├── .env.example
└── README.md
```

---

## Data Cleaning Process

1. **Duplicates** — check and remove (0 found)
2. **Missing values** — column-by-column with business logic (not blind fill-all)
3. **Text normalisation** — 12 weather variants → 6; 8 road condition variants → 4
4. **Invalid speed** — values < 1 or > 150 km/h replaced with city median
5. **Invalid congestion** — values < 0 or > 100% replaced with overall median
6. **Temperature outliers** — IQR + domain rule (−5 to 50°C for Indian cities)
7. **Datetime parsing** — creates 10 time-based features including `Peak_Hour_Flag`
8. **Accident flag** — binary `0`/`1` from `Accident_Severity`
9. **Severity score** — ordinal encoding: None=0, Minor=1, Moderate=2, Severe=3, Fatal=4

---

## Feature Engineering (24 New Columns)

| Category | Features |
|----------|---------|
| Time | `Year`, `Month`, `Month_Name`, `Quarter`, `Week`, `Day`, `Day_Name`, `Hour`, `Peak_Hour_Flag`, `Weekday_Weekend` |
| Risk Flags | `Poor_Road_Flag`, `Faulty_Signal_Flag`, `Low_Visibility_Flag`, `Weather_Risk_Flag` |
| Categories | `Traffic_Category`, `Congestion_Category`, `Speed_Category`, `Emergency_Response_Category` |
| Targets | `Accident_Flag`, `Severity_Score`, `Severe_Accident_Flag` |
| Numeric | `Weekend_Flag` |
| Composite | `Traffic_Risk_Score` (weighted 0–10 scale) |

---

## EDA — 20 Charts Generated

| # | Chart |
|---|-------|
| 01 | Traffic volume by city |
| 02 | Hourly traffic pattern |
| 03 | Traffic by day of week |
| 04 | Monthly traffic trend |
| 05 | Average speed by city |
| 06 | Congestion by city |
| 07 | Congestion by road type |
| 08 | Accidents by city |
| 09 | Accidents by road type |
| 10 | Accident type distribution |
| 11 | Accident severity distribution |
| 12 | Vehicle type vs accidents |
| 13 | Weather vs accident rate |
| 14 | Weather vs congestion |
| 15 | Visibility vs accident severity |
| 16 | Road condition vs accident rate |
| 17 | Traffic signal status vs accidents |
| 18 | Emergency response by city |
| 19 | Congestion vs emergency response |
| 20 | Geographic accident hotspot scatter |

---

## Statistical Analysis — 9 Tests

| Test | Method | Result |
|------|--------|--------|
| Vehicle Count vs Congestion | Pearson | **r = 0.81**, p < 0.001 — SIGNIFICANT |
| Speed vs Congestion | Pearson | **r = −0.31**, p < 0.001 — SIGNIFICANT |
| Congestion vs EMS Response | Spearman | rho = −0.003, p = 0.24 |
| Weather vs Accidents | Chi-square | X2 = 7.52, p = 0.18 |
| Road Condition vs Accidents | Chi-square | X2 = 2.66, p = 0.45 |
| Signal Status vs Accidents | Chi-square | X2 = 2.65, p = 0.27 |
| Weather vs Congestion | ANOVA | F = 0.66, p = 0.66 |
| Peak vs Off-Peak Severity | Welch t-test | **t = 2.00, p = 0.046** — SIGNIFICANT |
| Visibility vs Severity | Spearman | rho = −0.001, p = 0.63 |

---

## Machine Learning

| Model | Algorithm | Target | CV Metric |
|-------|-----------|--------|-----------|
| Congestion Prediction | Random Forest Regressor | `Congestion_Level_%` | **CV R² = 0.654** |
| Accident Risk | Gradient Boosting Classifier | `Accident_Flag` | **CV F1 = 0.800** |

**Data leakage prevention:** All accident outcome columns (`Accident_Type`, `Accident_Severity`, `Emergency_Response_Min`) are excluded from ML features — they are observed **after** an accident, not before.

---

## Geospatial Analysis

Three interactive HTML maps saved to `reports/`:

| Map | Description |
|-----|-------------|
| `map_accident_hotspots.html` | 10,000 accident points coloured by severity + heatmap |
| `map_congestion_hotspots.html` | 45,048 high-congestion readings weighted heatmap |
| `map_high_risk_locations.html` | 2,982 very high risk (score ≥ 6) locations |

Open any map directly in a browser — fully interactive (zoom, click, tooltip).

---

## Key Insights (From Real Data)

1. **r = 0.81** — Vehicle count and congestion are strongly correlated. Reducing peak-hour vehicles by 20% would cut congestion by ~16%.
2. **18.15 min** average emergency response — above the 15-minute urban EMS target for all 10 cities.
3. **Peak hours are statistically more severe** (Welch t-test p = 0.046) — rush-hour accidents require enhanced monitoring.
4. **5.96% fatal rate** — 9,820 fatal accidents out of 164,749 total.
5. **Outer Ring Road** is the most congested road type at 40.7% average congestion.
6. **2,982 very high risk readings** (score ≥ 6) — priority locations for preventive intervention.

---

## Business Recommendations

1. Pre-position ambulances at high-risk locations during 07:00–10:00 and 17:00–20:00
2. Deploy adaptive signal control on Outer Ring Road (most congested)
3. Use Traffic_Risk_Score ≥ 6 as a real-time alert threshold
4. Stagger office/school start times to reduce Hour 08 peak (avg 1,232 vehicles/location)
5. Designate emergency vehicle corridors with signal pre-emption on NH-27 and MG Road
6. Prioritise road safety audits in Kanpur (highest accident count: 16,696)
7. Deploy variable message signs on expressways during low-visibility conditions
8. Invest in real-time API pipeline for live dashboard updates

---

## Streamlit App — 6 Tabs

| Tab | Content |
|-----|---------|
| Overview | 8 KPI cards, monthly traffic, accident trends, congestion comparison |
| Traffic Intelligence | Hourly pattern, speed by city, road congestion, vehicle mix, weekday/weekend |
| Accident Intelligence | Severity, type, vehicle, weather, road condition, EMS response charts |
| Hotspot Map | Interactive folium map with severity colouring, heatmap, and tooltips |
| ML Prediction | Real-time congestion prediction with sliders and coloured result card |
| Data Explorer | Filterable table, statistics, CSV download |

---

## How to Run

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run the full pipeline (all steps)
```bash
python python/data_cleaning.py          # clean raw CSV
python python/feature_engineering.py   # add 24 features
python python/eda.py                    # generate 20 charts
python python/statistical_analysis.py  # run 9 statistical tests
python python/prediction.py            # train 2 ML models
python python/geospatial.py            # generate 3 HTML maps
```

### Launch the Streamlit app
```bash
streamlit run app/app.py
```
Opens at **http://localhost:8501** — no upload needed, loads data automatically.

### Open Jupyter notebooks
```bash
pip install notebook
jupyter notebook
```

### Optional — Real-time data pipeline
```bash
cp .env.example .env          # add your API keys
python app/pipeline.py        # starts 15-min collection loop
```

---

## Reports & Documentation

| File | Description |
|------|-------------|
| `reports/FULL_PROJECT_REPORT.html` | Complete HTML report — open in browser |
| `reports/data_dictionary.md` | All 48 columns documented |
| `reports/business_insights.md` | 15 data-grounded business insights |
| `reports/project_report.md` | Full project summary |
| `reports/resume_and_interview_prep.md` | 3 resume bullets + 20 interview Q&As |

---

## Resume Bullet Points

**Bullet 1 — Python & Data Engineering**
> Designed and executed an end-to-end Python data pipeline (pandas, numpy) to clean, validate, and transform 247,000+ real-world traffic and accident records across 10 Indian cities, resolving 73 invalid speed values, 101 invalid congestion entries, and 1,829 temperature IQR outliers through statistically justified imputation strategies.

**Bullet 2 — EDA & Statistical Analysis**
> Performed comprehensive EDA generating 20 publication-quality visualisations and conducted 9 formal statistical tests (Pearson, Spearman, Chi-square, ANOVA, Welch t-test), discovering a vehicle-congestion correlation of r=0.81 and statistically significant peak-hour accident severity differences (p=0.046).

**Bullet 3 — Machine Learning & Streamlit App**
> Built 2 ML models (Random Forest CV R²=0.654 for congestion prediction; Gradient Boosting CV F1=0.800 for accident risk) with explicit data-leakage prevention; developed a 6-tab interactive Streamlit analytics application with live congestion prediction, folium geospatial hotspot maps, and filter-responsive KPI dashboards.

---

## Future Improvements

- LSTM time-series model for hourly congestion forecasting
- Expand to 50+ cities with OpenStreetMap integration
- Live Power BI auto-refresh via REST API
- Email/SMS alert system for risk score spikes
- CCTV stream integration for real vehicle counting
