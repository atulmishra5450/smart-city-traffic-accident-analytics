# Data Dictionary — Smart City Traffic & Accident Analytics

**Dataset:** `smart_city_traffic_accident_messy_350k.csv`  
**Rows:** 247,204 | **Columns:** 24 (raw) → 48 (processed)  
**Date Range:** 2025-01-01 to 2025-12-31  
**Cities:** Bengaluru, Delhi, Hyderabad, Jaipur, Kanpur, Lucknow, Mumbai, Noida, Pune, Varanasi

---

## Raw Columns

| Column | Data Type | Description | Example | Business Meaning | Cleaning Rule | Used For |
|--------|-----------|-------------|---------|-----------------|---------------|----------|
| `Record_ID` | int | Unique identifier for each traffic reading | 1, 2, 3 | Primary key for joining and deduplication | Validate uniqueness; no imputation | All joins, deduplication |
| `City` | string | Name of the Indian city where the reading was taken | Lucknow, Delhi | Geographic dimension for city-level analysis | No missing values; no cleaning needed | City-level KPIs, filters |
| `Area` | string | Neighbourhood or locality within the city | Alambagh, Saket | Sub-city drill-down for hotspot analysis | No missing values; no cleaning needed | Area-level analysis |
| `Road_Name` | string | Type/name of road | Outer Ring Road, MG Road | Identifies which type of road has highest risk | No cleaning needed (8 unique values) | Road-level KPIs |
| `Date` | string (date) | Date of traffic reading | 2025-05-15 | Enables time-series analysis | Parsed to datetime; no missing values | Date dimension, trends |
| `Time` | string (time) | Time of traffic reading | 09:01:00 | Enables hourly analysis and peak-hour detection | Parsed to datetime; no missing values | Hour extraction, peak flag |
| `Day_Type` | string | Weekday or Weekend | Weekday, Weekend | Identifies weekend vs workday traffic behaviour | Title-case normalisation | Weekend comparisons |
| `Vehicle_Count` | float→int | Number of vehicles recorded in the reading period | 1152 | Core traffic volume metric | 998 missing → median imputation | Traffic volume KPIs |
| `Avg_Speed_kmph` | float | Average speed of vehicles | 51.2 | Inversely related to congestion | 1,463 missing → city-median; 73 invalid (<1 or >150) → city-median | Speed analysis, ML feature |
| `Congestion_Level_%` | float | Traffic congestion as a percentage (0–100) | 38.7 | Primary congestion KPI | 1 missing → median; 101 invalid (neg or >100) → median | Congestion analysis, ML target |
| `Weather` | string | Weather condition at time of reading | Clear, Fog, Heavy Rain | Environmental factor affecting traffic and accidents | 12 variants → normalised to 6 standard values | Weather analysis, risk flag |
| `Temperature_C` | float | Air temperature in Celsius | 31.1 | Extreme temperatures affect driver behaviour | 1,829 IQR outliers → monthly median | Statistical analysis |
| `Visibility_km` | float | Visibility distance in kilometres | 7.7 | Low visibility increases accident risk | 1 missing → weather-group median | Visibility flag, ML feature |
| `Vehicle_Type` | string | Type of vehicle | Car, Bike, Truck | Different vehicles have different risk profiles | 1 missing → mode | Vehicle analysis |
| `Accident_Type` | string | Type of accident that occurred | Head-on, Rear-end, None | Categorises accident patterns | 82,139 NaN (no accident) → "None"; variants normalised | Accident type analysis |
| `Accident_Severity` | string | Severity of the accident | Minor, Moderate, Severe, Fatal, None | Critical for emergency prioritisation | 82,455 NaN = no accident (correct) | Severity analysis, ML target |
| `Emergency_Response_Min` | float | Time taken for emergency services to respond | 12.1 | Measures emergency service efficiency | 0 where no accident; median for accident rows | Response time KPIs |
| `Road_Condition` | string | Quality of the road surface | Good, Fair, Poor, Under Construction | Poor roads increase accident probability | 8 variants → normalised to 4 standard values | Risk analysis |
| `Traffic_Signal_Status` | string | Status of traffic signal at the location | Working, Faulty, Maintenance | Faulty signals disrupt orderly traffic flow | 1 missing → mode | Signal analysis |
| `Road_Closure` | string | Whether the road is closed | Yes, No | Road closures divert traffic, increasing congestion | 1 missing → "No" (conservative) | Congestion analysis |
| `Public_Transport_Count` | float→int | Number of public transport vehicles in the reading | 41 | Higher PT reduces private vehicle demand | 1 missing → median | Transport planning |
| `Latitude` | float | Geographic latitude of the reading | 27.086475 | Required for geospatial mapping | 1 missing → area median | Maps, hotspot analysis |
| `Longitude` | float | Geographic longitude of the reading | 80.968068 | Required for geospatial mapping | 1 missing → area median | Maps, hotspot analysis |
| `Data_Source` | string | Source of the traffic reading | Police Report, GPS, CCTV | Data provenance and quality tracking | 1 missing → "Unknown" | Data quality monitoring |

---

## Engineered Columns (added during processing)

| Column | Type | Description | Business Value |
|--------|------|-------------|----------------|
| `Datetime` | datetime | Combined Date + Time as proper timestamp | Enables time-series operations |
| `Year` | int | Year of reading (2025) | Year-level aggregations |
| `Month` | int | Month number (1–12) | Monthly trends |
| `Month_Name` | string | Month abbreviation | Dashboard labels |
| `Quarter` | int | Quarter (1–4) | Quarterly reporting |
| `Week` | int | ISO week number | Weekly patterns |
| `Day` | int | Day of month | Calendar analysis |
| `Day_Name` | string | Name of day | Weekend identification |
| `Hour` | int | Hour of day (0–23) | Peak hour analysis |
| `Peak_Hour_Flag` | bit | 1 if hour is 7–10 or 17–20 | Rush-hour filtering |
| `Weekday_Weekend` | string | "Weekday" or "Weekend" | Period comparison |
| `Accident_Flag` | bit | 1 = accident occurred, 0 = no accident | Binary ML target, accident count KPI |
| `Severity_Score` | int (0–4) | None=0, Minor=1, Moderate=2, Severe=3, Fatal=4 | Weighted severity analysis, regression target |
| `Traffic_Category` | string | Low / Moderate / High / Very High | Readable traffic segmentation |
| `Congestion_Category` | string | Free Flow / Moderate / Heavy / Gridlock | Traffic management tiers |
| `Speed_Category` | string | Very Slow / Slow / Normal / Fast | Speed segmentation |
| `Severe_Accident_Flag` | bit | 1 if Severe or Fatal | High-priority incident filtering |
| `Weekend_Flag` | bit | 1 = weekend | ML numeric feature |
| `Poor_Road_Flag` | bit | 1 if Road_Condition is Poor or Under Construction | Infrastructure risk indicator |
| `Faulty_Signal_Flag` | bit | 1 if Traffic_Signal_Status is Faulty | Signal management KPI |
| `Low_Visibility_Flag` | bit | 1 if Visibility_km ≤ 3 | Hazardous visibility indicator |
| `Weather_Risk_Flag` | bit | 1 if Weather is Fog, Heavy Rain, or Storm | Adverse weather indicator |
| `Emergency_Response_Category` | string | No Accident / Excellent / Acceptable / Delayed / Critical | Response performance tier |
| `Traffic_Risk_Score` | float (0–10) | Composite risk score from 5 weighted factors | Holistic risk ranking |

---

## Data Quality Summary (Raw Dataset)

| Issue | Count | Treatment |
|-------|-------|-----------|
| Duplicate rows | 0 | N/A |
| Missing `Vehicle_Count` | 998 (0.4%) | Median imputation |
| Missing `Avg_Speed_kmph` | 1,463 (0.59%) | City-level median |
| Invalid speed (< 1 or > 150 km/h) | 73 | City-level median from valid rows |
| Invalid congestion (< 0 or > 100) | 101 | Overall median |
| Temperature IQR outliers | 1,829 | Monthly median |
| Missing `Accident_Type` / `Accident_Severity` | 82,139 / 82,455 | Correct — rows with no accident |
| Inconsistent weather values | ~50% of rows | Normalised to 6 standard categories |
| Inconsistent road condition values | ~50% of rows | Normalised to 4 standard categories |
| Inconsistent accident type values | ~50% of rows | Normalised to 7 standard categories |
