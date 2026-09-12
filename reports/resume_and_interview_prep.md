# Resume & Interview Preparation
## Smart City Traffic & Accident Analytics Project

---

## ATS-Ready Resume Bullet Points

### Bullet 1 — Python & Data Engineering
```
Designed and executed an end-to-end Python data pipeline (pandas, numpy) to clean,
validate, and transform 247,000+ real-world traffic and accident records across 10
Indian cities, resolving 73 invalid speed values, 101 invalid congestion entries, and
1,829 temperature IQR outliers through statistically justified imputation strategies.
```

### Bullet 2 — EDA & Statistical Analysis
```
Performed comprehensive exploratory data analysis generating 20 publication-quality
visualisations and conducted 9 formal statistical tests (Pearson, Spearman, Chi-square,
ANOVA, Welch t-test), discovering a vehicle-congestion correlation of r=0.81 (p<0.001)
and statistically significant peak-hour accident severity differences (p=0.046).
```

### Bullet 3 — Machine Learning & Streamlit App
```
Developed 2 ML models (Random Forest CV R²=0.654 for congestion prediction; Gradient
Boosting CV F1=0.800 for accident risk) with explicit data-leakage prevention; built a
6-tab interactive Streamlit analytics application with live congestion prediction, folium
geospatial hotspot maps, and filter-responsive KPI dashboards serving 247,204 records.
```

---

## Interview Questions & Answers

---

**Q1: Why did you choose this project?**

**A:** I wanted a project that reflects a real-world data analyst job — not just one clean dataset with simple charts. Traffic and accident analytics involves messy data, time-series analysis, geospatial thinking, and recommendations that affect public safety. It covers Python, statistics, ML, and a live Streamlit app in one cohesive problem, which mirrors what companies actually ask for in DA roles. The "smart city" domain also demonstrates awareness of modern urban challenges.

---

**Q2: How did you clean 247,000+ records?**

**A:** I built a modular cleaning pipeline in `data_cleaning.py` with explicit functions for each issue:
1. Identify and remove duplicates (0 found, Record_ID uniqueness check)
2. Impute missing values column-by-column — for example, speed was imputed using city-level medians because different cities have different speed patterns
3. Normalise text — weather had 12 variants like "clear", "Clear", "heavy_rain", "Heavy Rain" — all mapped to 6 standard values
4. Fix invalid speeds (73 values outside 1–150 km/h) and congestion (101 values outside 0–100%)
5. Treat temperature outliers using IQR method plus domain rules (−5 to 50°C for Indian cities)

---

**Q3: How did you handle missing values?**

**A:** I analysed each column separately before deciding on a strategy:
- `Vehicle_Count` (0.4% missing) — median imputation (safe for mildly skewed distribution)
- `Avg_Speed_kmph` (0.59% missing) — city-level median (different cities have different speed profiles)
- `Accident_Type` (33% missing) — set to "None" — these rows have no accident, so NULL is correct behaviour, not a real missing value
- `Emergency_Response_Min` — set to 0 where no accident occurred (no emergency was dispatched)
- `Temperature_C` — monthly median (temperatures are seasonal, month-level grouping preserves the pattern)

I never used blind `fillna(mean)` on all columns — that would distort categorical distributions and ignore domain context.

---

**Q4: How did you detect outliers?**

**A:** Two approaches combined:
1. **IQR method** — Lower = Q1 − 1.5×IQR, Upper = Q3 + 1.5×IQR. Distribution-based, works well for right-skewed data like temperature.
2. **Domain rules** — For Indian cities, temperatures below −5°C or above 50°C are physically impossible. Speeds above 150 km/h on urban roads are unrealistic.

For temperature, 1,829 outliers were replaced with monthly medians to preserve seasonal patterns.

---

**Q5: Explain your EDA approach.**

**A:** I structured EDA in three layers:
1. **Distribution analysis** — histograms and value counts for every key column
2. **Relationship analysis** — scatter plots (congestion vs vehicles), bar charts (weather vs accident rate)
3. **Time analysis** — hourly patterns, monthly trends, weekday vs weekend

I generated 20 charts covering traffic volume, speed, congestion, accidents, weather, road conditions, and emergency response. All charts are saved as PNGs in `reports/` for the portfolio.

---

**Q6: What statistical tests did you run and why?**

**A:** I ran 9 formal tests, choosing the method based on data type:
- **Pearson correlation** for two continuous variables (vehicle count vs congestion — r=0.81)
- **Spearman correlation** when the relationship might be non-linear (congestion vs response time)
- **Chi-square** for two categorical variables (weather vs accident occurrence)
- **ANOVA** to compare means across multiple groups (congestion across weather types)
- **Welch t-test** to compare two groups (peak vs off-peak severity — p=0.046, significant)

I always stated the null hypothesis, reported the exact p-value, and gave a business interpretation — not just whether it was significant.

---

**Q7: What was your biggest data quality challenge?**

**A:** The Accident_Type and Accident_Severity columns had ~82,000 NULL values — 33% of the dataset. At first this looks like a data quality problem. But analysing carefully: every one of those NULLs corresponded exactly to rows where Emergency_Response_Min = 0. These rows simply recorded traffic readings with no accident — NULLs were correct, not missing data. The real cleaning needed was text normalisation: "side_collision", "Side collision", "Side Collision" all meaning the same thing needed to be mapped to one standard value.

---

**Q8: Why did you use ML?**

**A:** ML is a supporting component — not the main focus. The business value is prediction: city authorities can input current conditions (vehicle count, weather, hour) and get a predicted congestion level or accident risk score BEFORE the event happens. This enables proactive management rather than reactive response. I used Random Forest for congestion (handles non-linear relationships, interpretable via feature importance) and Gradient Boosting for accident classification (good with moderately imbalanced targets).

---

**Q9: How did you ensure no data leakage in your ML models?**

**A:** I explicitly excluded all accident outcome variables from the feature set:
- `Accident_Type` — only known after the accident
- `Accident_Severity` — only known after the accident
- `Severity_Score` — derived from Severity, so excluded
- `Emergency_Response_Min` — only triggered by an accident

The features used are pre-accident observable conditions: vehicle count, speed, congestion level, hour, weather, road condition, signal status, visibility. This correctly simulates a real prediction scenario where you predict risk from observable inputs.

---

**Q10: How did you validate the results?**

**A:** Multiple validation layers:
1. **Count checks** — after cleaning, verify row counts match expectations (247,204 in, 247,204 out)
2. **Range validation** — verify speed is 1–150, congestion is 0–100, temperature is −5 to 50°C
3. **Business logic check** — rows with Accident_Flag=0 have Emergency_Response_Min=0 (100% consistent)
4. **Statistical sanity** — correlation direction matches domain expectation (more vehicles = more congestion ✓)
5. **ML cross-validation** — 3-fold CV on both models to ensure results generalise beyond training data

---

**Q11: What is your Traffic_Risk_Score and how did you derive it?**

**A:** It's a composite index on a 0–10 scale combining five weighted risk factors:
- Congestion level (normalised, weight 0.25) — high congestion increases rear-end risk
- Low visibility flag (weight 0.20) — reduces stopping distance
- Poor road flag (weight 0.20) — reduces vehicle control
- Weather risk flag (weight 0.20) — adverse weather increases accident probability
- Faulty signal flag (weight 0.15) — disrupts right-of-way

The weights are logically justified but not statistically derived — they represent domain expertise. In a production system, the weights could be calibrated using logistic regression coefficients from accident data.

---

**Q12: What business problem does the project solve?**

**A:** City transport authorities face three core problems: (1) they don't know in real-time which locations are highest risk today, (2) they allocate emergency resources based on intuition rather than data, and (3) infrastructure maintenance budgets are spread evenly rather than risk-weighted. This project solves all three: the Traffic_Risk_Score identifies high-risk locations, emergency response time analysis shows where ambulances should be pre-positioned, and the congestion prediction model lets planners forecast problems before they occur.

---

**Q13: Explain the Streamlit app.**

**A:** The app has 6 tabs:
1. **Overview** — 8 KPI cards (total vehicles, accidents, speed, congestion, accident rate, EMS time, fatal count, risk score) + 4 trend charts
2. **Traffic Intelligence** — hourly pattern with peak-hour shading, speed by city, road congestion, vehicle type, weekday/weekend comparison
3. **Accident Intelligence** — severity distribution with data labels, accident types, vehicle analysis, weather rates, EMS by city
4. **Hotspot Map** — live folium map with heatmap layer, severity colouring, interactive tooltips
5. **ML Prediction** — sliders for scenario input, large coloured result card (green=free flow to red=gridlock)
6. **Data Explorer** — column selector, statistics table, filtered CSV download

Data loads automatically from the processed CSV — no upload needed. Sidebar has City, Weather, Vehicle Type, Month Range, and Accidents-Only filters.

---

**Q14: What did you learn from this project?**

**A:** Three main learnings:
1. **Data cleaning is 60% of the work** — the most valuable insight isn't in the ML model, it's in understanding why data is missing or wrong
2. **Business context matters more than algorithms** — the Traffic_Risk_Score is more useful to a city manager than a 0.80 F1 score
3. **Statistical tests must be interpreted** — a p-value of 0.05 doesn't mean "important," it means "unlikely by chance." The magnitude (r=0.81 for vehicle-congestion) is what tells the real story

---

**Q15: How would you scale this to more cities?**

**A:** The architecture is already designed for scaling:
1. The cleaning pipeline is city-agnostic — it works on any city in the CSV
2. The Streamlit app filters by City dynamically — add more cities, the dropdown updates automatically
3. The real-time pipeline (`app/pipeline.py`) iterates over a `CITIES` dictionary — just add more city lat/lon pairs
4. The geospatial maps use lat/lon — they work for any city without code changes
5. The ML models use label encoding for City — retraining with more cities is one `python prediction.py` run

---

**Q16: What KPIs would you present to a city authority?**

**A:** Five headline KPIs with clear owners and targets:
1. **Accident Rate (%)** — safety performance (target: < 60%)
2. **Average Emergency Response Time (min)** — EMS efficiency (target: < 15 min)
3. **Average Traffic Risk Score** — composite proactive risk (alert: score ≥ 6)
4. **Peak Hour Congestion (%)** — traffic management effectiveness
5. **Fatal Accident Rate (% of all accidents)** — severity of outcomes

Each KPI has a defined target, a data source, and a named responsible department — making them actionable rather than just descriptive.

---

**Q17: Why didn't you use SQL or Power BI in this project?**

**A:** The project is focused on demonstrating Python-first analytics skills: data cleaning, statistical analysis, machine learning, and interactive visualisation through Streamlit. The Streamlit app replaces the need for a separate BI tool by providing interactive charts, filters, KPIs, and an ML prediction interface all in one Python application. For a larger enterprise deployment with multiple teams needing self-service reporting, SQL + Power BI would be the right addition — and the data pipeline is already structured to support that.

---

**Q18: Describe a specific challenge you faced.**

**A:** The temperature column had 1,829 outlier values — some as low as −25°C and as high as 80°C — clearly unrealistic for Indian cities. I had to decide: delete rows, cap at fixed bounds, or impute? Deleting would lose 0.74% of the dataset unnecessarily. Capping at a fixed value would distort seasonal analysis. I chose to replace outliers with the **monthly median** of valid values. This preserved seasonal patterns (January is colder than May) while removing the impossible extremes. The IQR bounds were [10.8°C, 43.2°C] and domain bounds were [−5°C, 50°C] — I used the intersection of both to avoid over-correcting valid edge values.

---

**Q19: How did you choose which ML algorithm to use?**

**A:** I evaluated based on three criteria:
1. **Data characteristics** — 247K rows, 10 numeric + categorical features, non-linear relationships (congestion doesn't scale linearly with vehicles)
2. **Interpretability** — Random Forest provides feature importance, which is directly useful to city planners ("vehicle count matters most")
3. **Performance** — tested Linear Regression as a baseline (R²=0.65) and Random Forest matched it with the same R²=0.65, confirming the relationship is largely linear despite being complex data

For classification, Gradient Boosting was chosen for its strong F1 performance on moderately imbalanced datasets (66% accident rate) and its ability to handle mixed feature types.

---

**Q20: What would you add to make this production-ready?**

**A:** Five additions for production:
1. **Live data ingestion** — the `pipeline.py` script already fetches OpenWeatherMap + TomTom data; need API credentials and a scheduler (cron or Airflow)
2. **Automated alert system** — email/SMS when Traffic_Risk_Score ≥ 6 at any location
3. **Model retraining pipeline** — weekly automatic retraining as new data accumulates
4. **User authentication** — restrict the Streamlit app to authorised city officials
5. **Containerisation** — package with Docker so the app runs consistently on any server without environment issues
