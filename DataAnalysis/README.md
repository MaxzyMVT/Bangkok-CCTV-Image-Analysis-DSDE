# Exploratory Data Analysis & Analytics (`DataAnalysis/`)

This directory contains analytical scripts, notebooks, and statistical workflows for analyzing vehicle density patterns across Bangkok's traffic surveillance network.

---

## 🔬 Analytical Objectives

The analysis focuses on extracting actionable insights from the unified vehicle count datasets:
- **Spatial Traffic Density**: Mapping vehicle concentration across major arterial intersections and identifying recurrent bottlenecks across Greater Bangkok.
- **Temporal Rhythm**: Analyzing hourly diurnal traffic curves, comparing morning rush hours (07:00–09:00), midday transitions, evening rush hours (17:00–20:00), and late-night conditions.
- **Modal Share Dynamics**: Calculating the ratio of motorcycles versus passenger cars and public transit across different urban corridors.
- **Transit Correlation**: Evaluating traffic density variations in proximity to rapid transit (BTS/MRT) stations.

---

## 📊 Primary Datasets Used

- `DataEngineering/cctv_images_metadata.csv`: Full spatial and temporal attribution for 61,319 observations.
- `DataEngineering/vehicle_counts.csv`: Model-inferred counts for *Cars*, *Motorcycles*, *Buses*, and *Trucks*.
- `Apps/Camera-Map/static/bangkok_rail.geojson`: Geographic alignments of Bangkok rapid transit lines and stations.

## Open the interactive rain and traffic dashboard

This dashboard uses the **committed result tables and figures** from `DataAnalysis/rain/`, including the latest vehicle recount, local rain gauges, and flood evidence. It does not rerun the model on page load.

```powershell
pip install -r DataAnalysis/requirements-dashboard.txt
streamlit run DataAnalysis/dashboard.py
```

Open the local URL printed by Streamlit (usually `http://localhost:8501`). The tabs show the citywide effect, hourly trends, camera map, tests of the idea-chart hypotheses, and source limitations. For methods and confidence intervals, see [the rain analysis](rain/README.md).

The main finding is a **21.0% decrease in motorcycles visible per frame during rain** (95% CI −27.4% to −14.4%) after matching the same camera, hour, and weekday/weekend type. Cars have no clear overall change. The same-hour nearby-camera comparison and other hypothesis checks are shown separately; one week of observations does not establish the full causal chain in the idea chart.

For Power BI Desktop, import `DataAnalysis/output/city_effect.csv` for headline effects, `camera_hour.csv` for a time series, and join `camera.csv` to `camera_effect.csv` on `camera_id` for a map. These CSVs are already generated and committed; no additional export step is needed.
