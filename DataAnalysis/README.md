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

## 📊 Primary Data Sources

- **PyDuckLake Lakehouse (`DataEngineering/lakehouse/bkk_cctv.ducklake`)**: Live runtime store queried by `DataAnalysis/dashboard.py` and `Apps/Traffic-Insights/app.py` (Silver vehicle counts, transit stations, and Gold rain/ML/correlation tables).
- `DataEngineering/cctv_images_metadata.csv`: Initialization seed with spatial and temporal attribution for 61,319 observations.
- `DataEngineering/vehicle_counts.csv`: Initialization seed of model-inferred counts for *Cars*, *Motorcycles*, *Buses*, and *Trucks*.
- `Apps/Camera-Map/static/bangkok_rail.geojson`: Geographic alignments of Bangkok rapid transit lines and stations (deduplicated into `lake.silver.transit_stations`).
- `DataAnalysis/output/`: Generated CSV extracts mirroring the Gold layer for Power BI Desktop (`executive_story.csv`, `city_effect.csv`, `camera_hour.csv`, `camera.csv`, `camera_effect.csv`, `stratified_effects.csv`, `rain_timing.csv`, `shelter_pairs.csv`, `shelter_share.csv`, `denser_share.csv`, `rebound.csv`, `natural_experiment.csv`, `policy_stations.csv`, `flood_areas.csv`, `flood_sensor_hours.csv`, `mealtime.csv`, `ml_metrics.csv`, `ml_feature_importance.csv`, `ml_predictions.csv`, `correlations.csv`).

## Open the interactive rain and traffic dashboard

This dashboard reads **Silver & Gold tables directly from PyDuckLake (`DataEngineering/lakehouse/bkk_cctv.ducklake`)**. Initialize PyDuckLake from the repository CSVs once (or regenerate Gold ML/analysis tables anytime), then launch Streamlit:

```powershell
pip install -r DataAnalysis/requirements-dashboard.txt
python DataEngineering/lakehouse_pipeline.py
python -m DataAnalysis.rain.run_all
python DataAnalysis/rain/ml_model.py
streamlit run DataAnalysis/dashboard.py
```

Open the local URL printed by Streamlit (usually `http://localhost:8501`). The dashboard features 7 interactive tabs powered live by **PyDuckLake** (`DataEngineering/lakehouse/bkk_cctv.ducklake`, reading from Gold and Silver tables with sidebar parameter controls):

1. **Citywide Overview** — Matched wet-vs-dry vehicle effects, severity breakdowns, and hourly profiles.
2. **Bangkok Map** — Interactive PyDeck OpenStreetMap view of all 256 cameras, nearby stations, and rain corridors.
3. **Hypothesis Checks** — Same-hour spatial control, corridor spillover, shelter & station tests, and post-rain rebound.
4. **Network Graph** — Sparse PyVis spatial connectivity linking cameras to their nearest BTS/MRT station (<=800 m) and neighbor cameras (<=900 m), with Louvain community clusters and drill-down tables.
5. **Correlations & Insights** — Stratified congestion-shift analysis (+12.6 pp car surge in high-baseline zones), feature correlation heatmap, and key analytical findings.
6. **ML Congestion & Traffic Simulator** — Supervised gradient-boosted regression & classification models with live what-if rain/rush-hour/zone simulation.
7. **Data & Caveats** — Hourly frame coverage, 07:00 Aug 27 1.4 KB blank-frame outage diagnosis, and source timing notes.

For methods and confidence intervals, see [the rain analysis](rain/README.md).

The headline citywide finding is a **21.0% decrease in motorcycles visible per frame during rain** (95% CI −27.4% to −14.4%) after matching the same camera, hour, and weekday/weekend type, while heavy/rush rain triggers a **+12.6 percentage-point localized car shift** in congested zones.

For **Power BI Desktop**, open [Bangkok_Rain_Traffic_Story.pbip](PowerBI/Bangkok_Rain_Traffic_Story.pbip) — a schema-validated 5-page analytical PBIP project pre-wired to `DataAnalysis/output/` (`executive_story.csv`, `city_effect.csv`, `camera_hour.csv`, `camera.csv`, `camera_effect.csv`, and related tables). See [the Power BI guide](PowerBI/README.md) for setup and opening instructions.
