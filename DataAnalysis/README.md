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
