# Rain & Traffic Analysis Pipeline (`DataAnalysis/rain`)

Modular statistical, spatial-network, and supervised ML pipeline quantifying how Bangkok rainfall changes urban road traffic across **100 BMA CCTV cameras** (`10,064` camera-hours, including `293` wet camera-hours across `94` cameras during late August 2026).

All pipeline outputs are written both to **PyDuckLake Gold tables** (`lake.gold.rain_*`, `lake.gold.ml_*` in `DataEngineering/lakehouse/bkk_cctv.ducklake`) and mirrored as CSV extracts in [`DataAnalysis/output/`](../output/) for Power BI Desktop and reproducibility.

---

## Core Pipeline Modules

| Module | Purpose |
| :--- | :--- |
| `common.py` | Shared paths, PyDuckLake Gold readers/writers (`load_gold`, `save_gold`), matched-baseline helpers, and cluster bootstrap CIs. |
| `recount.py` | Filters frozen camera frames (`> 30` px difference), counts stationary vs. moving boxes, and emits clean frame counts (`lake.silver.counts_v2`). |
| `prepare.py` | Joins frame counts (`counts_v2`), camera coordinates, BMA rain gauges (`gauge_dist_m`), BTS/MRT proximity (`rail_dist_m`), shelters, and Traffy floods into `rain_camera` & `rain_camera_hour`. |
| `effects.py` | Matched same-camera/hour/day-type rain effects citywide (`rain_city_effect`), per camera (`rain_camera_effect`), `±3 km` same-hour spatial comparison (`rain_natural_experiment`), denser camera share (`rain_denser_share`), mealtime delivery split (`rain_mealtime`), and post-rain recovery (`rain_rebound`). |
| `stratified.py` | Subgroup splits by weekday/weekend & light (`1–5 mm`) vs. heavy (`5+ mm`) rain (`rain_stratified_effects`) and rain-timing hour audit (`rain_rain_timing`). |
| `shelter.py` | Tests footbridge sheltering hypothesis by checking stationary motorcycles near shelters in wet vs. dry hours (`rain_shelter_pairs`, `rain_shelter_share`). |
| `traffy.py` | Clusters Bangkok flood complaints (`rain_flood_areas`) and audits BMA drainage flood sensors (`rain_flood_sensor_hours`). |
| `policy.py` | Ranks priority transit stations for feeder/shuttle readiness (`rain_policy_stations`). |
| `ml_model.py` | Computes feature correlations (`rain_correlations`) and trains Ridge, Random Forest, and HistGradientBoosting models on temporal (`Aug 30–31`) & 5-fold spatial splits (`ml_metrics`, `ml_feature_importance`, `ml_predictions`). |
| `network_viz.py` | Builds the sparse spatial connectivity graph linking cameras to their nearest rapid transit station (`<= 800 m`, `k_rail = 1`) and neighbor corridor cameras (`<= 900 m`, `k_cam = 2`), detects Louvain clusters, and produces `connection_summary_tables` + PyVis HTML. |
| `run_all.py` | End-to-end orchestrator regenerating all `rain_*` Gold tables and CSV exports. |
| `figures.py` | Renders publication-ready summary plots into `DataAnalysis/figures/`. |

---

## Key Verified Findings & Data Caveats

1. **Headline Citywide Impact (`rain_city_effect`):**
   - **Motorcycles:** `-21.0%` (`95% CI -27.4% to -14.4%`, `94` cameras, `293` wet camera-hours); steeper during **rush hours (`-30.5%`)** vs. other hours (`-18.3%`).
   - **Cars:** `-2.2%` overall (`95% CI -6.4% to +2.2%`), keeping total traffic at `-5.4%` (`-9.9%` in rush hours).
2. **Localized Rain Modal Shift (`rain_natural_experiment`):**
   - Across **15 hours** with simultaneous wet and dry cameras within `<= 3 km` (**172 pairs**), **cars surge `+12.6` percentage points** (`95% CI +2.2 to +20.7` pts) while motorcycles stay flat (`+0.9` pts), revealing localized shift from 2-wheelers to 4-wheelers.
3. **Subgroup & Rain-Timing Caveat (`rain_stratified_effects`, `rain_rain_timing`):**
   - Weekday rain appears sharper (`-34.3%` motorcycles, `69` wet camera-hours across `34` cameras) than weekend rain (`-13.9%`, `224` wet camera-hours across `93` cameras), **partially confounded by timing**: weekday rain occurred almost exclusively in the late-afternoon/evening rush (`16:00–20:00`), whereas weekend rain was concentrated midday/afternoon (`12:00–17:00`).
4. **Refuted Hypotheses (`rain_shelter_share`, `rain_denser_share`):**
   - **Footbridge sheltering:** While `61` cameras sit near a shelter (`56` with matched wet/dry frames in `rain_shelter_share`), stationary motorcycles remain rare across scenes (sidewalk parking/static parked bikes rather than commuters blocking traffic lanes citywide).
   - **Transit proximity:** Only `1` of `5` denser-during-rain cameras (`20.0%`) is near a BTS/MRT station (`39` cameras citywide are near rail) vs. `33.8%` of other cameras (`-13.8` percentage-point difference), and `0` of `5` denser cameras lie inside mapped flood clusters (`rain_flood_areas`).
5. **Drainage Flood Sensor Check (`rain_flood_sensor_hours`):**
   - BMA sensor `FL.RTW.02` on **Phetchaburi Road** (`Don Bosco`, `5.0–5.1 cm` on Aug 30–31) lies near Phetchaburi cameras **`250`** (Soi Phetchaburi 38) and **`256`** (Soi Phetchaburi 27), providing targeted corridor cross-checks.
6. **Data Quality Diagnostic (`2026-08-27 07:00` Drop):**
   - The sharp `07:00` traffic drop (`mean_total = 0.11` across `70` cameras / `72` frames in `rain_camera_hour`; `71` of `73` frames in `silver.vehicle_counts` had `0` detected vehicles, before recovering to `11.07` vehicles/frame at `08:00`) is caused by **upstream BMA blank/outage frames**, not an absence of morning traffic.
7. **Supervised ML Benchmarks (`ml_metrics`, `ml_feature_importance`):**
   - Without camera baselines (`temporal_aug30_31` split: `7,143` train / `2,853` test), `RandomForest` achieves `R² = 0.642` and `HistGradientBoosting` reaches `R² = 0.622` (`MAE = 2.39`) on total vehicles (`R² = 0.546` for motorcycles).
   - With camera baseline averages, `HistGradientBoosting` reaches **`R² = 0.830` (`total`)** and **`R² = 0.754` (`motorcycle`)** temporally (`R² = 0.894` / `0.788` across 5-fold unseen cameras), with `rain_mm` ranking as the **top weather feature** for motorcycle prediction after `motorcycle_base`.

---

## Reproduce From Scratch

```powershell
# 1. Clean & freeze-filter frame detections into silver.counts_v2 / counts_v2.csv
python -m DataAnalysis.rain.recount

# 2. Compute all rain statistical tables -> PyDuckLake Gold + DataAnalysis/output/*.csv
python -m DataAnalysis.rain.run_all

# 3. Compute correlations & train/evaluate supervised ML models -> PyDuckLake Gold + CSV
python -m DataAnalysis.rain.ml_model

# 4. Generate static matplotlib summary figures -> DataAnalysis/figures/
python -m DataAnalysis.rain.figures

# 5. Rebuild Power BI Desktop (.pbip) report & executive summary CSV
python DataAnalysis/PowerBI/build_powerbi_project.py
```
