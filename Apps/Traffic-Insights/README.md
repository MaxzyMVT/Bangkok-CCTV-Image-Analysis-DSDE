# Rain & Traffic Insights App (`Apps/Traffic-Insights/`)

Interactive web dashboard mounted at `/insights` by `Apps/main.py`.

## Launch
```bash
# Full unified web suite (Map + Detect + Lakehouse + Insights)
python Apps/main.py

# Lightweight Insights-only server
python Apps/main.py --insights-only
```

Open `http://localhost:8080/insights`. `GET /api/insights` queries **PyDuckLake (`lake.gold.rain_*` and `lake.silver.cameras`)** directly to serve preaggregated citywide, subgroup (`stratified`), hourly timeline, and per-camera (`location_th`, `location_en`, integer `camera_id`, and `no matched rainy hours` filtering) JSON payloads. Isolated unit tests can override `app.OUTPUT` when testing synthetic fixtures.

## Key Methodological Notes
- **Weekday vs. Weekend Timing Confounder:** The 69 weekday rainy camera-hours occur at 16:00 and 18:00–20:00 (only 8 distinct clock hours), whereas the 224 weekend rainy camera-hours occur at 12:00–17:00—overlapping only at **16:00 and 17:00**. Pooled weekday (`-34.3%`) vs. weekend (`-13.9%`) totals should not be interpreted as causal day-type differences.
- **Measurement Scope:** Still CCTV frames measure instantaneous vehicle density inside the camera field, not travel speed, flow volume, transit ridership, or ride-hailing pricing.
