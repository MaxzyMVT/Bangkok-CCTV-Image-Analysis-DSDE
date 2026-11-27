# Power BI Executive Storyboard (`DataAnalysis/PowerBI/`)

This folder contains a complete, ready-to-open **Microsoft Power BI Project (`.pbip`)** presenting our Bangkok Rain & Urban Mobility findings as a **5-page Hackathon Business Storytelling Deck**:

1. **`01. Executive Hook`** — Headline KPIs (`100` CCTV cameras, `293` wet camera-hours, `-21.0%` motorcycle drop, `-30.5%` rush-hour drop, `+12.6 pts` localized car mode shift) and full narrative arc.
2. **`02. Rain Effect & Timing`** — Mode breakdown (`city_effect`), weekday vs. weekend timing trap (`rain_timing`, showing overlap only at `16:00–17:00`), and post-rain `rebound` recovery curve.
3. **`03. Modal Shift & Spatial Proof`** — Natural experiment (`+12.6 pts` cars within 3 km), footbridge/station myth debunking (`2.5%` dry vs. `2.8%` wet), per-camera leaderboard (`camera.csv` with Thai street names), and `FL.RTW.02` Phetchaburi flood readings.
4. **`04. Supervised ML & Drivers`** — Model benchmarks (`RandomForest R² = 0.64`, `HistGradientBoosting R² = 0.83–0.89`), permutation feature importance (`rain_mm` #1 weather driver for motorcycles), and empirical correlations (`rail_dist_m r = -0.26`).
5. **`05. Action & Adaptability`** — Business actions (BTS/MRT feeder alerting, ride-hailing pre-positioning, automated `1.4 KB` blank-frame feed QA) and next-phase sensor roadmap.

---

## How to Install & Open Power BI on Windows

### Step 1: Install Power BI Desktop (Free)
Power BI Desktop is **not currently installed** on this machine. Install it using **one** of these methods:
- **Fastest (PowerShell / Terminal via winget):**
  ```powershell
  winget install Microsoft.PowerBI
  ```
- **Or via Microsoft Store:** Open **Microsoft Store** on Windows $\to$ search **Power BI Desktop** $\to$ click **Get / Install** (auto-updates cleanly).

### Step 2: Open the Power BI Project (`.pbip`) & Load Data
1. Double-click **`DataAnalysis/PowerBI/Bangkok_Rain_Traffic_Story.pbip`** (or open Power BI Desktop $\to$ **File $\to$ Open report $\to$ Browse** and select `Bangkok_Rain_Traffic_Story.pbip`).
2. **Important — Load Data on First Open:** `.pbip` files store schemas and visuals in plain text without committing a binary `.pbi/cache.abf` snapshot, so cards/charts open empty (`(Blank)`) until you click **Home $\to$ Refresh** in the top ribbon.
3. After **Refresh** finishes loading all 14 tables, press **Ctrl + S** (Save) so Power BI caches the loaded data locally in `.pbi/cache.abf` (gitignored).

### Step 3: Refresh / Update Data Path (If Opening on Another Laptop)
All table partitions point directly to `DataAnalysis/output/*.csv` (avoiding Power Query `Formula.Firewall` blocks). When moving to another machine/folder:
1. Run `python DataAnalysis/PowerBI/build_powerbi_project.py` (automatically rewrites every table path for your clone's location), **or** update paths under **Transform data $\to$ Data source settings** in Power BI Desktop.
2. Click **Refresh** in the Home ribbon.

### Step 4: Save as Standalone `.pbix` or Export PDF for Presentation
- Once loaded, click **File $\to$ Save As $\to$ Power BI file (`.pbix`)** to create a single compiled binary package for sharing, or **File $\to$ Export $\to$ Export to PDF** for presentation slides.
- If you regenerate analysis/ML tables (`python -m DataAnalysis.rain.run_all`, `python DataAnalysis/rain/ml_model.py`, `python DataAnalysis/PowerBI/build_powerbi_project.py`), simply click **Refresh** in Power BI Desktop.

> [!NOTE]
> Because Power BI Desktop was not installed on this workstation, the `.pbip`, TMDL semantic model, and PBIR report JSON files were validated against Microsoft's official JSON schemas (`developer.microsoft.com`) rather than opened inside the Power BI Desktop GUI.
