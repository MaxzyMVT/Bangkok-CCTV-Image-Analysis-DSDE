# Power BI Executive Report (`DataAnalysis/PowerBI/`)

This folder contains a ready-to-open **Microsoft Power BI Project (`.pbip`)** presenting our Bangkok Rain & Urban Mobility findings as a **5-page Bilingual (`English` / `ภาษาไทย`) Business Intelligence Report** with a synchronized **`Language / ภาษา` slicer** at the top-right of every page:

1. **`01. Executive Summary`** — Headline KPIs (`100` CCTV cameras, `293` wet camera-hours, `-21.0%` motorcycle drop, `-30.5%` rush-hour drop, `+12.6 pts` localized car mode shift) and key executive findings.
2. **`02. Rain Effect & Timing`** — Mode breakdown (`city_effect`), weekday vs. weekend timing trap (`rain_timing`, showing overlap only at `16:00–17:00`), and post-rain `rebound` recovery curve.
3. **`03. Modal Shift & Spatial Proof`** — Natural experiment (`+12.6 pts` cars within 3 km), footbridge sheltering & station proximity checks (`2.5%` dry vs. `2.8%` wet), per-camera leaderboard (`camera.csv` with Thai street names), and `FL.RTW.02` Phetchaburi flood readings.
4. **`04. Supervised ML & Drivers`** — Model benchmarks (`RandomForest R² = 0.64`, `HistGradientBoosting R² = 0.83–0.89`), permutation feature importance (`rain_mm` #1 weather driver for motorcycles), and empirical correlations (`rail_dist_m r = -0.26`).
5. **`05. Action & Adaptability`** — Operational recommendations (BTS/MRT feeder alerting, ride-hailing pre-positioning, automated `1.4 KB` blank-frame feed QA) and next-phase sensor roadmap.

---

## How to Install & Open Power BI on Windows

### Step 1: Install Power BI Desktop (Free, if needed)
Install Power BI Desktop using **one** of these methods:
- **Fastest (PowerShell / Terminal via winget):**
  ```powershell
  winget install Microsoft.PowerBI
  ```
- **Or via Microsoft Store:** Open **Microsoft Store** on Windows $\to$ search **Power BI Desktop** $\to$ click **Get / Install** (auto-updates cleanly).

### Step 2: Open the Power BI Project (`.pbip`) & Load Data
1. Double-click **`DataAnalysis/PowerBI/Bangkok_Rain_Traffic_Story.pbip`** (or open Power BI Desktop $\to$ **File $\to$ Open report $\to$ Browse** and select `Bangkok_Rain_Traffic_Story.pbip`).
2. **Important — Load Data on First Open:** `.pbip` files store schemas and visuals in plain text without committing a binary `.pbi/cache.abf` snapshot, so cards/charts open empty (`(Blank)`) until you click **Home $\to$ Refresh** in the top ribbon.
3. After **Refresh** finishes loading all 16 tables, press **Ctrl + S** (Save) so Power BI caches the loaded data locally in `.pbi/cache.abf` (gitignored).
4. Use the **`Language / ภาษา`** slicer at the top-right of any page to switch titles, headers, executive insights, and action cards between **English** and **ภาษาไทย** (synchronized across all 5 pages).

### Step 3: Refresh / Update Data Path (If Opening on Another Laptop)
All table partitions point directly to `DataAnalysis/output/*.csv` (avoiding Power Query `Formula.Firewall` blocks). When moving to another machine/folder:
1. Run `python DataAnalysis/PowerBI/build_powerbi_project.py` (automatically rewrites every table path for your clone's location), **or** update paths under **Transform data $\to$ Data source settings** in Power BI Desktop.
2. Click **Refresh** in the Home ribbon.

### Step 4: Save as Standalone `.pbix` or Export PDF for Presentation
- Once loaded, click **File $\to$ Save As $\to$ Power BI file (`.pbix`)** to create a single compiled binary package for sharing, or **File $\to$ Export $\to$ Export to PDF** for presentation slides.
- If you regenerate analysis/ML tables (`python -m DataAnalysis.rain.run_all`, `python DataAnalysis/rain/ml_model.py`, `python DataAnalysis/PowerBI/build_powerbi_project.py`), simply click **Refresh** in Power BI Desktop.

> [!NOTE]
> The `.pbip`, TMDL semantic model, and PBIR report JSON definitions are validated against Microsoft's Fabric schemas (`developer.microsoft.com`) and Power BI Desktop's `Microsoft.PowerBI.Tabular.dll` parser.
