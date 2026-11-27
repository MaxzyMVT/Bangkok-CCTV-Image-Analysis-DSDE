"""Bilingual (English / Thai) localization strings and lookup tables for Streamlit."""

from __future__ import annotations

import pandas as pd

LANG_OPTIONS = ("en", "th")

LABEL_TEXT_BY_LANG = {
    "en": {
        "clears": "Clears (-ve)",
        "denser": "Denser (+ve)",
        "unaffected": "Unaffected",
        "too few wet hours": "Too few wet hrs (<3h)",
        "no matched rainy hours": "No matched rainy hrs",
    },
    "th": {
        "clears": "ลดลงเมื่อฝนตก (-ve)",
        "denser": "หนาแน่นขึ้น (+ve)",
        "unaffected": "ไม่เปลี่ยนแปลงชัดเจน",
        "too few wet hours": "ชั่วโมงฝนตกน้อย (<3 ชม.)",
        "no matched rainy hours": "ไม่มีชั่วโมงฝนตกเปรียบเทียบ",
    },
}

CLASS_NAMES_BY_LANG = {
    "en": {
        "total": "All vehicles",
        "car": "Cars",
        "motorcycle": "Motorcycles",
        "bus": "Buses",
        "truck": "Trucks",
    },
    "th": {
        "total": "ยานพาหนะทั้งหมด",
        "car": "รถยนต์",
        "motorcycle": "รถจักรยานยนต์",
        "bus": "รถโดยสาร",
        "truck": "รถบรรทุก",
    },
}

SEGMENT_NAMES_BY_LANG = {
    "en": {
        "weekday": "Weekday (Mon–Fri)",
        "weekend": "Weekend (Sat–Sun)",
        "light_1_to_5_mm": "Light rain (1–5 mm/h)",
        "heavy_5plus_mm": "Heavy rain (≥5 mm/h)",
    },
    "th": {
        "weekday": "วันธรรมดา (จันทร์–ศุกร์)",
        "weekend": "วันหยุดสุดสัปดาห์ (เสาร์–อาทิตย์)",
        "light_1_to_5_mm": "ฝนตกเล็กน้อย (1–5 มม./ชม.)",
        "heavy_5plus_mm": "ฝนตกหนัก (≥5 มม./ชม.)",
    },
}

WINDOW_NAMES_BY_LANG = {
    "en": {
        "all": "All hours (06–21)",
        "rush": "Rush hours (07–09, 17–20)",
        "other": "Off-peak hours",
    },
    "th": {
        "all": "ทุกช่วงเวลา (06:00–21:00)",
        "rush": "ชั่วโมงเร่งด่วน (07–09, 17–20 น.)",
        "other": "นอกชั่วโมงเร่งด่วน",
    },
}

DAY_TYPE_BY_LANG = {
    "en": {"weekday": "Weekday", "weekend": "Weekend"},
    "th": {"weekday": "วันธรรมดา", "weekend": "เสาร์–อาทิตย์"},
}

HEAT_OPTIONS_BY_LANG = {
    "en": {
        "None": None,
        "Traffy: Flood Reports": "is_flood",
        "Traffy: Traffic Jams": "is_traffic",
        "Traffy: Road Issues": "is_road",
        "Traffy: Sidewalk Obstructions": "is_sidewalk",
    },
    "th": {
        "ไม่แสดง Heatmap": None,
        "Traffy: รายงานน้ำท่วม": "is_flood",
        "Traffy: รายงานรถติด": "is_traffic",
        "Traffy: ปัญหาผิวจราจร": "is_road",
        "Traffy: กีดขวางทางเท้า": "is_sidewalk",
    },
}

UI = {
    "en": {
        "page_title": "Bangkok Rain & Traffic Analytics (PyDuckLake)",
        "main_title": "Bangkok Rain, Urban Mobility & Traffic Analytics",
        "subtitle": "100 BMA CCTV Cameras · 25–31 Aug 2026 · Powered by PyDuckLake Silver & Gold Lakehouse Tables",
        "toggle_btn": "🇹🇭 ภาษาไทย",
        "toggle_help": "Switch dashboard interface to Thai (ภาษาไทย)",
        "sidebar_header": "Dashboard Parameters",
        "exp_ts": "1. Time Series & Data Quality",
        "dates": "Dates",
        "veh_classes": "Vehicle Classes",
        "excl_blank": "Exclude blank-feed placeholder hours (< 3 KB)",
        "excl_blank_help": "BMA snapshots around 07:21–07:26 (e.g., Aug 27 07:00) contain 1.4 KB solid white frames.",
        "exp_map": "2. Spatial Map (PyDeck)",
        "map_height": "Map Panel Height (px)",
        "cam_filter": "Camera Response Filter",
        "min_wet": "Minimum Wet Hours",
        "traffy_overlay": "Traffy Heatmap Overlay",
        "show_rail": "Show BTS / MRT Stations",
        "show_dist": "Show 50 Bangkok Districts",
        "exp_net": "3. Network Graph",
        "rail_rad": "Camera ↔ Rail Max Distance (m)",
        "cam_rad": "Camera ↔ Camera Max Distance (m)",
        "k_rail": "Nearest Rail Hubs per Camera",
        "k_cam": "Nearest Neighbor Cams per Camera",
        "net_comm": "Detect Communities (Greedy Modularity)",
        "net_layout": "Layout Algorithm",
        "net_size": "Node Size By",
        "net_color": "Node Color By",
        "net_scale": "Graph Spread Scale",
        "net_range": "Node Size Range",
        "net_noverlap": "Prevent Node Overlap (Noverlap)",
        "net_labels_n": "Label Top N Cameras",
        "net_font": "Label Font Size",
        "net_min_comp": "Hide Clusters Smaller Than (nodes)",
        "net_edges": "Show Network Edges",
        "exp_sim": "4. Supervised ML What-If Simulator",
        "sim_rain": "Simulated Rain (mm/hr)",
        "sim_hod": "Hour of Day",
        "sim_weekend": "Weekend Day",
        "sim_rail_dist": "Distance to Nearest BTS/MRT (m)",
        "sim_shelter": "Near Footbridge / Expressway Shelter",
        "sim_base": "Typical Dry Baseline Vehicles / Frame",
        "sim_temp": "Ambient Temperature (°C)",
        "sim_vis": "METAR Visibility (km)",
        "kpi_banner": (
            "**Key Finding:** Motorcycles visible per frame drop **-21.0%** (95% CI -27.4% to -14.4%, rush hour **-30.5%**) "
            "during rain compared to dry baselines at the exact same camera, clock hour, and weekday/weekend type. "
            "Cars overall stay near zero (-2.2%, CI -6.4% to +2.2%), pulling overall frame counts down by -5.4%."
        ),
        "kpi_all": "All Vehicles (Wet)",
        "kpi_cars": "Cars (Wet)",
        "kpi_moto": "Motorcycles (Wet)",
        "kpi_help": "95% CI {lo} to {hi} across {n} wet camera-hours",
        "kpi_overlap": "Weekday–Weekend Rain Overlap",
        "kpi_overlap_hrs": "{n} hrs",
        "kpi_overlap_help": "Rain fell weekdays 16:00–20:00 (8 clock hrs, 69 cam-hrs) vs weekends 12:00–17:00 (8 clock hrs, 224 cam-hrs)",
        "tabs": [
            "Overview & Timing",
            "Time Series & Anomalies",
            "Spatial Map (PyDeck)",
            "Network Analysis (PyVis)",
            "Supervised ML & Correlations",
            "Hypotheses, Flood & News",
            "Data Quality & Causal Audit",
        ],
        "ci_title_win": "Rain Effect by Time Window (95% CI)",
        "ci_title_seg": "Subgroup Splits: Weekday/Weekend & Rain Intensity",
        "ci_x": "Change when raining (%)",
        "timing_title": "Weekday vs Weekend Rain Exposure by Clock Hour (Overlap only at 16:00–17:00)",
        "timing_warn": (
            "**Why '69 camera-hours (8 clock hours, 16:00–20:00)' matters:** Across Mon–Fri, rain >= 1 mm only occurred during 8 distinct clock-hour timestamps, hitting multiple cameras at once (summing to 69 camera-hour pairs). "
            "Weekend rain fell almost exclusively earlier in the day (12:00–17:00), leaving only **16:00 and 17:00** to compare directly."
        ),
        "why_dec_exp": "Why do still-frame counts DECREASE when it rains instead of increasing congestion?",
        "why_dec_md": (
            "- **Frames measure spatial density, not travel speed:** A 10-minute still photo counts how many vehicles are inside the frame at that instant—it cannot measure travel time (km/h) or throughput (`flow = density × speed`).\n"
            "- **Motorcycles leave open roadways (-21.0% citywide, -30.5% rush hour):** Two-wheelers visibly disappear during rainfall (canceling trips, delaying departures until rain clears as seen in `Rebound`, or stopping under off-road cover). Because motorcycles make up ~20.5% of daytime counts (`113,310 / 551,501`), losing them pulls total vehicle counts down (`-5.4%`).\n"
            "- **Gridlock limits inflow:** Even when slow traffic builds up upstream, blocked corridors can feed *fewer* passing cars into open main-road frames. Only in our **15-hour same-hour neighbor comparison** (`natural_experiment`: rained-on cameras vs dry neighbors <= 3 km away) do cars read **+12.6 percentage points higher** (95% CI +2.2 to +20.7)."
        ),
        "ts_empty": "Select at least one date and one vehicle class in the sidebar.",
        "ts_title": "Hourly Traffic Density vs Mean Rainfall",
        "ts_y1": "Mean vehicles / frame",
        "ts_y2": "Mean gauge rain (mm/h)",
        "ts_rain_bar": "Mean Rain (mm/h)",
        "rebound_title": "Traffic Recovery Before and After Rain Stops (-60 to +180 min)",
        "rebound_x": "Minutes relative to first dry hour after rain (t = 0)",
        "rebound_y": "Ratio to matched dry baseline (1.0 = normal)",
        "rebound_cls": "Vehicle class",
        "outage_info": (
            "**Aug 27 07:00 (and 07:00–08:00) Drop Explanation:** Querying `lake.silver.cctv_frames` shows that on **Aug 27 at 07:21–07:26**, **71 of 73 captured frames** (across cameras like `#11, #18, #26`) have file sizes of **1,402–1,404 bytes**—completely solid-white BMA server outage placeholder images (normal frames are 25–35 KB). "
            "Thus neither YOLO detector failed; the source images were blank. Same 07:00–08:00 outage placeholders occurred across subsets of cameras on Aug 26, 28, 29, 30, and 31. Toggle `Exclude blank-feed placeholder hours` in the sidebar to inspect before vs after."
        ),
        "map_cap": (
            "Showing **{shown} / {total} cameras** · Height: `{h}px` · "
            "Cyan = Clears, Red = Denser, Green = Unaffected, Orange = <3 wet hrs, Slate = No matched rain"
        ),
        "map_empty": "No cameras match the current sidebar filter.",
        "net_subhdr": "Bangkok CCTV & Rapid Transit Corridor Network",
        "net_metrics": ("Nodes", "Edges", "Communities", "Diameter", "Avg Clustering", "Avg Path Length"),
        "net_zoom": "Zoom Into Community Cluster",
        "net_all_clusters": "All Clusters",
        "net_comm_fmt": "Community #{cid} — {cams} cams, {st} rail hubs ({hubs})",
        "net_corr_only": "corridor only",
        "net_dl": "Download Interactive Graph (HTML)",
        "net_rand": "Compare with Random Graph (Erdős–Rényi, same nodes & edges)",
        "net_links_hdr": "Who Connects to Whom (Camera ↔ Station ↔ Neighbors)",
        "net_comms_hdr": "Detected Spatial Corridor Communities",
        "net_deg_hdr": "Degree Distribution",
        "net_scale_radio": "Scale",
        "net_scale_opts": ["Histogram", "Log-log (CCDF)"],
        "net_cent_hdr": "Centrality Scores",
        "ml_subhdr": "Supervised Traffic Prediction Models (Trained from PyDuckLake)",
        "ml_missing": "ML tables not found in PyDuckLake. Run `python DataAnalysis/rain/ml_model.py`.",
        "ml_desc": (
            "Supervised **Regression** (`HistGradientBoostingRegressor`, `RandomForestRegressor`, `Ridge`) predicts hourly vehicle counts per camera, while "
            "**Classification** (`HistGradientBoostingClassifier`) flags `+25%` traffic surges vs matched dry baselines. Evaluated on **Chronological Split** (Train Aug 25–29 $\\to$ Test Aug 30–31) and **5-Fold Spatial GroupKFold** (unseen cameras)."
        ),
        "ml_imp_sel": "Permutation Importance Target",
        "ml_imp_x": "Mean Permutation Importance (R² drop)",
        "ml_imp_y": "Feature",
        "ml_imp_title": "Top Drivers: {tgt}",
        "ml_pred_wet": "Rain (>=1 mm)",
        "ml_pred_dry": "Dry (<1 mm)",
        "ml_pred_x": "Actual Vehicles / Frame",
        "ml_pred_y": "Predicted Vehicles / Frame",
        "ml_pred_title": "Held-Out Test Predictions (Aug 30–31): Actual vs Predicted",
        "sim_subhdr": "Interactive What-If Scenario Simulator (Sidebar Section 4)",
        "sim_cards": ("Predicted Total", "Predicted Cars", "Predicted Motorcycles"),
        "sim_unit": "{val:.2f} veh/frame",
        "sim_delta": "{delta:+.1f}% vs 0 mm scenario",
        "corr_subhdr": "Empirical Pearson Correlations Across Lakehouse Features",
        "corr_scope": "Correlation Scope",
        "corr_hourly": "Hourly Observations (n ≈ 9,996)",
        "corr_cam": "Camera Rain Effect (n = 94)",
        "corr_x": "Pearson Correlation (r)",
        "corr_y": "Variable",
        "corr_t": "Target",
        "corr_title": "Measured Linear Correlations Across Traffic & Weather Variables",
        "hyp_1": "1) Same-Hour Natural Experiment: Rained-on vs Dry Neighbor Cameras (≤ 3 km)",
        "hyp_1_title": "Natural Experiment (15 Clock Hours · 172 Camera Pairs)",
        "hyp_1_x": "Difference in % vs baseline (Percentage Points)",
        "hyp_1_cap": "Cars read +12.59 pts higher (CI +2.2 to +20.7) at wet cameras vs dry neighbors within 3 km in the exact same hour, though observed in only 15 hours without taxi/private splits.",
        "hyp_2": "2) Shelter & Transit Overlap vs Rain Density",
        "hyp_3": "3) DDS Flood Sensor ≥ 5 cm Matched to Phetchaburi Cameras",
        "hyp_3_cap": "`FL.RTW.02 Thanon Phetchaburi` recorded all 13 flood readings >= 5 cm (5.0–5.1 cm). Matched road cameras 250 & 256 did not show higher standing density.",
        "hyp_4": "4) Traffy Fondue Complaint Breakdown",
        "hyp_5": "5) Daily FM91 Traffic News Mentions (Aug 25–31)",
        "audit_rows": "DuckLake Frame Rows",
        "audit_cams": "Analyzed Cameras",
        "audit_wet": "Wet Camera-Hours",
        "audit_hdr": "Conceptual Causal Model: Evidence & Data Coverage Audit",
        "audit_exp": "View Original Static Verification Plots (DataAnalysis/figures/*.png)",
        "audit_footer": "All dashboard queries execute directly against PyDuckLake (`DataEngineering/lakehouse/bkk_cctv.ducklake`). Raw & Gold CSV files are only seed/export artifacts.",
        "scorecard": [
            {"Causal Node": "Rain Telemetry (>= 1 mm)", "Data Status": "Present (DDS Rain, 129 gauges)", "Observed Evidence & Caveats": "293 wet camera-hours matched; motorcycles drop -21.0%."},
            {"Causal Node": "Motorcycles Sheltering / Curb Block", "Data Status": "Tested (background-subtraction)", "Observed Evidence & Caveats": "Weak difference (2.5% dry vs 2.8% wet); mostly sidewalk parking, not 1–2 travel lanes blocked."},
            {"Causal Node": "DDS Flood >= 5 cm", "Data Status": "No GPS coordinates in DDS table", "Observed Evidence & Caveats": "All 13 >=5 cm readings at FL.RTW.02 Thanon Phetchaburi; matched road cams 250/256 showed no volume rise."},
            {"Causal Node": "Traffy Fondue (Flood / Road / Traffic)", "Data Status": "Present (7,247 geolocated tickets)", "Observed Evidence & Caveats": "Mostly inside soi/alleys (only 2 cameras within 100 m of DBSCAN flood clusters)."},
            {"Causal Node": "Transit Shift & +25% Ride Surge", "Data Status": "Not directly measured", "Observed Evidence & Caveats": "Natural experiment shows +12.6 pts cars in 15 hrs, but no BTS/MRT ridership, taxi, or surge-pricing feeds."},
            {"Causal Node": "Capacity Drop -> Road Gridlock", "Data Status": "Density only (no speed/flow)", "Observed Evidence & Caveats": "10-min still snapshots measure vehicle count in view, not km/h travel speed or queue delay."},
        ],
    },
    "th": {
        "page_title": "วิเคราะห์จราจรและฝนกรุงเทพฯ (PyDuckLake)",
        "main_title": "ระบบวิเคราะห์สภาพจราจร ฝน และความคล่องตัวในกรุงเทพฯ",
        "subtitle": "กล้อง CCTV กรุงเทพมหานคร 100 จุด · 25–31 ส.ค. 2569 · ขับเคลื่อนด้วย PyDuckLake (Silver & Gold)",
        "toggle_btn": "🇬🇧 English",
        "toggle_help": "สลับภาษาหน้าแดชบอร์ดเป็นภาษาอังกฤษ (English)",
        "sidebar_header": "พารามิเตอร์แดชบอร์ด",
        "exp_ts": "1. อนุกรมเวลาและคุณภาพข้อมูล",
        "dates": "วันที่วิเคราะห์",
        "veh_classes": "ประเภทยานพาหนะ",
        "excl_blank": "ตัดชั่วโมงที่ภาพกล้องขัดข้อง (< 3 KB)",
        "excl_blank_help": "ภาพจากกล้องช่วง 07:21–07:26 น. (เช่น 27 ส.ค. 07:00) เป็นภาพขาวขนาด 1.4 KB จึงควรกรองออก",
        "exp_map": "2. แผนที่เชิงพื้นที่ (PyDeck)",
        "map_height": "ความสูงแผนที่ (พิกเซล)",
        "cam_filter": "กรองสถานะผลกระทบกล้อง",
        "min_wet": "จำนวนชั่วโมงฝนตกขั้นต่ำ",
        "traffy_overlay": "ชั้นข้อมูล Heatmap (Traffy Fondue)",
        "show_rail": "แสดงสถานีรถไฟฟ้า (BTS / MRT)",
        "show_dist": "แสดงขอบเขต 50 เขต กทม.",
        "exp_net": "3. กราฟเครือข่ายคมนาคม (Network)",
        "rail_rad": "ระยะสูงสุด กล้อง ↔ สถานีรถไฟฟ้า (ม.)",
        "cam_rad": "ระยะสูงสุด กล้อง ↔ กล้อง (ม.)",
        "k_rail": "จำนวนสถานีใกล้ที่สุดต่อ 1 กล้อง",
        "k_cam": "จำนวนกล้องข้างเคียงต่อ 1 กล้อง",
        "net_comm": "จัดกลุ่มชุมชนพื้นที่ (Greedy Modularity)",
        "net_layout": "อัลกอริทึมจัดวางกราฟ",
        "net_size": "ปรับขนาดโหนดตาม",
        "net_color": "ลงสีโหนดตาม",
        "net_scale": "ระยะกระจายของกราฟ",
        "net_range": "ช่วงขนาดโหนด",
        "net_noverlap": "ป้องกันป้าย/โหนดซ้อนทับ",
        "net_labels_n": "แสดงชื่อกล้องสูงสุด N จุด",
        "net_font": "ขนาดตัวอักษรในกราฟ",
        "net_min_comp": "ซ่อนกลุ่มย่อยที่เล็กกว่า (โหนด)",
        "net_edges": "แสดงเส้นเชื่อม (Edges)",
        "exp_sim": "4. แบบจำลองคาดการณ์ ML แบบโต้ตอบ (What-If)",
        "sim_rain": "ปริมาณฝนจำลอง (มม./ชม.)",
        "sim_hod": "ชั่วโมงของวัน",
        "sim_weekend": "วันหยุดสุดสัปดาห์",
        "sim_rail_dist": "ระยะถึงสถานี BTS/MRT (ม.)",
        "sim_shelter": "ใกล้สะพานลอย / ทางด่วนหลบฝน",
        "sim_base": "ปริมาณรถเฉลี่ยช่วงฝนไม่ตก (คัน/เฟรม)",
        "sim_temp": "อุณหภูมิ (°C)",
        "sim_vis": "ทัศนวิสัย METAR (กม.)",
        "kpi_banner": (
            "**ข้อค้นพบหลัก:** ปริมาณรถจักรยานยนต์ที่ตรวจจับได้ต่อเฟรมลดลง **-21.0%** (ช่วงเชื่อมั่น 95%: -27.4% ถึง -14.4% และลดลงถึง **-30.5%** ในชั่วโมงเร่งด่วน) "
            "เมื่อเทียบกับช่วงเวลาเดียวกันและวันประเภทเดียวกันที่ฝนไม่ตก ส่วนรถยนต์โดยรวมเปลี่ยนแปลงใกล้เคียงศูนย์ (-2.2%) ส่งผลให้ยอดรถรวมลดลง -5.4%"
        ),
        "kpi_all": "ยานพาหนะรวม (ฝนตก)",
        "kpi_cars": "รถยนต์ (ฝนตก)",
        "kpi_moto": "รถจักรยานยนต์ (ฝนตก)",
        "kpi_help": "95% CI {lo} ถึง {hi} จาก {n} ชั่วโมงกล้องที่ฝนตก",
        "kpi_overlap": "ชั่วโมงฝนซ้อนทับ (วันทำงาน vs วันหยุด)",
        "kpi_overlap_hrs": "{n} ชม.",
        "kpi_overlap_help": "วันทำงานฝนตกช่วง 16:00–20:00 น. (69 ชม.กล้อง) ส่วนวันหยุดตกช่วง 12:00–17:00 น. (224 ชม.กล้อง)",
        "tabs": [
            "ภาพรวมและช่วงเวลา",
            "อนุกรมเวลาและจุดผิดปกติ",
            "แผนที่เชิงพื้นที่ (PyDeck)",
            "เครือข่ายคมนาคม (PyVis)",
            "ML และค่าสหสัมพันธ์",
            "สมมติฐาน น้ำท่วม และข่าว",
            "คุณภาพข้อมูลและข้อจำกัด",
        ],
        "ci_title_win": "ผลกระทบของฝนแยกตามช่วงเวลา (95% CI)",
        "ci_title_seg": "จำแนกตามวันทำงาน/วันหยุด และระดับความแรงของฝน",
        "ci_x": "การเปลี่ยนแปลงเมื่อฝนตก (%)",
        "timing_title": "ช่วงเวลาที่ฝนตก: วันทำงาน (16:00–20:00) vs วันหยุด (12:00–17:00) ซ้อนทับเฉพาะ 16:00–17:00 น.",
        "timing_warn": (
            "**ความหมายของ '69 ชม.กล้อง (8 ชม.นาฬิกา, ช่วง 16:00–20:00)':** ในวันจันทร์–ศุกร์ ฝน >= 1 มม. เกิดขึ้นเพียง 8 ชั่วโมงเวลาจริง แต่กระทบหลายกล้องพร้อมกัน (รวม 69 คู่กล้อง-ชั่วโมง) "
            "ขณะที่วันหยุดฝนตกช่วงกลางวัน (12:00–17:00) ทำให้มีเวลาเทียบตรงกันเพียง **16:00 และ 17:00 น.**"
        ),
        "why_dec_exp": "เหตุใดจำนวนรถในภาพนิ่งถึงลดลงเมื่อฝนตก แทนที่จะเพิ่มขึ้นเหมือนเวลารถติด?",
        "why_dec_md": (
            "- **ภาพนิ่งวัด 'ความหนาแน่นเชิงพื้นที่' ไม่ใช่ความเร็วรถ:** ภาพถ่ายทุก 10 นาทีมองเห็นเฉพาะจำนวนรถ ณ ขณะนั้น ไม่สามารถวัดความเร็ว (กม./ชม.) หรืออัตราการไหล (`flow = density × speed`) ได้\n"
            "- **มอเตอร์ไซค์หายออกจากถนนโล่ง (-21.0% ทั่วกรุง, -30.5% ชั่วโมงเร่งด่วน):** ผู้ขับขี่รถจักรยานยนต์เลี่ยงการเดินทางหรือรอให้ฝนหยุดดังเห็นในกราฟ `Rebound` และเนื่องจากมอเตอร์ไซค์คิดเป็น ~20.5% ของปริมาณรถกลางวัน ยอดรถรวมจึงดูลดลง `-5.4%`\n"
            "- **จุดคอขวดต้นน้ำชะลอปริมาณรถที่ไหลเข้ากล้อง:** แม้ถนนบางช่วงจะชะลอตัว แต่ในการทดลองธรรมชาติเปรียบเทียบ **กล้องที่ฝนตกกับกล้องข้างเคียงที่แห้ง (≤ 3 กม.) ในชั่วโมงเดียวกัน 15 ชม.** พบว่า **รถยนต์เพิ่มขึ้น +12.6 จุดร้อยละ** (95% CI +2.2 ถึง +20.7) สะท้อนการเปลี่ยนโหมดจาก 2 ล้อสู่ 4 ล้อ"
        ),
        "ts_empty": "กรุณาเลือกอย่างน้อย 1 วัน และ 1 ประเภทยานพาหนะจากแถบด้านซ้าย",
        "ts_title": "ความหนาแน่นการจราจรรายชั่วโมงเทียบกับปริมาณน้ำฝนเฉลี่ย",
        "ts_y1": "จำนวนรถเฉลี่ย / ภาพ",
        "ts_y2": "น้ำฝนเฉลี่ย (มม./ชม.)",
        "ts_rain_bar": "น้ำฝนเฉลี่ย (มม./ชม.)",
        "rebound_title": "การฟื้นตัวของปริมาณจราจรก่อนและหลังฝนหยุด (-60 ถึง +180 นาที)",
        "rebound_x": "นาทีเทียบกับชั่วโมงแรกหลังฝนหยุด (t = 0)",
        "rebound_y": "อัตราส่วนเทียบค่าฐานปกติ (1.0 = ปกติ)",
        "rebound_cls": "ประเภทยานพาหนะ",
        "outage_info": (
            "**คำอธิบายการตกวูบเวลา 07:00 น. (เช่น 27 ส.ค.):** จากการตรวจสอบตาราง `lake.silver.cctv_frames` พบว่าช่วง **07:21–07:26 น. ของวันที่ 27 ส.ค.** มีภาพมากถึง **71 จาก 73 เฟรม** มีขนาดไฟล์เพียง **1,402–1,404 ไบต์** "
            "(ภาพขาวล้วนจากเซิร์ฟเวอร์ BMA ขัดข้อง ไม่ใช่ความผิดพลาดของโมเดล YOLO) เหตุการณ์ลักษณะเดียวกันยังพบในบางกล้องช่วง 07:00–08:00 น. ของวันอื่น สามารถสลับตัวเลือกกรองภาพเสียได้ที่แถบด้านซ้าย"
        ),
        "map_cap": (
            "กำลังแสดง **{shown} / {total} กล้อง** · ความสูง: `{h}px` · "
            "ฟ้า = รถลดลง, แดง = หนาแน่นขึ้น, เขียว = ไม่เปลี่ยน, ส้ม = ฝนตก <3 ชม., เทา = ไม่มีข้อมูลฝนเปรียบเทียบ"
        ),
        "map_empty": "ไม่มีกล้องที่ตรงกับเงื่อนไขการกรองในแถบด้านซ้าย",
        "net_subhdr": "เครือข่ายเชิงพื้นที่ระหว่างกล้อง CCTV และสถานีรถไฟฟ้าในกรุงเทพฯ",
        "net_metrics": ("จำนวนโหนด", "เส้นเชื่อม", "จำนวนกลุ่มชุมชน", "เส้นผ่านศูนย์กลาง", "ค่า Clustering เฉลี่ย", "ระยะทางเฉลี่ย (Hops)"),
        "net_zoom": "เจาะลึกเฉพาะกลุ่มพื้นที่ (Community Cluster)",
        "net_all_clusters": "แสดงทุกกลุ่มพื้นที่",
        "net_comm_fmt": "กลุ่มที่ #{cid} — กล้อง {cams} ตัว, สถานีรถไฟฟ้า {st} แห่ง ({hubs})",
        "net_corr_only": "เฉพาะแนวถนน (ไม่มีสถานีใกล้เคียง)",
        "net_dl": "ดาวน์โหลดกราฟแบบโต้ตอบ (HTML)",
        "net_rand": "เปรียบเทียบกับกราฟสุ่ม (Erdős–Rényi ที่มีจำนวนโหนด/เส้นเชื่อมเท่ากัน)",
        "net_links_hdr": "ตารางจับคู่การเชื่อมต่อ: กล้อง ↔ สถานีรถไฟฟ้า ↔ กล้องข้างเคียง",
        "net_comms_hdr": "สรุปกลุ่มพื้นที่ (Spatial Corridor Communities) ที่ตรวจพบ",
        "net_deg_hdr": "การกระจายตัวของดีกรี (Degree Distribution)",
        "net_scale_radio": "รูปแบบสเกล",
        "net_scale_opts": ["ฮิสโทแกรม", "สเกลลอการิทึม (CCDF)"],
        "net_cent_hdr": "ค่าความเป็นศูนย์กลาง (Centrality Scores)",
        "ml_subhdr": "แบบจำลอง Machine Learning คาดการณ์ความหนาแน่นจราจร (จาก PyDuckLake)",
        "ml_missing": "ไม่พบตาราง ML ใน PyDuckLake กรุณารัน `python DataAnalysis/rain/ml_model.py`",
        "ml_desc": (
            "การถดถอยแบบมีผู้สอน (**Regression**: `HistGradientBoostingRegressor`, `RandomForestRegressor`, `Ridge`) ทำนายจำนวนรถต่อเฟรม "
            "และการจำแนกประเภท (**Classification**) เตือนจุดที่รถหนาแน่นพุ่งเกิน `+25%` ทดสอบทั้งแบบแบ่งตามเวลา (ฝึก 25–29 ส.ค. $\\to$ ทดสอบ 30–31 ส.ค.) และแบ่งตามกลุ่มพื้นที่ 5-Fold"
        ),
        "ml_imp_sel": "เลือกเป้าหมายสำหรับ Permutation Importance",
        "ml_imp_x": "ค่าความสำคัญเฉลี่ย (Permutation R² drop)",
        "ml_imp_y": "ตัวแปร (Feature)",
        "ml_imp_title": "ปัจจัยสำคัญที่สุด: {tgt}",
        "ml_pred_wet": "ฝนตก (>=1 มม.)",
        "ml_pred_dry": "ฝนไม่ตก (<1 มม.)",
        "ml_pred_x": "ปริมาณรถจริง / เฟรม",
        "ml_pred_y": "ปริมาณรถที่คาดการณ์ / เฟรม",
        "ml_pred_title": "ผลทำนายชุดทดสอบ 30–31 ส.ค.: ค่าจริงเทียบค่าคาดการณ์",
        "sim_subhdr": "แบบจำลองสถานการณ์แบบโต้ตอบ (ตั้งค่าได้ในแถบด้านซ้ายข้อ 4)",
        "sim_cards": ("คาดการณ์รวมทั้งหมด", "คาดการณ์รถยนต์", "คาดการณ์รถจักรยานยนต์"),
        "sim_unit": "{val:.2f} คัน/เฟรม",
        "sim_delta": "{delta:+.1f}% เทียบฝน 0 มม.",
        "corr_subhdr": "ค่าสหสัมพันธ์เชิงเส้น Pearson (r) ระหว่างตัวแปรสภาพอากาศกับจราจร",
        "corr_scope": "ขอบเขตข้อมูล",
        "corr_hourly": "ข้อมูลรายชั่วโมง (n ≈ 9,996)",
        "corr_cam": "ระดับผลกระทบรายกล้อง (n = 94)",
        "corr_x": "ค่าสหสัมพันธ์ Pearson (r)",
        "corr_y": "ตัวแปร",
        "corr_t": "เป้าหมาย",
        "corr_title": "สหสัมพันธ์เชิงเส้นของตัวแปรสภาพอากาศ พื้นที่ และปริมาณรถ",
        "hyp_1": "1) การทดลองธรรมชาติในชั่วโมงเดียวกัน: กล้องที่ฝนตก vs กล้องแห้งข้างเคียง (≤ 3 กม.)",
        "hyp_1_title": "การทดลองธรรมชาติ (15 ชั่วโมงเวลาจริง · 172 คู่กล้อง)",
        "hyp_1_x": "ส่วนต่างเป็น % เทียบค่าฐาน (Percentage Points)",
        "hyp_1_cap": "รถยนต์ในพื้นที่ฝนตกสูงกว่าพื้นที่แห้งใกล้เคียงถึง +12.59 จุดร้อยละ (CI +2.2 ถึง +20.7) สะท้อนการเปลี่ยนผ่านจากมอเตอร์ไซค์สู่รถยนต์เมื่อฝนตกเฉพาะจุด",
        "hyp_2": "2) สัดส่วนกล้องใกล้สะพานลอยหลบฝนและรถไฟฟ้าของจุดที่รถแน่นขึ้น",
        "hyp_3": "3) จุดตรวจวัดน้ำท่วม DDS ≥ 5 ซม. เทียบกล้องถนนเพชรบุรี",
        "hyp_3_cap": "สถานี `FL.RTW.02 ถนนเพชรบุรี` พบน้ำท่วม ≥ 5 ซม. ทั้งหมด 13 ชม. (5.0–5.1 ซม.) โดยกล้อง #250 และ #256 บนถนนเดียวกันไม่พบความหนาแน่นเพิ่มขึ้นผิดปกติ",
        "hyp_4": "4) จำแนกเรื่องร้องเรียน Traffy Fondue",
        "hyp_5": "5) ข่าวจราจร FM91 รายวัน (25–31 ส.ค.)",
        "audit_rows": "จำนวนเฟรมใน DuckLake",
        "audit_cams": "จำนวนกล้องที่วิเคราะห์",
        "audit_wet": "ชั่วโมงกล้องที่มีฝนตก",
        "audit_hdr": "การตรวจสอบหลักฐานเชิงสาเหตุและความครบถ้วนของข้อมูล",
        "audit_exp": "ดูกราฟภาพนิ่งต้นฉบับ (DataAnalysis/figures/*.png)",
        "audit_footer": "แดชบอร์ดนี้สืบค้นข้อมูลสดจาก PyDuckLake (`DataEngineering/lakehouse/bkk_cctv.ducklake`) โดยตรง",
        "scorecard": [
            {"โหนดเชิงสาเหตุ": "ข้อมูลน้ำฝน (>= 1 มม.)", "สถานะข้อมูล": "มีครบ (129 สถานีวัดฝน กทม.)", "หลักฐานและข้อจำกัดที่พบ": "จับคู่ได้ 293 ชม.กล้อง; รถจักรยานยนต์ลดลง -21.0%"},
            {"โหนดเชิงสาเหตุ": "มอเตอร์ไซค์จอดหลบฝนใต้สะพานลอย", "สถานะข้อมูล": "ทดสอบด้วย Background Subtraction", "หลักฐานและข้อจำกัดที่พบ": "ต่างกันเล็กน้อย (2.5% แห้ง vs 2.8% ฝน); ส่วนใหญ่เป็นรถจอดริมฟุตบาท ไม่ได้ขวาง 1–2 ช่องจราจรหลัก"},
            {"โหนดเชิงสาเหตุ": "จุดน้ำท่วมขัง DDS >= 5 ซม.", "สถานะข้อมูล": "ไม่มีพิกัด GPS ในตาราง DDS", "หลักฐานและข้อจำกัดที่พบ": "พบค่า >=5 ซม. เฉพาะสถานี FL.RTW.02 เพชรบุรี; กล้อง #250/#256 บนสายเดียวกันไม่พบรถสะสมเพิ่ม"},
            {"โหนดเชิงสาเหตุ": "Traffy Fondue (น้ำท่วม/ถนน/รถติด)", "สถานะข้อมูล": "มีครบ (7,247 เรื่องที่มีพิกัด)", "หลักฐานและข้อจำกัดที่พบ": "ส่วนใหญ่อยู่ในตรอกซอยย่อย (มีกล้องเพียง 2 จุดในระยะ 100 ม. จากคลัสเตอร์น้ำท่วม)"},
            {"โหนดเชิงสาเหตุ": "เปลี่ยนไปขึ้นรถไฟฟ้า & เรียกรถเพิ่ม +25%", "สถานะข้อมูล": "ยังไม่มีข้อมูลผู้โดยสารโดยตรง", "หลักฐานและข้อจำกัดที่พบ": "การทดลองธรรมชาติพบรถยนต์พุ่ง +12.6 จุดร้อยละ แต่ยังไม่มีสถิติผู้โดยสาร BTS/MRT หรือยอดเรียกรถแท็กซี่"},
            {"โหนดเชิงสาเหตุ": "ถนนลดความจุ -> รถติดขัด", "สถานะข้อมูล": "วัดเฉพาะความหนาแน่น (ไม่มีความเร็ว)", "หลักฐานและข้อจำกัดที่พบ": "ภาพนิ่งทุก 10 นาทีบอกจำนวนรถต่อเฟรมได้ แต่ไม่บอกความเร็ว (กม./ชม.) หรือเวลาหน่วงคิวสะสม"},
        ],
    },
}


def tr(key: str, lang: str = "en", **fmt):
    """Look up localized UI text with optional string formatting."""
    bucket = UI.get(lang, UI["en"])
    val = bucket.get(key, UI["en"].get(key, key))
    return val.format(**fmt) if (isinstance(val, str) and fmt) else val


def localize_network_tables(cam_links_df: pd.DataFrame, comm_summary_df: pd.DataFrame, lang: str):
    """Return display copies of network summary tables localized to English or Thai."""
    if lang != "th":
        return cam_links_df, comm_summary_df
    status_th = LABEL_TEXT_BY_LANG["th"]
    cam_out = cam_links_df.copy()
    if "Status" in cam_out.columns:
        cam_out["Status"] = cam_out["Status"].map(lambda s: status_th.get(s, s))
    cam_out = cam_out.rename(columns={
        "Camera": "รหัสกล้อง",
        "Location (TH)": "ตำแหน่งกล้อง",
        "Community": "กลุ่มชุมชน",
        "Linked Stations": "สถานีรถไฟฟ้าที่เชื่อมโยง",
        "Neighbor Cameras": "กล้องข้างเคียง",
        "Degree": "จำนวนการเชื่อม (Degree)",
        "Rain Effect (%)": "ผลกระทบช่วงฝน (%)",
        "Status": "สถานะ",
    })
    comm_out = comm_summary_df.rename(columns={
        "Community": "กลุ่มชุมชน",
        "Cameras": "จำนวนกล้อง",
        "Stations": "จำนวนสถานี",
        "Station Hubs": "จุดเชื่อมต่อรถไฟฟ้า",
        "Sample Cameras": "ตัวอย่างกล้องในกลุ่ม",
        "Mean Rain Effect (%)": "ผลกระทบฝนเฉลี่ย (%)",
    })
    return cam_out, comm_out
