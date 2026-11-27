"""Generate the Power BI Project (.pbip) backed by DataAnalysis/output/*.csv.

Produces:
  DataAnalysis/PowerBI/
  ├── Bangkok_Rain_Traffic_Story.pbip
  ├── Bangkok_Rain_Traffic_Story.SemanticModel/   (TMDL model + Business DAX measures)
  └── Bangkok_Rain_Traffic_Story.Report/          (PBIR 5-page analytical report)

Usage:
    python DataAnalysis/PowerBI/build_powerbi_project.py
"""

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "DataAnalysis" / "output"
PBI_DIR = ROOT / "DataAnalysis" / "PowerBI"
PROJECT_NAME = "Bangkok_Rain_Traffic_Story"


def _uid(seed: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"bkk-cctv-{seed}"))


def _vid(seed: str) -> str:
    """Short deterministic PBIR visual identifier safe under Windows MAX_PATH (260 chars)."""
    return seed.replace("-", "_")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _tmdl_dtype(dtype) -> tuple[str, str]:
    s = str(dtype).lower()
    if "int" in s:
        return "int64", "Int64.Type"
    if "float" in s or "double" in s:
        return "double", "type number"
    if "bool" in s:
        return "boolean", "type logical"
    if "datetime" in s:
        return "dateTime", "type datetime"
    return "string", "type text"


def _quote_col(col: str) -> str:
    return f"'{col}'" if any(ch in col for ch in (" ", "-", "/", "(", ")", "%", ".")) else col


def _build_table_tmdl(table_name: str, csv_name: str, csv_path: Path, extra_measures: str = "") -> str:
    df = pd.read_csv(csv_path, nrows=200, encoding="utf-8-sig")
    lines = [
        f"table {table_name}",
        f"\tlineageTag: {_uid('tbl-' + table_name)}",
        "",
    ]
    if extra_measures:
        lines.append(extra_measures.rstrip())
        lines.append("")

    type_pairs = []
    for col in df.columns:
        tmdl_t, m_t = _tmdl_dtype(df[col].dtype)
        if col == "hour_ts":
            tmdl_t, m_t = "dateTime", "type datetime"
        elif col == "camera_id":
            tmdl_t, m_t = "int64", "Int64.Type"
        summarize = "none" if (tmdl_t in ("string", "boolean", "dateTime") or col in ("camera_id", "latitude", "longitude", "cluster", "hod", "hour", "minute")) else "average"
        qname = _quote_col(col)
        lines.extend([
            f"\tcolumn {qname}",
            f"\t\tdataType: {tmdl_t}",
            f"\t\tlineageTag: {_uid('col-' + table_name + '-' + col)}",
            f"\t\tsummarizeBy: {summarize}",
            f"\t\tsourceColumn: {col}",
            "",
        ])
        type_pairs.append(f'{{"{col}", {m_t}}}')

    cols_count = len(df.columns)
    m_transform = ", ".join(type_pairs)
    csv_abs = str(csv_path.resolve())
    lines.extend([
        f"\tpartition {table_name} = m",
        "\t\tmode: import",
        "\t\tsource =",
        "\t\t\tlet",
        f'\t\t\t\tSource = Csv.Document(File.Contents("{csv_abs}"), [Delimiter=",", Columns={cols_count}, Encoding=65001, QuoteStyle=QuoteStyle.Csv]),',
        "\t\t\t\tPromotedHeaders = Table.PromoteHeaders(Source, [PromoteAllScalars=true]),",
        f"\t\t\t\tChangedType = Table.TransformColumnTypes(PromotedHeaders, {{{m_transform}}})",
        "\t\t\tin",
        "\t\t\t\tChangedType",
        "",
    ])
    return "\n".join(lines)


KPI_MEASURES_TMDL = f"""\tmeasure 'Moto Rain Effect %' = CALCULATE(AVERAGE(city_effect[effect_pct]), city_effect[cls] = "motorcycle", city_effect[window] = "all")
\t\tformatString: +0.0"%";-0.0"%";0.0"%"
\t\tlineageTag: {_uid('m-moto-eff')}

\tmeasure 'Rush Moto Drop %' = CALCULATE(AVERAGE(city_effect[effect_pct]), city_effect[cls] = "motorcycle", city_effect[window] = "rush")
\t\tformatString: +0.0"%";-0.0"%";0.0"%"
\t\tlineageTag: {_uid('m-rush-moto')}

\tmeasure 'Car Rain Effect %' = CALCULATE(AVERAGE(city_effect[effect_pct]), city_effect[cls] = "car", city_effect[window] = "all")
\t\tformatString: +0.0"%";-0.0"%";0.0"%"
\t\tlineageTag: {_uid('m-car-eff')}

\tmeasure 'Total Rain Effect %' = CALCULATE(AVERAGE(city_effect[effect_pct]), city_effect[cls] = "total", city_effect[window] = "all")
\t\tformatString: +0.0"%";-0.0"%";0.0"%"
\t\tlineageTag: {_uid('m-tot-eff')}

\tmeasure 'Localized Car Shift (pts)' = CALCULATE(AVERAGE(natural_experiment[effect_pct_points]), natural_experiment[cls] = "car")
\t\tformatString: +0.00" pts";-0.00" pts";0.00" pts"
\t\tlineageTag: {_uid('m-car-shift')}

\tmeasure 'Wet Camera-Hours' = CALCULATE(MAX(city_effect[n_wet_hours]), city_effect[cls] = "total", city_effect[window] = "all")
\t\tformatString: #,##0
\t\tlineageTag: {_uid('m-wet-hrs')}

\tmeasure 'Cameras Analyzed' = COUNTROWS(camera)
\t\tformatString: #,##0
\t\tlineageTag: {_uid('m-cam-cnt')}

\tmeasure 'ML Weekend R2 (Baseline)' = CALCULATE(MAX(ml_metrics[r2]), ml_metrics[target] = "total (with camera baseline)", ml_metrics[split] = "temporal_aug30_31", ml_metrics[model] = "HistGradientBoosting")
\t\tformatString: 0.00
\t\tlineageTag: {_uid('m-r2')}

\tmeasure 'Peak Flood Depth (cm)' = MAX(flood_sensor_hours[flood_cm])
\t\tformatString: 0.0" cm"
\t\tlineageTag: {_uid('m-flood-cm')}"""


def _textbox(vis_id: str, x: int, y: int, w: int, h: int, z: int, text: str, font_size: str = "13pt", bold: bool = False, color: str = "#1E293B") -> dict:
    run = {
        "value": text,
        "textStyle": {
            "fontSize": font_size,
            "fontFamily": "Segoe UI",
            "color": color,
        },
    }
    if bold:
        run["textStyle"]["fontWeight"] = "bold"
    return {
        "$schema": "https://developer.microsoft.com/json-schemas/fabric/item/report/definition/visualContainer/1.7.0/schema.json",
        "name": vis_id,
        "position": {"x": x, "y": y, "z": z, "height": h, "width": w, "tabOrder": z},
        "visual": {
            "visualType": "textbox",
            "objects": {
                "general": [{"properties": {"paragraphs": [{"textRuns": [run]}]}}]
            },
        },
    }


def _col_field(entity: str, prop: str) -> dict:
    return {"Column": {"Expression": {"SourceRef": {"Entity": entity}}, "Property": prop}}


def _meas_field(entity: str, prop: str) -> dict:
    return {"Measure": {"Expression": {"SourceRef": {"Entity": entity}}, "Property": prop}}


def _card_visual(vis_id: str, x: int, y: int, w: int, h: int, z: int, entity: str, measure_name: str, title: str) -> dict:
    return {
        "$schema": "https://developer.microsoft.com/json-schemas/fabric/item/report/definition/visualContainer/1.7.0/schema.json",
        "name": vis_id,
        "position": {"x": x, "y": y, "z": z, "height": h, "width": w, "tabOrder": z},
        "visual": {
            "visualType": "card",
            "query": {
                "queryState": {
                    "Values": {
                        "projections": [{
                            "field": _meas_field(entity, measure_name),
                            "queryRef": f"{entity}.{measure_name}",
                            "nativeQueryRef": measure_name,
                        }]
                    }
                }
            },
            "objects": {
                "categoryLabels": [{"properties": {"show": {"expr": {"Literal": {"Value": "false"}}}}}],
            },
            "visualContainerObjects": {
                "title": [{"properties": {
                    "show": {"expr": {"Literal": {"Value": "true"}}},
                    "text": {"expr": {"Literal": {"Value": f"'{title}'"}}},
                }}],
            },
        },
    }


def _chart_visual(vis_id: str, vtype: str, x: int, y: int, w: int, h: int, z: int, title: str,
                  cat_entity: str, cat_col: str, y_projections: list[tuple[str, str, bool]],
                  series_entity: str | None = None, series_col: str | None = None) -> dict:
    qs: dict = {
        "Category": {
            "projections": [{
                "field": _col_field(cat_entity, cat_col),
                "queryRef": f"{cat_entity}.{cat_col}",
                "nativeQueryRef": cat_col,
                "active": True,
            }]
        },
        "Y": {
            "projections": [
                {
                    "field": _meas_field(ent, col) if is_m else {
                        "Aggregation": {"Expression": _col_field(ent, col), "Function": 1}
                    },
                    "queryRef": f"{ent}.{col}" if is_m else f"Avg({ent}.{col})",
                    "nativeQueryRef": col,
                }
                for ent, col, is_m in y_projections
            ]
        },
    }
    if series_entity and series_col:
        qs["Series"] = {
            "projections": [{
                "field": _col_field(series_entity, series_col),
                "queryRef": f"{series_entity}.{series_col}",
                "nativeQueryRef": series_col,
            }]
        }
    return {
        "$schema": "https://developer.microsoft.com/json-schemas/fabric/item/report/definition/visualContainer/1.7.0/schema.json",
        "name": vis_id,
        "position": {"x": x, "y": y, "z": z, "height": h, "width": w, "tabOrder": z},
        "visual": {
            "visualType": vtype,
            "query": {"queryState": qs},
            "visualContainerObjects": {
                "title": [{"properties": {
                    "show": {"expr": {"Literal": {"Value": "true"}}},
                    "text": {"expr": {"Literal": {"Value": f"'{title}'"}}},
                }}],
            },
        },
    }


def _table_visual(vis_id: str, x: int, y: int, w: int, h: int, z: int, title: str,
                  cols: list[tuple[str, str]]) -> dict:
    return {
        "$schema": "https://developer.microsoft.com/json-schemas/fabric/item/report/definition/visualContainer/1.7.0/schema.json",
        "name": vis_id,
        "position": {"x": x, "y": y, "z": z, "height": h, "width": w, "tabOrder": z},
        "visual": {
            "visualType": "tableEx",
            "query": {
                "queryState": {
                    "Values": {
                        "projections": [{
                            "field": _col_field(ent, col),
                            "queryRef": f"{ent}.{col}",
                            "nativeQueryRef": col,
                        } for ent, col in cols]
                    }
                }
            },
            "visualContainerObjects": {
                "title": [{"properties": {
                    "show": {"expr": {"Literal": {"Value": "true"}}},
                    "text": {"expr": {"Literal": {"Value": f"'{title}'"}}},
                }}],
            },
        },
    }


def build_project() -> Path:
    """Create the full Power BI Project (.pbip + SemanticModel + 5-Page Report)."""
    # Ensure Executive Summary CSV exists for Page 1 & Page 5
    exec_csv = OUT_DIR / "executive_story.csv"
    pd.DataFrame([
        {"step": "1. Citywide Impact", "headline": "Total traffic falls -5.4% in rain", "evidence": "Motorcycles drop -21.0% (-30.5% in rush hours) across 293 wet camera-hours, masking a car shift."},
        {"step": "2. Localized Modal Shift", "headline": "Localized rain shifts travel from 2-wheelers to 4-wheelers", "evidence": "Across 15 hours comparing wet vs dry cameras <=3km apart (172 pairs), cars rise +12.6 percentage points."},
        {"step": "3. Sheltering & Station Test", "headline": "Footbridge sheltering & station proximity do not explain congestion", "evidence": "Stationary bikes near shelters shifted only 2.5% dry vs 2.8% wet (sidewalk parking); 20% of denser cams sit near rail."},
        {"step": "4. Predictive ML & QA", "headline": "Rain is the #1 weather feature for motorcycle prediction", "evidence": "HistGradientBoosting reaches R2=0.83 (total) & 0.75 (moto); BMA 07:00 drop proved to be 1.4 KB blank outage frames."},
        {"step": "5. Operational Actions", "headline": "Connect rain alerts to transit feeder & fleet dispatch", "evidence": "Pre-position 4-wheel ride-hailing + BTS/MRT feeder shuttles on high rail-density corridors ahead of 16:00-20:00 rain."},
    ]).to_csv(exec_csv, index=False, lineterminator="\n")

    sm_dir = PBI_DIR / f"{PROJECT_NAME}.SemanticModel"
    rep_dir = PBI_DIR / f"{PROJECT_NAME}.Report"
    PBI_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Root .pbip
    _write_json(PBI_DIR / f"{PROJECT_NAME}.pbip", {
        "$schema": "https://developer.microsoft.com/json-schemas/fabric/pbip/pbipProperties/1.0.0/schema.json",
        "version": "1.0",
        "artifacts": [{"report": {"path": f"{PROJECT_NAME}.Report"}}],
        "settings": {"enableAutoRecovery": True},
    })

    # 2. Semantic Model (.SemanticModel)
    _write_json(sm_dir / "definition.pbism", {
        "$schema": "https://developer.microsoft.com/json-schemas/fabric/item/semanticModel/definitionProperties/1.0.0/schema.json",
        "version": "4.2",
        "settings": {"qnaEnabled": True},
    })

    def_sm = sm_dir / "definition"
    def_sm.mkdir(parents=True, exist_ok=True)
    (def_sm / "database.tmdl").write_text(
        f"database {_uid('db')}\n\tcompatibilityLevel: 1702\n\tcompatibilityMode: powerBI\n\tlanguage: 1033\n",
        encoding="utf-8",
    )

    out_folder_escaped = str(OUT_DIR.resolve()).rstrip("\\/") + "\\"
    (def_sm / "expressions.tmdl").write_text(
        f'expression DataFolder = "{out_folder_escaped}" meta [IsParameterQuery=true, Type="Text", IsParameterQueryRequired=true]\n'
        f'\tlineageTag: {_uid("param-datafolder")}\n'
        '\tannotation PBI_ResultType = Text\n',
        encoding="utf-8",
    )

    tables_spec = [
        ("city_effect", "city_effect.csv", KPI_MEASURES_TMDL),
        ("stratified_effects", "stratified_effects.csv", ""),
        ("rain_timing", "rain_timing.csv", ""),
        ("natural_experiment", "natural_experiment.csv", ""),
        ("rebound", "rebound.csv", ""),
        ("camera", "camera.csv", ""),
        ("camera_effect", "camera_effect.csv", ""),
        ("camera_hour", "camera_hour.csv", ""),
        ("denser_share", "denser_share.csv", ""),
        ("flood_sensor_hours", "flood_sensor_hours.csv", ""),
        ("ml_metrics", "ml_metrics.csv", ""),
        ("ml_feature_importance", "ml_feature_importance.csv", ""),
        ("correlations", "correlations.csv", ""),
        ("executive_story", "executive_story.csv", ""),
    ]

    model_lines = [
        "model Model",
        "\tculture: en-US",
        "\tdefaultPowerBIDataSourceVersion: powerBI_V3",
        "\tsourceQueryCulture: en-US",
        '\tannotation PBI_ProTooling = ["DevMode"]',
        "",
    ]
    tables_dir = def_sm / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    for tbl, fname, measures in tables_spec:
        csv_path = OUT_DIR / fname
        if not csv_path.is_file():
            continue
        model_lines.append(f"ref table {tbl}")
        (tables_dir / f"{tbl}.tmdl").write_text(
            _build_table_tmdl(tbl, fname, csv_path, extra_measures=measures),
            encoding="utf-8",
        )
    (def_sm / "model.tmdl").write_text("\n".join(model_lines) + "\n", encoding="utf-8")

    (def_sm / "relationships.tmdl").write_text(
        "\n".join([
            f"relationship {_uid('rel-cam-hour')}",
            "\tfromColumn: camera_hour.camera_id",
            "\ttoColumn: camera.camera_id",
            "",
            f"relationship {_uid('rel-cam-eff')}",
            "\tfromColumn: camera_effect.camera_id",
            "\ttoColumn: camera.camera_id",
            "",
        ]),
        encoding="utf-8",
    )

    # 3. PBIR Report (.Report)
    _write_json(rep_dir / "definition.pbir", {
        "$schema": "https://developer.microsoft.com/json-schemas/fabric/item/report/definitionProperties/2.0.0/schema.json",
        "version": "4.0",
        "datasetReference": {"byPath": {"path": f"../{PROJECT_NAME}.SemanticModel"}},
    })

    def_rep = rep_dir / "definition"
    _write_json(def_rep / "version.json", {
        "$schema": "https://developer.microsoft.com/json-schemas/fabric/item/report/definition/versionMetadata/1.0.0/schema.json",
        "version": "2.0.0",
    })
    _write_json(def_rep / "report.json", {
        "$schema": "https://developer.microsoft.com/json-schemas/fabric/item/report/definition/report/1.0.0/schema.json",
        "themeCollection": {
            "baseTheme": {"name": "CY24SU08", "reportVersionAtImport": "5.58", "type": "SharedResources"}
        },
        "layoutOptimization": "None",
    })

    pages = [
        ("page01_summary", "01. Executive Summary", [
            _textbox(_vid("p1-hdr"), 24, 16, 1232, 54, 1,
                     "Bangkok Rain & Urban Mobility: Still Frames Miss the Modal Shift", "22pt", True, "#0F172A"),
            _textbox(_vid("p1-sub"), 24, 68, 1232, 34, 2,
                     "100 BMA CCTV Cameras (60,721 Clean Counts) + 129 Rain Gauges + 188 Transit Hubs + Supervised ML (25–31 Aug 2026)", "11pt", False, "#475569"),
            _card_visual(_vid("p1-c1"), 24, 114, 235, 118, 3, "city_effect", "Cameras Analyzed", "Cameras Monitored"),
            _card_visual(_vid("p1-c2"), 273, 114, 235, 118, 4, "city_effect", "Wet Camera-Hours", "Matched Rain Hours"),
            _card_visual(_vid("p1-c3"), 522, 114, 235, 118, 5, "city_effect", "Moto Rain Effect %", "Citywide Motorcycle Drop"),
            _card_visual(_vid("p1-c4"), 771, 114, 235, 118, 6, "city_effect", "Rush Moto Drop %", "Rush Hour Moto Drop"),
            _card_visual(_vid("p1-c5"), 1020, 114, 236, 118, 7, "city_effect", "Localized Car Shift (pts)", "Wet vs Dry Cars (<=3km)"),
            _table_visual(_vid("p1-story"), 24, 248, 740, 444, 8, "Executive Insights & Operational Actions",
                          [("executive_story", "step"), ("executive_story", "headline"), ("executive_story", "evidence")]),
            _chart_visual(_vid("p1-bar"), "clusteredColumnChart", 782, 248, 474, 444, 9,
                          "Headline Change in Vehicles Visible per Frame (%)", "city_effect", "cls",
                          [("city_effect", "effect_pct", False)], "city_effect", "window"),
        ]),
        ("page02_impact", "02. Rain Effect & Timing", [
            _textbox(_vid("p2-hdr"), 24, 16, 1232, 48, 1,
                     "Who Disappears in Rain? Motorcycles Plunge -21%, While Rain Timing Drives Weekday Differences", "18pt", True, "#0F172A"),
            _chart_visual(_vid("p2-eff"), "clusteredBarChart", 24, 76, 608, 300, 2,
                          "Rain Impact by Vehicle Mode & Time Window (%)", "city_effect", "window",
                          [("city_effect", "effect_pct", False)], "city_effect", "cls"),
            _chart_visual(_vid("p2-time"), "clusteredColumnChart", 648, 76, 608, 300, 3,
                          "Why Timing Matters: Weekday Rain (16-20h) vs Weekend Rain (12-17h)", "rain_timing", "hour",
                          [("rain_timing", "n_camera_hours", False)], "rain_timing", "day_type"),
            _chart_visual(_vid("p2-reb"), "lineChart", 24, 392, 608, 304, 4,
                          "Post-Rain Recovery Curve: Cancelled Trips Don't Bounce Back Immediately", "rebound", "minute",
                          [("rebound", "ratio", False)], "rebound", "cls"),
            _chart_visual(_vid("p2-strat"), "clusteredBarChart", 648, 392, 608, 304, 5,
                          "Subgroup Splits (Weekday/Weekend & Light/Heavy Rain)", "stratified_effects", "segment",
                          [("stratified_effects", "effect_pct", False)], "stratified_effects", "cls"),
        ]),
        ("page03_shift", "03. Modal Shift & Spatial Proof", [
            _textbox(_vid("p3-hdr"), 24, 16, 1232, 48, 1,
                     "Natural Experiment & Hypothesis Verification: Modal Shift & Spatial Controls", "18pt", True, "#0F172A"),
            _chart_visual(_vid("p3-nat"), "clusteredColumnChart", 24, 76, 580, 290, 2,
                          "Same-Hour Neighbor Test (<=3 km, 15 hrs): Cars +12.6 pts Higher Where Rain Falls",
                          "natural_experiment", "cls", [("natural_experiment", "effect_pct_points", False)]),
            _chart_visual(_vid("p3-den"), "clusteredBarChart", 620, 76, 636, 290, 3,
                          "Shelter & Station Share Comparison: No Congestion Concentration",
                          "denser_share", "feature", [("denser_share", "denser_share", False), ("denser_share", "other_share", False)]),
            _table_visual(_vid("p3-cams"), 24, 382, 740, 312, 4,
                          "Per-Camera Rain Effect Leaderboard (14 Clears, 5 Denser, 57 Unaffected)",
                          [("camera", "camera_id"), ("camera", "location_th"), ("camera_effect", "label"),
                           ("camera_effect", "effect_pct"), ("camera_effect", "n_wet_hours"), ("camera", "rail_dist_m")]),
            _table_visual(_vid("p3-flood"), 780, 382, 476, 312, 5,
                          "DDS Flood Sensors >= 5 cm (FL.RTW.02 Phetchaburi Rd)",
                          [("flood_sensor_hours", "station"), ("flood_sensor_hours", "hour_ts"), ("flood_sensor_hours", "flood_cm")]),
        ]),
        ("page04_ml", "04. Supervised ML & Drivers", [
            _textbox(_vid("p4-hdr"), 24, 16, 1232, 48, 1,
                     "Predictive ML Models & Spatial Drivers: What Drives Urban Volume vs Rain Drop?", "18pt", True, "#0F172A"),
            _card_visual(_vid("p4-c1"), 24, 72, 390, 100, 2, "city_effect", "ML Weekend R2 (Baseline)", "HistGB Test R2 (With Baseline)"),
            _card_visual(_vid("p4-c2"), 430, 72, 390, 100, 3, "city_effect", "Peak Flood Depth (cm)", "Max Recorded Flood Sensor"),
            _card_visual(_vid("p4-c3"), 836, 72, 420, 100, 4, "city_effect", "Total Rain Effect %", "Overall Frame Volume Delta"),
            _chart_visual(_vid("p4-imp"), "clusteredBarChart", 24, 186, 608, 255, 5,
                          "ML Permutation Importance (Rain_mm Drives Motorcycle Drops)",
                          "ml_feature_importance", "feature", [("ml_feature_importance", "importance_mean", False)],
                          "ml_feature_importance", "target"),
            _chart_visual(_vid("p4-corr"), "clusteredBarChart", 648, 186, 608, 255, 6,
                          "Linear Correlations (Rail Proximity r=-0.26, Weekend r=-0.18)",
                          "correlations", "feature_label", [("correlations", "pearson_r", False)],
                          "correlations", "target"),
            _table_visual(_vid("p4-met"), 24, 456, 1232, 240, 7,
                          "Cross-Validated Regression & Classification Benchmark (Aug 25–31, n=9,996 clean camera-hours)",
                          [("ml_metrics", "target"), ("ml_metrics", "model"), ("ml_metrics", "split"),
                           ("ml_metrics", "mae"), ("ml_metrics", "rmse"), ("ml_metrics", "r2"), ("ml_metrics", "roc_auc")]),
        ]),
        ("page05_action", "05. Action & Adaptability", [
            _textbox(_vid("p5-hdr"), 24, 16, 1232, 54, 1,
                     "Business Recommendations & Smart-City Data Adaptability", "20pt", True, "#0F172A"),
            _textbox(_vid("p5-act1"), 24, 86, 392, 250, 2,
                     "1. BTS/MRT Feeder Alerting\nDuring weekday 16:00–20:00 rain, motorcycle supply plunges -30.5% while station-adjacent cameras carry high volume (r = -0.26). Trigger dynamic bus/shuttle feeder dispatch on rail hubs.", "13pt", False, "#0F172A"),
            _textbox(_vid("p5-act2"), 444, 86, 392, 250, 3,
                     "2. Ride-Hailing Fleet Pre-Positioning\nWhere rain falls locally, car demand rises +12.6 pts vs dry neighbors within 3 km. Pair DDS 129-gauge rain alerts with ML forecasts to pre-stage 4-wheel drivers before surge spikes.", "13pt", False, "#0F172A"),
            _textbox(_vid("p5-act3"), 864, 86, 392, 250, 4,
                     "3. Automated Feed QA Guardrail\nBMA feeds emitted 1.4 KB solid-white outage frames at 07:21–07:26 (e.g., Aug 27 07:00). Filtering file_size_bytes < 3 KB in PyDuckLake Silver/Gold prevents false zero-traffic alarms.", "13pt", False, "#0F172A"),
            _textbox(_vid("p5-road"), 24, 356, 1232, 336, 5,
                     "Next-Phase Data Adaptability Roadmap:\n"
                     "• Add BTS/MRT Turnstile Tap-In Telemetry -> Directly validate passenger transit shift volumes.\n"
                     "• Add GPS Coordinates to DDS Flood Sensors -> Enable point-to-camera flood/congestion causality beyond road-name joins.\n"
                     "• Capture 1-Minute Video / Optical Flow Speed -> Measure true queue delay & capacity drop alongside frame vehicle counts.", "13pt", False, "#334155"),
        ]),
    ]

    pages_dir = def_rep / "pages"
    if pages_dir.exists():
        shutil.rmtree(pages_dir)
    page_ids = [p[0] for p in pages]
    _write_json(pages_dir / "pages.json", {
        "$schema": "https://developer.microsoft.com/json-schemas/fabric/item/report/definition/pagesMetadata/1.0.0/schema.json",
        "pageOrder": page_ids,
        "activePageName": page_ids[0],
    })

    for pid, title, visuals in pages:
        pdir = pages_dir / pid
        _write_json(pdir / "page.json", {
            "$schema": "https://developer.microsoft.com/json-schemas/fabric/item/report/definition/page/1.2.0/schema.json",
            "name": pid,
            "displayName": title,
            "displayOption": "FitToPage",
            "height": 720,
            "width": 1280,
        })
        for vis in visuals:
            _write_json(pdir / "visuals" / vis["name"] / "visual.json", vis)

    return PBI_DIR / f"{PROJECT_NAME}.pbip"


if __name__ == "__main__":
    pbip = build_project()
    print(f"[+] Built Power BI Project -> {pbip}")
