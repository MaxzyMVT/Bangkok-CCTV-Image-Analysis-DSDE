"""Generate the bilingual (English / Thai) Power BI Project (.pbip) backed by DataAnalysis/output/*.csv.

Produces:
  DataAnalysis/PowerBI/
  ├── Bangkok_Rain_Traffic_Story.pbip
  ├── Bangkok_Rain_Traffic_Story.SemanticModel/   (TMDL model + Business & i18n DAX measures)
  └── Bangkok_Rain_Traffic_Story.Report/          (PBIR 5-page analytical report with Language slicer)

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
        elif col in ("camera_id", "step_no"):
            tmdl_t, m_t = "int64", "Int64.Type"
        summarize = (
            "none"
            if (tmdl_t in ("string", "boolean", "dateTime")
                or col in ("camera_id", "step_no", "latitude", "longitude", "cluster", "hod", "hour", "minute"))
            else "average"
        )
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


STORY_MEASURES_TMDL = f"""\tmeasure 'Active Step' = IF(SELECTEDVALUE(ui_lang[lang], "en") = "th", MAX(executive_story[step_th]), MAX(executive_story[step]))
\t\tlineageTag: {_uid('m-act-step')}

\tmeasure 'Active Headline' = IF(SELECTEDVALUE(ui_lang[lang], "en") = "th", MAX(executive_story[headline_th]), MAX(executive_story[headline]))
\t\tlineageTag: {_uid('m-act-head')}

\tmeasure 'Active Evidence' = IF(SELECTEDVALUE(ui_lang[lang], "en") = "th", MAX(executive_story[evidence_th]), MAX(executive_story[evidence]))
\t\tlineageTag: {_uid('m-act-evid')}"""


def _lang_measure(name: str, en_text: str, th_text: str) -> str:
    e_esc = en_text.replace('"', '""')
    t_esc = th_text.replace('"', '""')
    return (
        f'\tmeasure {name} = IF(SELECTEDVALUE(ui_lang[lang], "en") = "th", "{t_esc}", "{e_esc}")\n'
        f'\t\tlineageTag: {_uid("m-ui-" + name)}'
    )


UI_LANG_SPECS = [
    ("T_Lang_Slicer", "Language / ภาษา", "ภาษา / Language"),
    ("H_P1_Title", "Bangkok Rain & Urban Mobility: Still Frames Miss the Modal Shift",
     "ฝนตกกับการจราจร กทม.: ภาพนิ่งเผยการเปลี่ยนโหมดสู่รถยนต์"),
    ("H_P1_Sub", "100 BMA CCTV Cameras (60,721 Counts) | 129 Rain Gauges | 188 Transit Hubs | 25-31 Aug 2026",
     "กล้อง CCTV กทม. 100 จุด (60,721 เฟรม) | 129 สถานีฝน | 188 สถานีรถไฟฟ้า | 25-31 ส.ค. 2569"),
    ("H_P2_Title", "Motorcycle Drop (-21%) & Rain Timing Analysis",
     "ปริมาณมอเตอร์ไซค์ลดลง -21% และวิเคราะห์ความต่างช่วงเวลาฝนตก"),
    ("H_P3_Title", "Localized Modal Shift (+12.6 pts Cars) & Spatial Hypothesis Verification",
     "การเปลี่ยนโหมดสู่รถยนต์ (+12.6 จุด) และการทดสอบสมมติฐานเชิงพื้นที่"),
    ("H_P4_Title", "Supervised ML Predictions & Spatial-Weather Correlations",
     "แบบจำลอง ML คาดการณ์จราจร และค่าสหสัมพันธ์เชิงพื้นที่-สภาพอากาศ"),
    ("H_P5_Title", "Operational Recommendations & Smart-City Data Adaptability",
     "ข้อเสนอแนะเชิงปฏิบัติการและแนวทางต่อยอดข้อมูลเมืองอัจฉริยะ"),
    ("T_P1_C1", "Cameras Monitored", "จำนวนกล้องที่วิเคราะห์"),
    ("T_P1_C2", "Matched Rain Hours", "ชั่วโมงกล้องที่มีฝนตก"),
    ("T_P1_C3", "Citywide Motorcycle Drop", "มอเตอร์ไซค์ลดลงทั่วกรุง"),
    ("T_P1_C4", "Rush Hour Moto Drop", "มอเตอร์ไซค์ลดช่วงเร่งด่วน"),
    ("T_P1_C5", "Wet vs Dry Cars (<=3km)", "ส่วนต่างรถยนต์จุดฝน vs แห้ง"),
    ("T_P1_Story", "Executive Insights & Operational Actions", "สรุปข้อค้นพบหลักและแนวทางปฏิบัติ"),
    ("T_P1_Bar", "Headline Change in Vehicles Visible per Frame (%)", "การเปลี่ยนแปลงจำนวนรถเฉลี่ยต่อเฟรมเมื่อฝนตก (%)"),
    ("T_P2_Eff", "Rain Impact by Vehicle Mode & Time Window (%)", "ผลกระทบฝนแยกตามประเภทรถและช่วงเวลา (%)"),
    ("T_P2_Time", "Rain Timing: Weekday (16-20h) vs Weekend (12-17h)", "ช่วงเวลาฝนตก: วันทำงาน (16-20 น.) vs วันหยุด (12-17 น.)"),
    ("T_P2_Reb", "Post-Rain Recovery Curve (-60 to +180 min)", "การฟื้นตัวของการจราจรก่อนและหลังฝนหยุด"),
    ("T_P2_Strat", "Subgroup Splits (Weekday/Weekend & Light/Heavy Rain)", "ผลกระทบแยกกลุ่มย่อย (วันทำงาน/หยุด และระดับฝน)"),
    ("T_P3_Nat", "Same-Hour Neighbor Test (<=3 km, 15 hrs): Cars +12.6 pts", "การทดลองธรรมชาติในชั่วโมงเดียวกัน (<=3 กม.): รถยนต์ +12.6 จุด"),
    ("T_P3_Den", "Shelter & Station Share Comparison (5 Denser vs 71 Other)", "สัดส่วนใกล้สะพานลอยและรถไฟฟ้า (จุดหนาแน่นขึ้น vs อื่นๆ)"),
    ("T_P3_Cams", "Per-Camera Leaderboard (Thai & English Road Names)", "ตารางผลกระทบรายกล้อง (ชื่อถนนไทยและอังกฤษ)"),
    ("T_P3_Flood", "DDS Flood Sensors >= 5 cm (FL.RTW.02 Phetchaburi Rd)", "จุดวัดน้ำท่วม DDS >= 5 ซม. (ถนนเพชรบุรี)"),
    ("T_P4_C1", "HistGB Test R2 (With Baseline)", "ค่า R2 ทดสอบ HistGB (มีค่าฐาน)"),
    ("T_P4_C2", "Max Recorded Flood Sensor (cm)", "ระดับน้ำสูงสุดจากเซนเซอร์ (ซม.)"),
    ("T_P4_C3", "Overall Frame Volume Delta (%)", "การเปลี่ยนแปลงปริมาณรถรวมทั่วกรุง (%)"),
    ("T_P4_Imp", "ML Permutation Importance (Rain_mm Drives Moto Drops)", "ปัจจัยสำคัญใน ML (ฝนมีผลต่อมอเตอร์ไซค์สูงสุด)"),
    ("T_P4_Corr", "Linear Correlations (Rail Distance r=-0.26)", "สหสัมพันธ์เชิงเส้น (ระยะรถไฟฟ้า r=-0.26)"),
    ("T_P4_Met", "Regression & Classification Benchmark (n=9,996 camera-hours)", "เปรียบเทียบประสิทธิภาพโมเดล ML (9,996 ชม.กล้อง)"),
    ("T_P5_Act1", "1. BTS/MRT Feeder Alerting", "1. แจ้งเตือนรถฟีดเดอร์ BTS/MRT ล่วงหน้า"),
    ("T_P5_Act2", "2. Ride-Hailing Fleet Pre-Positioning", "2. กระจายรถ 4 ล้อรับช่วงฝนตกเฉพาะจุด"),
    ("T_P5_Act3", "3. Automated Feed QA Guardrail", "3. กรองภาพกล้องขาวขัดข้องอัตโนมัติ"),
    ("T_P5_Road", "Next-Phase Data Adaptability Roadmap", "แผนต่อยอดข้อมูลเมืองอัจฉริยะระยะถัดไป"),
    ("M_P5_Act1",
     "During weekday 16:00-20:00 rain, motorcycle supply plunges -30.5% near transit hubs (r = -0.26). Dispatch feeder shuttles when gauges cross 1 mm/h.",
     "ช่วงฝนตกวันทำงาน 16:00-20:00 น. มอเตอร์ไซค์ลดลง -30.5% รอบสถานีรถไฟฟ้า (r = -0.26) ควรเสริมรถฟีดเดอร์ทันทีเมื่อฝนเกิน 1 มม./ชม."),
    ("M_P5_Act2",
     "Locally wet roads show +12.6 pts higher car share vs dry roads within 3 km. Pair 129 BMA rain gauges with ML forecasts to pre-stage 4-wheel supply.",
     "ถนนจุดที่ฝนตกมีสัดส่วนรถยนต์สูงกว่าพื้นที่แห้งใกล้เคียง +12.6 จุด ใช้สัญญาณวัดฝน 129 จุดร่วมกับ ML จัดส่งรถยนต์ก่อนดีมานด์พุ่ง"),
    ("M_P5_Act3",
     "BMA cameras emitted 1.4 KB solid-white outage frames around 07:21-07:26. Filtering file_size_bytes < 3 KB in Silver/Gold prevents false zero alerts.",
     "กล้อง กทม. ส่งภาพขาวขนาด 1.4 KB ช่วง 07:21-07:26 น. การกรอง file_size_bytes < 3 KB ใน PyDuckLake ช่วยป้องกันสัญญาณเตือนรถหายผิดพลาด"),
    ("M_P5_Road",
     "1) Add BTS/MRT turnstile tap-in telemetry to verify rail passenger shift. 2) Add GPS coordinates to DDS flood gauges for point-level causality. 3) Sample 1-min video optical flow to capture travel speed.",
     "1) เชื่อมข้อมูลจำนวนผู้โดยสารเข้าสถานี BTS/MRT 2) เพิ่มพิกัด GPS ในตารางเซนเซอร์น้ำท่วม DDS 3) เพิ่มความถี่ภาพ 1 นาทีหรือ Optical Flow เพื่อวัดความเร็วรถจริง"),
]

UI_LANG_MEASURES_TMDL = "\n\n".join(_lang_measure(n, en, th) for n, en, th in UI_LANG_SPECS)


def _col_field(entity: str, prop: str) -> dict:
    return {"Column": {"Expression": {"SourceRef": {"Entity": entity}}, "Property": prop}}


def _meas_field(entity: str, prop: str) -> dict:
    return {"Measure": {"Expression": {"SourceRef": {"Entity": entity}}, "Property": prop}}


def _dynamic_title(title_prop: str | None, fallback_title: str = "") -> dict:
    expr = _meas_field("ui_lang", title_prop) if title_prop else {"Literal": {"Value": f"'{fallback_title}'"}}
    return {
        "title": [{"properties": {
            "show": {"expr": {"Literal": {"Value": "true"}}},
            "text": {"expr": expr},
        }}],
    }


def _slicer_visual(vis_id: str, x: int, y: int, w: int, h: int, z: int) -> dict:
    return {
        "$schema": "https://developer.microsoft.com/json-schemas/fabric/item/report/definition/visualContainer/1.7.0/schema.json",
        "name": vis_id,
        "position": {"x": x, "y": y, "z": z, "height": h, "width": w, "tabOrder": z},
        "visual": {
            "visualType": "slicer",
            "query": {
                "queryState": {
                    "Values": {
                        "projections": [{
                            "field": _col_field("ui_lang", "Language"),
                            "queryRef": "ui_lang.Language",
                            "nativeQueryRef": "Language",
                            "active": True,
                        }]
                    }
                }
            },
            "syncGroup": {"groupName": "LangSync", "fieldChanges": True, "filterChanges": True},
            "objects": {
                "data": [{"properties": {"mode": {"expr": {"Literal": {"Value": "'Basic'"}}}}}],
                "selection": [{"properties": {"singleSelect": {"expr": {"Literal": {"Value": "true"}}}}}],
                "general": [{"properties": {"orientation": {"expr": {"Literal": {"Value": "1D"}}}}}],
            },
            "visualContainerObjects": _dynamic_title("T_Lang_Slicer", "Language / ภาษา"),
        },
    }


def _card_visual(vis_id: str, x: int, y: int, w: int, h: int, z: int,
                 entity: str, measure_name: str, title_prop: str, font_size: int | None = None) -> dict:
    objs: dict = {
        "categoryLabels": [{"properties": {"show": {"expr": {"Literal": {"Value": "false"}}}}}],
    }
    if font_size is not None:
        objs["labels"] = [{"properties": {"fontSize": {"expr": {"Literal": {"Value": f"{font_size}D"}}}}}]
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
            "objects": objs,
            "visualContainerObjects": _dynamic_title(title_prop),
        },
    }


def _header_card(vis_id: str, x: int, y: int, w: int, h: int, z: int,
                 measure_name: str, font_size: int = 18) -> dict:
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
                            "field": _meas_field("ui_lang", measure_name),
                            "queryRef": f"ui_lang.{measure_name}",
                            "nativeQueryRef": measure_name,
                        }]
                    }
                }
            },
            "objects": {
                "categoryLabels": [{"properties": {"show": {"expr": {"Literal": {"Value": "false"}}}}}],
                "labels": [{"properties": {"fontSize": {"expr": {"Literal": {"Value": f"{font_size}D"}}},
                                          "wordWrap": {"expr": {"Literal": {"Value": "true"}}}}}],
            },
            "visualContainerObjects": {
                "title": [{"properties": {"show": {"expr": {"Literal": {"Value": "false"}}}}}],
            },
        },
    }


def _chart_visual(vis_id: str, vtype: str, x: int, y: int, w: int, h: int, z: int, title_prop: str,
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
            "visualContainerObjects": _dynamic_title(title_prop),
        },
    }


def _table_visual(vis_id: str, x: int, y: int, w: int, h: int, z: int, title_prop: str,
                  cols: list[tuple[str, str, bool]]) -> dict:
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
                            "field": _meas_field(ent, col) if is_m else _col_field(ent, col),
                            "queryRef": f"{ent}.{col}",
                            "nativeQueryRef": col,
                        } for ent, col, is_m in cols]
                    }
                }
            },
            "visualContainerObjects": _dynamic_title(title_prop),
        },
    }


def build_project() -> Path:
    """Create the bilingual Power BI Project (.pbip + SemanticModel + 5-Page Report)."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        {"lang": "en", "Language": "English"},
        {"lang": "th", "Language": "ภาษาไทย"},
    ]).to_csv(OUT_DIR / "ui_lang.csv", index=False, lineterminator="\n")

    exec_csv = OUT_DIR / "executive_story.csv"
    pd.DataFrame([
        {
            "step_no": 1,
            "step": "1. Citywide Impact",
            "headline": "Total traffic falls -5.4% in rain",
            "evidence": "Motorcycles drop -21.0% (-30.5% in rush hours) across 293 wet camera-hours, masking a car shift.",
            "step_th": "1. ภาพรวมทั่วกรุงเทพฯ",
            "headline_th": "ปริมาณรถต่อเฟรมลดลง -5.4% เมื่อฝนตก",
            "evidence_th": "มอเตอร์ไซค์ลดลง -21.0% (-30.5% ช่วงเร่งด่วน) ใน 293 ชม.กล้องที่ฝนตก",
        },
        {
            "step_no": 2,
            "step": "2. Localized Modal Shift",
            "headline": "Localized rain shifts travel from 2-wheelers to 4-wheelers",
            "evidence": "Across 15 hours comparing wet vs dry cameras <=3km apart (172 pairs), cars rise +12.6 percentage points.",
            "step_th": "2. การเปลี่ยนโหมดสู่รถยนต์",
            "headline_th": "จุดที่ฝนตกมีสัดส่วนรถยนต์เพิ่มขึ้นเทียบจุดแห้งใกล้เคียง",
            "evidence_th": "เทียบ 15 ชม. ระหว่างกล้องฝนตกกับกล้องแห้งในระยะ <=3 กม. (172 คู่) รถยนต์พุ่ง +12.6 จุดร้อยละ",
        },
        {
            "step_no": 3,
            "step": "3. Sheltering & Station Test",
            "headline": "Footbridge sheltering & station proximity do not explain congestion",
            "evidence": "Stationary bikes near shelters shifted only 2.5% dry vs 2.8% wet (sidewalk parking); 20% of denser cams sit near rail.",
            "step_th": "3. ตรวจสอบจุดหลบฝนและรถไฟฟ้า",
            "headline_th": "สะพานลอยและสถานีรถไฟฟ้าไม่ได้เป็นสาเหตุหลักของรถติด",
            "evidence_th": "รถจอดนิ่งใกล้จุดหลบฝนต่างกันเพียง 2.5% vs 2.8%; กล้องที่รถแน่นขึ้นอยู่ใกล้รถไฟฟ้าเพียง 20%",
        },
        {
            "step_no": 4,
            "step": "4. Predictive ML & QA",
            "headline": "Rain is the #1 weather feature for motorcycle prediction",
            "evidence": "HistGradientBoosting reaches R2=0.83 (total) & 0.75 (moto); BMA 07:00 drop proved to be 1.4 KB blank outage frames.",
            "step_th": "4. แบบจำลอง ML และคุณภาพข้อมูล",
            "headline_th": "ปริมาณฝนเป็นตัวแปรอากาศอันดับ 1 ในการทำนายมอเตอร์ไซค์",
            "evidence_th": "HistGradientBoosting ได้ R2=0.83 (รวม) และ 0.75 (มอเตอร์ไซค์); ภาพช่วง 07:00 น. ที่หายคือไฟล์ขาว 1.4 KB",
        },
        {
            "step_no": 5,
            "step": "5. Operational Actions",
            "headline": "Connect rain alerts to transit feeder & fleet dispatch",
            "evidence": "Pre-position 4-wheel ride-hailing + BTS/MRT feeder shuttles on high rail-density corridors ahead of 16:00-20:00 rain.",
            "step_th": "5. แนวทางปฏิบัติการ",
            "headline_th": "เชื่อมสัญญาณเตือนฝนกับการเสริมรถฟีดเดอร์และรถ 4 ล้อ",
            "evidence_th": "เตรียมรถฟีดเดอร์ BTS/MRT และรถยนต์รับจ้างรอบสถานีหลักล่วงหน้าก่อนช่วงฝนเย็น 16:00-20:00 น.",
        },
    ]).to_csv(exec_csv, index=False, lineterminator="\n")

    sm_dir = PBI_DIR / f"{PROJECT_NAME}.SemanticModel"
    rep_dir = PBI_DIR / f"{PROJECT_NAME}.Report"
    PBI_DIR.mkdir(parents=True, exist_ok=True)

    _write_json(PBI_DIR / f"{PROJECT_NAME}.pbip", {
        "$schema": "https://developer.microsoft.com/json-schemas/fabric/pbip/pbipProperties/1.0.0/schema.json",
        "version": "1.0",
        "artifacts": [{"report": {"path": f"{PROJECT_NAME}.Report"}}],
        "settings": {"enableAutoRecovery": True},
    })

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
        ("ui_lang", "ui_lang.csv", UI_LANG_MEASURES_TMDL),
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
        ("executive_story", "executive_story.csv", STORY_MEASURES_TMDL),
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
            _header_card(_vid("p1-hdr"), 24, 12, 1020, 52, 1, "H_P1_Title", font_size=18),
            _slicer_visual(_vid("p1-lang"), 1056, 12, 200, 88, 10),
            _header_card(_vid("p1-sub"), 24, 66, 1020, 34, 2, "H_P1_Sub", font_size=11),
            _card_visual(_vid("p1-c1"), 24, 114, 235, 118, 3, "city_effect", "Cameras Analyzed", "T_P1_C1"),
            _card_visual(_vid("p1-c2"), 273, 114, 235, 118, 4, "city_effect", "Wet Camera-Hours", "T_P1_C2"),
            _card_visual(_vid("p1-c3"), 522, 114, 235, 118, 5, "city_effect", "Moto Rain Effect %", "T_P1_C3"),
            _card_visual(_vid("p1-c4"), 771, 114, 235, 118, 6, "city_effect", "Rush Moto Drop %", "T_P1_C4"),
            _card_visual(_vid("p1-c5"), 1020, 114, 236, 118, 7, "city_effect", "Localized Car Shift (pts)", "T_P1_C5"),
            _table_visual(_vid("p1-story"), 24, 248, 740, 444, 8, "T_P1_Story", [
                ("executive_story", "step_no", False),
                ("executive_story", "Active Step", True),
                ("executive_story", "Active Headline", True),
                ("executive_story", "Active Evidence", True),
            ]),
            _chart_visual(_vid("p1-bar"), "clusteredColumnChart", 782, 248, 474, 444, 9,
                          "T_P1_Bar", "city_effect", "cls",
                          [("city_effect", "effect_pct", False)], "city_effect", "window"),
        ]),
        ("page02_impact", "02. Rain Effect & Timing", [
            _header_card(_vid("p2-hdr"), 24, 12, 1020, 54, 1, "H_P2_Title", font_size=16),
            _slicer_visual(_vid("p2-lang"), 1056, 12, 200, 88, 10),
            _chart_visual(_vid("p2-eff"), "clusteredBarChart", 24, 106, 608, 286, 2,
                          "T_P2_Eff", "city_effect", "window",
                          [("city_effect", "effect_pct", False)], "city_effect", "cls"),
            _chart_visual(_vid("p2-time"), "clusteredColumnChart", 648, 106, 608, 286, 3,
                          "T_P2_Time", "rain_timing", "hour",
                          [("rain_timing", "n_camera_hours", False)], "rain_timing", "day_type"),
            _chart_visual(_vid("p2-reb"), "lineChart", 24, 404, 608, 292, 4,
                          "T_P2_Reb", "rebound", "minute",
                          [("rebound", "ratio", False)], "rebound", "cls"),
            _chart_visual(_vid("p2-strat"), "clusteredBarChart", 648, 404, 608, 292, 5,
                          "T_P2_Strat", "stratified_effects", "segment",
                          [("stratified_effects", "effect_pct", False)], "stratified_effects", "cls"),
        ]),
        ("page03_shift", "03. Modal Shift & Spatial Proof", [
            _header_card(_vid("p3-hdr"), 24, 12, 1020, 54, 1, "H_P3_Title", font_size=16),
            _slicer_visual(_vid("p3-lang"), 1056, 12, 200, 88, 10),
            _chart_visual(_vid("p3-nat"), "clusteredColumnChart", 24, 106, 580, 274, 2,
                          "T_P3_Nat", "natural_experiment", "cls",
                          [("natural_experiment", "effect_pct_points", False)]),
            _chart_visual(_vid("p3-den"), "clusteredBarChart", 620, 106, 636, 274, 3,
                          "T_P3_Den", "denser_share", "feature",
                          [("denser_share", "denser_share", False), ("denser_share", "other_share", False)]),
            _table_visual(_vid("p3-cams"), 24, 394, 740, 300, 4, "T_P3_Cams", [
                ("camera", "camera_id", False),
                ("camera", "location_th", False),
                ("camera", "location_en", False),
                ("camera_effect", "label", False),
                ("camera_effect", "effect_pct", False),
                ("camera_effect", "n_wet_hours", False),
                ("camera", "rail_dist_m", False),
            ]),
            _table_visual(_vid("p3-flood"), 780, 394, 476, 300, 5, "T_P3_Flood", [
                ("flood_sensor_hours", "station", False),
                ("flood_sensor_hours", "hour_ts", False),
                ("flood_sensor_hours", "flood_cm", False),
            ]),
        ]),
        ("page04_ml", "04. Supervised ML & Drivers", [
            _header_card(_vid("p4-hdr"), 24, 12, 1020, 52, 1, "H_P4_Title", font_size=16),
            _slicer_visual(_vid("p4-lang"), 1056, 12, 200, 88, 10),
            _card_visual(_vid("p4-c1"), 24, 104, 390, 94, 2, "city_effect", "ML Weekend R2 (Baseline)", "T_P4_C1"),
            _card_visual(_vid("p4-c2"), 430, 104, 390, 94, 3, "city_effect", "Peak Flood Depth (cm)", "T_P4_C2"),
            _card_visual(_vid("p4-c3"), 836, 104, 420, 94, 4, "city_effect", "Total Rain Effect %", "T_P4_C3"),
            _chart_visual(_vid("p4-imp"), "clusteredBarChart", 24, 208, 608, 240, 5,
                          "T_P4_Imp", "ml_feature_importance", "feature",
                          [("ml_feature_importance", "importance_mean", False)],
                          "ml_feature_importance", "target"),
            _chart_visual(_vid("p4-corr"), "clusteredBarChart", 648, 208, 608, 240, 6,
                          "T_P4_Corr", "correlations", "feature_label",
                          [("correlations", "pearson_r", False)],
                          "correlations", "target"),
            _table_visual(_vid("p4-met"), 24, 460, 1232, 236, 7, "T_P4_Met", [
                ("ml_metrics", "target", False),
                ("ml_metrics", "model", False),
                ("ml_metrics", "split", False),
                ("ml_metrics", "mae", False),
                ("ml_metrics", "rmse", False),
                ("ml_metrics", "r2", False),
                ("ml_metrics", "roc_auc", False),
            ]),
        ]),
        ("page05_action", "05. Action & Adaptability", [
            _header_card(_vid("p5-hdr"), 24, 12, 1020, 54, 1, "H_P5_Title", font_size=18),
            _slicer_visual(_vid("p5-lang"), 1056, 12, 200, 88, 10),
            _card_visual(_vid("p5-act1"), 24, 114, 392, 246, 2, "ui_lang", "M_P5_Act1", "T_P5_Act1", font_size=13),
            _card_visual(_vid("p5-act2"), 444, 114, 392, 246, 3, "ui_lang", "M_P5_Act2", "T_P5_Act2", font_size=13),
            _card_visual(_vid("p5-act3"), 864, 114, 392, 246, 4, "ui_lang", "M_P5_Act3", "T_P5_Act3", font_size=13),
            _card_visual(_vid("p5-road"), 24, 376, 1232, 316, 5, "ui_lang", "M_P5_Road", "T_P5_Road", font_size=14),
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
