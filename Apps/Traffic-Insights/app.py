"""Rain and traffic insights app, mounted at /insights by Apps/main.py.

Reads all preaggregated Gold insight marts and camera metadata directly from PyDuckLake.
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from DataEngineering.lakehouse_pipeline import lake_has_table, query_lake  # noqa: E402

DEFAULT_OUTPUT = ROOT / "DataAnalysis" / "output"
OUTPUT = DEFAULT_OUTPUT
TEMPLATE = Path(__file__).resolve().parent / "templates" / "index.html"
IDEA_CHART = Path(__file__).resolve().parent / "static" / "idea-chart.png"


def _rows(name):
    """Load an insight table from PyDuckLake gold.rain_<name> (or test-overridden OUTPUT dir)."""
    if OUTPUT == DEFAULT_OUTPUT and lake_has_table("gold", f"rain_{name}"):
        df = query_lake(f"SELECT * FROM lake.gold.rain_{name}")
        if "hour_ts" in df.columns:
            df["hour_ts"] = df["hour_ts"].astype(str).str.slice(0, 19)
        return [{k: (None if v is None else str(v)) for k, v in rec.items()} for rec in df.to_dict(orient="records")]
    with (OUTPUT / f"{name}.csv").open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _number(value):
    try:
        return round(float(value), 4) if value not in (None, "", "nan", "None", "NaT") else None
    except (TypeError, ValueError):
        return None


def _typed(rows, numeric):
    return [{key: _number(value) if key in numeric else value for key, value in row.items()} for row in rows]


def _is_true(value) -> bool:
    return value is True or str(value).strip().lower() == "true"


def _camera_locations() -> dict[str, tuple[str, str]]:
    """Query Thai and English camera names from PyDuckLake silver.cameras."""
    if lake_has_table("silver", "cameras"):
        df = query_lake("SELECT camera_id, location_th, location_en FROM lake.silver.cameras")
        return {
            str(r.camera_id).strip(): (r.location_th or "", r.location_en or "")
            for r in df.itertuples(index=False)
        }
    return {}


@lru_cache(maxsize=1)
def insight_payload():
    """Small, preaggregated response; the 10k camera-hours stay in PyDuckLake."""
    locations = _camera_locations()

    cameras = {str(row["camera_id"]).strip(): dict(row) for row in _rows("camera")}
    for effect in _rows("camera_effect"):
        cid = str(effect["camera_id"]).strip()
        if cid in cameras:
            cameras[cid].update(effect)
    camera_rows = _typed(list(cameras.values()), {
        "latitude", "longitude", "gauge_dist_m", "rail_dist_m",
        "effect_pct", "ci_low", "ci_high", "n_wet_hours"})
    for row in camera_rows:
        cid_str = str(row["camera_id"]).strip()
        row["camera_id"] = int(cid_str) if cid_str.isdigit() else cid_str
        row["near_rail"] = _is_true(row.get("near_rail"))
        row["near_shelter"] = _is_true(row.get("near_shelter"))
        row["in_flood_area"] = _is_true(row.get("in_flood_area"))
        row["label"] = row.get("label") or "no matched rainy hours"
        row["n_wet_hours"] = int(row["n_wet_hours"]) if row.get("n_wet_hours") is not None else 0
        row.setdefault("effect_pct", None)
        row.setdefault("ci_low", None)
        row.setdefault("ci_high", None)
        loc_th, loc_en = locations.get(cid_str, ("", ""))
        row["location_th"] = loc_th
        row["location_en"] = loc_en

    hours = defaultdict(lambda: {"cameras": 0, "frames": 0, "wet_cameras": 0,
                                 "total": 0.0, "car": 0.0, "motorcycle": 0.0, "rain_mm": 0.0})
    for row in _rows("camera_hour"):
        ts = str(row["hour_ts"])[:19]
        target = hours[ts]
        target["cameras"] += 1
        target["frames"] += int(float(row["n_frames"]))
        target["wet_cameras"] += int(_is_true(row.get("wet")))
        for column in ("total", "car", "motorcycle", "rain_mm"):
            target[column] += float(row[column])
    timeline = []
    for hour, values in sorted(hours.items()):
        n = values["cameras"]
        timeline.append({"hour": hour, "date": hour[:10], "cameras": n,
                         "frames": values["frames"], "wet_cameras": values["wet_cameras"],
                         **{column: round(values[column] / n, 3)
                            for column in ("total", "car", "motorcycle", "rain_mm")}})

    return {
        "period": "25–31 Aug 2026",
        "city_effect": _typed(_rows("city_effect"), {"effect_pct", "ci_low", "ci_high", "n_cameras", "n_wet_hours"}),
        "stratified": _typed(_rows("stratified_effects"), {"effect_pct", "ci_low", "ci_high", "n_cameras", "n_wet_hours", "n_clock_hours"}),
        "rain_timing": _typed(_rows("rain_timing"), {"hour", "n_camera_hours", "n_clock_hours", "motorcycle_effect_pct"}),
        "natural_experiment": _typed(_rows("natural_experiment"), {"effect_pct_points", "ci_low", "ci_high", "n_hours", "n_pairs"}),
        "cameras": camera_rows,
        "timeline": timeline,
    }


def _send(handler, body, content_type, status=200):
    handler.send_response(status)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(body)


def handle_insights_get(handler, path, query=None):
    """Return True only for routes owned by this app."""
    if path in ("/insights", "/insights/"):
        _send(handler, TEMPLATE.read_bytes(), "text/html; charset=utf-8")
        return True
    if path == "/insights/idea-chart.png":
        _send(handler, IDEA_CHART.read_bytes(), "image/png")
        return True
    if path == "/api/insights":
        try:
            body = json.dumps(insight_payload(), ensure_ascii=False, allow_nan=False).encode("utf-8")
            _send(handler, body, "application/json; charset=utf-8")
        except (FileNotFoundError, ValueError) as error:
            body = json.dumps({"error": str(error)}, ensure_ascii=False).encode("utf-8")
            _send(handler, body, "application/json; charset=utf-8", 503)
        return True
    return False


class InsightsHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if not handle_insights_get(self, urlparse(self.path).path):
            self.send_error(404)


def serve(port=8083):
    print(f"Traffic insights: http://localhost:{port}/insights", flush=True)
    ThreadingHTTPServer(("", port), InsightsHandler).serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone traffic insights dashboard")
    parser.add_argument("--port", type=int, default=8083)
    args = parser.parse_args()
    serve(args.port)
