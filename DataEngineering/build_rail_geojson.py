"""
Build the BTS / MRT layer for the Camera Map from OpenStreetMap.

Downloads the real route alignments and stations for Bangkok's BTS and MRT lines
from the Overpass API and writes one GeoJSON file the map serves itself:

    Apps/Camera-Map/static/bangkok_rail.geojson

Run again whenever you want fresher data:
    python DataEngineering/build_rail_geojson.py

Data © OpenStreetMap contributors, available under the Open Database License (ODbL).
"""
import json
import math
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
OUT_FILE = ROOT_DIR / "Apps" / "Camera-Map" / "static" / "bangkok_rail.geojson"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
BBOX = (13.45, 100.25, 14.05, 100.95)  # south, west, north, east (greater Bangkok)

# One entry per line. "match" is checked against the route's English name.
LINES = [
    {"key": "bts-sukhumvit", "match": "Sukhumvit", "system": "BTS", "name_th": "บีทีเอส สายสุขุมวิท", "name_en": "BTS Sukhumvit Line", "colour": "#65B724"},
    {"key": "bts-silom",     "match": "Silom",     "system": "BTS", "name_th": "บีทีเอส สายสีลม",     "name_en": "BTS Silom Line",     "colour": "#02817D"},
    {"key": "bts-gold",      "match": "Gold",      "system": "BTS", "name_th": "สายสีทอง",            "name_en": "Gold Line",          "colour": "#A58704"},
    {"key": "mrt-blue",      "match": "Blue",      "system": "MRT", "name_th": "เอ็มอาร์ที สายสีน้ำเงิน", "name_en": "MRT Blue Line",   "colour": "#1E398D"},
    {"key": "mrt-purple",    "match": "Purple",    "system": "MRT", "name_th": "เอ็มอาร์ที สายสีม่วง",  "name_en": "MRT Purple Line",    "colour": "#893B90"},
    {"key": "mrt-yellow",    "match": "Yellow",    "system": "MRT", "name_th": "เอ็มอาร์ที สายสีเหลือง", "name_en": "MRT Yellow Line",   "colour": "#FFD400"},
    {"key": "mrt-pink",      "match": "Pink",      "system": "MRT", "name_th": "เอ็มอาร์ที สายสีชมพู",  "name_en": "MRT Pink Line",      "colour": "#E76589"},
]
STATION_MAX_DIST_M = 150  # a station belongs to a line if its node is this close to the line's track


def candidate_lines(tags):
    """Lines a station node can belong to, from its OSM tags (None = not a BTS/MRT station).

    Distance alone mislabels interchanges with different names (BTS Asok sits right next to
    MRT Sukhumvit), so first narrow by station type and operator, then match by distance.
    """
    kind = tags.get("station") or next((k for k in ("subway", "light_rail", "monorail") if tags.get(k) == "yes"), None)
    if kind not in ("subway", "light_rail", "monorail") or tags.get("ferry") == "yes":
        return None  # piers, SRT / Airport Rail Link stations, ...
    if kind == "monorail":
        return ["mrt-yellow", "mrt-pink"]
    if kind == "light_rail":
        return ["bts-gold"]
    who = " ".join(tags.get(k, "") for k in ("operator", "network", "operator:en", "network:en")).lower()
    if any(s in who for s in ("ทางด่วนและรถไฟฟ้า", "bem", "metropolitan rapid transit", "mrt")):
        return ["mrt-blue", "mrt-purple"]
    if any(s in who for s in ("ระบบขนส่งมวลชนกรุงเทพ", "bts", "บีทีเอส", "bangkok mass transit")):
        return ["bts-sukhumvit", "bts-silom", "bts-gold"]
    return ["bts-sukhumvit", "bts-silom", "mrt-blue", "mrt-purple"]


def overpass(query):
    body = urllib.parse.urlencode({"data": query}).encode()
    req = urllib.request.Request(OVERPASS_URL, data=body, headers={"User-Agent": "bkk-cctv-class-project/1.0"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                return json.loads(resp.read())
        except Exception as e:  # Overpass is busy now and then; back off and retry
            if attempt == 2:
                raise
            print(f"[!] Overpass error ({e}); retrying...")
            time.sleep(10 * (attempt + 1))


def meters(a, b):
    """Approximate distance in meters between (lat, lon) points (fine at city scale)."""
    lat = math.radians((a[0] + b[0]) / 2)
    dx = (a[1] - b[1]) * 111_320 * math.cos(lat)
    dy = (a[0] - b[0]) * 110_540
    return math.hypot(dx, dy)


def point_to_segment_m(p, a, b):
    lat = math.radians(p[0])
    kx, ky = 111_320 * math.cos(lat), 110_540
    px, py = p[1] * kx, p[0] * ky
    ax, ay, bx, by = a[1] * kx, a[0] * ky, b[1] * kx, b[0] * ky
    dx, dy = bx - ax, by - ay
    t = 0 if dx == dy == 0 else max(0, min(1, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def main():
    s, w, n, e = BBOX
    print("[*] Downloading BTS/MRT routes from OpenStreetMap (Overpass)...")
    routes = overpass(f'[out:json][timeout:120];relation["route"~"^(subway|light_rail|monorail)$"]({s},{w},{n},{e});out geom;')
    print("[*] Downloading stations...")
    stations_raw = overpass(
        f'[out:json][timeout:120];'
        f'(node["railway"="station"]({s},{w},{n},{e});node["public_transport"="station"]({s},{w},{n},{e}););out body;'
    )

    # 1) Track geometry per line (each OSM way once, both directions share tracks)
    tracks = {line["key"]: {} for line in LINES}
    for rel in routes["elements"]:
        name_en = rel.get("tags", {}).get("name:en", "")
        line = next((l for l in LINES if l["match"] in name_en), None)
        if not line:
            continue  # e.g. the airport people mover
        for m in rel.get("members", []):
            if m["type"] == "way" and not m.get("role", "").startswith("platform") and m.get("geometry"):
                tracks[line["key"]][m["ref"]] = [(round(p["lat"], 6), round(p["lon"], 6)) for p in m["geometry"]]

    features = []
    for line in LINES:
        ways = list(tracks[line["key"]].values())
        if not ways:
            print(f"[!] No track found for {line['name_en']}")
            continue
        features.append({
            "type": "Feature",
            "properties": {"kind": "line", **{k: line[k] for k in ("key", "system", "name_th", "name_en", "colour")}},
            "geometry": {"type": "MultiLineString", "coordinates": [[[lon, lat] for lat, lon in way] for way in ways]},
        })

    # 2) Stations: real station nodes, matched to every line whose track passes close by
    stations = []
    for node in stations_raw["elements"]:
        tags = node.get("tags", {})
        allowed = candidate_lines(tags)
        if not allowed:
            continue
        p = (node["lat"], node["lon"])
        lines_here = []
        for line in LINES:
            if line["key"] not in allowed:
                continue
            near = any(point_to_segment_m(p, a, b) <= STATION_MAX_DIST_M
                       for way in tracks[line["key"]].values() for a, b in zip(way, way[1:]))
            if near:
                lines_here.append(line["key"])
        if not lines_here:
            continue  # SRT / Airport Rail Link / other stations
        name_th = tags.get("name:th") or tags.get("name") or ""
        name_en = tags.get("name:en") or ""
        if not name_th:
            continue
        stations.append({"p": p, "name_th": name_th, "name_en": name_en, "lines": lines_here})

    # Merge duplicates (same Thai name within 300 m) into one interchange station
    merged = []
    for st in stations:
        twin = next((m for m in merged if m["name_th"] == st["name_th"] and meters(m["p"], st["p"]) < 300), None)
        if twin:
            twin["lines"] = sorted(set(twin["lines"]) | set(st["lines"]), key=[l["key"] for l in LINES].index)
            twin["name_en"] = twin["name_en"] or st["name_en"]
        else:
            merged.append(st)

    for st in merged:
        features.append({
            "type": "Feature",
            "properties": {"kind": "station", "name_th": st["name_th"], "name_en": st["name_en"], "lines": st["lines"]},
            "geometry": {"type": "Point", "coordinates": [round(st["p"][1], 6), round(st["p"][0], 6)]},
        })

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    collection = {
        "type": "FeatureCollection",
        "attribution": "© OpenStreetMap contributors (ODbL)",
        "source": "OpenStreetMap via Overpass API",
        "generated": time.strftime("%Y-%m-%d"),
        "lines": [{k: l[k] for k in ("key", "system", "name_th", "name_en", "colour")} for l in LINES],
        "features": features,
    }
    OUT_FILE.write_text(json.dumps(collection, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    n_lines = sum(1 for f in features if f["properties"]["kind"] == "line")
    print(f"[✓] Wrote {n_lines} lines and {len(merged)} stations to {OUT_FILE} "
          f"({OUT_FILE.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
