"""
Backward-compatibility proxy.
The rail GeoJSON builder has moved to: DataEngineering/WebScraping/build_rail_geojson.py
"""
import sys
from pathlib import Path

target_script = Path(__file__).resolve().parent / "WebScraping" / "build_rail_geojson.py"

if __name__ == "__main__":
    import runpy
    print(f"[*] Redirecting to {target_script} ...")
    runpy.run_path(str(target_script), run_name="__main__")
