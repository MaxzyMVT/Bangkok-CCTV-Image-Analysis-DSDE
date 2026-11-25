"""Regression tests for bug fixes in rain/common.py, Traffic-Insights API, and Streamlit dashboard."""

import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
INSIGHTS_DIR = ROOT / "Apps" / "Traffic-Insights"


def _load_insights_module():
    sys.path.insert(0, str(INSIGHTS_DIR))
    try:
        spec = importlib.util.spec_from_file_location("insights_app_test_fixes", INSIGHTS_DIR / "app.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


class PathAndInsightsApiTests(unittest.TestCase):
    def test_dataset_root_resolves_without_hardcoded_missing_drive(self):
        from DataAnalysis.rain import common

        self.assertTrue((common.DATASET_ROOT / "dataset" / "100_cameras").exists())

    def test_insight_payload_fixes_camera_id_locations_and_zero_wet_default(self):
        module = _load_insights_module()
        module.insight_payload.cache_clear()
        payload = module.insight_payload()
        self.assertEqual(len(payload["cameras"]), 100)
        cam11 = next(c for c in payload["cameras"] if c["camera_id"] == 11)
        self.assertIsInstance(cam11["camera_id"], int)
        self.assertTrue(cam11["location_th"])
        no_match = [c for c in payload["cameras"] if c["label"] == "no matched rainy hours"]
        self.assertEqual(len(no_match), 6)
        for cam in no_match:
            self.assertEqual(cam["n_wet_hours"], 0)
            self.assertIsNone(cam["effect_pct"])

    def test_insights_route_and_html_fixes(self):
        import json
        import threading
        import urllib.request

        module = _load_insights_module()
        server = module.ThreadingHTTPServer(("127.0.0.1", 0), module.InsightsHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            base = f"http://127.0.0.1:{server.server_port}"
            with urllib.request.urlopen(f"{base}/api/insights") as resp:
                self.assertEqual(resp.status, 200)
                payload = json.loads(resp.read().decode("utf-8"))
            cam11 = next(c for c in payload["cameras"] if c["camera_id"] == 11)
            self.assertEqual(type(cam11["camera_id"]), int)
            self.assertTrue(cam11["location_th"])

            with urllib.request.urlopen(f"{base}/insights") as resp:
                self.assertEqual(resp.status, 200)
                html = resp.read().decode("utf-8")
            self.assertIn("no matched rainy hours", html)
            self.assertIn("stratified-chart", html)
            self.assertNotIn("วงสีส้มแสดงสัดส่วนกล้องที่ฝนตก", html)

            with urllib.request.urlopen(f"{base}/insights/idea-chart.png") as resp:
                self.assertEqual(resp.status, 200)
                self.assertGreater(len(resp.read()), 1000)
        finally:
            server.shutdown()
            server.server_close()


class DashboardAppTests(unittest.TestCase):
    def test_streamlit_dashboard_renders_without_errors(self):
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(str(ROOT / "DataAnalysis" / "dashboard.py"), default_timeout=25)
        at.run()
        self.assertFalse(list(at.exception), msg=f"Unexpected exceptions: {list(at.exception)}")
        self.assertFalse(list(at.error), msg=f"Unexpected errors: {list(at.error)}")
        self.assertEqual(len(at.tabs), 6)


if __name__ == "__main__":
    unittest.main()
