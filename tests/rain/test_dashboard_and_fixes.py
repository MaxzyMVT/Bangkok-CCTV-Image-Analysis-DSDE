"""Regression tests for PyDuckLake integration, ML Gold tables, Traffic-Insights API, and Streamlit dashboard."""

import importlib.util
import json
from pathlib import Path
import sys
import threading
import unittest
import urllib.request

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


class PyDuckLakeAndApiTests(unittest.TestCase):
    def test_dataset_root_resolves_without_hardcoded_missing_drive(self):
        from DataAnalysis.rain import common

        self.assertTrue((common.DATASET_ROOT / "dataset" / "100_cameras").exists())

    def test_gold_ml_and_correlation_tables_exist_in_ducklake(self):
        from DataEngineering.lakehouse_pipeline import lake_has_table, query_lake

        for tbl in ("ml_metrics", "ml_feature_importance", "ml_predictions", "rain_correlations", "rain_camera_hour"):
            self.assertTrue(lake_has_table("gold", tbl), f"Missing lake.gold.{tbl}")
        self.assertGreater(len(query_lake("SELECT * FROM lake.gold.ml_metrics")), 5)
        self.assertGreater(len(query_lake("SELECT * FROM lake.gold.rain_correlations")), 10)

    def test_insights_route_reads_from_ducklake_with_fixes(self):
        module = _load_insights_module()
        module.insight_payload.cache_clear()
        server = module.ThreadingHTTPServer(("127.0.0.1", 0), module.InsightsHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            base = f"http://127.0.0.1:{server.server_port}"
            with urllib.request.urlopen(f"{base}/api/insights") as resp:
                self.assertEqual(resp.status, 200)
                payload = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(len(payload["cameras"]), 100)
            cam11 = next(c for c in payload["cameras"] if c["camera_id"] == 11)
            self.assertEqual(type(cam11["camera_id"]), int)
            self.assertTrue(cam11["location_th"])
            no_match = [c for c in payload["cameras"] if c["label"] == "no matched rainy hours"]
            self.assertEqual(len(no_match), 6)
            for cam in no_match:
                self.assertEqual(cam["n_wet_hours"], 0)
                self.assertIsNone(cam["effect_pct"])

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
    def test_dashboard_has_no_mermaid_text(self):
        src = (ROOT / "DataAnalysis" / "dashboard.py").read_text(encoding="utf-8").lower()
        self.assertNotIn("mermaid", src)

    def test_streamlit_dashboard_renders_all_tabs_without_errors(self):
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(str(ROOT / "DataAnalysis" / "dashboard.py"), default_timeout=45)
        at.run()
        self.assertFalse(list(at.exception), msg=f"Unexpected exceptions: {list(at.exception)}")
        self.assertFalse(list(at.error), msg=f"Unexpected errors: {list(at.error)}")
        self.assertEqual(len(at.tabs), 7)


class NetworkAndPowerBITests(unittest.TestCase):
    def test_sparse_network_graph_has_clean_edges_and_communities(self):
        from DataAnalysis.rain import network_viz
        from DataEngineering.lakehouse_pipeline import query_lake

        cams = query_lake("SELECT * FROM lake.gold.rain_camera").merge(
            query_lake("SELECT camera_id, effect_pct, label FROM lake.gold.rain_camera_effect"),
            on="camera_id",
            how="left",
        )
        rail = query_lake(
            "SELECT name_th AS name, name_en, primary_system AS network, latitude, longitude "
            "FROM lake.silver.transit_stations WHERE latitude IS NOT NULL AND longitude IS NOT NULL"
        )
        rail["station_label"] = rail["name"].fillna(rail["name_en"]).fillna("Rail Station")
        G = network_viz.build_bangkok_network(cams, rail, rail_radius_m=800, cam_radius_m=900, k_rail=1, k_cam=2)
        self.assertGreater(G.number_of_nodes(), 50)
        self.assertGreater(G.number_of_edges(), 50)
        self.assertLess(G.number_of_edges(), 300)

        comms = network_viz.detect_communities(G)
        self.assertGreaterEqual(len(set(comms.values())), 10)
        cam_df, comm_df = network_viz.connection_summary_tables(G, comms)
        self.assertFalse(cam_df.empty)
        self.assertFalse(comm_df.empty)
        self.assertIn("Linked Stations", cam_df.columns)
        self.assertIn("Neighbor Cameras", cam_df.columns)
        self.assertIn("Station Hubs", comm_df.columns)
        self.assertIn("Mean Rain Effect (%)", comm_df.columns)

    def test_powerbi_pbip_project_and_story_files_valid(self):
        pb_dir = ROOT / "DataAnalysis" / "PowerBI"
        pbip = json.loads((pb_dir / "Bangkok_Rain_Traffic_Story.pbip").read_text(encoding="utf-8"))
        self.assertIn("artifacts", pbip)

        sm_def = pb_dir / "Bangkok_Rain_Traffic_Story.SemanticModel" / "definition"
        for req in (
            sm_def / "model.tmdl",
            sm_def / "relationships.tmdl",
            sm_def / "expressions.tmdl",
            sm_def / "database.tmdl",
            sm_def / "tables" / "executive_story.tmdl",
            sm_def / "tables" / "city_effect.tmdl",
            sm_def / "tables" / "ml_metrics.tmdl",
        ):
            self.assertTrue(req.exists(), f"Missing SemanticModel file: {req}")

        rel_txt = (sm_def / "relationships.tmdl").read_text(encoding="utf-8")
        self.assertIn("fromColumn: camera_hour.camera_id", rel_txt)
        self.assertIn("toColumn: camera.camera_id", rel_txt)
        self.assertNotIn("fromTable:", rel_txt)
        self.assertNotIn("toTable:", rel_txt)

        rep_def = pb_dir / "Bangkok_Rain_Traffic_Story.Report" / "definition"
        pages = json.loads((rep_def / "pages" / "pages.json").read_text(encoding="utf-8"))
        self.assertEqual(len(pages["pageOrder"]), 5)
        for pid in pages["pageOrder"]:
            page_json = json.loads((rep_def / "pages" / pid / "page.json").read_text(encoding="utf-8"))
            self.assertEqual(page_json["name"], pid)

        visual_files = list((rep_def / "pages").rglob("visual.json"))
        self.assertGreaterEqual(len(visual_files), 30)
        for vf in visual_files:
            self.assertLess(len(str(vf)), 255, f"Windows MAX_PATH risk: {vf}")
            vjson = json.loads(vf.read_text(encoding="utf-8"))
            self.assertIn("visual", vjson)

        story_csv = (ROOT / "DataAnalysis" / "output" / "executive_story.csv").read_text(encoding="utf-8")
        self.assertIn("Localized Modal Shift", story_csv)
        for forbidden in ("hackathon", "storytelling", "story arc"):
            self.assertNotIn(forbidden, story_csv.lower())


if __name__ == "__main__":
    unittest.main()
