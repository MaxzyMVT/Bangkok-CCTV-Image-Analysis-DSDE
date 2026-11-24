import importlib.util
import io
import json
import unittest
from pathlib import Path


APP = Path(__file__).resolve().parents[2] / "Apps" / "Traffic-Insights" / "app.py"
spec = importlib.util.spec_from_file_location("traffic_insights_test_app", APP)
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)


class FakeHandler:
    def __init__(self):
        self.wfile = io.BytesIO()
        self.status = None
        self.headers = {}

    def send_response(self, status):
        self.status = status

    def send_header(self, name, value):
        self.headers[name] = value

    def end_headers(self):
        pass


class InsightsAppTest(unittest.TestCase):
    def test_api_and_idea_chart_routes(self):
        handler = FakeHandler()
        self.assertTrue(app.handle_insights_get(handler, "/api/insights"))
        self.assertEqual(handler.status, 200)
        payload = json.loads(handler.wfile.getvalue())
        self.assertEqual(len(payload["cameras"]), 100)
        self.assertTrue(payload["rain_timing"])
        self.assertTrue(payload["timeline"])

        image_handler = FakeHandler()
        self.assertTrue(app.handle_insights_get(image_handler, "/insights/idea-chart.png"))
        self.assertEqual(image_handler.headers["Content-Type"], "image/png")
        self.assertEqual(image_handler.wfile.getvalue()[:8], b"\x89PNG\r\n\x1a\n")
        self.assertFalse(app.handle_insights_get(FakeHandler(), "/other"))


if __name__ == "__main__":
    unittest.main()
