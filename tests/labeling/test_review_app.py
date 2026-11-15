import json
import shutil
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "DataEngineering" / "labeling"))

from common import find_image, resolve_dataset_root  # noqa: E402
from prelabel import load_items  # noqa: E402
from review_app import make_server  # noqa: E402
from review_store import ReviewStore  # noqa: E402


class ReviewAppTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        tmp = Path(cls._tmp.name)
        items = load_items("test")
        cls.stem, cls.stem_noimg = items["stem"].iloc[0], items["stem"].iloc[1]
        rel = items["relative_path"].iloc[0]
        src = find_image(resolve_dataset_root(None), rel)
        dst = tmp / "data" / rel
        dst.parent.mkdir(parents=True)
        shutil.copy2(src, dst)
        cls.store = ReviewStore(tmp / "reviews")
        cls.store.write_prelabel("test", cls.stem, [(0, .5, .5, .1, .1, .91), (1, .3, .6, .05, .1, .44)])
        cls.server = make_server(0, cls.store, tmp / "data")
        cls.base = f"http://127.0.0.1:{cls.server.server_address[1]}"
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls._tmp.cleanup()

    def _get(self, path):
        with urllib.request.urlopen(self.base + path) as r:
            return r.status, r.headers.get("Content-Type"), r.read()

    def _json(self, path):
        return json.loads(self._get(path)[2])

    def _post(self, body):
        req = urllib.request.Request(self.base + "/api/save", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req) as r:
            return json.loads(r.read())

    def _status_code(self, path, body=None):
        try:
            if body is None:
                self._get(path)
            else:
                req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(), method="POST")
                urllib.request.urlopen(req)
            return 200
        except urllib.error.HTTPError as e:
            return e.code

    def test_page_served(self):
        status, ctype, body = self._get("/")
        self.assertEqual(status, 200)
        self.assertIn("text/html", ctype)

    def test_queue_lists_items_with_status(self):
        q = self._json("/api/queue?set=test")
        self.assertEqual(len(q["items"]), 1000)
        first = next(i for i in q["items"] if i["stem"] == self.stem)
        self.assertEqual(first["n_boxes"], 2)
        self.assertAlmostEqual(first["min_conf"], 0.44)
        self.assertIn(first["status"], ("pending", "approved", "corrected", "rejected"))

    def test_item_returns_prelabel_and_counts(self):
        it = self._json(f"/api/item?set=test&stem={self.stem}")
        self.assertEqual(len(it["prelabel"]), 2)
        self.assertEqual(it["counts"]["car"], 1)
        self.assertEqual(it["counts"]["motorcycle"], 1)
        self.assertIn("time_period", it["meta"])

    def test_save_then_stats_shows_approved(self):
        boxes = [[0, .5, .5, .1, .1], [1, .3, .6, .05, .1]]
        self.assertEqual(self._post({"set": "test", "stem": self.stem, "boxes": boxes, "action": "save"})["status"],
                         "approved")
        self.assertEqual(self._json("/api/stats?set=test")["approved"], 1)
        self._post({"set": "test", "stem": self.stem, "boxes": [], "action": "reset"})
        self.assertEqual(self._json("/api/stats?set=test")["pending"], 1000)

    def test_unknown_stem_404_and_nothing_written(self):
        for stem in ("nope", "../x", "cam9999_img99999"):
            self.assertEqual(self._status_code(f"/api/item?set=test&stem={stem}"), 404)
            self.assertEqual(self._status_code("/api/save", {"set": "test", "stem": stem, "boxes": [],
                                                             "action": "save"}), 404)
        self.assertEqual(list((self.store.root / "test").glob("labels/*")), [])

    def test_missing_image_404_but_item_ok(self):
        self.assertEqual(self._status_code(f"/api/image?set=test&stem={self.stem_noimg}"), 404)
        self.assertEqual(self._json(f"/api/item?set=test&stem={self.stem_noimg}")["prelabel"], [])

    def test_image_served_as_jpeg(self):
        status, ctype, body = self._get(f"/api/image?set=test&stem={self.stem}")
        self.assertEqual(ctype, "image/jpeg")
        self.assertTrue(body.startswith(b"\xff\xd8"))


if __name__ == "__main__":
    unittest.main()
