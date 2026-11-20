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

    def test_item_with_missing_location_is_valid_json(self):
        # camera 1589 has no location_en in the metadata; NaN is not valid JSON for the browser
        stem = next(s for s in load_items("test")["stem"] if s.startswith("cam1589_"))
        def reject(c):
            raise ValueError(f"invalid JSON constant {c}")
        it = json.loads(self._get(f"/api/item?set=test&stem={stem}")[2], parse_constant=reject)
        self.assertIsNone(it["meta"]["location_en"])
        self.assertIsInstance(it["boxes"], list)

    def test_queue_flags_overlaps_and_val(self):
        stem = load_items("test")["stem"].iloc[3]
        # truck scored higher than car: too doubtful to merge automatically, so it stays flagged
        self.store.write_prelabel("test", stem, [(0, .5, .5, .2, .2, .6), (3, .505, .5, .21, .2, .9)], overwrite=True)
        try:
            items = {i["stem"]: i for i in self._json("/api/queue?set=test")["items"]}
            self.assertEqual(items[stem]["n_overlaps"], 1)
            self.assertEqual(items[self.stem]["n_overlaps"], 0)
            saved = self._post({"set": "test", "stem": stem, "boxes": [[0, .5, .5, .2, .2]], "action": "save"})
            self.assertEqual(saved["n_overlaps"], 0)   # the page updates its flag from this
            self.assertFalse(any(i["val"] for i in self._json("/api/queue?set=test")["items"]))
            train = self._json("/api/queue?set=train")["items"]
            self.assertEqual(sum(i["val"] for i in train), 173)   # 31 Aug frames in train_manifest.csv
        finally:
            self._post({"set": "test", "stem": stem, "boxes": [], "action": "reset"})
            self.store.write_prelabel("test", stem, [], overwrite=True)

    def test_queue_shows_assigned_reviewer(self):
        items = load_items("test")
        a, b = items["stem"].iloc[0], items["stem"].iloc[1]
        path = self.store.set_dir("test") / "assignments.csv"
        rows = ["stem,camera_id,reviewer", f"{a},1,ปัน", f"{b},1,ครีม"]
        path.write_text("\n".join(rows) + "\n", encoding="utf-8-sig")
        try:
            q = {i["stem"]: i for i in self._json("/api/queue?set=test")["items"]}
            self.assertEqual((q[a]["reviewer"], q[b]["reviewer"]), ("ปัน", "ครีม"))
            self.assertIsNone(q[items["stem"].iloc[2]]["reviewer"])
        finally:
            path.unlink()
        self.assertIsNone(self._json("/api/queue?set=test")["items"][0]["reviewer"])   # no file: no reviewer

    def test_queue_and_stats_mark_core_subset(self):
        items = self._json("/api/queue?set=test")["items"]
        self.assertEqual(sum(i["core"] for i in items), 300)
        s = self._json("/api/stats?set=test")
        self.assertEqual(s["core_total"], 300)
        self.assertIn("core_done", s)
        train = self._json("/api/queue?set=train")["items"]
        self.assertFalse(any(i["core"] for i in train))

    def test_queue_flags_bus_and_truck_from_prelabel(self):
        stem_bus = load_items("test")["stem"].iloc[2]
        self.store.write_prelabel("test", stem_bus, [(2, .5, .5, .2, .2, .8), (0, .2, .2, .1, .1, .9)], overwrite=True)
        try:
            items = {i["stem"]: i for i in self._json("/api/queue?set=test")["items"]}
            self.assertTrue(items[stem_bus]["has_bus"])
            self.assertFalse(items[stem_bus]["has_truck"])
            self.assertFalse(items[self.stem]["has_bus"])
            self.assertFalse(items[self.stem]["has_truck"])
            self.store.write_prelabel("test", stem_bus, [(3, .5, .5, .2, .2, .8)], overwrite=True)
            items = {i["stem"]: i for i in self._json("/api/queue?set=test")["items"]}
            self.assertFalse(items[stem_bus]["has_bus"])
            self.assertTrue(items[stem_bus]["has_truck"])
        finally:
            self.store.write_prelabel("test", stem_bus, [], overwrite=True)

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
