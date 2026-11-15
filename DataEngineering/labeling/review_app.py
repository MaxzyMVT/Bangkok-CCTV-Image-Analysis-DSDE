"""
Review tool: approve or fix YOLO pre-labels in the browser.

    conda activate dsde
    python DataEngineering/labeling/review_app.py          # http://localhost:8090

Results are written to DataEngineering/reviews/<set>/ (labels/ + status/). Commit and push them.
"""
import argparse
import json
import sys
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import find_image, resolve_dataset_root  # noqa: E402
from prelabel import load_items  # noqa: E402
from review_store import VALID_SETS, ReviewStore, counts  # noqa: E402

TEMPLATE = Path(__file__).resolve().parent / "templates" / "review.html"
META_COLS = ["camera_id", "timestamp", "time_period", "location_en"]


class NotFound(Exception):
    pass


class BadRequest(Exception):
    pass


class ReviewApp:
    def __init__(self, store: ReviewStore, dataset_root: Path):
        self.store, self.dataset_root = store, Path(dataset_root)
        self._items = {}
        self.lock = threading.Lock()

    def items(self, set_name):
        if set_name not in VALID_SETS:
            raise BadRequest(f"unknown set {set_name!r}")
        if set_name not in self._items:
            df = load_items(set_name)
            self._items[set_name] = {r["stem"]: r for r in df.to_dict("records")}
        return self._items[set_name]

    def row(self, set_name, stem):
        row = self.items(set_name).get(stem)
        if row is None:
            raise NotFound(f"unknown frame {stem!r}")
        return row

    def queue(self, set_name, camera=None):
        statuses = self.store.all_status(set_name)
        out = []
        for stem, r in self.items(set_name).items():
            if camera and str(r["camera_id"]) != str(camera):
                continue
            pre = self.store.read_prelabel(set_name, stem)
            out.append({"stem": stem, "camera_id": int(r["camera_id"]), "timestamp": r["timestamp"],
                        "time_period": r["time_period"], "status": statuses.get(stem, "pending"),
                        "n_boxes": len(self.store.current_boxes(set_name, stem)),
                        "min_conf": round(min((b[5] for b in pre), default=1.0), 4)})
        return {"items": out}

    def item(self, set_name, stem):
        r = self.row(set_name, stem)
        boxes = self.store.current_boxes(set_name, stem)
        return {"stem": stem, "meta": {k: r.get(k) for k in META_COLS},
                "prelabel": [list(b) for b in self.store.read_prelabel(set_name, stem)],
                "boxes": [list(b) for b in boxes], "status": self.store.read_status(set_name, stem)["status"],
                "counts": counts(boxes)}

    def image_path(self, set_name, stem) -> Path:
        path = find_image(self.dataset_root, self.row(set_name, stem)["relative_path"])
        if not path.exists():
            raise NotFound(f"image not found: {path}")
        return path

    def stats(self, set_name):
        total = len(self.items(set_name))
        out = {"approved": 0, "corrected": 0, "rejected": 0}
        for s in self.store.all_status(set_name).values():
            out[s] = out.get(s, 0) + 1
        out["total"] = total
        out["pending"] = total - out["approved"] - out["corrected"] - out["rejected"]
        return out

    def save(self, body: dict):
        set_name, stem = body.get("set"), body.get("stem")
        self.row(set_name, stem)
        action = body.get("action", "save")
        if action not in ("save", "reject", "reset"):
            raise BadRequest(f"unknown action {action!r}")
        try:
            boxes = [tuple(b[:5]) for b in body.get("boxes", [])]
        except TypeError as e:
            raise BadRequest(f"bad boxes: {e}")
        with self.lock:
            status = self.store.save(set_name, stem, boxes, action)
        return {**status, "counts": counts(self.store.current_boxes(set_name, stem))}


def make_server(port: int, store: ReviewStore, dataset_root: Path) -> ThreadingHTTPServer:
    app = ReviewApp(store, dataset_root)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def _send(self, body: bytes, ctype: str, status=200):
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, data, status=200):
            self._send(json.dumps(data).encode("utf-8"), "application/json; charset=utf-8", status)

        def _run(self, fn):
            try:
                fn()
            except NotFound as e:
                self._json({"error": str(e)}, 404)
            except (BadRequest, ValueError, json.JSONDecodeError) as e:
                self._json({"error": str(e)}, 400)
            except Exception as e:  # keep the server alive for the reviewer
                self._json({"error": f"{type(e).__name__}: {e}"}, 500)

        def do_GET(self):
            url = urllib.parse.urlparse(self.path)
            q = {k: v[0] for k, v in urllib.parse.parse_qs(url.query).items()}

            def handle():
                if url.path == "/":
                    self._send(TEMPLATE.read_bytes(), "text/html; charset=utf-8")
                elif url.path == "/api/queue":
                    self._json(app.queue(q.get("set", "test"), q.get("camera")))
                elif url.path == "/api/item":
                    self._json(app.item(q.get("set", "test"), q.get("stem")))
                elif url.path == "/api/image":
                    self._send(app.image_path(q.get("set", "test"), q.get("stem")).read_bytes(), "image/jpeg")
                elif url.path == "/api/stats":
                    self._json(app.stats(q.get("set", "test")))
                else:
                    raise NotFound(url.path)
            self._run(handle)

        def do_POST(self):
            def handle():
                if urllib.parse.urlparse(self.path).path != "/api/save":
                    raise NotFound(self.path)
                length = int(self.headers.get("Content-Length") or 0)
                self._json(app.save(json.loads(self.rfile.read(length) or b"{}")))
            self._run(handle)

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--dataset-root", default=None, help="Folder that contains dataset/.")
    args = ap.parse_args()
    server = make_server(args.port, ReviewStore(), resolve_dataset_root(args.dataset_root))
    print(f"[+] Review tool: http://localhost:{args.port}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
