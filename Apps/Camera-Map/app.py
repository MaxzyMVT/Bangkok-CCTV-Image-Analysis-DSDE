"""
Bangkok CCTV Camera Map - lightweight web server.

Serves an interactive Leaflet + OpenStreetMap map of Bangkok.
Uses only the Python standard library (no extra installs needed).

Usage:
    python Apps/Camera-Map/app.py --port 8081
"""
import argparse
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler

TEMPLATE_FILE = Path(__file__).resolve().parent / "templates" / "index.html"


class MapRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            body = TEMPLATE_FILE.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404, "Not found")

    def log_message(self, fmt, *args):
        # Keep the console quiet except for errors
        pass


def start_server(port=8081):
    server = HTTPServer(("", port), MapRequestHandler)
    print("=" * 60)
    print(" Bangkok CCTV Camera Map")
    print(f" Local URL: http://localhost:{port}")
    print(" Press Ctrl+C to stop")
    print("=" * 60)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
        server.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bangkok CCTV Camera Map server")
    parser.add_argument("--port", type=int, default=8081, help="Web server port (default: 8081)")
    args = parser.parse_args()
    start_server(port=args.port)
