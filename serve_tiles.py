"""
Simple HTTP server to serve the tile editor.
Opens the browser automatically.

Usage:
    serve_tiles.bat
"""

import http.server
import json
import os
import sys
import webbrowser
import threading

PORT = 8090
TILES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tiles")


class TileHandler(http.server.SimpleHTTPRequestHandler):
    def do_POST(self):
        if self.path == "/save_geojson":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            out_path = os.path.join(TILES_DIR, "false_positives_background.geojson")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(body.decode("utf-8"))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"saved": out_path}).encode())
            print(f"Saved GeoJSON: {out_path}")
        else:
            self.send_error(404)


def main() -> int:
    os.chdir(TILES_DIR)
    server = http.server.HTTPServer(("localhost", PORT), TileHandler)

    url = f"http://localhost:{PORT}/editor.html"
    print(f"Serving tiles at {url}")
    print("Press Ctrl+C to stop")

    # Open browser after short delay
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
