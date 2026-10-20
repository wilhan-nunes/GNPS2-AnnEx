#!/usr/bin/env python3
"""Minimal CORS proxy for the MASST records API (local testing only).

masst-records.gnps2.org (Datasette) sends no Access-Control-Allow-Origin
header, so gnps2_annex_standalone.html cannot query it from the browser.
This forwards GET /masst_records.json?... to the upstream unchanged and adds
CORS headers. Only that one path is forwarded, so it is not an open proxy.

    python webapp/masst_cors_proxy.py               # http://localhost:8787
    python webapp/masst_cors_proxy.py --port 9000   # then ?masst_proxy=http://localhost:9000

The standalone page's MASST_CORS_PROXY defaults to http://localhost:8787.
Once GNPS2 hosts a proxy (or enables --cors on the Datasette), point
MASST_CORS_PROXY at it and this script is no longer needed.
"""
import argparse
import json
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UPSTREAM = "https://masst-records.gnps2.org"
ALLOWED_PATHS = {"/masst_records.json"}
TIMEOUT_SEC = 60


class MasstProxyHandler(BaseHTTPRequestHandler):
    def _send(self, status, body, content_type="application/json; charset=utf-8"):
        self.send_response(status)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        # Chrome's Private Network Access: lets an https page reach localhost.
        self.send_header("Access-Control-Allow-Private-Network", "true")
        if body:
            self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self._send(204, b"")

    def do_GET(self):
        parsed = urllib.parse.urlsplit(self.path)
        if parsed.path not in ALLOWED_PATHS:
            self._send(404, json.dumps({"ok": False, "error": "path not proxied"}).encode())
            return

        url = UPSTREAM + parsed.path + ("?" + parsed.query if parsed.query else "")
        req = urllib.request.Request(url, headers={"User-Agent": "GNPS2-AnnEx-cors-proxy/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT_SEC) as resp:
                self._send(resp.status, resp.read(), resp.headers.get("Content-Type"))
        except urllib.error.HTTPError as exc:
            self._send(exc.code, exc.read(), exc.headers.get("Content-Type"))
        except Exception as exc:
            self._send(502, json.dumps({"ok": False, "error": f"upstream unreachable: {exc}"}).encode())


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), MasstProxyHandler)
    print(f"Proxying {UPSTREAM}{sorted(ALLOWED_PATHS)} on http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
