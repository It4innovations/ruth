"""HTTP server for the console: a stdlib ThreadingHTTPServer in a daemon thread.

The handler never touches simulation objects — it serves the console's latest
pre-encoded snapshot and enqueues syntactically validated commands.
"""

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from urllib.parse import urlparse, parse_qs

from .commands import validate_param, validate_restriction, validate_restriction_delete

logger = logging.getLogger(__name__)

_STATIC_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "application/javascript",
    ".css": "text/css",
    ".png": "image/png",
}


def _static_bytes(relpath):
    base = resources.files("ruth.console") / "static"
    target = (base / relpath)
    # forbid path escape
    if ".." in relpath or relpath.startswith("/"):
        return None, None
    try:
        data = target.read_bytes()
    except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
        return None, None
    for ext, ctype in _STATIC_TYPES.items():
        if relpath.endswith(ext):
            return data, ctype
    return data, "application/octet-stream"


class ConsoleHandler(BaseHTTPRequestHandler):
    # set by start_server
    console = None

    def log_message(self, fmt, *args):  # keep sim logs clean
        logger.debug("console http: " + fmt, *args)

    def _send(self, code, body, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, code, obj):
        self._send(code, json.dumps(obj).encode("utf-8"))

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/" or path == "/index.html":
            data, ctype = _static_bytes("index.html")
            if data is None:
                self._send_json(500, {"error": "index.html missing from package"})
                return
            self._send(200, data, ctype)
        elif path.startswith("/static/"):
            data, ctype = _static_bytes(path[len("/static/"):])
            if data is None:
                self._send_json(404, {"error": "not found"})
            else:
                self._send(200, data, ctype)
        elif path == "/api/state":
            self._send(200, self.console.snapshot_bytes)
        elif path == "/api/commands":
            self._send_json(200, self.console.command_results())
        elif path == "/api/nearest-segment":
            q = parse_qs(parsed.query)
            try:
                lat = float(q["lat"][0])
                lon = float(q["lon"][0])
            except (KeyError, ValueError):
                self._send_json(400, {"error": "lat and lon query params required"})
                return
            seg = self.console.nearest_segment(lat, lon)
            if seg is None:
                self._send_json(503, {"error": "edge index not ready yet"})
            else:
                self._send_json(200, seg)
        elif path == "/api/geometry":
            ids = parse_qs(parsed.query).get("ids", [""])[0].split(",")
            self._send_json(200, self.console.geometry_for_ids([i for i in ids if i]))
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            self._send_json(400, {"error": "invalid JSON body"})
            return

        try:
            if self.path == "/api/param":
                name, value = validate_param(body)
                cmd_id = self.console.submit("set-param", {"name": name, "value": value})
            elif self.path == "/api/restriction":
                payload = validate_restriction(body)
                cmd_id = self.console.submit("restriction", payload)
            elif self.path == "/api/restriction-delete":
                payload = validate_restriction_delete(body)
                cmd_id = self.console.submit("remove-restriction", payload)
            elif self.path in ("/api/pause", "/api/resume", "/api/checkpoint", "/api/stop"):
                cmd_id = self.console.submit(self.path.rsplit("/", 1)[1])
            else:
                self._send_json(404, {"error": "not found"})
                return
        except ValueError as e:
            self._send_json(400, {"error": str(e)})
            return
        self._send_json(202, {"id": cmd_id})


def start_server(console):
    """Bind, start the daemon serve thread, return the server object."""
    handler = type("BoundConsoleHandler", (ConsoleHandler,), {"console": console})
    server = ThreadingHTTPServer((console.args.host, console.args.port), handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, name="ruth-console-http", daemon=True)
    thread.start()
    return server
