"""Labelling page server: stdlib http.server, 127.0.0.1 only, static page in label_static/.

    tmux new -d -s footyviz-label 'cd ~/football-vision-bench && PYTHONPATH=src \
        ~/football-vision/.venv/bin/python -m fv.bench label'
    # from the Mac: ssh -L 5191:127.0.0.1:5191 zulzi-gpu  ->  http://localhost:5191

GET  /                         the page
GET  /api/clips                clips that have a prelabel
GET  /api/clip/<id>            {"prelabel": ..., "work": ...}
GET  /data/<id>/frames|crops/<file>.jpg
POST /api/save/<id>            body = work.json; writes it, derives clip.json + keyframes.json
"""
from __future__ import annotations

import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import labels as L

STATIC = Path(__file__).with_name("label_static")
SAFE = re.compile(r"^[A-Za-z0-9_]+$")
FILE = re.compile(r"^/data/([A-Za-z0-9_]+)/(frames|crops)/([A-Za-z0-9_.-]+\.jpg)$")


class H(BaseHTTPRequestHandler):
    def log_message(self, fmt, *a):  # quiet
        pass

    def _send(self, code: int, body: bytes, ctype: str, cache: bool = False):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "max-age=3600" if cache else "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj).encode(), "application/json")

    def do_GET(self):
        p = self.path.split("?")[0]
        if p in ("/", "/index.html"):
            return self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
        if p == "/api/clips":
            out = []
            for d in sorted(L.LABELS.glob("*/prelabel.json")):
                cid = d.parent.name
                w = L.load_work(cid)
                out.append({"id": cid, "kf_done": len(w["kf_done"])})
            return self._json(out)
        m = re.match(r"^/api/clip/([A-Za-z0-9_]+)$", p)
        if m and (L.LABELS / m[1] / "prelabel.json").exists():
            pl = json.loads((L.LABELS / m[1] / "prelabel.json").read_text())
            return self._json({"prelabel": pl, "work": L.load_work(m[1])})
        m = FILE.match(p)
        if m:
            f = L.LABELS / m[1] / m[2] / m[3]
            if f.is_file():
                return self._send(200, f.read_bytes(), "image/jpeg", cache=True)
        self._send(404, b"not found", "text/plain")

    def do_POST(self):
        m = re.match(r"^/api/save/([A-Za-z0-9_]+)$", self.path)
        if not m or not (L.LABELS / m[1] / "prelabel.json").exists():
            return self._send(404, b"not found", "text/plain")
        n = int(self.headers.get("Content-Length", 0))
        try:
            w = json.loads(self.rfile.read(n))
            assert isinstance(w, dict) and "people" in w
        except Exception:
            return self._send(400, b"bad json", "text/plain")
        d = L.LABELS / m[1]
        tmp = d / "work.json.tmp"
        tmp.write_text(json.dumps(w))
        tmp.replace(d / "work.json")
        try:
            info = L.derive(m[1])
        except Exception as e:  # keep work.json even if derive breaks
            info = {"derive_error": repr(e)}
        self._json({"ok": True, **info})


def main(port: int = 5191) -> int:
    srv = ThreadingHTTPServer(("127.0.0.1", port), H)
    print(f"fvlabel on http://127.0.0.1:{port}  (labels: {L.LABELS})", flush=True)
    srv.serve_forever()
    return 0
