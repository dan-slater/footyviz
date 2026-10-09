"""Optional local HTTP front for fv.job (contract: API.md).

    python -m fv.api --port 8790          # binds 127.0.0.1 only

Not needed to use footyviz — `python -m fv.job` is the real entry point. Run
this only if you want a browser front on your own machine. Jobs run one at a
time in a worker thread, each as a `python -m fv.job` subprocess; all state is
on disk under data/jobs/<id>/ (job.json, progress.json, job.log, results).
"""
from __future__ import annotations

import argparse
import json
import queue
import re
import secrets
import subprocess
import sys
import threading
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response

from .job import MAX_CLIP_S, REPO

JOBS = REPO / "data" / "jobs"
_FILE_RE = re.compile(r"^(overlay\.mp4|stats\.json|poster\.jpg|roster\.jpg|crop_[A-Za-z0-9_.-]+\.jpg)$")
_MIME = {".mp4": "video/mp4", ".json": "application/json", ".jpg": "image/jpeg"}

app = FastAPI(title="footyviz", version="1")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST", "OPTIONS"],
                   allow_headers=["*"], expose_headers=["Content-Range", "Accept-Ranges", "Content-Length"])
_q: "queue.Queue[str]" = queue.Queue()
_waiting: list[str] = []
_lock = threading.Lock()


def _read(p: Path, default=None):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return default


def _worker():
    while True:
        jid = _q.get()
        with _lock:
            if jid in _waiting:
                _waiting.remove(jid)
        d = JOBS / jid
        job = _read(d / "job.json")
        cmd = [sys.executable, "-m", "fv.job", "--api", "--url", job["url"],
               "--start-s", str(job["start_s"]), "--end-s", str(job["end_s"]), "--out", str(d)]
        with open(d / "job.log", "w") as log:
            rc = subprocess.run(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT).returncode
        prog = _read(d / "progress.json", {})
        if rc != 0 and prog.get("status") != "error":
            prog.update(status="error", error=f"job exited with code {rc}; see job.log")
            (d / "progress.json").write_text(json.dumps(prog))


def _recover():
    """On start: re-queue queued jobs, fail jobs that were mid-run when we died."""
    JOBS.mkdir(parents=True, exist_ok=True)
    pending = []
    for d in JOBS.iterdir():
        prog, job = _read(d / "progress.json"), _read(d / "job.json")
        if not prog or not job:
            continue
        if prog["status"] == "queued":
            pending.append((job.get("created_at", 0), d.name))
        elif prog["status"] not in ("done", "error"):
            prog.update(status="error", error="interrupted by API restart")
            (d / "progress.json").write_text(json.dumps(prog))
    for _, jid in sorted(pending):
        _waiting.append(jid)
        _q.put(jid)


@app.post("/jobs", status_code=202)
async def create_job(req: Request):
    try:
        body = await req.json()
    except ValueError:
        raise HTTPException(400, "body must be JSON")
    url = str(body.get("url") or "").strip()
    if not url or len(url) > 500:
        raise HTTPException(400, "url is required")
    try:
        start_s = float(body.get("start_s", 0) or 0)
        end_s = float(body.get("end_s", start_s + MAX_CLIP_S))
    except (TypeError, ValueError):
        raise HTTPException(400, "start_s / end_s must be numbers")
    if start_s < 0 or end_s <= start_s:
        raise HTTPException(400, "need 0 <= start_s < end_s")
    clamped = end_s - start_s > MAX_CLIP_S
    end_s = min(end_s, start_s + MAX_CLIP_S)
    jid = f"j_{time.strftime('%Y%m%d')}_{secrets.token_hex(3)}"
    d = JOBS / jid
    d.mkdir(parents=True)
    (d / "job.json").write_text(json.dumps({"id": jid, "url": url, "start_s": start_s, "end_s": end_s,
                                            "clamped": clamped, "created_at": time.time()}))
    (d / "progress.json").write_text(json.dumps({"status": "queued", "progress": 0.0,
                                                 "stage_note": "", "error": None}))
    with _lock:
        _waiting.append(jid)
    _q.put(jid)
    return {"id": jid, "status": "queued", "start_s": start_s, "end_s": end_s, "clamped": clamped}


@app.get("/jobs/{jid}")
def get_job(jid: str):
    d = JOBS / jid
    if not re.fullmatch(r"j_[0-9]{8}_[0-9a-f]{6}", jid) or not (d / "job.json").exists():
        raise HTTPException(404, "unknown job")
    prog = _read(d / "progress.json", {})
    status = prog.get("status", "queued")
    with _lock:
        qpos = _waiting.index(jid) + 1 if jid in _waiting else None
    result = None
    if status == "done":
        result = {"overlay_mp4": f"/jobs/{jid}/files/overlay.mp4", "stats": f"/jobs/{jid}/files/stats.json",
                  "poster": f"/jobs/{jid}/files/poster.jpg", "roster": f"/jobs/{jid}/files/roster.jpg"}
    return {"id": jid, "status": status, "progress": prog.get("progress", 0.0),
            "stage_note": prog.get("stage_note", ""), "queue_position": qpos,
            "error": prog.get("error"), "result": result}


@app.get("/jobs/{jid}/files/{name}")
def get_file(jid: str, name: str, request: Request):
    p = JOBS / jid / name
    if not re.fullmatch(r"j_[0-9]{8}_[0-9a-f]{6}", jid) or not _FILE_RE.match(name) or not p.is_file():
        raise HTTPException(404, "no such file")
    mime = _MIME[p.suffix]
    size = p.stat().st_size
    rng = request.headers.get("range")
    m = re.fullmatch(r"bytes=(\d*)-(\d*)", rng.strip()) if rng else None
    if not m or (not m.group(1) and not m.group(2)):
        return FileResponse(p, media_type=mime, headers={"Accept-Ranges": "bytes"})
    if m.group(1):
        start = int(m.group(1))
        end = min(int(m.group(2)), size - 1) if m.group(2) else size - 1
    else:                                               # suffix range: last N bytes
        start, end = max(0, size - int(m.group(2))), size - 1
    if start >= size or start > end:
        return Response(status_code=416, headers={"Content-Range": f"bytes */{size}"})
    with open(p, "rb") as fh:
        fh.seek(start)
        data = fh.read(end - start + 1)
    return Response(data, status_code=206, media_type=mime, headers={
        "Accept-Ranges": "bytes", "Content-Range": f"bytes {start}-{end}/{size}",
        "Content-Length": str(len(data))})


def _gpu_name() -> str:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=5).stdout.strip()
        return out.splitlines()[0] if out else "cpu"
    except (OSError, subprocess.SubprocessError):
        return "cpu"


_GPU = None


@app.get("/health")
def health():
    global _GPU
    _GPU = _GPU or _gpu_name()
    with _lock:
        n = len(_waiting)
    return JSONResponse({"ok": True, "gpu": _GPU, "queue": n})


def main() -> None:
    import uvicorn
    ap = argparse.ArgumentParser(description="Optional local job API for footyviz (see API.md).")
    ap.add_argument("--port", type=int, default=8790)
    args = ap.parse_args()
    _recover()
    threading.Thread(target=_worker, daemon=True).start()
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="info")


if __name__ == "__main__":
    main()
