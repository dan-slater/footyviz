# footyviz job API — contract v1

The contract between the footyviz pipeline (this repo, `src/fv/api.py`) and the
web front end (`footyviz-web`). **Stable**: fields are only ever *added*; a
breaking change bumps `schema_version` and is announced upstream first.

- Base URL (dev, on the GPU box): `http://127.0.0.1:8790`
- All bodies are JSON (`Content-Type: application/json`), UTF-8.
- CORS: `Access-Control-Allow-Origin: *` on every route (GET/POST/OPTIONS).
- Times are seconds (float). Fractions are 0..1 floats.
- One job runs on the GPU at a time; the rest queue FIFO.

---

## `POST /jobs`

Submit a clip for analysis.

Request
```json
{ "url": "https://www.youtube.com/watch?v=hhaWMwLlugE", "start_s": 0, "end_s": 120 }
```
| field | type | notes |
|---|---|---|
| `url` | string, required | A YouTube or Veo share URL (anything yt-dlp can fetch). |
| `start_s` | number, default `0` | Clip start within the source video. |
| `end_s` | number, default `start_s + max` | Clip end. |

The clip length is capped at **`FOOTYVIZ_MAX_CLIP_S` (default 180 s)**. A longer
request is **clamped** (not rejected): `end_s = start_s + max`; the response
says so.

Response `202`
```json
{ "id": "j_20261009_1a2b3c", "status": "queued", "start_s": 0, "end_s": 120, "clamped": false }
```
Only `id` is required for the client; the rest is informational.

Errors: `400 {"detail": "..."}` for a missing/invalid url or `end_s <= start_s`.

---

## `GET /jobs/{id}`

Poll every 1–2 s.

```json
{
  "id": "j_20261009_1a2b3c",
  "status": "tracking",
  "progress": 0.42,
  "stage_note": "detect+track 760/1800 frames",
  "queue_position": null,
  "error": null,
  "result": null
}
```
| field | type | notes |
|---|---|---|
| `status` | enum | `queued` → `downloading` → `tracking` → `teams` → `stats` → `rendering` → `done`, or `error` at any point. |
| `progress` | 0..1 | Whole-job progress (monotonic). `1.0` at `done`. |
| `stage_note` | string | Human-readable detail for the current stage. |
| `queue_position` | int \| null | 1 = next to run. `null` once started. |
| `error` | null \| string | Set when `status == "error"`. |
| `result` | null \| object | Set only when `status == "done"`: |

```json
"result": {
  "overlay_mp4": "/jobs/j_20261009_1a2b3c/files/overlay.mp4",
  "stats":       "/jobs/j_20261009_1a2b3c/files/stats.json",
  "poster":      "/jobs/j_20261009_1a2b3c/files/poster.jpg",
  "roster":      "/jobs/j_20261009_1a2b3c/files/roster.jpg"
}
```
Paths are relative to the API base URL. `404 {"detail": "unknown job"}` for an unknown id.

---

## `GET /jobs/{id}/files/{name}`

Serves a result file. Supports **HTTP Range** (`206 Partial Content`,
`Accept-Ranges: bytes`) so `<video>` seeks on phones. Names:

| name | type | what |
|---|---|---|
| `overlay.mp4` | `video/mp4` | H.264 (yuv420p, faststart), source resolution & fps, no audio. Team-coloured chevrons over each player; honesty line burned in as a footer. |
| `stats.json` | `application/json` | Schema below. |
| `poster.jpg` | `image/jpeg` | A representative overlaid frame (same size as the video). |
| `roster.jpg` | `image/jpeg` | Grid of best crops, one per roster entry, grouped by team. |
| `crop_<rosterId>.jpg` | `image/jpeg` | One player crop (referenced from `stats.json` `roster[].crop`). |

`404` if the file does not exist (yet).

---

## `GET /health`

```json
{ "ok": true, "gpu": "NVIDIA GeForce RTX 4090", "queue": 1 }
```
`queue` = jobs waiting (not counting the one running). `gpu` is `"cpu"` if no CUDA.

---

## `stats.json` (schema_version 1)

Example values are illustrative. Team keys are always `"A"` and `"B"` (the two
outfield kits). Goalkeepers / referees / odd kits are cluster `"O"` and are
**excluded** from team stats.

```jsonc
{
  "schema_version": 1,

  "source": {
    "url": "https://www.youtube.com/watch?v=hhaWMwLlugE",
    "title": "…",            // yt-dlp title, or the file name for a local file
    "fps": 30.0,
    "width": 1280, "height": 720,
    "start_s": 666.7, "end_s": 726.7,   // the analysed clip, in source-video seconds
    "frames": 1800                       // frames analysed
  },

  "teams": {
    "A": { "label": "White", "color": "#e4e6e8", "tracks": 41 },
    "B": { "label": "Pink",  "color": "#d9708f", "tracks": 37 }
  },
  // label = a plain colour name derived from the kit colour; color = median kit colour (hex RGB).
  // tracks = raw tracker IDs assigned to that team (diagnostic, not a player count).

  "possession": {
    "A": 0.48, "B": 0.37, "unresolved": 0.15,   // sum to 1 over analysed frames
    "ball_frames_frac": 0.34,                   // fraction of frames with a ball detection
    "method": "Proxy: in each frame with a ball detection the team of the nearest player (foot point) within 3 player-heights owns the ball; frames without a ball inherit the last owner for up to 1 s, otherwise count as unresolved. Ball detected in 34% of frames."
  },

  "distance": {
    "method": "No homography. Player foot-point tracks in image pixels, camera pan removed by global phase correlation, smoothed, jumps > 0.5 frame-widths/s dropped. Metres = pixels / (frame width / 25 m), i.e. assumes the visible pitch width is 25 m. Expect ±50%.",
    "assumed_visible_width_m": 25,
    "teams": {
      "A": { "px": 182340.0, "m": 3561.3 },
      "B": { "px": 160221.0, "m": 3129.3 }
    },
    "players": [
      { "id": "A-3", "team": "A", "px": 21034.0, "m": 410.8, "frames": 1712 }
    ]
    // players: one per roster entry (same ids as roster[]), sorted by m descending.
  },

  "heatmaps": {
    "grid": [32, 18],        // [cols, rows]
    "space": "image",        // image coordinates of the (moving) camera view, NOT pitch coordinates
    "A": [[0.0, 0.1, "…32 values"], "…18 rows"],
    "B": [["…"]]
    // A[row][col], row 0 = top of frame, col 0 = left. Foot-point counts per cell,
    // normalised so the team's busiest cell = 1.0. Draw as 32x18 on a 16:9 canvas.
  },

  "roster_level": "identity",   // "identity" = stable IDs from the identity pass; "track" = raw tracker IDs (fallback)
  "roster": [
    {
      "id": "A-3",                // "A#7" when a confident jersey number anchors it
      "team": "A",                // "A" | "B" | "O"
      "frames": 1712,             // frames this identity was seen
      "crop": "/jobs/j_…/files/crop_A-3.jpg",
      "jersey": null              // string like "7" only when OCR was confident, else null
    }
  ],
  // sorted by team then frames descending.

  "identity_note": "Identity is capped by the footage: this pipeline resolved 23 stable identities; on the reference night slice it found ~23 stable vs ~14–16 true players. Team-level stats do not depend on perfect identity.",
  // N (here 23) = len(roster). Show this sentence verbatim near any per-player number.

  "events": [
    { "type": "whistle", "t_s": 12.4, "frame": 372 }
    // t_s / frame are relative to the clip start. May be empty (no audio, or no whistles).
  ],

  "pipeline": {
    "footyviz": "0.2.0",
    "yolo": "yolo11m.pt (ultralytics 8.x)",
    "tracker": "ByteTrack (supervision 0.x)",
    "reid": "BoT-SORT + OSNet x1_0 msmt17 (boxmot x)",   // null if the identity pass was skipped
    "vlm": null,                                         // VLM layer is off by default
    "run_time_s": 143.2
  }
}
```

### Notes for the front end
- Team stats (possession, team distance, heatmaps) are the trustworthy layer.
  Per-player numbers inherit identity errors — always pair them with `identity_note`.
- `possession` can be dominated by `unresolved` on clips where the ball is
  small/occluded; show `ball_frames_frac` alongside.
- Pre-rendered demo results (same file set) live at
  `~/orch-scratch/footyviz-demo/{day,night}/` on the GPU box.
