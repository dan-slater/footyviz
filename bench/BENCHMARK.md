# footyviz-bench: design (v0.1, DRAFT for review)

Status: **design only, nothing built.** Phase 2 (manifest, fetch, pre-labels, label page,
`fv.bench`, baselines, leaderboard) starts after this is approved.
Owner line: ORCH-fv-bench (branch `bench`). Date: 2026-10-09.

The benchmark is the measuring stick for everything that follows: comparing pipeline approaches,
swapping models, and later fine-tuning. So this doc puts **ground truth, metric definitions and
variance control** ahead of speed. Each decision lists its reason.

---

## 0. What the footage looks like (probed, not assumed)

**Sources are full matches.** The control plane relays whole 1080p matches with audio
(about 40 min and about 1.1 GB each) to `~/footyviz-bench-data/raw/<tier>_<videoid>.mp4`.
A `.tmp` suffix means the copy is still in progress. The reference match is already at
`~/football-vision/data/raw/hhaWMwLlugE_1080.mp4`. `fv.bench` cuts every benchmark window from
these files (§1, §7). So far only the reference window `A1` (900–1020 s, cut from the local
1080p) has been probed; each relayed match is probed once it lands. Findings:

- 1920×1080, 30 fps, H.264, AAC stereo. YouTube's ceiling here is 1080p at about 2 Mb/s, so the
  source is already heavily compressed.
- Veo panoramic auto-pan. Roughly half the pitch is in view at a time, with 5–9 people on screen.
  Players are about **40–90 px tall**, and smaller at the far side.
- The ball is **about 8–12 px**, often motion-blurred, and often merged with a foot.
- **Stitch/lens ghosts at the frame edges**: half-transparent duplicate people near the goal
  when the camera turns (end of the clip). Floodlight flare and bloom sit over the right half.
- **Kits are not clean.** One side wears pink or orange bibs over mixed shirts. The other wears
  white shirts and grey hoodies, plus a maroon shirt. Keepers wear their own colours.
- **Non-players are always in view**: subs and spectators behind the fence and on the walkway,
  and a person standing by the goal. In this window, frame 0 is already floodlit night.
- A permanent "veo" watermark sits bottom-right.
- **Video titles carry team names and the final score**, e.g. "Swans 4 vs Mamelodi Sundowners 8".
  That gives free match-level labels, but the same teams recur across matches (see §3).

Consequences that shape the design:

- Small objects and soft labels make IoU 0.5 the right threshold. AP@0.5:0.95 would mostly
  measure how sloppy the labels are.
- Off-pitch people and ghosts need an explicit **ignore** class.
- Team labels must be kit **descriptions** per clip, not fixed colours.
- Splits must be **team-disjoint**, not only match-disjoint.

Per-clip objective quality stats go into the manifest, so the tiers can be checked against
numbers rather than eyeballed: mean luma, RMS contrast, Laplacian sharpness, median GT player box
height, ball visibility rate and colour cast. Phase 2 also re-checks each relayed clip by eye.

---

## 1. Match set and windows v1

The unit of the benchmark is the **match**: one full video, one tier, one split. Inside a match
there are three kinds of window:
- **Labelled eval windows**: 120 s, human GT (§3). Default `900–1020 s`. The table below lists
  these.
- **Full-match runs**: the method runs over the whole match. There is no box GT, but these runs
  measure throughput, cap hits and lighting drift across the match, and they score the one
  full-match label we get for free: the **final score in the title**, kept for the later
  goals/events task. The audio also makes whistles scoreable later.
- **Train material**: *the whole* of every train match (4 × about 40 min ≈ 160 min,
  pseudo-labelled). The unlabelled parts of **eval** matches are **eval-reserved** and never
  used for training, because they are the same people.

**Choosing the window.** The 900 s default is fixed in advance, *before* any method sees the
footage, so the window is not picked to flatter a method. Two exceptions, recorded in the
manifest:
- If 900–1020 is mostly a stoppage (more than 30 s with the ball out of play or teams
  regrouping), shift the window +60 s at a time until it isn't.
- A match whose lighting changes during the game gets a **second** labelled window in the other
  lighting, scored under that window's tier. The reference match's dusk-to-night change is the
  obvious case; whether it adds a window is decided when the full matches are probed.

Phase 2 may add more 120 s eval windows per eval match. Each one costs about 35 Daniel-minutes
(§3), so v1 starts with one per match.

| id | tier | video | window (source s) | teams (from title) | role |
|---|---|---|---|---|---|
| A1 | A night | hhaWMwLlugE | 900–1020 | Expected Toulouse v Friski Diski | eval |
| A0 | A night | hhaWMwLlugE | 0–60 (f0–1800) | same match | eval, *continuity only* |
| A2 | A night | O4YGl9Ps93A | 900–1020 | goal diggers v sky 5 | train |
| B1 | B dusk | mzcJ8_wyOaw | 900–1020 | Friski Diski v Retention United | eval (forced: shares Friski Diski with A1) |
| B2 | B dusk | vS4NP3wkBIA | 900–1020 | Swans v Mamelodi Sundowners | train |
| C1 | C day/shadow | GQfni7JdYvY | 900–1020 | Nose Beers Utd v Cesc Pistols | eval |
| C2 | C day/shadow | 2u32W0O7ZfQ | 900–1020 | Hellas v FC Broadsville | train |
| D1 | D easy | c6IV1mQ3GjM | 900–1020 | Salahds Wins Scrums v Pique Blinders | eval |
| D2 | D easy | gjd12dkMNNI | 900–1020 | Salahds win scram v Sibas kids fc | eval (forced: shares Salahds with D1) |
| D3 | D easy | *replacement, TBD* | 900–1020 | must not share a team with any eval clip | train |

- **A0** is the night slice used by runs 1–13. It is scored and shown separately so we keep
  continuity with that history. It is **excluded from the tier-A and overall aggregates**, so one
  match does not count twice.
- **D3**: both D clips share the Salahds team, so neither can be train while the other is eval.
  Candidates from `thumb_scores.csv`: league-style title, luma > 95, edges > 25, no team shared
  with the eval set. `YwkdvL6Un3g` (Unathletico v The Wizards), `b6_9iO20jGQ` (Jenna Tolls v
  Flappys) and `vWyHM8saydI` (Banana Banana v Stiles). These need an eyeball check and a relay
  of the full match from the Mac.
- **Why 2-minute windows at 900 s**: about 15 minutes in, so play has settled and the window is
  inside the first half. Two minutes is long enough for real identity churn: players leave and
  re-enter view, and subs come on. It is short enough to label. Identity over 40 minutes, with
  rolling subs, is a different and harder problem. The full-match runs show it qualitatively;
  labelling it is out of scope for v1.

---

## 2. Tasks and metrics

### Conventions
- Frames are **0-based indices into the clip** at the clip's fps. Each label also stores the
  source time. Boxes are 1080p pixel coordinates `[x1,y1,x2,y2]`.
- **Keyframes** are every 3 s (frames 0, 90, 180, …): 40 per 120 s clip and 20 for A0.
  - Why 3 s: views decorrelate enough between keyframes that each one adds information.
  - 40 × about 8 people ≈ 320 boxes per clip, which is enough to measure recall to about ±3 pp.
- **Team permutation.** The predicted "A"/"B" labels are arbitrary. Per clip, predicted teams are
  mapped to GT teams by Hungarian matching on matched detections. That one mapping is reused for
  the team, identity and possession metrics, so a label swap is never penalised twice.
- **Detection matching** uses greedy IoU ≥ 0.5, highest confidence first. Every downstream
  metric (team, identity) is scored **on matched GT boxes only**. Detection misses are reported
  separately as *coverage*, so a stage is not charged for an upstream stage's error.
- **Reporting.** Every metric is reported per clip, per tier (A/B/C/D, eval clips) and overall:
  the mean over eval clips, weighted equally per clip, not per box. The primary view is **per
  tier**, because "does it degrade gracefully from D to A?" is the question. Overall is a single
  number for the headline only.
- **Uncertainty.** A 95% bootstrap CI over keyframes within a clip, and over clips for the
  aggregates. With about 5 eval clips, overall CIs will be wide. That is honest, and it is why
  the per-tier view and the paired comparisons in §5 matter more than the headline.

### T1: person detection
- **GT.** On keyframes, every person gets a box and a role:
  - `player`: an on-pitch participant, including keepers.
  - `referee`.
  - `ignore`: off-pitch people, subs behind the fence, stitch ghosts, people under 15 px tall,
    and anyone heavily cut off at the frame edge.
- **Metric.**
  - **AP50** for the `player` class. Predictions matched to `ignore` are neither TP nor FP.
  - **Recall@0.5** at the method's own operating threshold, since that is what downstream
    stages consume.
  - Precision at the operating point.
  - Recall split by box height: < 40 px, 40–80 px, > 80 px.
- **Why not AP50:95**: hand boxes on 40 px, blurred players carry about 3–5 px of label noise.
  Tight IoU levels would score label noise.
- **Why ignore off-pitch people here**: a COCO person detector is *right* to find them.
  Rejecting non-participants is scored in T3 (role), where that decision actually lives.

### T2: ball
- **GT.** Every 1 s (120 per clip), a ball centre point plus a state: `visible`, `occluded`
  (position known from context, e.g. at a player's feet) or `out_of_view`.
- **Metric.**
  - **Ball recall**: share of `visible` labels with a predicted ball centre within
    **max(12 px, 0.25 × median player height)**.
  - **Precision**: share of predictions on labelled frames that land near a visible or occluded
    ball.
  - **F1**.
  - The error rate on `out_of_view` frames, i.e. predicting a ball when none is in view.
- **Why points, not boxes**: at 8–12 px the box is the point, and clicking is 3× faster.
  Per-second sampling covers the whole clip, not just keyframes.

### T3: team assignment and role
- **GT.** Each `player`/`referee` box on keyframes carries `team` ∈ {T1, T2} and `role` ∈
  {outfield, keeper, referee}. The clip header describes each team's kit in words
  (e.g. "pink/orange bibs").
- **Metric.**
  - **Team accuracy** on matched outfield-player boxes, after the permutation mapping.
  - **Keeper handling**: our pipelines put keepers and odd kits into "O". Keepers get their own
    accuracy, where "O" or the keeper's own team both count as correct, but the *other* team does
    not.
  - **Non-player rejection**: a prediction matched to an `ignore` box (spectator or sub) that the
    method assigned to T1/T2 is a *false participant*. Report the false-participant rate, a known
    defect class (night roster B-8/B-13).
- **Why on GT boxes, not on a method's tracklet crops**: crops tied to one method's tracker make
  the GT method-dependent. GT boxes let any pipeline be scored.

### T4: tracking and identity (the hard one)
GT for this task is **person IDs on the keyframe boxes**. Each clip gets a roster gallery
(P01…Pn, one exemplar crop each, team, role, optional jersey number), and every keyframe box is
assigned a P-id or `unsure`. A predicted box inherits its method's identity label (stable ID, or
raw track ID for track-level methods). Metrics:

1. **Identity count error** per team: `|pred − true|`, and signed `pred − true`.
   - True = the number of distinct GT people who were on-pitch participants for that team during
     the window.
   - Pred = the number of predicted identities with at least one matched keyframe box on that
     team, plus unmatched identities that persist for ≥ 2 s, so phantoms count.
   - *Cheap and readable, but gameable*: an over-merger can hit the count by luck. **It is never
     reported alone**; it is always paired with (2).
2. **Pairwise same-person precision/recall/F1** over *all* pairs of matched GT boxes from
   **different keyframes** (about 50k pairs per clip, free once the IDs are labelled):
   - precision = P(same GT person | method says same);
   - recall = P(method says same | same GT person).
   Over-merging lowers precision and over-splitting lowers recall, so it separates the two
   failure modes the identity work keeps swinging between (19 over-merged vs 24–43 over-split).
   This is the cheap IDF1 proxy. It is computed on all pairs instead of sampled tracklet pairs,
   because sampled pairs depend on one method's tracklets.
3. **Keyframe IDF1**: IDF1 computed on the sparse keyframe sequence (Hungarian GT-ID ↔ pred-ID
   over keyframe boxes). It is close to the real IDF1 for long-range identity, which is what the
   roster stage changes. It is blind to short-range ID switches between keyframes; the dense
   subset covers that.
4. **Dense subset** (§3, L4), 20 s per tier with every frame labelled:
   - **HOTA (+ DetA/AssA), IDF1, MOTA, ID switches**, via TrackEval in MOT format;
   - plus the same identity metrics on these frames.
   These are the standard numbers for comparing against published trackers.

`unsure` GT boxes are excluded from the identity metrics but kept for detection and team. The
`unsure` rate per tier is reported as a footage-difficulty statistic in its own right.

### T5: possession
- **GT.** Per second of the clip: `T1`, `T2`, `contested` (no team in control, a loose ball or a
  50/50) or `dead` (ball out of play, restart pending, or ball out of view with no inference
  possible).
- **Pred.** The method's per-frame owner, reduced to the per-second majority, after the
  permutation mapping.
- **Metric.**
  - **Possession agreement**: accuracy on seconds where the GT is T1/T2.
  - **Share error**: `|pred share T1 − true share T1|` over those seconds. This is what a user
    sees on the stats page.
  - Abstention: the share of GT-T1/T2 seconds where the method says unresolved.
- **Why seconds, not frames**: humans judge control at about 1 s granularity, and frame labels
  would be fake precision.

### Out of scope for v1 (noted for later)
Events (whistles, goals: the title gives the final score but not the times), jersey OCR
(an optional `jersey` field on the gallery records it if legible), distance and heatmap
(needs pitch homography GT).

---

## 3. Ground-truth plan, costed in Daniel-minutes

### Principle
Models pre-label and Daniel verifies. **Every label Daniel touches records `status`
(`prelabel` → `verified`) and the seconds spent**, so the costs below become measured, not
guessed.

**Anti-bias rule.** Pre-labels anchor the human toward the pre-labeller, which would flatter any
benchmarked method that resembles it. So:
- the pre-labeller is a configuration that is **not a leaderboard entry** (an ensemble, see
  below);
- **3 keyframes per eval clip are labelled blind, from scratch.** The difference between blind
  and verified labels is reported as the pre-label bias estimate. If it is large, we label more
  frames blind.

**Pre-labellers.** These are offline, slow, high-recall and allowed to be expensive:
- Boxes: the union of YOLO11x at imgsz 1280 (conf 0.10) and RT-DETR-x, NMS-merged. The human
  mostly deletes, which is faster than drawing.
- Box ↔ ID: BoT-SORT + OSNet tracklets over the whole clip propagate gallery IDs. The human fixes
  the groups, not individual boxes.
- Team: the k-means kit colour, plus a Gemini neutral question per gallery person ("which of
  these two kit descriptions does this person wear?"), loaded from `~/football-vision/.env` by
  path.
- Ball: the detector plus linear interpolation over 1 s gaps, with the state guessed from the
  detector confidence.
- Possession: the current `fv.stats` proxy on the pre-label boxes.
- Dense subset: tracker boxes and IDs on every frame. SAM2, prompted from the corrected keyframe
  boxes, propagates masks to tighten boxes in between. The human corrects every 10th frame and
  fixes ID switches; the frames in between are interpolated.

### Label layers and cost (per 120 s clip; A0 at about half)

| layer | what | how | Daniel-min / clip |
|---|---|---|---|
| L0 | roster gallery: who played (P-ids, team, role, kit text, jersey if legible) → true count | page pre-fills candidate people from tracklet clusters; Daniel merges/splits/discards, watching the clip at 2× | 6 |
| L1 | keyframe boxes + role (40 kf) | delete/adjust pre-label boxes, add misses, mark ignore | 10 |
| L1b | keyframe person-ID (+ team via gallery) | pre-filled from tracklets; fix wrong assignments, mark `unsure` | 8 |
| L2 | ball point + state, 1 Hz | accept/drag/flip state; keyboard-driven | 4 |
| L3 | possession per second | play the clip at 1×, hold `1`/`2`, tap `c`/`d`; then one pass to fix | 4 |
| blind | 3 keyframes from scratch (bias audit) | | 3 |
| **cheap total** | | | **≈ 35** |
| L4 | dense box + ID, 20 s (600 frames), one eval clip per tier | correct every 10th frame + ID switches, interpolate | **≈ 40 per segment** |

- **v1 eval set**: 5 clips (A1, B1, C1, D1, D2) × 35 min, plus A0 × about 18 min ≈ **3.2 h**.
- **Dense subset**: 4 segments × 40 min ≈ **2.7 h**. I propose doing it **after** the cheap
  layers, once we see whether the keyframe metrics already separate the methods.
- The cheap layers can be done in 20–30 minute sittings, one clip each.
- **Train matches need no Daniel time in v1.** Each full train match gets pre-labels for
  fine-tuning, i.e. pseudo labels: about 72k frames per match. A verified train set (e.g. a few
  sampled windows per train match through the same page) is a later, separate budget.

Expected noise, recorded and not hidden:
- the person count on A-tier night footage may itself be uncertain (grey hoodies);
- the gallery allows `unsure` merges;
- the tier's `unsure` rate is a reported statistic.

### Label files (JSON, committed under `bench/labels/<clip_id>/`)
These contain no pixels and no names, only box coordinates and anonymous P-ids. Jersey *numbers*
are allowed. Printed shirt *names* (e.g. "Elliot 7") are **not** recorded.

```jsonc
// clip.json: L0 header + gallery
{
  "schema": "fvbench.clip/1",
  "clip_id": "A1", "video_id": "hhaWMwLlugE", "start_s": 900.0, "end_s": 1020.0,
  "fps": 30.0, "n_frames": 3600, "width": 1920, "height": 1080,
  "teams": {
    "T1": {"kit": "pink or orange bibs over mixed shirts", "title_name": "Friski Diski"},  // title_name optional
    "T2": {"kit": "white shirts / grey hoodies", "title_name": null}
  },
  "people": [
    {"pid": "P01", "team": "T1", "role": "outfield", "jersey": "11", "exemplar": {"frame": 450, "box": [812,233,861,341]},
     "on_pitch": [[0, 3599]]},                       // frame ranges as participant (subs!)
    {"pid": "P07", "team": "T2", "role": "keeper", "jersey": null, "exemplar": {...}, "on_pitch": [[0, 3599]]}
  ],
  "true_count": {"T1": 6, "T2": 7, "referee": 0},  // derived from people[], stored for readability
  "provenance": {"status": "verified", "labeller": "daniel", "seconds": 410, "tool": "fvlabel 0.1",
                 "prelabel": "ens-yolo11x1280+rtdetrx/botsort-osnet", "date": "2026-10-xx"}
}
```
```jsonc
// keyframes.json: L1 + L1b
{
  "schema": "fvbench.keyframes/1", "clip_id": "A1", "stride": 90,
  "frames": [
    {"frame": 0, "blind": false, "seconds": 14,
     "boxes": [
       {"box": [812,233,861,341], "role": "player", "pid": "P01"},     // role: player|referee|ignore
       {"box": [100,40,121,90],  "role": "ignore", "why": "off_pitch"}, // why: off_pitch|ghost|tiny|truncated
       {"box": [400,300,450,410],"role": "player", "pid": "unsure", "team": "T2"} // team only needed when pid is unsure
     ]}
  ],
  "provenance": {...}
}
```
```jsonc
// ball.json: L2 (1 Hz)
{ "schema": "fvbench.ball/1", "clip_id": "A1",
  "points": [ {"frame": 0, "state": "visible", "xy": [1043, 151]},
              {"frame": 30, "state": "occluded", "xy": [998, 170]},
              {"frame": 60, "state": "out_of_view", "xy": null} ],
  "provenance": {...} }
```
```jsonc
// possession.json: L3 (per second, index = clip second)
{ "schema": "fvbench.possession/1", "clip_id": "A1",
  "seconds": ["T1","T1","contested","T2","dead", "..."],   // len = 120
  "provenance": {...} }
```
```
dense/<segment>.txt   : MOT-challenge gt.txt  (frame,pid_int,x,y,w,h,conf=1,class,visibility)
dense/<segment>.json  : {"schema":"fvbench.dense/1","clip_id","start_frame","end_frame","pid_map":{"1":"P01",...},"provenance"}
```

### Labelling page
- A small FastAPI app with a static front end (`fv.bench.label`) in tmux **`footyviz-label`**,
  bound to **127.0.0.1:5191**. Daniel reaches it with `ssh -L 5191:127.0.0.1:5191 zulzi-gpu`.
- Five views, matching the layers: Gallery, Keyframes, Ball, Possession, Dense.
- It is keyboard-first and autosaves to `bench/labels/`.
- Frames are served from `~/footyviz-bench-data/`; nothing leaves the box.
- Screenshot-checked with Playwright before Daniel sees it.

---

## 4. Splits (for fine-tuning later)

**Rule: train and eval are team-disjoint.** No team named in an eval match title may appear in a
train match.
- **Why match-disjoint is not enough**: this is a league, and teams recur constantly
  (Friski Diski 39 uploads, Mamelodi Sundowners 90, Pique Blinders 61). The same people in the
  same kits would leak identity and team appearance into a fine-tuned ReID or team model.
- **Enforcement**: the manifest stores normalised team names, parsed from titles and
  hand-checked. `fv.bench` refuses to score a model whose `trained_on` manifest field overlaps
  the eval teams.
- **Venue**: every Rookie clip is the same few Cape Town venues, so venue leakage is unavoidable
  inside this channel. That is documented. An out-of-channel eval clip is the future fix.
- **v1**: eval matches = hhaWMwLlugE (A1, A0), mzcJ8_wyOaw (B1), GQfni7JdYvY (C1),
  c6IV1mQ3GjM (D1), gjd12dkMNNI (D2). Train matches = O4YGl9Ps93A, vS4NP3wkBIA, 2u32W0O7ZfQ and
  the D3 replacement, each used **in full**.
  The split is decided per match, never per frame or window.
- **Growth**: `thumb_scores.csv` has 1,631 league-style titles. Filter out rugby, school
  festivals and 11-a-side. New train matches are drawn from teams that never appear in eval;
  new eval matches move their teams' other matches out of train.
- **Frozen eval**: eval labels are versioned (`bench/labels/VERSION`). A leaderboard row records
  the label version it was scored against.

---

## 5. Harness contract

```
python -m fv.bench run   --method bench/methods/<m>.yaml --clips bench/manifest.json \
                         --out ~/footyviz-bench-data/results/<m>/<batch>/ [--split eval] [--clip A1] [--repeats N]
python -m fv.bench score --results ~/footyviz-bench-data/results/<m>/<batch>/ --labels bench/labels/
python -m fv.bench board                      # -> bench/leaderboard/LEADERBOARD.md + leaderboard.html
python -m fv.bench compare --a <m1>/<batch> --b <m2>/<batch>    # paired diff, see variance
```

### Method config (swap a model = a new yaml)
```yaml
name: run13-gklock
inherits: base-cv.yaml            # shallow per-stage override
stochastic: true                  # => repeats required
repeats: 3
detector: {weights: yolo11m.pt, imgsz: 640, conf: 0.30, ball: {weights: yolo11m.pt, class: 32, conf: 0.10}}
tracker:  {name: botsort, params: {track_buffer: 90, appearance_thresh: 0.25}}   # bytetrack | botsort | ocsort | ...
reid:     {model: osnet_x1_0, weights: data/models/osnet_x1_0_msmt17.pt}
teams:    {method: kmeans_color, k: null}                                      # kmeans_color | embedding | vlm
identity:
  method: gk_lock                # none | cv_fusion | roster_v3 | gk_lock
  params: {emb_gate: 0.55, min_conf: 0.6, max_calls: 40}
  vlm: {provider: gemini, model: gemini-3.1-pro-preview, temperature: null}
possession: {method: nearest_player, params: {max_dist_heights: 3, hold_s: 1.0}}
```

**Prediction format.** Each method writes this per clip and repeat; the scorer only reads this
format:
```
<out>/<clip>/r<k>/
  dets.parquet      frame, x1,y1,x2,y2, conf, track_id, identity_id, team(A|B|O), role(outfield|keeper|referee|other|null)
  ball.parquet      frame, x, y, conf                 (≤1 per frame)
  possession.parquet frame, owner(A|B|null)
  meta.json         git sha, config hash, versions, wall time per stage, GPU, VLM {calls, cap_hit, tokens, cost_usd}, seed
```

**Stages are adapters.** Each stage is a function with parquet in and parquet out, registered
by name, so a new model is a new adapter plus a yaml.
- `roster_v3` and `gk_lock` are currently hard-wired to night-slice paths
  (`data/tracks/context_night`, …). Phase 2 wraps them in per-clip adapters inside
  `fv/bench/`, and touches their modules as little as possible to avoid conflicts with
  ORCH-fv-pipeline.
- **Stage-isolated (oracle) runs**: `--oracle detect` feeds GT keyframe boxes, or dense-subset
  GT boxes, into the downstream stages. That separates "the identity method is bad" from "the
  detections it got were bad".

**Score output.** `scores.json` holds every metric × clip × repeat, the aggregates, the CIs and
the label version. It is small, has no pixels, and is committed under
`bench/leaderboard/runs/`. Raw predictions stay in `~/footyviz-bench-data/`.

### Variance and same-day control (from exp 13: run 11 re-run went 24 → 43 on identical inputs)
1. **Repeats.** Any config with a stochastic stage (any VLM call; also flagged automatically if
   two repeats' `dets.parquet` hashes differ) runs **N = 3 repeats per clip by default** (5 for
   final claims). Deterministic configs run once, plus one determinism check on first use.
2. **Reporting.** Each metric is reported as **mean ± sd over repeats**, plus the min–max range.
   The leaderboard shows `mean ± sd`, never a single run. Run-to-run sd is itself a column
   (*stability*): a method that gives 24 or 43 on the same input is worse than one that gives
   30 ± 1.
3. **Batches and controls.** Every `run` belongs to a **batch** (a date plus a short tag).
   - `compare` and the leaderboard's "Δ vs control" column **refuse to compare across batches**
     unless `--allow-cross-batch` is given, and then they print a warning.
   - Each batch that contains a stochastic method must include the **control config** (for
     identity: the current reference, run 11) re-run in the same batch.
4. **Paired statistics.** Δ = method − control, per clip and repeat. Report the mean Δ with a
   bootstrap 95% CI over clips × repeats, and the number of clips where the method wins.
   "Better" requires the CI to exclude 0. Otherwise it reads "no detectable difference".
5. **What gets logged** (in `meta.json`, aggregated on the board):
   - VLM call counts, **cap hits** (exp 13 hit the 40-call cap on every team);
   - rejected-merge reasons;
   - provider/model version strings.
   A run that hit a cap is flagged on the board.
6. **Cache.** The VLM response cache (keyed on model, prompt and image hashes) is **off for
   scored runs** because it would hide the variance. It is allowed for `--replay` debugging only,
   and that is marked on the result.

Leaderboard layout: one table per task. Rows are methods. Columns are D | C | B | A | overall
(eval), plus A0 continuity, stability (sd), runtime/clip and VLM $/clip. The HTML version adds
per-clip drill-down.

---

## 6. Baselines (first batch, same day)

| # | config | stochastic | why |
|---|---|---|---|
| B0 | `cv-job`: exactly `fv.job` (yolo11m@640, ByteTrack → BoT-SORT/OSNet, kmeans teams, `identity.fuse` CV) | no (×1) | the shipped product. Anchor row. |
| B1 | `run11`: B0 + `roster_v3` neutral VLM grouping (gemini-3.1-pro-preview) | yes (×3) | best published identity method; **the identity control** |
| B2 | `run13`: B1 + `gk_lock` pre-pass | yes (×3) | the latest experiment, scored against B1 in the same batch |
| B3 | detector: yolo11m@640 vs **yolo11x@640** vs **yolo11x@1280** (rest = B0) | no | at 40 px players, input resolution probably matters more than model size; this tests both |
| B4 | tracker: **ByteTrack vs BoT-SORT(+OSNet)** as the only tracker, no identity fusion (rest = B0) | no | isolates the tracker's own contribution to the T4 metrics |

Rough cost:
- Deterministic rows are about 1–2 GPU-min per clip, so about 1 h for the whole batch over
  6 eval clips.
- B1/B2 run 6 clips × 3 repeats × up to about 120 VLM calls. That is about 4.3k Gemini 3.1 Pro
  calls, an estimated **$30–60**. The exact rate will be measured on the first clip.
- GPU jobs run one at a time and only when `nvidia-smi` shows ORCH-fv-pipeline is idle.

Note: the pre-labeller ensemble (yolo11x@1280 ∪ RT-DETR-x) overlaps B3's yolo11x@1280. The blind
keyframes (§3) quantify how much that row is flattered. If needed, the pre-label boxes switch to
RT-DETR-x + SAM2 only.

---

## 7. Data handling and licence

- Pixels (clips, frames, crops) and all parquet stay in `~/footyviz-bench-data/` on zulzi, which
  is private. **Git gets**:
  - the manifest: video id, title, window, tier, teams, quality stats, the expected frame count,
    and a 64-bit perceptual hash of 3 frames, so a re-fetched clip can be checked for frame
    alignment;
  - labels (JSON);
  - code and docs.
- **Data layout (on the box, outside git):**
  - `raw/<tier>_<videoid>.mp4` holds the full matches.
  - `clips/<clip_id>.mp4` holds the windows cut from them.
  - `frames/`, `prelabels/` and `results/` hold everything derived.
- **Fetch script**: `python -m fv.bench fetch --manifest bench/manifest.json [--cookies …]`
  works in two steps.
  1. Get each **full match** with yt-dlp (1080p plus audio), or skip that if
     `raw/<tier>_<id>.mp4` already exists, as on zulzi where the control plane relays them.
  2. Cut every window in the manifest with a frame-accurate re-encode (libx264 crf 12, the
     recipe used for A1), and verify each window's frame count and pHashes.

  If the YouTube encode has changed, it fails loudly. The manifest also stores the full match's
  duration and frame count, so a truncated relay is caught.
- **Licence note** in the manifest:
  - Footage © the uploader (Rookie / Veo) and is **not redistributed**.
  - Labels are CC-BY-4.0, code is MIT.
  - Labels hold no names and no pixels. People are anonymous P-ids.
  - A takedown request removes a clip from the manifest and its labels.
- The Gemini key is read from `~/football-vision/.env` by path at runtime. It is never copied
  or logged.

---

## 8. Decisions for Daniel

1. **Approve the metric set**, in particular:
   - keyframe person-ID labels (all-pairs P/R + keyframe IDF1) *instead of* sampled tracklet
     pairs;
   - off-pitch people as `ignore` for detection but scored as false participants in T3.
2. **Approve the time budget**: about 3.2 h of cheap labels now, about 2.7 h dense later.
   Alternatively, start with D1 + A1 only (about 70 min) to shake out the page.
3. **Team-disjoint splits**: this forces both D clips into eval and needs a **D3 replacement**
   relayed (candidates in §1).
4. **VLM budget** for the B1/B2 repeats (est. $30–60), and N = 3 repeats as the default.
5. Is **A0** (the run 1–13 slice) worth labelling, at about 18 min?
6. Full matches: is **one labelled 120 s window per eval match** (default 900 s, shift only for
   stoppages, plus a second window if the lighting changes) the right v1? And should whole-match
   runs be on the board as an unlabelled stability and throughput check?
