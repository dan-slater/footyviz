#!/usr/bin/env python3
"""
football-vision EXPERIMENT TRACKER — a tabbed, self-contained HTML lab notebook.
One tab per experiment run (setup / result / notes), plus an Overview tab.
Runs on zulzi-gpu (has data + imageio-ffmpeg). Regenerate after each new run:
add an entry to VARIANTS and re-run -> pull dashboard_experiments.html -> redeploy Artifact.
"""
import base64, glob, os, subprocess, sys, io, datetime
import pandas as pd, imageio_ffmpeg
from PIL import Image

ROOT = os.path.expanduser("~/football-vision")
OUT  = os.path.join(ROOT, "data/out"); TRK = os.path.join(ROOT, "data/tracks")
CACHE = os.path.join(OUT, "_dash_cache"); os.makedirs(CACHE, exist_ok=True)
FF = imageio_ffmpeg.get_ffmpeg_exe()
TRUE_LO, TRUE_HI = 14, 16
vid_bytes = 0
# Full-res served build (references H.264 renders by filename, for a cloudflared-tunnel
# server) is now the DEFAULT and only maintained doc. Pass --inline to build the legacy
# crushed-300p self-contained Artifact (deprecated; base64-inlined to fit the <16MB cap).
SERVED = "--inline" not in sys.argv
SERVEDIR = os.path.join(OUT, "_served"); SERVE_FILES = set()
if SERVED: os.makedirs(SERVEDIR, exist_ok=True)

def b64_img(path, maxw=820, q=72):
    if not path or not os.path.exists(path): return None
    try:
        im = Image.open(path).convert("RGB")
        if im.width > maxw: im = im.resize((maxw, round(im.height*maxw/im.width)), Image.LANCZOS)
        buf = io.BytesIO(); im.save(buf, "JPEG", quality=q, optimize=True); data = buf.getvalue()
    except Exception:
        data = open(path, "rb").read()
    return "data:image/jpeg;base64," + base64.b64encode(data).decode()

def compress_video(src, tag, height=300, crf=34, fps=10):
    global vid_bytes
    if not os.path.exists(src): return None
    if SERVED:
        # full-res: reference the real render by filename; server hosts _served/
        bn = os.path.basename(src); SERVE_FILES.add(src)
        vid_bytes += os.path.getsize(src)
        return f"./{bn}"
    dst = os.path.join(CACHE, f"{tag}_h{height}c{crf}f{fps}.mp4")
    if not os.path.exists(dst) or os.path.getmtime(dst) < os.path.getmtime(src):
        subprocess.run([FF,"-y","-i",src,"-vf",f"scale=-2:{height},fps={fps}","-c:v","libx264",
            "-crf",str(crf),"-preset","veryfast","-pix_fmt","yuv420p","-an","-movflags","+faststart",dst],
            capture_output=True)
    if not os.path.exists(dst): return None
    vid_bytes += os.path.getsize(dst)
    return "data:video/mp4;base64," + base64.b64encode(open(dst,"rb").read()).decode()

def count_from_parquet(name):
    p = os.path.join(TRK, name)
    if not os.path.exists(p): return None
    df = pd.read_parquet(p)
    return len(df) if name.endswith("_table.parquet") else int(df["stable_id"].nunique())

def frames(prefix, nums, n=2):
    out=[]
    for x in nums:
        p=os.path.join(OUT, f"{prefix}{x}.jpg")
        if os.path.exists(p): out.append(p)
    return out[:n]

# run#, label, tag, count_parquet, video, frame_prefix, frame_nums, [montages], setup(what changed), family
VARIANTS = [
 (1,"Baseline · CV-only ReID","v_base","identities_night_table.parquet","ident_night_h264.mp4",
  "ident_frame",["0200","1100"],[], "OSNet + BoT-SORT appearance ReID only; no VLM.","cv"),
 (2,"VLM cluster","v_vlmc","identities_vlmcluster_night_table.parquet","ident_vlmcluster_night_h264.mp4",
  "ident_vlmcluster_frame",["0200","1100"],[], "Vision-LLM clustering of raw tracklets by appearance.","vlm"),
 (3,"VLM primary","v_vlmp","identities_vlmprimary_night_table.parquet","ident_vlmprimary_night_h264.mp4",
  "ident_vlmprimary_frame",["0200","1100"],
  ["ident_vlmprimary_montage_A.jpg","ident_vlmprimary_montage_B.jpg","ident_vlmprimary_montage_O.jpg"],
  "VLM primary grouping. Peak over-split — worst over-count of the run.","vlm"),
 (4,"Team-bucketed","v_buck","identities_bucket_night_table.parquet","ident_bucket_night_h264.mp4",
  "ident_bucket_frame",["0200","1100"],[], "Grouping constrained within team buckets (A/B/O).","vlm"),
 (5,"+ Participant filter","v_filt","identities_vlmprimary_players_table.parquet","ident_filtered_night_h264.mp4",
  "ident_filtered_frame",["0200","1100"],[], "Drop referees / spectators / non-players before grouping. Big drop toward truth.","filter"),
 (6,"1080p + reprompt","v_1080","identities_1080_players_table.parquet","ident_1080_night_h264.mp4",
  "ident_1080_frame",["0200","1100"],[], "Sharper 1080p crops + reworked prompt.","res"),
 (7,"Holistic roster dedup","v_rost","identities_roster_night_table.parquet","ident_roster_fixed_fifa_h264.mp4",
  "ident_roster_fixed_frame",["0200","1100"],[], "Whole-team roster grid handed to the VLM at once (vs pairwise).","vlm"),
 (8,"+ Super-res","v_sr","identities_sr_night_table.parquet","ident_sr_fifa_h264.mp4",
  "ident_sr_frame",["0200","1100"],[], "Real-ESRGAN (spandrel) super-res on crops.","res"),
 (9,"Super-res v2","v_sr2","identities_sr2_night_table.parquet","ident_sr2_fifa_h264.mp4",
  "ident_sr2_frame",["0200","1100"],[], "Super-res tuning pass.","res"),
 (10,"Grid dedup + number anchor","v_dedup","identities_dedup_night_table.parquet","ident_dedup_fifa_h264.mp4",
  "ident_dedup_frame",["0200","1100"],[], "Fix: only confident jersey numbers anchor a split; A#1+A#11 (partial '11') now merge.","fix"),
 (11,"Neutral + context + veto","v_neut","identities_neutral_night.parquet","ident_neutral_fifa_h264.mp4",
  "ident_neutral_frame",["0200","1100"],
  ["roster_montage_A.jpg","roster_montage_B.jpg","roster_montage_O.jpg"],
  "LATEST. Neutral prompt + full-frame context crops + co-occurrence veto (two tracklets >100px apart on-screen can't be one player).","latest"),
]

# Upcoming experiments — rendered as "planned" tabs (hypothesis / method / expected), no data yet.
# run#, label, status, hypothesis, method, expected
PLANNED = [
 (12,"Daytime slice","NEXT",
  "The ~23 night ceiling is a <b>footage limit, not a method flaw</b>. Every identity run so far was measured on frames 0–1800 — the single <b>darkest, grainiest 60&nbsp;s</b> of the match (dusk, floodlit, bloomed, ~35&nbsp;px figures). Clear golden-hour daylight footage exists later in the video and has never been tested.",
  "Re-run the full identity pipeline (reid → teams → jersey → participant → roster_v3) with <b>identical run-11 settings</b> on a clear-daylight window (frames ~20000–21800, 60&nbsp;s — matched to the night-slice length so <b>lighting is the only changed variable</b>). Reuses the existing full-match detections; no re-detect.",
  "≈14–17 stable IDs ⇒ the night ~23 is a <b>footage ceiling</b> and the methodology is validated. Still ≈23 ⇒ a <b>systematic method over-count</b> — the real bug to chase. Either outcome is decisive."),
 (13,"GK-role pre-pass","QUEUED",
  "The goalkeeper (track 14) tangles into a <b>duplicate A-4 / keeper row</b> because grouping runs before the keeper's role is locked, so the GK gets merged/split as if an outfield grey-hoodie player.",
  "Run a <b>GK-role lock as a pre-pass</b>: identify the keeper by standing-in-goal geometry via full-frame context crops, <b>exclude that track from the outfield grey-hoodie merges</b>, then run the neutral grouping. Night slice; reuses existing data (cheap).",
  "Removes the duplicate keeper identity → count down ~1 and a <b>cleaner, correct roster</b>. A quality fix, judged on roster correctness, not just the raw count."),
]

# ── Algorithm model ──────────────────────────────────────────────────────────
# Canonical identity pipeline (superset of all runs). Each run's tab renders this
# same diagram with the stages IT introduces/changes highlighted (green), stages
# it inherits shown normal, and stages it doesn't use dimmed — so the method
# evolution is diffable tab-to-tab. Facts grounded in src/fv/*.py (real params).
STAGES = [
 ("01","Acquire","yt-dlp 720p/1080p"),
 ("02","Detect","YOLO11m · conf 0.30"),
 ("03","Track","BoT-SORT + OSNet ReID"),
 ("04","Team gate","KMeans-HSV → A/B/O"),
 ("05","Crops","sharpest · 1080p/SR"),
 ("06","Jersey OCR","easyocr ×4 · confident"),
 ("07","Participant","geometry + VLM"),
 ("08","Descriptor","Lab bands + OSNet"),
 ("09","Grouping","cluster / VLM engine"),
 ("10","Constraints","gates + vetoes"),
 ("11","Roster","→ FIFA overlay"),
]
# per run: new = stages this run introduces/changes (green); off = stages unused (dim).
# everything else renders as inherited/on. prose = detailed method; params = chips.
ALGO = {
 1:{"new":{"08","09","10"},"off":{"06","07"},
    "prose":"Tracklets from <b>BoT-SORT + OSNet</b> ReID are collapsed offline. Each track's descriptor = the <b>OSNet embedding</b> (osnet_x1_0_msmt17, L2-normalised median over its crops) fused with a <b>part-based Lab colour fingerprint</b> — median Lab of horizontal bands (hair·shirt·shorts·socks·shoes·skin) after masking pitch-green &amp; shadow, socks/shorts weighted highest, shirt down-weighted (≈ team) — plus build (bbox aspect). Pair distance <b>d = 0.50·(1−cos emb) + 0.45·band ΔE + 0.05·aspect</b>. <b>Constrained average-linkage agglomerative</b> merge while linkage &lt; 0.42, with hard cannot-links for <b>different team</b> and <b>temporal co-occurrence</b>. No VLM — over-merges distinct players.",
    "params":["OSNet x1_0 msmt17","threshold 0.42","w 0.50/0.45/0.05","avg-linkage"]},
 2:{"new":{"08","09"},"off":{"06","07"},
    "prose":"Adds a <b>vision-LLM appearance layer</b>: one <code>describe_player</code> call per track (on its 2–3 clearest crops) returns attribute tokens (hair·socks·shoes·build·shorts·skin·facial-hair·shirt). Tracklets are clustered by <b>weighted token-Jaccard distance</b> over those attributes, VLM-led. Same team + temporal cannot-links. VLM attributes are richer but noisier across a track → over-splits.",
    "params":["gemini-3.1-pro","token-Jaccard attrs","VLM-led"]},
 3:{"new":{"09"},"off":{"06","07"},
    "prose":"<b>VLM-primary</b> — the vision-LLM does the grouping directly (team montage + per-track hints), and the OSNet embedding is demoted to a <b>singleton-rescue tiebreak</b> (a leftover track joins a group only if its embedding cosine-distance to the centroid ≤ 0.15). Hard temporal/number splits still apply. Maximal reliance on VLM judgement on ~35 px night crops <b>fragments</b> identities the worst of the whole run.",
    "params":["VLM leads grouping","OSNet tiebreak cos≤0.15"]},
 4:{"new":{"04","09"},"off":{"06","07"},
    "prose":"Fingerprints are <b>discretised into buckets</b> (colour-canon, hair length, beard, skin, sleeve, build) and each team is clustered independently on a <b>weighted bucket-Hamming distance</b> (socks/shorts weighted highest, shirt lowest; threshold 0.30) — plus a <code>min_frames</code> noise-track drop. Same team/temporal/number hard gates.",
    "params":["discrete buckets","bucket-Hamming 0.30","per-team"]},
 5:{"new":{"07"},"off":{"06"},
    "prose":"Introduces the <b>participant filter</b>: geometry priors (feet-on-green fraction ≥ 0.5, centroid motion &gt; 150 px) are handed to the VLM, which labels each track <b>player / goalkeeper / referee / spectator</b> — with a code override that flips any 'spectator' who is on the grass AND moving back to player. Non-players are dropped before grouping; the GK is kept as its own identity. The biggest single drop toward truth.",
    "params":["geometry-primed VLM","feet-on-green + motion","non-players dropped"]},
 6:{"new":{"05","09"},"off":{"06"},
    "prose":"Crops are now sampled from the <b>1080p</b> source (vs 720p) and the grouping prompt is reworked. Sharper evidence per track, but the count drifts back up.",
    "params":["1080p crops","reworked prompt"]},
 7:{"new":{"09","10"},"off":{"06"},
    "prose":"<b>Holistic roster grouping.</b> Instead of pairwise compares, each track becomes a <b>multi-view grid</b> (sharpest crop per temporal bin, captioned with its id) and the VLM is handed <b>all of a team's grids at once</b> with the prior '~7–8 players' and jersey numbers as hard anchors, returning groups. A <b>pairwise-verify</b> pass then merges the two most-similar groups while a team still has &gt; ~8.",
    "params":["multi-view grids","whole-team review","pairwise verify"]},
 8:{"new":{"05"},"off":{"06"},
    "prose":"Adds real <b>super-resolution</b> — Real-ESRGAN ×4 (RealESRGAN_x4plus loaded via <b>spandrel</b>, not basicsr) on crops before the VLM sees them, to test whether image clarity was the limiter.",
    "params":["Real-ESRGAN ×4","spandrel loader"]},
 9:{"new":{"05"},"off":{"06"},
    "prose":"Super-res <b>tuning</b> pass (crop selection + upscale settings). Count holds — confirming clarity is no longer the bottleneck; the ceiling is the footage.",
    "params":["SR tuning"]},
 10:{"new":{"06","10"},
    "prose":"Fixes the <b>jersey-number anchor</b> bug: only <code>confident=True</code> OCR numbers anchor a split, and single digits are treated as <b>prefix-compatible</b> with multi-digit reads (a partial '1' no longer hard-splits '11'). So A#1 + A#11 now correctly merge into one player.",
    "params":["confident-only anchor","prefix-compatible digits"]},
 11:{"new":{"09","10"},
    "prose":"<b>Neutral prompt + full-frame context + co-occurrence veto</b> (LATEST). Prompts are stripped of leading hypotheses (grey-hoodie people ARE players on a cold night); each track is shown as <b>2–3 full 1080p frames with the player boxed</b> so the VLM reasons about position/role, not a tight crop alone. A precomputed <b>co-occurrence veto</b> hard-blocks merging any two tracklets that are ever on-screen together and &gt; 100 px apart (median). A correct 24 beats an over-merged 19.",
    "params":["neutral prompts","full-frame context crops","co-occur veto >100px","emb_gate 0.55"]},
 12:{"new":{"01"},
    "prose":"<b>Identical to run 11's algorithm</b> — the only variable changed is the <b>input footage</b>. Re-run the whole pipeline on a clear-daylight 60 s slice (frames ~20000–21800) instead of the dark floodlit opening (frames 0–1800). This controls for method and isolates lighting/clarity: ≈14–17 ⇒ footage ceiling (thesis confirmed); still ≈23 ⇒ systematic method over-count.",
    "params":["run-11 pipeline","daytime frames 20000–21800","lighting = only variable"]},
 13:{"new":{"07","09"},
    "prose":"<b>GK-role pre-pass</b> before grouping. Lock the keeper by <b>standing-in-goal geometry</b> (decided from full-frame context crops), <b>exclude that track from the outfield grey-hoodie merges</b>, then run the neutral grouping — so the goalkeeper (track 14) can no longer tangle into a duplicate A-4 outfield row.",
    "params":["GK geometry lock","exclude GK from outfield merges","pre-pass"]},
}

# ── Glossary ─────────────────────────────────────────────────────────────────
# Plain-English definitions for the CV/ML jargon used in the tabs. Under every
# block of text, a "jargon in this section" box auto-lists the terms that block
# uses (de-duped per tab — a term is defined the first time it appears on a tab).
# Each entry: canonical term -> (definition, [detection aliases]).
import re as _re
GLOSSARY = {
 "tracklet":("A single unbroken track of one detected person — one player usually breaks into many tracklets as they leave/re-enter frame.",["tracklet"]),
 "ReID":("Re-identification — matching the same person across separate appearances using an appearance model, not just motion.",["reid","re-id","re-identification","re-identify"]),
 "OSNet":("A convolutional neural network that turns a person crop into an appearance 'embedding' vector; here trained on the MSMT17 person-ReID dataset.",["osnet"]),
 "BoT-SORT":("A multi-object tracker that combines motion prediction with an appearance model (OSNet) and camera-motion compensation to keep IDs stable.",["bot-sort","botsort"]),
 "ByteTrack":("A fast motion-only multi-object tracker (no appearance model) — cheap but churns IDs when players cross or leave frame.",["bytetrack"]),
 "YOLO":("A real-time object detector; 'YOLO11m' is the medium variant used here to find people frame-by-frame.",["yolo"]),
 "embedding":("A fixed-length vector summarising a crop's appearance; similar-looking crops have vectors that are close together.",["embedding","embed "]),
 "cosine distance":("A measure of how different two embedding vectors are by the angle between them (0 = identical direction, 1 = unrelated).",["cosine","cos emb","1−cos","1-cos","cos≤","cos ≤"]),
 "L2-normalise":("Scaling a vector to unit length so only its direction matters, making cosine comparisons fair.",["l2-normal","l2 normal","l2-norm"]),
 "agglomerative clustering":("A bottom-up grouping method: start with every item alone, then repeatedly merge the closest two groups.",["agglomerative"]),
 "average linkage":("A rule for agglomerative clustering — the distance between two groups is the average of all cross-pair distances.",["average-linkage","avg-linkage","average linkage"]),
 "KMeans":("A clustering algorithm that partitions points into k groups by nearest centre; used here to separate kit colours into teams.",["kmeans"]),
 "silhouette score":("A score (−1..1) rating how cleanly points fit their cluster; used to pick whether there are 3 or 4 kit-colour groups.",["silhouette"]),
 "HSV":("A colour space (hue, saturation, value) that separates colour from brightness — handy for masking green pitch and shadow.",["hsv"]),
 "Lab colour":("A colour space (lightness + two colour axes) built so numeric distance ≈ perceived colour difference.",["lab colour","lab color","median lab","band lab","lab)"]),
 "ΔE (deltaE)":("The straight-line distance between two colours in Lab space — a perceptual 'how different are these colours' number.",["δe","deltae","lab-euclidean","lab euclidean","band δe"]),
 "Jaccard":("A similarity score between two sets = shared items ÷ total distinct items; here over descriptive word-tokens.",["jaccard"]),
 "Hamming distance":("A count of how many discrete attributes two items differ on (weighted here by attribute importance).",["hamming"]),
 "bounding box":("The rectangle a detector draws around a person; 'bbox aspect' is its height÷width (build/pose proxy).",["bbox","bounding box"]),
 "aspect ratio":("Box height divided by width — a rough proxy for a player's build and stance.",["aspect"]),
 "centroid":("The average position (or average vector) of a set — e.g. a group's mean embedding, or a box's centre point.",["centroid"]),
 "VLM":("Vision-language model — an LLM that accepts images and answers questions about them (here Gemini, for appearance/role judgements).",["vlm","vision-llm","vision llm","vision-language"]),
 "Gemini":("Google's vision-language model; the specific VLM queried for player descriptions, roles and same/different judgements.",["gemini"]),
 "OCR":("Optical character recognition — reading text (here jersey numbers) out of an image; via the easyocr library.",["easyocr","ocr"]),
 "CLAHE":("Contrast-Limited Adaptive Histogram Equalisation — a local contrast boost that helps OCR read faint digits.",["clahe"]),
 "Otsu":("An automatic threshold that splits an image into black/white by finding the best cut in its brightness histogram.",["otsu"]),
 "Laplacian variance":("A sharpness score — the variance of edge response; higher = crisper crop, used to pick the clearest frames.",["laplacian"]),
 "super-resolution":("Upscaling a small crop to more pixels while inventing plausible detail; here Real-ESRGAN, a neural upscaler.",["super-res","super res","super-resolution","real-esrgan","esrgan"]),
 "spandrel":("A Python library that loads pretrained image-model weights cleanly (used instead of the broken basicsr path).",["spandrel"]),
 "cannot-link":("A hard rule forbidding two tracklets from ever being merged (e.g. different team, or seen at once).",["cannot-link","cannot link","cannot-merge"]),
 "co-occurrence":("Two tracklets appearing in the same frame — proof they're different people (one player can't be in two places).",["co-occurrence","co-occur","cooccur","coexist","co-present","co present"]),
 "veto":("A hard code-level block that overrides the model — e.g. refusing a merge the geometry proves impossible.",["veto"]),
 "hard gate":("A non-negotiable pre-filter on merges (team / time / number) applied before any similarity scoring.",["hard gate","team gate","hard-gate","hard cannot-link"]),
 "prior":("A built-in expectation fed to the model — e.g. 'a team has ~7–8 players' — to steer its grouping.",["prior"]),
 "prefix-compatible":("Treating a partial number read as agreeing with a longer one — '1' is compatible with '11' (OCR saw half).",["prefix-compat","prefix compat","prefix-compatible","prefix"]),
 "singleton":("A group of one — a lone tracklet that no merge absorbed; sometimes rescued by an embedding tiebreak.",["singleton"]),
 "participant filter":("A stage that labels each track player / goalkeeper / referee / spectator and drops non-players.",["participant filter","participant"]),
 "descriptor / fingerprint":("The compact appearance summary of a track (colour bands + embedding, or VLM attribute words) used to compare tracks.",["descriptor","fingerprint"]),
 "multi-view grid":("A tile of several crops of one tracklet across its life, so the VLM sees varied angles/poses at once.",["multi-view","montage","grid image","captioned grid"]),
}
_GORDER = sorted(GLOSSARY, key=lambda t: -max(len(a) for a in GLOSSARY[t][1]))
def _has(alias, text_l):
    return _re.search(r'(?<![a-z0-9])'+_re.escape(alias.lower())+r'(?![a-z0-9])', text_l) is not None
def jargon_box(html_text, seen):
    plain_l = _re.sub(r'<[^>]+>', ' ', html_text).lower()
    found = []
    for term in _GORDER:
        if term in seen: continue
        if any(_has(a, plain_l) for a in GLOSSARY[term][1]):
            found.append(term); seen.add(term)
    if not found: return ""
    items = "".join(f'<div class="jg"><dt>{t}</dt><dd>{GLOSSARY[t][0]}</dd></div>' for t in found)
    return f'<div class="jargon"><span class="k">JARGON IN THIS SECTION</span><dl>{items}</dl></div>'

def pipe_html(run, seen=None):
    a = ALGO.get(run)
    if not a: return ""
    new, off = a.get("new",set()), a.get("off",set())
    nodes=""
    for sid,lab,det in STAGES:
        st = "new" if sid in new else ("off" if sid in off else "on")
        nodes += (f'<div class="st {st}"><span class="sti">{sid}</span>'
                  f'<span class="stl">{lab}</span><span class="std">{det}</span></div>')
    chips="".join(f'<span class="chip">{c}</span>' for c in a.get("params",[]))
    jarg = jargon_box(a["prose"], seen) if seen is not None else ""
    return (f'<div class="algo"><span class="k">ALGORITHM · PIPELINE</span>'
            f'<div class="pipe">{nodes}</div>'
            f'<div class="legend"><span class="lg new">introduced / changed this run</span>'
            f'<span class="lg on">inherited</span><span class="lg off">not used</span></div>'
            f'<p class="algotext">{a["prose"]}</p>'
            f'{f"""<div class="chips">{chips}</div>""" if chips else ""}'
            f'{jarg}</div>')

def tone(cnt):
    if cnt is None: return "na"
    d = cnt - TRUE_HI
    return "good" if d<=6 else ("warn" if d<=9 else "hot")

print("Building experiment tabs (reusing video cache)...")
tabs, panels, rows = [], [], []
counts = []
for run,label,tag,cq,vid,fpre,fnums,montages,setup,family in VARIANTS:
    cnt = count_from_parquet(cq); counts.append(cnt or 0)
    vuri = compress_video(os.path.join(OUT,vid), tag)
    d = (cnt-TRUE_HI) if cnt else None
    latest = family=="latest"
    print(f"  run{run} {label:32s} count={cnt}")
    tabs.append(f'<button class="tab {tone(cnt)}" data-p="p{run}">'
                f'<span class="tn">{run:02d}</span>{label.split("·")[0].strip() if "·" in label else label}'
                f'<span class="tc">{cnt if cnt is not None else "—"}</span></button>')
    # per-run figure IDs — F<run>.<n>, assigned in panel order (video, frames, montages).
    _fig = [0]
    def fid():
        _fig[0]+=1; return f"F{run}.{_fig[0]}"
    vid_id = fid()
    if vuri:
        vhtml = (f'<figure class="fig"><span class="figid">{vid_id}</span>'
                 f'<video class="clip" controls preload="none" playsinline src="{vuri}"></video>'
                 f'<figcaption>{vid_id} · overlay render — {label}</figcaption></figure>')
    else:
        vhtml = (f'<figure class="fig"><span class="figid">{vid_id}</span>'
                 f'<div class="clip miss">no render</div><figcaption>{vid_id} · overlay render (n/a)</figcaption></figure>')
    fr = ""
    for p in frames(fpre,fnums):
        tok = os.path.basename(p)[len(fpre):].replace(".jpg","")
        i = fid()
        fr += (f'<figure class="fig shot"><span class="figid">{i}</span>'
               f'<img class="zoom" loading="lazy" src="{b64_img(p,760,70)}">'
               f'<figcaption>{i} · frame {tok}</figcaption></figure>')
    mg = ""
    for m in montages:
        u=b64_img(os.path.join(OUT,m),1000,80)
        if u:
            t=m.replace(".jpg","").split("_")[-1]
            i=fid()
            mg += (f'<figure class="fig mon"><span class="figid">{i}</span>'
                   f'<img class="zoom" loading="lazy" src="{u}"><figcaption>{i} · roster montage — team {t}</figcaption></figure>')
    seen = set()
    setup_jarg = jargon_box(setup, seen)
    panels.append(f'''<section class="panel{' active' if run==11 else ''}" id="p{run}">
      <div class="phead"><div><span class="run">RUN {run:02d}{' · LATEST' if latest else ''}</span><h2>{label}</h2></div>
        <div class="bignum {tone(cnt)}"><span>{cnt if cnt is not None else '—'}</span><em>stable IDs · truth {TRUE_LO}–{TRUE_HI}{f' · +{d}' if d is not None else ''}</em></div></div>
      <div class="setup"><span class="k">SETUP / WHAT CHANGED</span><p>{setup}</p></div>
      {setup_jarg}
      {pipe_html(run, seen)}
      <div class="cols"><div class="vwrap">{vhtml}<div class="shots">{fr}</div></div>
        {f'<div class="mons">{mg}</div>' if mg else ''}</div>
    </section>''')
    rows.append((run,label,cnt,d,setup))

# planned/upcoming experiments — distinct "planned" tabs + panels (no data yet)
for run,label,status,hyp,method,expected in PLANNED:
    print(f"  run{run} {label:32s} [PLANNED · {status}]")
    tabs.append(f'<button class="tab plan" data-p="p{run}">'
                f'<span class="tn">{run:02d}</span>{label}'
                f'<span class="tc plan">{status}</span></button>')
    seen = set()
    hyp_jarg = jargon_box(hyp, seen); method_jarg = jargon_box(method, seen)
    panels.append(f'''<section class="panel planned" id="p{run}">
      <div class="phead"><div><span class="run plan">RUN {run:02d} · PLANNED · {status}</span><h2>{label}</h2></div>
        <div class="bignum na"><span>◇</span><em>not yet run · truth {TRUE_LO}–{TRUE_HI}</em></div></div>
      <div class="setup"><span class="k">HYPOTHESIS</span><p>{hyp}</p></div>
      {hyp_jarg}
      <div class="setup"><span class="k">METHOD</span><p>{method}</p></div>
      {method_jarg}
      {pipe_html(run, seen)}
      <div class="setup exp"><span class="k">EXPECTED OUTCOME</span><p>{expected}</p></div>
      {jargon_box(expected, seen)}
    </section>''')

# overview: progression bar chart (CSS) + table
maxc = max([c for c in counts if c]+[1])
bars=""
for run,label,cnt,d,setup in rows:
    if not cnt: continue
    bars+=f'''<div class="brow"><div class="bl">{run:02d} {label.split("·")[0].strip() if "·" in label else label}</div>
      <div class="bar"><span class="bf {tone(cnt)}" style="width:{int(100*cnt/maxc)}%"></span>
      <span class="btruth" style="left:{int(100*TRUE_HI/maxc)}%"></span></div><div class="bn">{cnt}</div></div>'''
trows="".join(f'<tr><td class="mono">{run:02d}</td><td>{label}</td><td class="mono num {tone(cnt)}">{cnt if cnt else "—"}</td>'
              f'<td class="mono num">{("+"+str(d)) if d is not None else "—"}</td><td>{setup}</td></tr>' for run,label,cnt,d,setup in rows)
trows+="".join(f'<tr class="planrow"><td class="mono">{run:02d}</td><td>{label} <span class="mono" style="color:var(--dim)">· planned</span></td>'
              f'<td class="mono num">{status}</td><td class="mono num">—</td><td>{hyp}</td></tr>'
              for run,label,status,hyp,method,expected in PLANNED)
best=min([c for c in counts if c]); ts=datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

HTML = r"""<style>
:root{--bg:#0A0D0B;--card:#12161A;--card2:#171c20;--line:#232a2d;--line2:#2e3739;--ink:#DBE2E1;--mut:#8A94A0;--dim:#616b70;
--pitch:#5FB878;--pitchdim:#3c6f4c;--good:#5FB878;--warn:#E0B23C;--hot:#E0663C;--A:#EEF2EC;--B:#F5679E;--O:#4EA8E0;
--mono:"SF Mono","JetBrains Mono","Fira Code",ui-monospace,Menlo,monospace;--sans:"SF Pro Text",-apple-system,"Segoe UI",Roboto,system-ui,sans-serif}
*{box-sizing:border-box}
.ex{background:radial-gradient(1100px 500px at 80% -10%,#10201540,transparent 60%),var(--bg);color:var(--ink);font-family:var(--sans);min-height:100vh;font-size:15px;padding-bottom:60px}
.ex ::selection{background:var(--pitch);color:#06110a}
.wrap{max-width:1080px;margin:0 auto;padding:0 20px}
img{max-width:100%;display:block}
.top{border-bottom:1px solid var(--line);padding:26px 0 0}
.eyebrow{font-family:var(--mono);font-size:11px;letter-spacing:.22em;text-transform:uppercase;color:var(--pitch)}
h1{font-family:var(--mono);font-weight:600;font-size:26px;letter-spacing:-.02em;margin:10px 0 4px}
.sub{color:var(--mut);font-size:13.5px;margin-bottom:18px}
.tabs{display:flex;gap:6px;overflow-x:auto;padding-bottom:0;scrollbar-width:thin}
.tab{flex:0 0 auto;background:var(--card);border:1px solid var(--line);border-bottom:none;color:var(--mut);
  font-family:var(--mono);font-size:12px;padding:9px 12px 8px;border-radius:8px 8px 0 0;cursor:pointer;display:flex;align-items:center;gap:7px;white-space:nowrap}
.tab .tn{color:var(--dim);font-size:10px}
.tab .tc{font-weight:600;font-variant-numeric:tabular-nums;padding:1px 6px;border-radius:10px;background:#0a0e0c;font-size:11px}
.tab.good .tc{color:var(--good)} .tab.warn .tc{color:var(--warn)} .tab.hot .tc{color:var(--hot)}
.tab.plan{border-style:dashed;color:var(--dim)}
.tab .tc.plan{background:transparent;border:1px dashed var(--line2);color:var(--mut);font-size:9px;letter-spacing:.1em}
.tab:hover{color:var(--ink)}
.tab.on{background:var(--card2);color:var(--ink);border-color:var(--line2);box-shadow:0 -2px 0 var(--pitch) inset}
.body{background:var(--card2);border:1px solid var(--line2);border-radius:0 10px 10px 10px;min-height:400px;padding:22px}
.panel{display:none} .panel.show{display:block}
.phead{display:flex;justify-content:space-between;align-items:flex-start;gap:16px;flex-wrap:wrap;border-bottom:1px solid var(--line);padding-bottom:14px;margin-bottom:16px}
.run{font-family:var(--mono);font-size:11px;letter-spacing:.12em;color:var(--pitch)}
.run.plan{color:var(--mut)}
.bignum.na span{color:var(--dim)}
.panel.planned .setup.exp{border-color:var(--pitchdim);background:#0c1512}
.panel.planned .setup.exp .k{color:var(--pitch)}
tr.planrow td{color:var(--mut)} tr.planrow .mono{color:var(--dim)}
.phead h2{font-family:var(--mono);font-weight:600;font-size:19px;margin:4px 0 0;letter-spacing:-.01em}
.bignum{text-align:right}
.bignum span{font-family:var(--mono);font-size:34px;font-weight:600;font-variant-numeric:tabular-nums;line-height:1}
.bignum.good span{color:var(--good)} .bignum.warn span{color:var(--warn)} .bignum.hot span{color:var(--hot)}
.bignum em{display:block;font-style:normal;color:var(--dim);font-size:11px;font-family:var(--mono);margin-top:4px}
.setup{background:#0d1210;border:1px solid var(--line);border-radius:10px;padding:12px 15px;margin-bottom:16px}
.setup .k{font-family:var(--mono);font-size:10px;letter-spacing:.14em;color:var(--pitch)}
.setup p{margin:6px 0 0;color:var(--ink);font-size:14px}
/* algorithm pipeline diagram */
.algo{background:#0b100e;border:1px solid var(--line2);border-radius:10px;padding:14px 15px;margin-bottom:16px}
.algo>.k{font-family:var(--mono);font-size:10px;letter-spacing:.14em;color:var(--pitch)}
.pipe{display:flex;flex-wrap:wrap;gap:7px 12px;margin:13px 0 11px}
.st{flex:0 0 auto;min-width:98px;max-width:150px;background:#0a0e0c;border:1px solid var(--line);border-radius:8px;padding:7px 9px;position:relative}
.st .sti{font-family:var(--mono);font-size:9px;color:var(--dim);display:block;letter-spacing:.08em}
.st .stl{font-family:var(--mono);font-size:12.5px;color:var(--ink);font-weight:600;display:block;margin-top:1px;letter-spacing:-.01em}
.st .std{font-size:10px;color:var(--mut);display:block;margin-top:3px;line-height:1.3}
.st.new{border-color:var(--pitch);background:linear-gradient(180deg,#122a1c,#0c1712);box-shadow:0 0 0 1px var(--pitch) inset,0 0 14px #5fb87833}
.st.new .stl{color:var(--pitch)} .st.new .sti{color:#8fd6a5}
.st.off{opacity:.34;border-style:dashed}
.pipe .st:not(:last-child)::after{content:"›";position:absolute;right:-9px;top:50%;transform:translateY(-50%);color:var(--dim);font-size:13px;font-family:var(--mono)}
.legend{display:flex;flex-wrap:wrap;gap:6px 16px;font-family:var(--mono);font-size:10px;color:var(--mut);margin:2px 0 4px}
.lg::before{content:"";display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;vertical-align:-1px}
.lg.new::before{background:var(--pitch);box-shadow:0 0 8px #5fb87866}
.lg.on::before{background:#0a0e0c;border:1px solid var(--line)}
.lg.off::before{background:transparent;border:1px dashed var(--line2);opacity:.6}
.algotext{margin:10px 0 0;font-size:13.5px;color:var(--ink);line-height:1.55}
.algotext code{font-family:var(--mono);font-size:12px;color:var(--pitch);background:#0a0e0c;padding:1px 4px;border-radius:4px}
.chips{display:flex;flex-wrap:wrap;gap:6px;margin-top:11px}
.chip{font-family:var(--mono);font-size:10px;color:var(--mut);background:#0a0e0c;border:1px solid var(--line);border-radius:20px;padding:3px 10px}
/* jargon glossary box — sits directly under the block whose terms it defines */
.jargon{background:#0a0f0d;border:1px solid var(--line);border-left:2px solid var(--pitchdim);
  border-radius:0 8px 8px 0;padding:9px 14px 10px;margin:-8px 0 16px}
.jargon .k{font-family:var(--mono);font-size:9px;letter-spacing:.16em;color:var(--pitchdim);text-transform:uppercase}
.jargon dl{margin:7px 0 0;display:grid;gap:5px}
.jg{display:grid;grid-template-columns:150px 1fr;gap:4px 14px;align-items:baseline}
.jg dt{font-family:var(--mono);font-size:11px;color:var(--pitch);font-weight:600}
.jg dd{margin:0;color:var(--mut);font-size:12px;line-height:1.45}
.algo .jargon{margin-top:12px;margin-bottom:0}
@media(max-width:640px){.jg{grid-template-columns:1fr}.jg dt{margin-bottom:1px}}
.cols{display:grid;grid-template-columns:1fr;gap:16px}
.vwrap .clip{width:100%;background:#000;border-radius:8px;aspect-ratio:16/9}
.clip.miss{display:flex;align-items:center;justify-content:center;color:var(--dim);font-family:var(--mono);font-size:12px}
.shots{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:8px}
.shots .shot img{aspect-ratio:16/9;object-fit:cover;border-radius:6px;border:1px solid var(--line);background:#000}
.mons{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}
.mon{margin:0}.mon img{border-radius:6px;border:1px solid var(--line);background:#000}
.zoom{cursor:zoom-in}
/* figure identifiers — cite as F<run>.<n> */
.fig{position:relative;margin:0}
.figid{position:absolute;top:6px;left:6px;z-index:3;font-family:var(--mono);font-size:10px;font-weight:600;
  letter-spacing:.05em;color:#06110a;background:var(--pitch);padding:2px 6px;border-radius:5px;box-shadow:0 1px 5px #0009;pointer-events:none}
.fig figcaption{font-family:var(--mono);font-size:10px;color:var(--dim);margin-top:4px;letter-spacing:.02em}
/* overview */
.ov h3{font-family:var(--mono);font-size:14px;text-transform:uppercase;letter-spacing:.06em;color:var(--mut);margin:0 0 14px}
.chart{background:#0d1210;border:1px solid var(--line);border-radius:10px;padding:16px 18px;margin-bottom:20px}
.brow{display:grid;grid-template-columns:210px 1fr 34px;gap:12px;align-items:center;padding:4px 0}
.bl{font-family:var(--mono);font-size:12px;color:var(--mut);text-align:right}
.bar{position:relative;height:15px;background:#0a0e0c;border:1px solid var(--line);border-radius:4px}
.bf{position:absolute;left:0;top:0;bottom:0;border-radius:3px 0 0 3px;opacity:.9}
.bf.good{background:linear-gradient(90deg,#3c6f4c,var(--good))}.bf.warn{background:linear-gradient(90deg,#7a6320,var(--warn))}.bf.hot{background:linear-gradient(90deg,#7a3320,var(--hot))}
.btruth{position:absolute;top:-3px;bottom:-3px;width:2px;background:var(--pitch);box-shadow:0 0 8px var(--pitch)}
.bn{font-family:var(--mono);font-variant-numeric:tabular-nums;font-size:13px;text-align:right}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{text-align:left;padding:7px 10px;border-bottom:1px solid var(--line);vertical-align:top}
th{font-family:var(--mono);font-size:10px;text-transform:uppercase;letter-spacing:.1em;color:var(--dim)}
.mono{font-family:var(--mono)}.num{text-align:right;font-variant-numeric:tabular-nums}
.num.good{color:var(--good)}.num.warn{color:var(--warn)}.num.hot{color:var(--hot)}
.howto{background:#0d1210;border:1px dashed var(--line2);border-radius:10px;padding:14px 16px;margin-top:20px;color:var(--mut);font-size:13px}
.howto code{font-family:var(--mono);color:var(--ink);background:#0a0e0c;padding:1px 5px;border-radius:4px}
#lb{position:fixed;inset:0;background:#000e;display:none;align-items:center;justify-content:center;z-index:99;padding:28px;cursor:zoom-out}
#lb.on{display:flex}#lb img{max-width:96vw;max-height:94vh;border-radius:6px}
.foot{color:var(--dim);font-size:11px;font-family:var(--mono);text-align:center;padding-top:22px}
@media(max-width:720px){.brow{grid-template-columns:120px 1fr 30px}.mons{grid-template-columns:1fr}}
</style>
<div class="ex"><div class="wrap">
 <div class="top">
   <div class="eyebrow">football-vision · player-identity experiments</div>
   <h1>Experiment Tracker</h1>
   <div class="sub">Night slice · one tab per run · stable-ID count vs true roster __LO__–__HI__ · best so far __BEST__. Every figure has a citable ID (<b>F&lt;run&gt;.&lt;n&gt;</b>, top-left). Click a tab; click any image to zoom.</div>
   <div class="tabs"><button class="tab on" data-p="pov"><span class="tn">＊</span>Overview</button>__TABS__</div>
 </div>
 <div class="body">
   <section class="panel show" id="pov"><div class="ov">
     <h3>Stable-ID count per run (green line = truth __HI__)</h3>
     <figure class="chart fig"><span class="figid">F0.1</span>__BARS__<figcaption>F0.1 · stable-ID count per run vs true roster</figcaption></figure>
     <h3>All runs</h3>
     <figure class="fig"><span class="figid" style="position:static;display:inline-block;margin-bottom:6px">F0.2</span>
     <table><thead><tr><th>#</th><th>Run</th><th class="num">IDs</th><th class="num">Δ</th><th>Setup / what changed</th></tr></thead><tbody>__TROWS__</tbody></table>
     <figcaption>F0.2 · all runs — counts, deltas, and what changed</figcaption></figure>
     <div class="howto"><b>Citing figures:</b> every figure has a stable ID — <code>F&lt;run&gt;.&lt;n&gt;</code> (e.g. <code>F11.1</code> = run 11's overlay video, <code>F11.2</code>/<code>F11.3</code> its frames, <code>F11.4</code>+ its montages); overview = <code>F0.1</code>/<code>F0.2</code>. IDs are per-run, so adding an experiment never renumbers older figures. <b>To add an experiment:</b> run the variant on zulzi → add a <code>VARIANTS</code> entry → re-run <code>build_exp_tracker.py</code> → refresh the served copy behind the tunnel. Non-monotonic on purpose: a <em>correct</em> 23 beats an over-merged 19.</div>
   </div></section>
   __PANELS__
 </div>
 <div class="foot">generated __TS__ · zulzi-gpu:~/football-vision · night slice frames 0–1800</div>
</div>
<div id="lb"><img alt="zoom"></div></div>
<script>
(function(){
 var tabs=[].slice.call(document.querySelectorAll('.tab')),panels=[].slice.call(document.querySelectorAll('.panel'));
 tabs.forEach(function(t){t.addEventListener('click',function(){
   tabs.forEach(function(x){x.classList.remove('on')});t.classList.add('on');
   var id=t.getAttribute('data-p');panels.forEach(function(p){p.classList.toggle('show',p.id===id)});
   document.querySelector('.body').scrollIntoView({block:'start',behavior:'smooth'});
 })});
 var lb=document.getElementById('lb'),lbi=lb.querySelector('img');
 document.querySelectorAll('.zoom').forEach(function(im){im.addEventListener('click',function(){lbi.src=im.currentSrc||im.src;lb.classList.add('on')})});
 lb.addEventListener('click',function(){lb.classList.remove('on');lbi.src=''});
 document.addEventListener('keydown',function(e){if(e.key==='Escape'){lb.classList.remove('on');lbi.src=''}});
})();
</script>"""
HTML = (HTML.replace("__TABS__","".join(tabs)).replace("__PANELS__","".join(panels))
        .replace("__BARS__",bars).replace("__TROWS__",trows)
        .replace("__LO__",str(TRUE_LO)).replace("__HI__",str(TRUE_HI)).replace("__BEST__",str(best)).replace("__TS__",ts))
# first real panel visible by default = overview; make sure no variant panel forced show
HTML = HTML.replace('class="panel active"','class="panel"')
if SERVED:
    import shutil
    # Standalone page (served raw by http.server, no claude.ai wrapper) → needs its own
    # <head> with an explicit UTF-8 charset, else the browser decodes ·→≥× as mojibake.
    DOC = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
           '<meta name="viewport" content="width=device-width,initial-scale=1">'
           '<title>football-vision · experiment tracker</title></head><body>'
           + HTML + '</body></html>')
    outp = os.path.join(SERVEDIR,"index.html"); open(outp,"w",encoding="utf-8").write(DOC)
    for src in SERVE_FILES:
        dst = os.path.join(SERVEDIR, os.path.basename(src))
        if not os.path.exists(dst) or os.path.getsize(dst)!=os.path.getsize(src):
            shutil.copy2(src, dst)
    print(f"\n[SERVED] Wrote {outp}  ({len(DOC.encode())/1e6:.1f} MB HTML) · {len(SERVE_FILES)} full-res videos, {vid_bytes/1e6:.1f} MB total")
    print(f"[SERVED] serve with:  cd {SERVEDIR} && python3 -m http.server 8137")
else:
    outp = os.path.join(OUT,"dashboard_experiments.html"); open(outp,"w",encoding="utf-8").write(HTML)
    print(f"\nWrote {outp}  ({len(HTML.encode())/1e6:.1f} MB HTML) · video payload {vid_bytes/1e6:.1f} MB")
