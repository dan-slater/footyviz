#!/usr/bin/env python3
"""
football-vision review dashboard generator.
Runs on zulzi-gpu (has the data + imageio-ffmpeg). Produces a single
self-contained HTML fragment (no <html>/<head>/<body>) with every artifact
inlined as base64 data URIs, ready to publish as a claude.ai Artifact.

Re-runnable: recompresses videos into a cache dir, recomputes ID counts from
parquet, and rebuilds the HTML. Add new variants to VARIANTS and re-run.
"""
import base64, glob, os, subprocess, sys, mimetypes, datetime, io
import pandas as pd
import imageio_ffmpeg
from PIL import Image

ROOT = os.path.expanduser("~/football-vision")
OUT  = os.path.join(ROOT, "data/out")
TRK  = os.path.join(ROOT, "data/tracks")
CACHE = os.path.join(OUT, "_dash_cache")
os.makedirs(CACHE, exist_ok=True)
FF = imageio_ffmpeg.get_ffmpeg_exe()

TRUE_LO, TRUE_HI = 14, 16   # true roster size (5-a-side ~5-8/team + keepers)

# ---------- helpers ----------
_img_bytes = {}   # path -> recompressed byte length (for accounting)
def b64_img(path, maxw=820, q=72):
    if not path or not os.path.exists(path): return None
    try:
        im = Image.open(path).convert("RGB")
        if im.width > maxw:
            im = im.resize((maxw, round(im.height*maxw/im.width)), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=q, optimize=True)
        data = buf.getvalue()
    except Exception as e:
        with open(path, "rb") as f: data = f.read()
    _img_bytes[path] = len(data)
    return "data:image/jpeg;base64," + base64.b64encode(data).decode()

def compress_video(src, tag, height=300, crf=34, fps=10):
    """Re-encode to a small browser-safe mp4, cache by tag+params. Returns data URI."""
    if not os.path.exists(src): return None, 0
    dst = os.path.join(CACHE, f"{tag}_h{height}c{crf}f{fps}.mp4")
    if not os.path.exists(dst) or os.path.getmtime(dst) < os.path.getmtime(src):
        cmd = [FF, "-y", "-i", src,
               "-vf", f"scale=-2:{height},fps={fps}",
               "-c:v", "libx264", "-crf", str(crf), "-preset", "veryfast",
               "-pix_fmt", "yuv420p", "-an", "-movflags", "+faststart", dst]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  ! ffmpeg failed for {src}:\n{r.stderr[-500:]}", file=sys.stderr)
            return None, 0
    sz = os.path.getsize(dst)
    with open(dst, "rb") as f:
        return "data:video/mp4;base64," + base64.b64encode(f.read()).decode(), sz

def count_from_parquet(name):
    """ID count: table parquet -> row count; raw parquet -> nunique(stable_id)."""
    p = os.path.join(TRK, name)
    if not os.path.exists(p): return None
    df = pd.read_parquet(p)
    if name.endswith("_table.parquet"):
        return len(df)
    col = "stable_id" if "stable_id" in df.columns else df.columns[0]
    return int(df[col].nunique())

def frames(prefix, nums):
    out = []
    for n in nums:
        p = os.path.join(OUT, f"{prefix}{n}.jpg")
        if os.path.exists(p): out.append((n, p))
    return out

# ---------- identity variants, in experiment order ----------
# (label, tag, count_parquet, video_file, frame_prefix, frame_nums, [montage files], note)
VARIANTS = [
 ("Baseline (CV-only)", "v_base", "identities_night_table.parquet",
  "ident_night_h264.mp4", "ident_frame", ["0200","0650","1100","1500","1750"], [],
  "OSNet + BoT-SORT ReID only. Over-merged — collapses distinct players into too few."),
 ("VLM cluster", "v_vlmc", "identities_vlmcluster_night_table.parquet",
  "ident_vlmcluster_night_h264.mp4", "ident_vlmcluster_frame", ["0200","1100"], [],
  "Vision-LLM clustering of raw tracklets by appearance."),
 ("VLM primary", "v_vlmp", "identities_vlmprimary_night_table.parquet",
  "ident_vlmprimary_night_h264.mp4", "ident_vlmprimary_frame", ["0200","1100"],
  ["ident_vlmprimary_montage_A.jpg","ident_vlmprimary_montage_B.jpg","ident_vlmprimary_montage_O.jpg"],
  "VLM primary grouping. Peak over-split — highest count of the run."),
 ("Team-bucketed", "v_buck", "identities_bucket_night_table.parquet",
  "ident_bucket_night_h264.mp4", "ident_bucket_frame", ["0200","1100"], [],
  "Grouping constrained within team buckets (A / B / O)."),
 ("+ Participant filter", "v_filt", "identities_vlmprimary_players_table.parquet",
  "ident_filtered_night_h264.mp4", "ident_filtered_frame", ["0200","0650","1100","1500"], [],
  "Drop referees / spectators / mis-bucketed non-players. Count falls toward truth."),
 ("1080p + reprompt", "v_1080", "identities_1080_players_table.parquet",
  "ident_1080_night_h264.mp4", "ident_1080_frame", ["0200","0650","1100"], [],
  "Sharper 1080p crops + reworked prompt."),
 ("Holistic roster dedup", "v_rost", "identities_roster_night_table.parquet",
  "ident_roster_fixed_fifa_h264.mp4", "ident_roster_fixed_frame", ["0200","0650","1100"], [],
  "Whole-team roster grid handed to the VLM at once — spots duplicates a pairwise sweep misses."),
 ("+ Super-res", "v_sr", "identities_sr_night_table.parquet",
  "ident_sr_fifa_h264.mp4", "ident_sr_frame", ["0200","0650","1100"], [],
  "Real-ESRGAN (spandrel) super-res on crops. Proves clarity is no longer the limiter."),
 ("Super-res v2", "v_sr2", "identities_sr2_night_table.parquet",
  "ident_sr2_fifa_h264.mp4", "ident_sr2_frame", ["0200","1100"], [],
  "Super-res tuning pass."),
 ("Grid dedup + number anchor", "v_dedup", "identities_dedup_night_table.parquet",
  "ident_dedup_fifa_h264.mp4", "ident_dedup_frame", ["0200","0650","1100"], [],
  "Fix: only confident jersey numbers anchor a split. A#1 + A#11 (partial ‘11’) now merge."),
 ("Neutral + context + veto", "v_neut", "identities_neutral_night.parquet",
  "ident_neutral_fifa_h264.mp4", "ident_neutral_frame", ["0200","1100"],
  ["roster_montage_A.jpg","roster_montage_B.jpg","roster_montage_O.jpg"],
  "LATEST. Neutral prompt + full-frame context crops + co-occurrence veto (two tracklets on-screen >100px apart cannot be one player)."),
]

total_bytes = 0
def track(sz):
    global total_bytes; total_bytes += sz

# ---------- build variant blocks ----------
print("Compressing identity-variant videos + reading counts...")
variant_html = []
timeline = []   # (label, count) for the progression gauge
for label, tag, cq, vid, fpre, fnums, montages, note in VARIANTS:
    cnt = count_from_parquet(cq)
    timeline.append((label, cnt))
    vsrc = os.path.join(OUT, vid)
    vuri, vsz = compress_video(vsrc, tag)
    track(vsz)
    print(f"  {label:32s} count={cnt}  video={vsz//1024}KB")
    # gauge: how far above truth (delta over TRUE_HI)
    delta = (cnt - TRUE_HI) if cnt else None
    # frames
    fr = frames(fpre, fnums)[:2]
    fr_html = "".join(
        f'<img class="zoom shot" loading="lazy" src="{u}" alt="{label} frame {n}">'
        for n, p in fr for u in [b64_img(p, maxw=760, q=70)] if u)
    # montages
    mg_html = ""
    for m in montages:
        mp = os.path.join(OUT, m)
        u = b64_img(mp, maxw=1000, q=80)
        if u:
            team = m.replace(".jpg","").split("_")[-1]
            mg_html += f'<figure class="montage"><img class="zoom" loading="lazy" src="{u}" alt="{label} roster {team}"><figcaption>Roster grid — team {team}</figcaption></figure>'
    vid_html = (f'<video class="clip" controls preload="none" playsinline src="{vuri}"></video>'
                if vuri else '<div class="clip missing">no render</div>')
    is_latest = "LATEST" in note
    is_best = (cnt == min([c for _,c in timeline if c]))
    cls = "variant" + (" latest" if is_latest else "")
    variant_html.append(f"""
    <article class="{cls}" id="{tag}">
      <header class="v-head">
        <div class="v-title">{label}{' <span class="tag-latest">latest</span>' if is_latest else ''}</div>
        <div class="v-count" title="stable identities">
          <span class="num">{cnt if cnt is not None else '—'}</span><span class="lbl">IDs</span>
        </div>
      </header>
      <div class="v-gauge" aria-hidden="true"><span class="v-gauge-fill" data-count="{cnt or 0}"></span></div>
      <div class="v-delta">{('+'+str(delta)+' over truth' ) if delta is not None else ''}</div>
      {vid_html}
      <p class="v-note">{note}</p>
      <div class="shots">{fr_html}</div>
      {('<div class="montages">'+mg_html+'</div>') if mg_html else ''}
    </article>""")

# ---------- other stages ----------
def img_fig(fname, cap):
    p = os.path.join(OUT, fname)
    u = b64_img(p, maxw=900, q=74)
    if not u: return ""
    return f'<figure class="tile"><img class="zoom" loading="lazy" src="{u}" alt="{cap}"><figcaption>{cap}</figcaption></figure>'

def vid_fig(fname, tag, cap, height=300):
    src = os.path.join(OUT, fname)
    u, sz = compress_video(src, tag, height=height)
    track(sz)
    if not u: return ""
    return f'<figure class="tile"><video class="clip" controls preload="none" playsinline src="{u}"></video><figcaption>{cap}</figcaption></figure>'

print("Building stage sections (detect / teams / participant / events)...")
detect_html = (
  vid_fig("slice900_annotated.mp4","s_slice","30s tracked slice — boxes + IDs")
  + img_fig("slice900_frame300.jpg","Tracked slice — frame 300")
  + img_fig("full_frame21000.jpg","Full match — daylight sample")
  + img_fig("full_frame52000.jpg","Full match — floodlit sample"))

teams_html = (
  vid_fig("teams_day_h264.mp4","s_teamday","Team assignment — daytime")
  + vid_fig("teams_night_h264.mp4","s_teamnight","Team assignment — night")
  + img_fig("teams_day_frame20300.jpg","Teams day — A white / B pink / O blue")
  + img_fig("teams_night_frame00500.jpg","Teams night — holds under floodlights"))

part_html = (
  img_fig("participant_montage_player.jpg","Classified players (vs ref / spectator / GK)")
  + img_fig("pair_6v20_f462.jpg","Co-occurrence veto — track 6 vs 20, frame 462")
  + img_fig("pair_6v20_f718.jpg","Same instant, 317px apart — cannot be one person"))

# scoreboard evidence
sb_files = [
  ("sb3_montage.jpg","Top-12 sharpest crops of the lit board"),
  ("sb3_panel_compare.jpg","Registered straddle test — averaged, deblurred"),
  ("sb3_early_vs_late.jpg","Early vs late — the shapes never change"),
  ("sb3_digits_compare.jpg","Digit-region compare — nothing under the bloom"),
  ("sb3_temporal_std.jpg","Temporal std-dev — no digit-like variation"),
  ("sb2_topcenter_contact.jpg","Contact sheet — board through the match"),
]
sb_html = "".join(img_fig(f,c) for f,c in sb_files)

# ---------- timeline / progression gauge ----------
counts = [c for _,c in timeline if c]
maxc = max(counts) if counts else 1
tl_html = ""
for label, cnt in timeline:
    if not cnt: continue
    pct = int(100*cnt/maxc)
    near = cnt - TRUE_HI
    tone = "good" if near<=6 else ("warn" if near<=9 else "hot")
    tl_html += f"""<div class="tl-row">
      <div class="tl-label">{label}</div>
      <div class="tl-bar"><span class="tl-fill {tone}" style="width:{pct}%"></span>
        <span class="tl-truth" style="left:{int(100*TRUE_HI/maxc)}%"></span></div>
      <div class="tl-num">{cnt}</div></div>"""

gen_ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
best = min(counts) if counts else "—"
img_total = sum(_img_bytes.values())
print(f"Payload pre-base64: video {total_bytes/1e6:.1f} MB + images {img_total/1e6:.1f} MB = {(total_bytes+img_total)/1e6:.1f} MB")

# ---------- assemble HTML fragment ----------
HTML = r"""<style>
:root{
  --bg:#0A0D0B; --bg2:#0d110f; --card:#12161A; --card2:#171c20;
  --line:#232a2d; --line2:#2e3739;
  --ink:#DBE2E1; --mut:#8A94A0; --dim:#616b70;
  --pitch:#5FB878; --pitchdim:#3c6f4c;
  --A:#EEF2EC; --B:#F5679E; --O:#4EA8E0;
  --good:#5FB878; --warn:#E0B23C; --hot:#E0663C;
  --mono:"SF Mono","JetBrains Mono","Fira Code",ui-monospace,Menlo,monospace;
  --sans:"SF Pro Text",-apple-system,"Segoe UI",Roboto,system-ui,sans-serif;
  --maxw:1180px;
}
*{box-sizing:border-box}
.fv-root{background:
   radial-gradient(1200px 600px at 78% -10%, #10201580 0%, transparent 60%),
   var(--bg);
  color:var(--ink);font-family:var(--sans);line-height:1.55;
  -webkit-font-smoothing:antialiased;font-size:15px;padding-bottom:80px;min-height:100vh}
.fv-root ::selection{background:var(--pitch);color:#06110a}
.wrap{max-width:var(--maxw);margin:0 auto;padding:0 22px}
img{max-width:100%;display:block}
a{color:var(--pitch)}
h1,h2,h3{text-wrap:balance;margin:0}
.mono{font-family:var(--mono)}
.eyebrow{font-family:var(--mono);font-size:11px;letter-spacing:.22em;text-transform:uppercase;color:var(--pitch)}

/* ---- top HUD ---- */
.hud{border-bottom:1px solid var(--line);background:linear-gradient(180deg,#0c110e,#0a0d0b);
  position:relative;overflow:hidden}
.hud .wrap{padding-top:40px;padding-bottom:30px}
.hud:before{content:"";position:absolute;inset:0;
  background-image:linear-gradient(90deg,#ffffff06 1px,transparent 1px);
  background-size:34px 100%;pointer-events:none}
.kicker{display:flex;gap:14px;align-items:center;margin-bottom:16px}
.kicker .dot{width:8px;height:8px;border-radius:50%;background:var(--pitch);box-shadow:0 0 12px var(--pitch)}
h1{font-family:var(--mono);font-weight:600;font-size:clamp(26px,4.4vw,44px);
  letter-spacing:-.02em;line-height:1.02}
h1 .thin{color:var(--mut);font-weight:400}
.sub{color:var(--mut);max-width:62ch;margin-top:12px;font-size:15px}
.metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));
  gap:1px;background:var(--line);border:1px solid var(--line);margin-top:28px;border-radius:10px;overflow:hidden}
.metric{background:var(--card);padding:16px 16px}
.metric .v{font-family:var(--mono);font-size:23px;font-variant-numeric:tabular-nums;letter-spacing:-.01em}
.metric .k{font-size:11px;text-transform:uppercase;letter-spacing:.13em;color:var(--dim);margin-top:5px}
.metric .v.pitch{color:var(--pitch)}

/* ---- layout with rail ---- */
.shell{display:grid;grid-template-columns:210px 1fr;gap:34px;align-items:start;
  max-width:var(--maxw);margin:0 auto;padding:0 22px}
.rail{position:sticky;top:22px;padding-top:38px}
.rail ol{list-style:none;margin:0;padding:0;display:flex;flex-direction:column;gap:2px}
.rail a{display:flex;gap:11px;align-items:baseline;text-decoration:none;color:var(--mut);
  font-family:var(--mono);font-size:12.5px;padding:8px 10px;border-radius:7px;border-left:2px solid transparent}
.rail a .n{color:var(--dim);font-size:11px;min-width:20px}
.rail a:hover{background:#ffffff08;color:var(--ink)}
.rail a.active{color:var(--ink);border-left-color:var(--pitch);background:#5fb87814}
.rail a.active .n{color:var(--pitch)}
main{padding-top:38px;min-width:0}
section{scroll-margin-top:20px;padding:26px 0 40px;border-top:1px solid var(--line)}
section:first-child{border-top:none}
.s-head{display:flex;align-items:baseline;gap:14px;margin-bottom:6px}
.s-head .no{font-family:var(--mono);color:var(--pitchdim);font-size:13px;font-weight:600}
h2{font-family:var(--mono);font-weight:600;font-size:21px;letter-spacing:-.01em}
.s-desc{color:var(--mut);max-width:70ch;margin:10px 0 22px}
.chip{display:inline-block;font-family:var(--mono);font-size:11px;padding:3px 9px;border-radius:20px;
  border:1px solid var(--line2);color:var(--mut);margin-left:auto}
.chip.done{color:var(--good);border-color:#2c5238}
.chip.ceiling{color:var(--hot);border-color:#5a3324}
.chip.ruled{color:var(--warn);border-color:#5a4a24}

/* tiles / grids */
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:16px}
.tile{background:var(--card);border:1px solid var(--line);border-radius:10px;overflow:hidden;margin:0}
.tile img,.tile video{width:100%;background:#000;aspect-ratio:16/9;object-fit:cover}
.tile figcaption{font-size:12.5px;color:var(--mut);padding:10px 12px}
.clip{width:100%;background:#000;border-radius:6px;aspect-ratio:16/9}
.clip.missing{display:flex;align-items:center;justify-content:center;color:var(--dim);font-family:var(--mono);font-size:12px}

/* identity progression gauge */
.timeline{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:20px 22px;margin-bottom:26px}
.tl-head{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:16px;flex-wrap:wrap;gap:8px}
.tl-head .t{font-family:var(--mono);font-size:13px;letter-spacing:.04em;text-transform:uppercase;color:var(--mut)}
.tl-legend{font-size:12px;color:var(--dim);font-family:var(--mono)}
.tl-legend b{color:var(--pitch);font-weight:600}
.tl-row{display:grid;grid-template-columns:200px 1fr 40px;gap:14px;align-items:center;padding:5px 0}
.tl-label{font-size:12.5px;color:var(--mut);text-align:right;font-family:var(--mono)}
.tl-bar{position:relative;height:16px;background:#0a0e0c;border-radius:4px;overflow:visible;border:1px solid var(--line)}
.tl-fill{position:absolute;left:0;top:0;bottom:0;border-radius:3px 0 0 3px;opacity:.9}
.tl-fill.good{background:linear-gradient(90deg,#3c6f4c,var(--good))}
.tl-fill.warn{background:linear-gradient(90deg,#7a6320,var(--warn))}
.tl-fill.hot{background:linear-gradient(90deg,#7a3320,var(--hot))}
.tl-truth{position:absolute;top:-4px;bottom:-4px;width:2px;background:var(--pitch);box-shadow:0 0 8px var(--pitch)}
.tl-num{font-family:var(--mono);font-variant-numeric:tabular-nums;font-size:14px;text-align:right}

/* variant cards */
.variants{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:18px}
.variant{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px;
  display:flex;flex-direction:column;gap:12px}
.variant.latest{border-color:var(--pitchdim);box-shadow:0 0 0 1px #5fb87830, 0 10px 40px -20px #5fb87880}
.v-head{display:flex;justify-content:space-between;align-items:flex-start;gap:10px}
.v-title{font-family:var(--mono);font-size:14px;font-weight:600;line-height:1.25}
.tag-latest{display:inline-block;font-size:9.5px;letter-spacing:.14em;text-transform:uppercase;
  color:#06110a;background:var(--pitch);padding:2px 6px;border-radius:4px;vertical-align:middle;margin-left:4px}
.v-count{text-align:right;white-space:nowrap}
.v-count .num{font-family:var(--mono);font-size:26px;font-variant-numeric:tabular-nums;letter-spacing:-.02em}
.v-count .lbl{font-size:10px;color:var(--dim);text-transform:uppercase;letter-spacing:.1em;margin-left:3px}
.v-gauge{height:4px;background:#0a0e0c;border-radius:3px;overflow:hidden}
.v-gauge-fill{display:block;height:100%;background:linear-gradient(90deg,var(--pitchdim),var(--pitch))}
.v-delta{font-family:var(--mono);font-size:11px;color:var(--dim);margin-top:-6px}
.v-note{font-size:13px;color:var(--mut);margin:0;line-height:1.5}
.variant .clip{margin:0}
.shots{display:grid;grid-template-columns:repeat(auto-fit,minmax(90px,1fr));gap:6px}
.shots .shot{aspect-ratio:16/9;object-fit:cover;border-radius:5px;background:#000;border:1px solid var(--line)}
.montages{display:grid;grid-template-columns:repeat(3,1fr);gap:6px}
.montage{margin:0}
.montage img{border-radius:5px;background:#000;border:1px solid var(--line)}
.montage figcaption{font-size:10px;color:var(--dim);font-family:var(--mono);margin-top:3px}

/* finding callout */
.finding{background:linear-gradient(180deg,#191410,#141210);border:1px solid #4a3a1c;
  border-radius:12px;padding:20px 22px;margin-bottom:22px}
.finding .verdict{font-family:var(--mono);color:var(--warn);font-size:13px;letter-spacing:.08em;
  text-transform:uppercase;margin-bottom:8px}
.finding p{margin:0;color:var(--ink);max-width:74ch}
.finding .hint{color:var(--mut);font-size:13px;margin-top:10px}

.zoom{cursor:zoom-in}
/* lightbox */
#lb{position:fixed;inset:0;background:#000000ee;display:none;align-items:center;justify-content:center;
  z-index:99;padding:30px;cursor:zoom-out}
#lb.on{display:flex}
#lb img{max-width:96vw;max-height:94vh;width:auto;border-radius:6px;box-shadow:0 20px 80px #000}
.foot{color:var(--dim);font-size:12px;font-family:var(--mono);text-align:center;padding-top:30px;border-top:1px solid var(--line);margin-top:10px}

@media (max-width:820px){
  .shell{grid-template-columns:1fr}
  .rail{display:none}
  .tl-row{grid-template-columns:120px 1fr 32px;gap:8px}
  .tl-label{font-size:11px}
}
@media (prefers-reduced-motion:reduce){*{scroll-behavior:auto!important}}
</style>

<div class="fv-root">
<div class="hud"><div class="wrap">
  <div class="kicker"><span class="dot"></span><span class="eyebrow">Monocular Sports-CV Pipeline &nbsp;/&nbsp; Review Console</span></div>
  <h1>football&#8209;vision <span class="thin">// intermediate artifact review</span></h1>
  <p class="sub">Every stage of the pipeline that turns an amateur Veo match off YouTube into tracked players, team labels, whistle events and a broadcast overlay &mdash; laid out as reviewable evidence. The identity stage carries an experiment log: each variant's stable-ID count against the true roster of __TRUELO__&ndash;__TRUEHI__.</p>
  <div class="metrics">
    <div class="metric"><div class="v pitch">~120</div><div class="k">detect+track fps</div></div>
    <div class="metric"><div class="v">488k</div><div class="k">detections</div></div>
    <div class="metric"><div class="v">98.6%</div><div class="k">frames tracked</div></div>
    <div class="metric"><div class="v">39.5m</div><div class="k">match length</div></div>
    <div class="metric"><div class="v pitch">__BEST__</div><div class="k">best stable IDs</div></div>
    <div class="metric"><div class="v">__TRUELO__&ndash;__TRUEHI__</div><div class="k">true roster</div></div>
  </div>
</div></div>

<div class="shell">
  <nav class="rail"><ol>
    <li><a href="#s-detect"><span class="n">01</span> Detect &amp; Track</a></li>
    <li><a href="#s-teams"><span class="n">02</span> Teams</a></li>
    <li><a href="#s-identity"><span class="n">03</span> Identity</a></li>
    <li><a href="#s-participant"><span class="n">04</span> Participant &amp; Veto</a></li>
    <li><a href="#s-events"><span class="n">05</span> Events</a></li>
    <li><a href="#s-scoreboard"><span class="n">06</span> Scoreboard</a></li>
  </ol></nav>

  <main>
    <section id="s-detect">
      <div class="s-head"><span class="no">01</span><h2>Detect &amp; Track</h2><span class="chip done">done</span></div>
      <p class="s-desc">YOLO11m + ByteTrack / BoT-SORT, ~120&nbsp;fps on the 4090. 488k detections across the full 39.5&nbsp;min, 98.6% of frames with a track. GPU is decode-bound, not compute-bound.</p>
      <div class="grid">__DETECT__</div>
    </section>

    <section id="s-teams">
      <div class="s-head"><span class="no">02</span><h2>Team assignment</h2><span class="chip done">done</span></div>
      <p class="s-desc">Three buckets &mdash; <b style="color:var(--A)">A white</b>, <b style="color:var(--B)">B pink</b>, <b style="color:var(--O)">O blue + keepers</b>. Holds through the day&rarr;night transition under floodlights.</p>
      <div class="grid">__TEAMS__</div>
    </section>

    <section id="s-identity">
      <div class="s-head"><span class="no">03</span><h2>Player identity &mdash; experiment log</h2><span class="chip ceiling">at ceiling</span></div>
      <p class="s-desc">The hard stage: collapse thousands of churning track IDs into a stable per-player roster. Every variant below is one experiment; the goal is the true roster of __TRUELO__&ndash;__TRUEHI__. Lower isn't always better &mdash; a correct 23 beats an over-merged 19. The count is capped by the footage (dark, near-identical kits, hoodies, ~35px figures), not the method.</p>
      <div class="timeline">
        <div class="tl-head"><span class="t">Stable ID count per variant</span>
          <span class="tl-legend">green marker = truth (<b>__TRUEHI__</b>) &nbsp;·&nbsp; bar tone = distance from truth</span></div>
        __TIMELINE__
      </div>
      <div class="variants">__VARIANTS__</div>
    </section>

    <section id="s-participant">
      <div class="s-head"><span class="no">04</span><h2>Participant filter &amp; co-occurrence veto</h2><span class="chip done">done</span></div>
      <p class="s-desc">Classify player / referee / spectator / keeper by geometry, then the geometric arbiter: two tracklets on-screen together and &gt;100px apart cannot be one person &mdash; a hard block on impossible merges (proved track&nbsp;6&nbsp;&ne;&nbsp;20, 317px apart over 464 shared frames).</p>
      <div class="grid">__PART__</div>
    </section>

    <section id="s-events">
      <div class="s-head"><span class="no">05</span><h2>Events &mdash; whistles</h2><span class="chip done">done</span></div>
      <p class="s-desc">Audio-detected referee whistles: 12 blasts located, second-half kickoff and some goal-pairs isolated &rarr; <span class="mono">whistles.csv</span>. Goals &amp; assists (whistle + audio-spike + ball-near-goal) are designed, not yet built.</p>
    </section>

    <section id="s-scoreboard">
      <div class="s-head"><span class="no">06</span><h2>Scoreboard OCR</h2><span class="chip ruled">ruled out</span></div>
      <p class="s-desc">A hard investigation that reached a negative result &mdash; kept here as evidence so it isn't re-run.</p>
      <div class="finding">
        <div class="verdict">Verdict &mdash; likely a drinks advert, not a scoreboard</div>
        <p>The lit top-center board is ~140px wide even at the sharpest pan and heavily bloomed. Registration + temporal averaging recovers no digits under the bloom, and the two shapes never change between early and late in the match. There is nothing there to read.</p>
        <p class="hint">Flips only if you point at a specific timestamp where you can personally see two clearly-different digits (e.g. a &ldquo;2&rdquo; then a &ldquo;3&rdquo;) &mdash; then it gets zoomed straight to that frame.</p>
      </div>
      <div class="grid">__SB__</div>
    </section>

    <div class="foot">generated __TS__ &nbsp;·&nbsp; zulzi-gpu:~/football-vision &nbsp;·&nbsp; night slice, frames 0&ndash;1800 (~60s) &nbsp;·&nbsp; regenerate: <span style="color:var(--mut)">build_dashboard.py</span></div>
  </main>
</div>

<div id="lb"><img alt="zoom"></div>
</div>

<script>
(function(){
  var lb=document.getElementById('lb'),lbi=lb.querySelector('img');
  document.querySelectorAll('.zoom').forEach(function(im){
    im.addEventListener('click',function(){lbi.src=im.currentSrc||im.src;lb.classList.add('on');});
  });
  lb.addEventListener('click',function(){lb.classList.remove('on');lbi.src='';});
  document.addEventListener('keydown',function(e){if(e.key==='Escape'){lb.classList.remove('on');lbi.src='';}});
  // gauge fills (count -> width, capped)
  document.querySelectorAll('.v-gauge-fill').forEach(function(g){
    var c=+g.getAttribute('data-count')||0; var pct=Math.min(100,Math.round(c/28*100));
    g.style.width=pct+'%';
  });
  // scrollspy
  var links=[].slice.call(document.querySelectorAll('.rail a'));
  var secs=links.map(function(a){return document.querySelector(a.getAttribute('href'));});
  function spy(){
    var y=window.scrollY+140,cur=0;
    secs.forEach(function(s,i){if(s&&s.offsetTop<=y)cur=i;});
    links.forEach(function(a,i){a.classList.toggle('active',i===cur);});
  }
  window.addEventListener('scroll',spy,{passive:true});spy();
})();
</script>
"""

HTML = (HTML
  .replace("__DETECT__", detect_html)
  .replace("__TEAMS__", teams_html)
  .replace("__PART__", part_html)
  .replace("__SB__", sb_html)
  .replace("__TIMELINE__", tl_html)
  .replace("__VARIANTS__", "".join(variant_html))
  .replace("__TRUELO__", str(TRUE_LO)).replace("__TRUEHI__", str(TRUE_HI))
  .replace("__BEST__", str(best))
  .replace("__TS__", gen_ts))

outp = os.path.join(OUT, "dashboard.html")
with open(outp, "w") as f:
    f.write(HTML)
mb = len(HTML.encode())/1e6
print(f"\nWrote {outp}  ({mb:.1f} MB HTML)")
if mb > 15.5: print("WARNING: >15.5MB — over the 16MB Artifact limit; compress further")
else: print("OK: under the 16MB Artifact limit")
