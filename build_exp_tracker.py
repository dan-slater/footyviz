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
    fr = "".join(f'<img class="zoom shot" loading="lazy" src="{b64_img(p,760,70)}">' for p in frames(fpre,fnums))
    mg = ""
    for m in montages:
        u=b64_img(os.path.join(OUT,m),1000,80)
        if u:
            t=m.replace(".jpg","").split("_")[-1]
            mg += f'<figure class="mon"><img class="zoom" loading="lazy" src="{u}"><figcaption>team {t}</figcaption></figure>'
    vhtml = f'<video class="clip" controls preload="none" playsinline src="{vuri}"></video>' if vuri else '<div class="clip miss">no render</div>'
    panels.append(f'''<section class="panel{' active' if run==11 else ''}" id="p{run}">
      <div class="phead"><div><span class="run">RUN {run:02d}{' · LATEST' if latest else ''}</span><h2>{label}</h2></div>
        <div class="bignum {tone(cnt)}"><span>{cnt if cnt is not None else '—'}</span><em>stable IDs · truth {TRUE_LO}–{TRUE_HI}{f' · +{d}' if d is not None else ''}</em></div></div>
      <div class="setup"><span class="k">SETUP / WHAT CHANGED</span><p>{setup}</p></div>
      <div class="cols"><div class="vwrap">{vhtml}<div class="shots">{fr}</div></div>
        {f'<div class="mons">{mg}</div>' if mg else ''}</div>
    </section>''')
    rows.append((run,label,cnt,d,setup))

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
.tab:hover{color:var(--ink)}
.tab.on{background:var(--card2);color:var(--ink);border-color:var(--line2);box-shadow:0 -2px 0 var(--pitch) inset}
.body{background:var(--card2);border:1px solid var(--line2);border-radius:0 10px 10px 10px;min-height:400px;padding:22px}
.panel{display:none} .panel.show{display:block}
.phead{display:flex;justify-content:space-between;align-items:flex-start;gap:16px;flex-wrap:wrap;border-bottom:1px solid var(--line);padding-bottom:14px;margin-bottom:16px}
.run{font-family:var(--mono);font-size:11px;letter-spacing:.12em;color:var(--pitch)}
.phead h2{font-family:var(--mono);font-weight:600;font-size:19px;margin:4px 0 0;letter-spacing:-.01em}
.bignum{text-align:right}
.bignum span{font-family:var(--mono);font-size:34px;font-weight:600;font-variant-numeric:tabular-nums;line-height:1}
.bignum.good span{color:var(--good)} .bignum.warn span{color:var(--warn)} .bignum.hot span{color:var(--hot)}
.bignum em{display:block;font-style:normal;color:var(--dim);font-size:11px;font-family:var(--mono);margin-top:4px}
.setup{background:#0d1210;border:1px solid var(--line);border-radius:10px;padding:12px 15px;margin-bottom:16px}
.setup .k{font-family:var(--mono);font-size:10px;letter-spacing:.14em;color:var(--pitch)}
.setup p{margin:6px 0 0;color:var(--ink);font-size:14px}
.cols{display:grid;grid-template-columns:1fr;gap:16px}
.vwrap .clip{width:100%;background:#000;border-radius:8px;aspect-ratio:16/9}
.clip.miss{display:flex;align-items:center;justify-content:center;color:var(--dim);font-family:var(--mono);font-size:12px}
.shots{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:8px}
.shots .shot{aspect-ratio:16/9;object-fit:cover;border-radius:6px;border:1px solid var(--line);background:#000}
.mons{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}
.mon{margin:0}.mon img{border-radius:6px;border:1px solid var(--line);background:#000}
.mon figcaption{font-family:var(--mono);font-size:10px;color:var(--dim);margin-top:3px;text-align:center}
.zoom{cursor:zoom-in}
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
   <div class="sub">Night slice · one tab per run · stable-ID count vs true roster __LO__–__HI__ · best so far __BEST__. Click a tab; click any image to zoom.</div>
   <div class="tabs"><button class="tab on" data-p="pov"><span class="tn">＊</span>Overview</button>__TABS__</div>
 </div>
 <div class="body">
   <section class="panel show" id="pov"><div class="ov">
     <h3>Stable-ID count per run (green line = truth __HI__)</h3>
     <div class="chart">__BARS__</div>
     <h3>All runs</h3>
     <table><thead><tr><th>#</th><th>Run</th><th class="num">IDs</th><th class="num">Δ</th><th>Setup / what changed</th></tr></thead><tbody>__TROWS__</tbody></table>
     <div class="howto"><b>To add an experiment:</b> run the pipeline variant on zulzi → add an entry to <code>VARIANTS</code> in <code>build_exp_tracker.py</code> → re-run it → pull <code>dashboard_experiments.html</code> → redeploy this Artifact. The tab appears automatically. Non-monotonic on purpose: a <em>correct</em> 23 beats an over-merged 19.</div>
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
outp = os.path.join(OUT,"dashboard_experiments.html"); open(outp,"w").write(HTML)
print(f"\nWrote {outp}  ({len(HTML.encode())/1e6:.1f} MB HTML) · video payload {vid_bytes/1e6:.1f} MB")
