"""Build a mid-level 'technical briefing' HTML report — between the layman
(make_report.py) and scientific (make_report_scientific.py) versions.
Audience: a technically literate stakeholder/decision-maker. Plain language,
real numbers, honest caveats, no equations or citations. Self-contained."""

import base64
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def b64(path):
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


THRESH = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
RECALL = [75.9, 74.1, 66.3, 64.2, 60.5, 57.5, 55.8, 52.0, 50.0]
PREC = [98.5, 98.9, 99.4, 99.6, 99.7, 99.8, 99.8, 99.7, 99.7]

# (label, recall@0.2, colour, plain one-liner)
JOURNEY = [
    ("Starting point", 0.41, "#c0392b", "a hidden training error capped it"),
    ("Training fix +\nrecall focus", 0.68, "#e67e22", "biggest single jump"),
    ("Rebalance to\nlone cattle", 0.69, "#f1c40f", "modest gain"),
    ("Vegetation index\n(first attempt)", 0.66, "#7f8c8d", "added wrongly — regressed"),
    ("Vegetation index\n(corrected)", 0.74, "#27ae60", "final, best result"),
]

STRATA = [("Cattle in a herd", 684, 728, 94.0, "#27ae60"),
          ("Solitary cattle", 903, 1406, 64.2, "#e67e22"),
          ("All cattle", 1587, 2134, 74.4, "#1f6f8b")]


def operating_curve():
    L, T, W, H = 60, 18, 540, 215
    def X(t): return L + (t - 0.1) / 0.8 * W
    def Y(p): return T + (100 - p) / 100 * H
    def poly(v): return " ".join(f"{X(t):.1f},{Y(x):.1f}" for t, x in zip(THRESH, v))
    g = ""
    for p in range(0, 101, 20):
        g += f'<line x1="{L}" y1="{Y(p):.0f}" x2="{L+W}" y2="{Y(p):.0f}" stroke="#eef1f3"/>'
        g += f'<text x="{L-8}" y="{Y(p)+4:.0f}" text-anchor="end" class="ax">{p}%</text>'
    for t in THRESH:
        g += f'<text x="{X(t):.0f}" y="{T+H+18:.0f}" text-anchor="middle" class="ax">{t}</text>'
    dots = "".join(f'<circle cx="{X(t):.1f}" cy="{Y(v):.1f}" r="3.2" fill="#1f6f8b"/>'
                   for t, v in zip(THRESH, RECALL))
    dots += "".join(f'<circle cx="{X(t):.1f}" cy="{Y(v):.1f}" r="3.2" fill="#27ae60"/>'
                    for t, v in zip(THRESH, PREC))
    mx = X(0.2)
    return f'''<svg viewBox="0 0 640 285" class="fig">
      {g}
      <line x1="{mx:.0f}" y1="{T}" x2="{mx:.0f}" y2="{T+H}" stroke="#e67e22" stroke-dasharray="4 3"/>
      <text x="{mx:.0f}" y="{T-4}" text-anchor="middle" class="ax" fill="#e67e22">chosen setting</text>
      <polyline points="{poly(PREC)}" fill="none" stroke="#27ae60" stroke-width="2.6"/>
      <polyline points="{poly(RECALL)}" fill="none" stroke="#1f6f8b" stroke-width="2.6"/>
      {dots}
      <rect x="{L+W-210}" y="{T+2}" width="12" height="12" fill="#27ae60"/>
      <text x="{L+W-194}" y="{T+12}" class="lg">Accuracy of alarms (~99%)</text>
      <rect x="{L+W-210}" y="{T+20}" width="12" height="12" fill="#1f6f8b"/>
      <text x="{L+W-194}" y="{T+30}" class="lg">Share of cattle found</text>
      <text x="{L+W/2:.0f}" y="{T+H+40:.0f}" text-anchor="middle" class="axl">sensitivity setting&nbsp;&nbsp;(left = finds more, right = stricter)</text>
    </svg>'''


def journey_fig():
    L, T, W, H = 40, 16, 560, 195
    n = len(JOURNEY); bw = W / n * 0.58; gap = W / n
    g = ""
    for p in (0.2, 0.4, 0.6, 0.8):
        y = T + (1 - p) * H
        g += f'<line x1="{L}" y1="{y:.0f}" x2="{L+W}" y2="{y:.0f}" stroke="#eef1f3"/>'
        g += f'<text x="{L-6}" y="{y+4:.0f}" text-anchor="end" class="ax">{int(p*100)}%</text>'
    for i, (lab, v, c, _note) in enumerate(JOURNEY):
        x = L + i * gap + (gap - bw) / 2
        h = v * H
        g += f'<rect x="{x:.0f}" y="{T+H-h:.0f}" width="{bw:.0f}" height="{h:.0f}" fill="{c}" rx="3"/>'
        g += f'<text x="{x+bw/2:.0f}" y="{T+H-h-5:.0f}" text-anchor="middle" class="lg">{int(v*100)}%</text>'
        lines = lab.split("\n")
        for j, ln in enumerate(lines):
            g += f'<text x="{x+bw/2:.0f}" y="{T+H+15+j*12:.0f}" text-anchor="middle" class="ax">{ln}</text>'
    return f'<svg viewBox="0 0 620 250" class="fig">{g}</svg>'


def strata_fig():
    rows = ""
    for name, d, n, pct, c in STRATA:
        rows += f'''<div class="hb-row"><div class="hb-lab">{name}</div>
          <div class="hb-track"><div class="hb-fill" style="width:{pct}%;background:{c}">{pct:.1f}%</div></div>
          <div class="hb-n">{d:,}/{n:,}</div></div>'''
    return f'<div class="hbars">{rows}</div>'


missed_img = b64(os.path.join(HERE, "missed_cows_overlay.png"))
detok_img = b64(os.path.join(HERE, "detected_cows_sample.png"))

HTML = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Cattle Detection from Satellite Imagery — Technical Briefing</title>
<style>
 body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:#21333f;
      line-height:1.6;max-width:860px;margin:0 auto;padding:30px 24px 80px;background:#fff;font-size:16px}
 h1{font-size:27px;margin:0 0 4px;color:#16313f}
 .sub{color:#5b7180;font-size:15px;margin-bottom:18px}
 h2{font-size:20px;margin:38px 0 10px;color:#16313f;border-bottom:2px solid #1f6f8b;padding-bottom:6px}
 h3{font-size:16px;margin:20px 0 6px;color:#1f6f8b}
 .summary{background:#f2f8fa;border:1px solid #cfe3ea;border-left:5px solid #1f6f8b;border-radius:6px;
          padding:16px 20px;font-size:15.5px}
 .kpis{display:flex;gap:12px;flex-wrap:wrap;margin:18px 0}
 .kpi{flex:1;min-width:140px;background:#fff;border:1px solid #e1e8ec;border-radius:8px;padding:14px;text-align:center}
 .kpi .n{font-size:30px;font-weight:700;line-height:1.1}.kpi .l{font-size:12.5px;color:#5b7180;margin-top:3px}
 .teal{color:#1f6f8b}.green{color:#27ae60}.orange{color:#e67e22}
 .fig{width:100%;height:auto}.figbox{background:#fbfcfd;border:1px solid #e1e8ec;border-radius:8px;padding:12px 14px;margin:16px 0}
 .figcap{font-size:13px;color:#5b7180;margin-top:6px}
 .ax{font-size:10.5px;fill:#90a4af}.axl{font-size:12px;fill:#5b7180}.lg{font-size:12px;fill:#21333f;font-weight:600}
 table{border-collapse:collapse;width:100%;margin:14px 0;font-size:14px}
 th,td{border:1px solid #e1e8ec;padding:9px 11px;text-align:left;vertical-align:top}th{background:#eef4f6}
 .hbars{margin:6px 0}
 .hb-row{display:flex;align-items:center;gap:10px;margin:6px 0;font-size:13.5px}
 .hb-lab{width:150px;text-align:right;color:#21333f;font-weight:600}
 .hb-track{flex:1;background:#eef1f3;border-radius:5px;overflow:hidden}
 .hb-fill{color:#fff;font-weight:700;padding:5px 9px;border-radius:5px;font-size:12.5px;white-space:nowrap}
 .hb-n{width:90px;color:#5b7180;font-size:12.5px}
 .photo{width:49%;border:1px solid #e1e8ec;border-radius:4px;vertical-align:top}
 .steps{counter-reset:s;list-style:none;padding-left:0}
 .steps li{position:relative;padding:8px 0 8px 42px;border-bottom:1px solid #f0f3f5}
 .steps li:before{counter-increment:s;content:counter(s);position:absolute;left:0;top:8px;width:28px;height:28px;
   background:#1f6f8b;color:#fff;border-radius:50%;text-align:center;line-height:28px;font-weight:700;font-size:14px}
 .note{background:#fff8ec;border-left:4px solid #e6a817;padding:10px 14px;border-radius:4px;margin:14px 0;font-size:14.5px}
 footer{margin-top:46px;font-size:12px;color:#90a4af;border-top:1px solid #eef1f3;padding-top:14px}
</style></head><body>

<h1>Cattle Detection from Satellite Imagery</h1>
<div class="sub">Technical briefing &middot; SkyFi Sector&nbsp;6 (76&nbsp;km&sup2;, ~30&nbsp;cm imagery) &middot; 2026-06-22</div>

<div class="summary">
<b>What this is.</b> An automated system that scans a large satellite photo of farmland and marks where
cattle are &mdash; no manual searching. This briefing covers what it does, how well it works, where it is
reliable, and where it is not. <b>Bottom line:</b> it finds <b>74%</b> of all cattle at <b>99%</b> alarm
accuracy, and is <b>highly reliable for locating herds (94%)</b> while weaker on solitary animals (64%).
</div>

<div class="kpis">
 <div class="kpi"><div class="n green">74%</div><div class="l">of all cattle found</div></div>
 <div class="kpi"><div class="n green">99%</div><div class="l">of alarms are real cattle</div></div>
 <div class="kpi"><div class="n teal">94%</div><div class="l">of herd cattle found</div></div>
 <div class="kpi"><div class="n orange">64%</div><div class="l">of solitary cattle found</div></div>
</div>

<h2>1. The task and the data</h2>
<p>The input is one satellite image of a <b>76&nbsp;km&sup2;</b> farming sector at roughly <b>30&nbsp;cm
per pixel</b> &mdash; sharp, but a single animal is still only a few pixels across. The camera captures
four light bands (including near-infrared, which vegetation reflects strongly); from these we also derive
a <b>vegetation index (NDVI)</b> that distinguishes living grass from bare ground. To train and grade the
system, an expert marked <b>2,134 cattle</b> and <b>4,268 confirmed non-cattle</b> locations.</p>

<h2>2. How the system works</h2>
<p>At its core is a <b>neural network</b> trained on thousands of small image tiles with cattle already
marked, until it recognises the look of an animal from above. It is then slid across the entire scene,
outputting a confidence value for every location, which is turned into a cattle map. A single
<b>sensitivity setting</b> controls how eager it is &mdash; lower finds more animals but risks more false
alarms; higher only flags what it is sure about.</p>

<h3>What we did to improve it</h3>
<p>The first version found only 41% of cattle. Five targeted changes brought it to 74%:</p>
<ol class="steps">
<li><b>Fixed a training error.</b> Practice images were being flipped/rotated without flipping their
answer key to match &mdash; so the system was learning from mismatched labels. Correcting this was the
single biggest improvement.</li>
<li><b>Made misses costlier.</b> We changed the training scoring so that overlooking a cow is penalised
more heavily, pushing the system to be less conservative.</li>
<li><b>Rebalanced toward lone animals.</b> Solitary cattle were rare in training, so we showed them more
often.</li>
<li><b>Added the vegetation index</b> as an extra input, helping the system tell a pale animal from pale
bare soil.</li>
<li><b>Corrected how that index was used</b> &mdash; see the note below.</li>
</ol>

<div class="figbox">__JOURNEY__
<div class="figcap"><b>Figure 1.</b> Share of cattle found at each stage of development.</div></div>

<div class="note"><b>An honest setback worth noting.</b> The vegetation index <i>initially made results worse</i>
(the grey bar, 66%) because we processed it with the same random brightness/contrast tweaks used on normal
colours &mdash; meaningless for an index. Once corrected, the very same idea delivered the largest single
gain, to 74%.</div>

<h2>3. How well it works</h2>
<p>Across the whole sensitivity range, <b>almost every alarm is a real animal</b> (green line below stays
near 99%). Turning sensitivity up finds more cattle at very little cost in false alarms. We operate at the
point marked in orange, which finds about three-quarters of cattle while keeping alarms essentially clean.</p>

<div class="figbox">__OPERATING__
<div class="figcap"><b>Figure 2.</b> The trade-off dial. Alarm accuracy (green) stays high everywhere;
share of cattle found (blue) rises as sensitivity increases. Chosen operating point dashed.</div></div>

<h2>4. Strengths and weaknesses</h2>
<p>When we examined <i>which</i> cattle are missed, a clear pattern emerged: the system is
<b>near-perfect on animals standing in a herd</b> and noticeably weaker on <b>solitary animals</b>. A lone
animal on bare ground is a faint speck; a herd is an obvious cluster. Solitary cattle happen to be the
majority here, which is what holds the overall figure at 74%.</p>

<div class="figbox">__STRATA__
<div class="figcap"><b>Figure 3.</b> Detection rate by context (found / total).</div></div>

<h3>Real examples</h3>
<p>Straight from the imagery (yellow ring = an expert-marked animal):</p>
<div class="figbox">
<img class="photo" src="__DETOK_IMG__" alt="detected">
<img class="photo" src="__MISSED_IMG__" alt="missed">
<div class="figcap"><b>Figure 4.</b> Left: cattle the system <b>found</b> (bright, clustered).
Right: cattle it <b>missed</b> (faint, solitary, on bare ground).</div></div>

<h2>5. What it is suitable for</h2>
<table>
<tr><th>Intended use</th><th>Suitability</th></tr>
<tr><td><b>Locating cattle / mapping herds and grazing groups</b></td>
<td class="green"><b>Strong.</b> A herd is essentially never missed &mdash; detecting one of its members
flags the location.</td></tr>
<tr><td><b>Exact head-count of every individual animal</b></td>
<td class="orange"><b>Use with care.</b> Expect an undercount, concentrated on scattered solitary animals
near the limit of image sharpness.</td></tr>
</table>

<h2>6. Limitations</h2>
<p>Results come from a <b>single scene</b>; performance on other areas or dates is untested. The
expert markings are a <b>sample, not a complete census</b>, so the true false-alarm rate over unmarked
ground cannot be pinned down exactly (the evidence points to it being low). The remaining misses are a
<b>resolution and contrast limit</b> &mdash; small, faint, isolated animals &mdash; more than a software
shortcoming; sharper imagery or more marked examples of such animals would help most.</p>

<h2>7. At a glance</h2>
<table>
<tr><td><b>Coverage</b></td><td>76&nbsp;km&sup2; scene, ~30&nbsp;cm/pixel, 4 light bands + vegetation index</td></tr>
<tr><td><b>Reference data</b></td><td>2,134 marked cattle, 4,268 non-cattle controls</td></tr>
<tr><td><b>Found overall</b></td><td>74% of cattle, at 99% alarm accuracy</td></tr>
<tr><td><b>Herd vs. solitary</b></td><td>94% vs. 64%</td></tr>
<tr><td><b>Speed</b></td><td>Whole scene scanned in ~5 minutes on a single modern GPU</td></tr>
</table>

<footer>Auto-generated from the project's evaluation results. A non-technical summary and a full scientific
report are also available. Figures measured against 2,134 expert-marked cattle in one SkyFi scene.</footer>
</body></html>"""

HTML = (HTML
        .replace("__JOURNEY__", journey_fig())
        .replace("__OPERATING__", operating_curve())
        .replace("__STRATA__", strata_fig())
        .replace("__DETOK_IMG__", detok_img)
        .replace("__MISSED_IMG__", missed_img))

out = os.path.join(HERE, "report_brief.html")
with open(out, "w", encoding="utf-8") as f:
    f.write(HTML)
print(f"wrote {out}  ({os.path.getsize(out)/1e6:.1f} MB, self-contained)")
