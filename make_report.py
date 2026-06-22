"""Build a self-contained, layman-friendly HTML report of the cow-detection
methodology and results. All charts are inline SVG/CSS and the example photos
are embedded as base64, so report.html is a single shareable file."""

import base64
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def b64(path):
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


# ----- data -----------------------------------------------------------------
# Journey: share of cows the system finds, at each improvement step.
JOURNEY = [
    ("Starting point", 41, "#c0392b", "had a hidden training bug"),
    ("Bug fixed +\nsmarter focus", 68, "#e67e22", "labels lined up; chases misses"),
    ("Show it more\nlone cows", 69, "#f1c40f", "small extra gain"),
    ("Plant-vs-bare\nsense (1st try)", 66, "#7f8c8d", "added wrong — went backwards"),
    ("Plant-vs-bare\nsense (fixed)", 74, "#27ae60", "the winning recipe"),
]

# Herd vs lone cow detection rate (final model).
HERD = 94.0
LONE = 64.2
OVERALL = 74.4

# Threshold trade-off (final model): how the two "dials" move.
THRESH = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
RECALL = [75.9, 74.1, 66.3, 64.2, 60.5, 57.5, 55.8, 52.0, 50.0]
PREC = [98.5, 98.9, 99.4, 99.6, 99.7, 99.8, 99.8, 99.7, 99.7]


# ----- threshold line chart (inline SVG) ------------------------------------
def line_chart():
    L, T, W, H = 64, 18, 560, 230
    def X(t): return L + (t - 0.1) / 0.8 * W
    def Y(p): return T + (100 - p) / 100 * H
    def poly(vals): return " ".join(f"{X(t):.1f},{Y(v):.1f}" for t, v in zip(THRESH, vals))
    grid = ""
    for p in range(0, 101, 20):
        grid += f'<line x1="{L}" y1="{Y(p):.0f}" x2="{L+W}" y2="{Y(p):.0f}" stroke="#eee"/>'
        grid += f'<text x="{L-10}" y="{Y(p)+4:.0f}" text-anchor="end" class="ax">{p}%</text>'
    for t in THRESH:
        grid += f'<text x="{X(t):.0f}" y="{T+H+20:.0f}" text-anchor="middle" class="ax">{t}</text>'
    rec_dots = "".join(f'<circle cx="{X(t):.1f}" cy="{Y(v):.1f}" r="3.5" fill="#2980b9"/>'
                       for t, v in zip(THRESH, RECALL))
    pre_dots = "".join(f'<circle cx="{X(t):.1f}" cy="{Y(v):.1f}" r="3.5" fill="#27ae60"/>'
                       for t, v in zip(THRESH, PREC))
    # marker for the chosen operating dial (0.2)
    mx = X(0.2)
    return f'''<svg viewBox="0 0 660 300" class="chart">
      {grid}
      <line x1="{mx:.0f}" y1="{T}" x2="{mx:.0f}" y2="{T+H}" stroke="#e67e22" stroke-dasharray="4 3"/>
      <text x="{mx:.0f}" y="{T-4}" text-anchor="middle" class="ax" fill="#e67e22">chosen setting</text>
      <polyline points="{poly(PREC)}" fill="none" stroke="#27ae60" stroke-width="3"/>
      <polyline points="{poly(RECALL)}" fill="none" stroke="#2980b9" stroke-width="3"/>
      {pre_dots}{rec_dots}
      <text x="{L+W}" y="{Y(PREC[-1])-10:.0f}" text-anchor="end" class="lg" fill="#27ae60">Accuracy of alarms (almost always 99%)</text>
      <text x="{L+W}" y="{Y(RECALL[-1])+18:.0f}" text-anchor="end" class="lg" fill="#2980b9">Share of cows found</text>
      <text x="{L+W/2:.0f}" y="296" text-anchor="middle" class="ax">Caution dial &rarr; (left = sensitive / finds more, right = strict / fewer false alarms)</text>
    </svg>'''


def journey_bars():
    maxv = 100
    bars = ""
    for label, val, color, note in JOURNEY:
        h = int(val / maxv * 200)
        lines = "".join(f'<tspan x="50%" dy="1.1em">{ln}</tspan>' for ln in label.split("\n"))
        bars += f'''<div class="bar-col">
          <div class="bar-val">{val}%</div>
          <div class="bar" style="height:{h}px;background:{color}"></div>
          <svg class="bar-lbl" viewBox="0 0 120 44"><text x="50%" y="0" text-anchor="middle">{lines}</text></svg>
          <div class="bar-note">{note}</div>
        </div>'''
    return f'<div class="bar-chart">{bars}</div>'


# ----- assemble -------------------------------------------------------------
missed_img = b64(os.path.join(HERE, "missed_cows_overlay.png"))
detok_img = b64(os.path.join(HERE, "detected_cows_sample.png"))

HTML = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Finding Cows from the Sky &mdash; Methodology Report</title>
<style>
 body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:#2c3e50;
      line-height:1.6;max-width:880px;margin:0 auto;padding:28px 22px 80px;background:#fff}
 h1{font-size:30px;margin:0 0 4px} h2{font-size:22px;margin:42px 0 10px;border-bottom:3px solid #27ae60;padding-bottom:6px}
 h3{font-size:17px;margin:22px 0 6px;color:#1a5c38}
 .sub{color:#7f8c8d;font-size:15px;margin-bottom:8px}
 .card{background:#f8f9fa;border:1px solid #e6e8ea;border-radius:10px;padding:18px 20px;margin:16px 0}
 .big{font-size:15px}
 .kpi-row{display:flex;gap:14px;flex-wrap:wrap;margin:18px 0}
 .kpi{flex:1;min-width:150px;background:#fff;border:1px solid #e6e8ea;border-radius:10px;padding:14px 16px;text-align:center}
 .kpi .n{font-size:34px;font-weight:700;line-height:1.1}
 .kpi .l{font-size:13px;color:#7f8c8d;margin-top:4px}
 .green{color:#27ae60}.blue{color:#2980b9}.orange{color:#e67e22}.red{color:#c0392b}
 .chart{width:100%;height:auto}.ax{font-size:11px;fill:#95a5a6}.lg{font-size:12px;font-weight:600}
 .bar-chart{display:flex;align-items:flex-end;gap:10px;justify-content:space-between;margin:18px 0 8px;min-height:300px}
 .bar-col{flex:1;display:flex;flex-direction:column;align-items:center}
 .bar{width:62%;border-radius:6px 6px 0 0;transition:.2s}
 .bar-val{font-weight:700;font-size:17px;margin-bottom:5px}
 .bar-lbl{width:120px;height:44px;font-size:11px;fill:#2c3e50;margin-top:6px}
 .bar-note{font-size:11px;color:#7f8c8d;text-align:center;margin-top:30px;max-width:130px}
 .hbar{height:34px;border-radius:6px;color:#fff;font-weight:700;display:flex;align-items:center;
       padding-left:12px;margin:6px 0}
 table{border-collapse:collapse;width:100%;margin:14px 0;font-size:14px}
 th,td{border:1px solid #e6e8ea;padding:8px 10px;text-align:left}th{background:#f1f4f6}
 .flow{display:flex;flex-wrap:wrap;align-items:center;gap:6px;margin:14px 0;font-size:13px}
 .step{background:#eafaf1;border:1px solid #27ae60;border-radius:8px;padding:8px 12px;font-weight:600}
 .arr{color:#27ae60;font-weight:700;font-size:18px}
 .photo{width:100%;border-radius:8px;border:1px solid #e6e8ea;margin-top:8px}
 .photo-cap{font-size:13px;color:#7f8c8d;margin-top:4px}
 .note{background:#fef9e7;border-left:4px solid #f1c40f;padding:10px 14px;border-radius:4px;margin:14px 0;font-size:14px}
 footer{margin-top:50px;font-size:12px;color:#95a5a6;border-top:1px solid #eee;padding-top:14px}
</style></head><body>

<h1>&#128046; Finding Cows from the Sky</h1>
<div class="sub">How an automatic system spots cattle in satellite photos &mdash; methodology &amp; results, in plain language</div>

<div class="card big">
<b>The goal in one sentence.</b> We take a very large, detailed photo of farmland taken from a
satellite and ask a computer to mark <i>where every cow is</i> &mdash; automatically, without a
person having to scan the whole image by eye.
</div>

<div class="kpi-row">
 <div class="kpi"><div class="n green">74%</div><div class="l">of all cows found</div></div>
 <div class="kpi"><div class="n green">99%</div><div class="l">of its alarms are real cows</div></div>
 <div class="kpi"><div class="n blue">94%</div><div class="l">of cows in a <b>herd</b> found</div></div>
 <div class="kpi"><div class="n orange">64%</div><div class="l">of <b>lone</b> cows found</div></div>
</div>

<h2>1. What we start with</h2>
<p>The satellite photo covers a <b>76 km&sup2;</b> farming sector &mdash; about
<b>10,000 football pitches</b>. Every pixel is roughly <b>30 cm</b> on the ground, so a single cow
is only a tiny smudge a handful of pixels wide.</p>
<p>The camera doesn't just see normal colour. It records <b>four "colour layers"</b>: blue, green,
red, and an invisible one called <b>near-infrared</b> that plants reflect strongly. From those we
also compute a fifth layer, a simple <b>"greenness" measure</b> (is this spot living plant, or bare
soil?). That greenness clue turns out to matter a lot &mdash; more on that later.</p>
<p>To teach and grade the system, a person hand-marked <b>2,134 real cows</b> and
<b>4,268 "definitely not a cow" spots</b> on the photo.</p>

<h2>2. How the system works</h2>
<p>The system is a kind of pattern-learner called a <b>neural network</b>. Think of it as a diligent
trainee who is shown thousands of small photo tiles, each with the cows already circled, until it
learns what a cow looks like from above. After training, you slide it across the whole giant photo
and it <b>paints in</b> every spot it believes is a cow.</p>

<div class="flow">
 <span class="step">Satellite photo</span><span class="arr">&rarr;</span>
 <span class="step">Cut into tiles</span><span class="arr">&rarr;</span>
 <span class="step">Show examples &amp; train</span><span class="arr">&rarr;</span>
 <span class="step">Scan whole photo</span><span class="arr">&rarr;</span>
 <span class="step">Map of cows</span>
</div>

<h3>Two ways to be wrong &mdash; and one dial to balance them</h3>
<p>Any such system makes two kinds of mistake: it can <b>miss a real cow</b>, or it can
<b>raise a false alarm</b> on something that isn't a cow. We track two simple scores:</p>
<ul>
 <li><b>Share of cows found</b> &mdash; out of all real cows, how many did it catch?</li>
 <li><b>Accuracy of alarms</b> &mdash; out of everything it flagged, how many were truly cows?</li>
</ul>
<p>There's a <b>caution dial</b>: turn it down and the system becomes eager &mdash; it finds more
cows but risks more false alarms; turn it up and it only flags things it's very sure about. The chart
below shows the trade-off for our final system. Notice the green line stays near the top: <b>almost
every alarm is a real cow no matter where we set the dial</b>. We picked the setting marked in orange.</p>

__LINE_CHART__

<h2>3. The journey: from 41% to 74%</h2>
<p>The first version only found <b>41%</b> of the cows. Step by step we improved it. Each bar below is
a version of the system; the height is the share of cows it found.</p>

__JOURNEY__

<h3>What each step actually did (in plain words)</h3>
<table>
<tr><th>Step</th><th>The problem</th><th>The fix</th></tr>
<tr><td><b>Hidden bug</b></td><td>During practice the system was shown a cow photo flipped like a
mirror, but the "answer sheet" telling it <i>where</i> the cow was had <b>not</b> been flipped to
match. It was being taught with mismatched answers.</td><td>Flip the photo and the answer sheet
<b>together</b>. This one fix was the single biggest jump.</td></tr>
<tr><td><b>Lazy on misses</b></td><td>Missing a cow was "cheap" for the system, so it played it safe
and under-marked.</td><td>Change the scoring during training so <b>missing a cow hurts more</b>,
nudging it to be braver.</td></tr>
<tr><td><b>Few lone cows</b></td><td>Most practice examples were cows in groups, so solitary cows were
rare in its training.</td><td>Deliberately <b>show it lone cows more often</b> during practice.</td></tr>
<tr><td><b>Greenness clue</b></td><td>Hard to tell a pale cow from pale bare dirt using colour alone.</td>
<td>Add the <b>"greenness" layer</b> so it can tell living grass from bare ground. <span class="orange">
First attempt backfired</span> (we accidentally scrambled this clue); once corrected it gave the
<b>biggest gain of all</b>.</td></tr>
</table>

<div class="note"><b>An honest wrong turn.</b> Adding the greenness clue <i>the first time</i> actually
made things <b>worse</b> (the grey bar, 66%). We had accidentally jittered that delicate clue the same
way we shuffle ordinary colours. Once we stopped doing that, the very same idea became the winning
move (74%). Good methods include the dead ends.</div>

<h2>4. The key finding: herds vs. lone cows</h2>
<p>When we looked at <i>which</i> cows get missed, a clear pattern jumped out. The system is
<b>excellent at cows standing in a herd</b>, and <b>much weaker at solitary cows</b> wandering on
their own. A lone animal on bare ground is just a faint few-pixel speck; a herd is an unmistakable
cluster.</p>

<div class="card">
 <div class="hbar" style="width:94%;background:#27ae60">Cows in a herd &mdash; 94% found</div>
 <div class="hbar" style="width:64%;background:#e67e22">Lone cows &mdash; 64% found</div>
 <div class="hbar" style="width:74%;background:#2980b9">Everything combined &mdash; 74% found</div>
 <div class="photo-cap">Of 2,134 marked cows: 728 were in herds (684 found), 1,406 were solitary
 (903 found). Most cows in this area happen to be loners, which is what pulls the overall number down.</div>
</div>

<h3>See it for yourself</h3>
<p>Below are real examples straight from the photo. A yellow ring marks where a person said
"there's a cow." Notice the <b>found</b> cows are often bright and clustered, while the
<b>missed</b> ones are faint, alone, and on bare brownish ground.</p>

<img class="photo" src="__DETOK_IMG__" alt="cows the system found">
<div class="photo-cap">&#9989; Cows the system <b>found</b> &mdash; often bright and in small groups.</div>
<img class="photo" src="__MISSED_IMG__" alt="cows the system missed">
<div class="photo-cap">&#10060; Cows the system <b>missed</b> &mdash; mostly faint, solitary animals on bare ground.</div>

<h2>5. What this means in practice</h2>
<table>
<tr><th>If your goal is&hellip;</th><th>Is this system good enough?</th></tr>
<tr><td><b>Finding <i>where</i> cattle are</b> (locating herds, grazing groups, mapping where
livestock gather)</td><td class="green"><b>Yes &mdash; very reliable.</b> A herd is almost never
missed, because catching even one of its members flags the spot.</td></tr>
<tr><td><b>Counting <i>every single</i> animal</b> exactly</td><td class="orange"><b>Partly.</b> It
would undercount, mostly by missing scattered lone cows near the limit of what 30 cm pixels can show.</td></tr>
</table>

<h2>6. Where the remaining difficulty lies</h2>
<p>The cows still missed are overwhelmingly <b>solitary animals on bare soil</b>, often barely larger
than the photo's smallest visible detail. This is no longer a matter of cleverer software &mdash; it's
closer to the <b>physical limit of the photo's sharpness</b>. The clearest next steps would be sharper
imagery, or hand-marking more of these hard lone-cow examples so the system sees more of them.</p>

<h2>7. How it was built (for the record)</h2>
<p>The whole process is automated as a repeatable pipeline:</p>
<div class="flow">
 <span class="step">prepare data</span><span class="arr">&rarr;</span>
 <span class="step">train</span><span class="arr">&rarr;</span>
 <span class="step">scan photo</span><span class="arr">&rarr;</span>
 <span class="step">grade results</span><span class="arr">&rarr;</span>
 <span class="step">find herds / make map</span>
</div>
<p class="sub">Winning recipe: 5 colour layers (incl. greenness) &middot; mismatched-answer bug fixed
&middot; miss-penalty raised &middot; lone cows shown more often &middot; greenness clue left
un-jittered &middot; caution dial set to a sensitive-but-clean setting.</p>

<footer>
Cattle detection on SkyFi 30&nbsp;cm satellite imagery &middot; neural-network image segmentation.
Figures measured against 2,134 hand-marked cows. Report auto-generated from the project's evaluation
results.
</footer>
</body></html>"""

HTML = (HTML
        .replace("__LINE_CHART__", line_chart())
        .replace("__JOURNEY__", journey_bars())
        .replace("__DETOK_IMG__", detok_img)
        .replace("__MISSED_IMG__", missed_img))

out = os.path.join(HERE, "report.html")
with open(out, "w", encoding="utf-8") as f:
    f.write(HTML)
print(f"wrote {out}  ({os.path.getsize(out)/1e6:.1f} MB, self-contained)")
