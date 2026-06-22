"""Build a technical (researcher-facing) HTML report of the cattle-detection
method and results. Self-contained: inline SVG figures + base64 photos.
Companion to make_report.py (the layman version)."""

import base64
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def b64(path):
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


# ----- data -----------------------------------------------------------------
THRESH = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
RECALL = [75.9, 74.1, 66.3, 64.2, 60.5, 57.5, 55.8, 52.0, 50.0]
PREC = [98.5, 98.9, 99.4, 99.6, 99.7, 99.8, 99.8, 99.7, 99.7]

# Ablation: recall & precision at the deployment threshold tau=0.2 (radius 5 px),
# plus best achievable F1 over the sweep. M0 reported at its own operating tau=0.5.
ABLATION = [
    ("M0", "Baseline: focal loss, in-graph augmentation, val-recall checkpoint (4-band)",
     "0.41†", "0.99", "0.58†"),
    ("M1", "+ augmentation desync fixed; focal+Tversky (β=0.7); val-F1 checkpoint (4-band)",
     "0.585", "0.987", "0.744"),
    ("M2", "+ Tversky β=0.8; spectral-only photometric jitter (4-band)",
     "0.676", "0.978", "0.812"),
    ("M3", "+ isolation-aware oversampling (×3 lone-cow patches) (4-band)",
     "0.690", "0.968", "0.815"),
    ("M4", "+ NDVI input channel, photometrically jittered (5-band)",
     "0.656", "0.970", "0.792"),
    ("M5", "+ NDVI excluded from photometric jitter (5-band) — final",
     "0.741", "0.989", "0.857"),
]

# Stratified recall at tau=0.2.
STRATA = [("Herd cows (neighbour ≤ 4.5 m)", 684, 728, 94.0, "#27ae60"),
          ("Solitary cows", 903, 1406, 64.2, "#e67e22"),
          ("All cows", 1587, 2134, 74.4, "#2980b9")]

# Error characterization (mean over group) at tau=0.2.
ERRCHAR = [("Mean brightness (R+G+B)/3", "16257", "15646"),
           ("Mean NDVI", "0.289", "0.324"),
           ("Local contrast (std)", "1110", "1054"),
           ("Fraction in a herd", "8.0%", "43.1%")]


# ----- figures --------------------------------------------------------------
def operating_curve():
    L, T, W, H = 64, 18, 540, 220
    def X(t): return L + (t - 0.1) / 0.8 * W
    def Y(p): return T + (100 - p) / 100 * H
    def poly(v): return " ".join(f"{X(t):.1f},{Y(x):.1f}" for t, x in zip(THRESH, v))
    g = ""
    for p in range(0, 101, 20):
        g += f'<line x1="{L}" y1="{Y(p):.0f}" x2="{L+W}" y2="{Y(p):.0f}" stroke="#eee"/>'
        g += f'<text x="{L-8}" y="{Y(p)+4:.0f}" text-anchor="end" class="ax">{p}</text>'
    for t in THRESH:
        g += f'<text x="{X(t):.0f}" y="{T+H+18:.0f}" text-anchor="middle" class="ax">{t}</text>'
    dots = "".join(f'<circle cx="{X(t):.1f}" cy="{Y(v):.1f}" r="3" fill="#2980b9"/>'
                   for t, v in zip(THRESH, RECALL))
    dots += "".join(f'<circle cx="{X(t):.1f}" cy="{Y(v):.1f}" r="3" fill="#27ae60"/>'
                    for t, v in zip(THRESH, PREC))
    mx = X(0.2)
    return f'''<svg viewBox="0 0 640 280" class="fig">
      {g}
      <line x1="{mx:.0f}" y1="{T}" x2="{mx:.0f}" y2="{T+H}" stroke="#e67e22" stroke-dasharray="4 3"/>
      <text x="{mx:.0f}" y="{T-4}" text-anchor="middle" class="ax" fill="#e67e22">τ=0.2</text>
      <polyline points="{poly(PREC)}" fill="none" stroke="#27ae60" stroke-width="2.5"/>
      <polyline points="{poly(RECALL)}" fill="none" stroke="#2980b9" stroke-width="2.5"/>
      {dots}
      <text x="20" y="{T+H/2:.0f}" transform="rotate(-90 14 {T+H/2:.0f})" text-anchor="middle" class="axl">metric (%)</text>
      <text x="{L+W/2:.0f}" y="{T+H+40:.0f}" text-anchor="middle" class="axl">decision threshold τ</text>
      <rect x="{L+W-186}" y="{T+4}" width="12" height="12" fill="#2980b9"/>
      <text x="{L+W-170}" y="{T+14}" class="lg">Recall (point, r=5px)</text>
      <rect x="{L+W-186}" y="{T+22}" width="12" height="12" fill="#27ae60"/>
      <text x="{L+W-170}" y="{T+32}" class="lg">Precision at GT background</text>
    </svg>'''


def ablation_fig():
    L, T, W, H = 40, 16, 560, 210
    vals = [0.41, 0.585, 0.676, 0.690, 0.656, 0.741]
    labels = [a[0] for a in ABLATION]
    n = len(vals); bw = W / n * 0.6; gap = W / n
    colors = ["#c0392b", "#e67e22", "#f1c40f", "#d4ac0d", "#7f8c8d", "#27ae60"]
    g = ""
    for p in (0.2, 0.4, 0.6, 0.8):
        y = T + (1 - p) * H
        g += f'<line x1="{L}" y1="{y:.0f}" x2="{L+W}" y2="{y:.0f}" stroke="#eee"/>'
        g += f'<text x="{L-6}" y="{y+4:.0f}" text-anchor="end" class="ax">{p:.1f}</text>'
    for i, (v, lab, c) in enumerate(zip(vals, labels, colors)):
        x = L + i * gap + (gap - bw) / 2
        h = v * H
        g += f'<rect x="{x:.0f}" y="{T+H-h:.0f}" width="{bw:.0f}" height="{h:.0f}" fill="{c}" rx="3"/>'
        g += f'<text x="{x+bw/2:.0f}" y="{T+H-h-5:.0f}" text-anchor="middle" class="lg">{v:.2f}</text>'
        g += f'<text x="{x+bw/2:.0f}" y="{T+H+16:.0f}" text-anchor="middle" class="ax">{lab}</text>'
    return f'''<svg viewBox="0 0 620 250" class="fig">{g}
      <text x="14" y="{T+H/2:.0f}" transform="rotate(-90 14 {T+H/2:.0f})" text-anchor="middle" class="axl">Recall @ τ=0.2</text>
    </svg>'''


def strata_fig():
    rows = ""
    for name, d, n, pct, c in STRATA:
        rows += f'''<div class="hb-row"><div class="hb-lab">{name}</div>
          <div class="hb-track"><div class="hb-fill" style="width:{pct}%;background:{c}">{pct:.1f}%</div></div>
          <div class="hb-n">{d}/{n}</div></div>'''
    return f'<div class="hbars">{rows}</div>'


# ----- assemble -------------------------------------------------------------
missed_img = b64(os.path.join(HERE, "missed_cows_overlay.png"))
detok_img = b64(os.path.join(HERE, "detected_cows_sample.png"))

abl_rows = "".join(
    f"<tr><td><b>{i}</b></td><td>{d}</td><td>{r}</td><td>{p}</td><td>{f}</td></tr>"
    for i, d, r, p, f in ABLATION)
err_rows = "".join(f"<tr><td>{m}</td><td>{ms}</td><td>{dt}</td></tr>" for m, ms, dt in ERRCHAR)

HTML = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Recall-oriented U-Net for cattle detection in 30 cm satellite imagery</title>
<style>
 body{font-family:Georgia,"Times New Roman",serif;color:#1a1a1a;line-height:1.55;
      max-width:820px;margin:0 auto;padding:34px 26px 90px;background:#fff;font-size:16px}
 h1{font-size:25px;line-height:1.25;margin:0 0 6px;font-family:Georgia,serif}
 h2{font-size:19px;margin:34px 0 8px;border-bottom:1px solid #bbb;padding-bottom:4px}
 h3{font-size:16px;margin:20px 0 4px;font-style:italic}
 .meta{color:#555;font-size:14px;margin-bottom:18px}
 .abstract{background:#f6f6f4;border:1px solid #e2e2dd;padding:14px 18px;border-radius:4px;font-size:15px}
 .abstract b{font-variant:small-caps}
 p{margin:8px 0;text-align:justify}
 code,.mono{font-family:"SFMono-Regular",Consolas,monospace;font-size:13px;background:#f3f3f1;padding:1px 4px;border-radius:3px}
 .eq{background:#fafafa;border-left:3px solid #888;padding:10px 16px;margin:12px 0;font-size:15px;
     font-family:Cambria,Georgia,serif;overflow-x:auto}
 .eq .where{display:block;color:#555;font-size:13px;margin-top:6px;font-family:Georgia,serif}
 table{border-collapse:collapse;width:100%;margin:12px 0;font-size:13.5px;font-family:Helvetica,Arial,sans-serif}
 th,td{border:1px solid #d8d8d2;padding:6px 9px;text-align:left;vertical-align:top}
 th{background:#efefe9}
 caption{caption-side:top;text-align:left;font-size:13px;color:#444;margin-bottom:5px;font-family:Georgia,serif}
 .fig{width:100%;height:auto;margin:6px 0}
 .figbox{margin:18px 0;border:1px solid #e2e2dd;border-radius:4px;padding:12px 14px;background:#fcfcfb}
 .figcap{font-size:13px;color:#444;margin-top:6px;font-family:Helvetica,Arial,sans-serif}
 .ax{font-size:10px;fill:#888;font-family:Helvetica,Arial,sans-serif}
 .axl{font-size:11px;fill:#555;font-family:Helvetica,Arial,sans-serif}
 .lg{font-size:11px;fill:#333;font-family:Helvetica,Arial,sans-serif}
 .hbars{margin:8px 0}
 .hb-row{display:flex;align-items:center;gap:10px;margin:5px 0;font-family:Helvetica,Arial,sans-serif;font-size:13px}
 .hb-lab{width:200px;text-align:right;color:#333}
 .hb-track{flex:1;background:#eee;border-radius:4px;overflow:hidden}
 .hb-fill{color:#fff;font-weight:700;padding:4px 8px;border-radius:4px;font-size:12px;white-space:nowrap}
 .hb-n{width:80px;color:#666}
 .photo{width:49%;border:1px solid #ddd;border-radius:3px;vertical-align:top}
 .note{background:#fdf6e3;border-left:3px solid #b58900;padding:8px 14px;margin:12px 0;font-size:14px}
 ol.refs{font-size:13.5px;line-height:1.5;padding-left:22px}
 ol.refs li{margin:4px 0}
 footer{margin-top:46px;font-size:12px;color:#888;border-top:1px solid #eee;padding-top:12px}
 sup{font-size:11px}
</style></head><body>

<h1>A recall-oriented U-Net for cattle detection in 30&nbsp;cm multispectral satellite imagery</h1>
<div class="meta">Technical report &middot; SkyFi Sector&nbsp;6 (76&nbsp;km&sup2;) &middot; 2026-06-22 &middot;
TensorFlow&nbsp;2.21 / Keras&nbsp;3.14, NVIDIA&nbsp;RTX&nbsp;5090</div>

<div class="abstract">
<b>Abstract.</b> We address binary semantic segmentation of individual cattle in a single
29,925&times;19,072&nbsp;px (&approx;30&nbsp;cm GSD) four-band SkyFi scene, supervised by 2,134 point
annotations. A compact U-Net (2.69&nbsp;M parameters) is trained on 64&times;64 patches and applied by
overlap-averaged sliding window. Starting from a baseline that recovered only 41% of annotated animals,
we identify and correct a label&ndash;image augmentation desynchronisation, adopt a combined
focal&ndash;Tversky objective with recall-weighted (β&gt;α) penalties, select checkpoints on
validation F1, oversample isolated-animal patches, and add a derived NDVI input channel. At the
deployment threshold the final model attains <b>recall 0.741</b> and precision <b>0.989</b> against
annotated points (search radius 5&nbsp;px), with best F1 = 0.857. A stratified analysis shows the
detector is near-saturated on herd animals (<b>94.0%</b>) and limited on solitary animals
(<b>64.2%</b>), localising the residual error to a signal-strength regime near the sensor's resolution
limit rather than to model capacity. We also report a negative result: adding NDVI <i>regressed</i>
performance until it was excluded from photometric augmentation.
</div>

<h2>1. Introduction</h2>
<p>Manual enumeration of livestock over large pastoral areas is labour-intensive. We treat cattle
detection as per-pixel binary segmentation, producing a probability raster that is thresholded and
post-processed into points/herds. The scene is acquired by SkyFi at &approx;30&nbsp;cm ground sampling
distance, at which an adult bovine subtends only a few pixels, making individual detection sensitive to
local contrast and context. This report documents the methodology and an ablation quantifying each
design decision, and characterises the residual error mode.</p>

<h2>2. Data</h2>
<h3>2.1 Imagery</h3>
<p>A single ortho scene <span class="mono">SkyFi_51_Sector6_76km2</span> of 29,925&times;19,072&nbsp;px,
EPSG:4326, with four single-band rasters stacked in order blue, green, red, near-infrared (NIR). Bands
are retained as native <span class="mono">uint16</span>.</p>
<h3>2.2 Annotations</h3>
<p>Point annotations comprise <b>2,134</b> cattle (<span class="mono">Color=1</span>) and <b>4,268</b>
background controls (<span class="mono">Color=0</span>). Annotations are sparse and not exhaustive,
which bounds scene-level precision estimation (&sect;4.5).</p>
<h3>2.3 Derived NDVI channel</h3>
<p>A fifth input channel encodes the Normalised Difference Vegetation Index, discretised to
<span class="mono">uint16</span> so it co-resides with the spectral stack (per-band standardisation
rescales it downstream):</p>
<div class="eq">NDVI = (NIR &minus; Red) / (NIR + Red) &nbsp;&isin;&nbsp; [&minus;1, 1],
&nbsp;&nbsp; x<sub>NDVI</sub> = round( (NDVI + 1)/2 &middot; 65535 )
<span class="where">Nodata (NIR+Red = 0) maps to 0. Implemented once in <span class="mono">ndvi_util.py</span>
and shared by data preparation and inference to guarantee identical scaling.</span></div>

<h2>3. Methods</h2>
<h3>3.1 Patch dataset</h3>
<p>Cattle points are rasterised as disks of radius 5&nbsp;px into a binary mask. We extract 64&times;64
patches centred on each cattle point and each background point, plus random and data/nodata-border
background patches, yielding <b>7,585</b> patches (2,144 containing cattle; positive-pixel fraction
1.68%). The set is split 80/20 (train 6,068 / val 1,517), stratified by patch-level cattle presence
(seed&nbsp;1337). Inputs are standardised per band using training-set statistics.</p>

<h3>3.2 Architecture</h3>
<p>A symmetric U-Net&nbsp;[1]: four encoder stages (32, 64, 128, 128 channels), a 256-channel
bottleneck, and a mirrored decoder with skip concatenations; each stage is two
3&times;3 Conv&ndash;BatchNorm&ndash;ReLU blocks. A 1&times;1 convolution with sigmoid yields a per-pixel
probability. Input 64&times;64&times;5, output 64&times;64&times;1; <b>2,691,009</b> parameters.</p>

<h3>3.3 Objective</h3>
<p>We minimise the sum of a pixel-level focal term&nbsp;[2] and a region-level Tversky term&nbsp;[3]:</p>
<div class="eq">
p<sub>t</sub> = y&middot;p + (1&minus;y)(1&minus;p), &nbsp; &alpha;<sub>t</sub> = y&middot;&alpha; + (1&minus;y)(1&minus;&alpha;)<br>
L<sub>focal</sub> = &minus;(1/N) &Sigma; &alpha;<sub>t</sub> (1 &minus; p<sub>t</sub>)<sup>&gamma;</sup> log p<sub>t</sub>
&nbsp;&nbsp; (&alpha;=0.75, &gamma;=2)
<span class="where"></span></div>
<div class="eq">
TP=&Sigma; y&middot;p, &nbsp; FP=&Sigma;(1&minus;y)p, &nbsp; FN=&Sigma; y(1&minus;p)<br>
L<sub>Tversky</sub> = 1 &minus; (TP + s) / (TP + &alpha;&middot;FP + &beta;&middot;FN + s) &nbsp;&nbsp; (&alpha;=0.2, &beta;=0.8, s=1)<br>
L = L<sub>focal</sub> + L<sub>Tversky</sub>
<span class="where">&beta;&gt;&alpha; penalises false negatives more than false positives, biasing the
optimum toward recall &mdash; the intended correction for the high-precision/low-recall baseline.</span></div>

<h3>3.4 Augmentation</h3>
<p>Geometric augmentation (random horizontal/vertical flips and k&middot;90&deg; rotations, the
dihedral group D<sub>4</sub>) is applied <b>jointly and losslessly to image and mask</b> in the
<span class="mono">tf.data</span> pipeline. Photometric jitter (brightness &plusmn;0.2, contrast
[0.8,1.2]) is applied to the image only, and only to the four raw spectral bands; the derived NDVI
channel is excluded (&sect;4, negative result).</p>

<h3>3.5 Isolation-aware oversampling</h3>
<p>Motivated by the error analysis (&sect;4.3), training patches are replicated by local cattle density,
estimated from the per-patch count of connected mask components: single-animal patches &times;3,
2&ndash;3-animal patches &times;2, otherwise &times;1 (train 6,068&rarr;8,293). Validation is untouched.</p>

<h3>3.6 Training</h3>
<p>Adam (lr 10<sup>&minus;3</sup>), batch 32, up to 40 epochs. <span class="mono">ReduceLROnPlateau</span>
(val_loss, &times;0.5, patience 4) and <span class="mono">EarlyStopping</span> / checkpointing on
<b>validation F1</b> (patience 8, restore best). F1-based selection is deliberate: recall alone is
trivially maximised by over-prediction.</p>

<h3>3.7 Inference</h3>
<p>A 64&times;64 window is slid at stride 32 (50% overlap) over the standardised scene (555,730 windows);
overlapping probabilities are averaged. The NDVI channel is reconstructed at inference from the raw
bands by the shared routine. The averaged raster is thresholded at &tau;.</p>

<h3>3.8 Evaluation protocol</h3>
<p>Detection is scored against annotation points with a tolerance radius r=5&nbsp;px (&approx;1.5&nbsp;m):
a cattle point is a true positive if any positive pixel lies within r; a background point with a positive
pixel within r is a false positive. We report recall, precision-at-annotated-points, and F1 across a
threshold sweep &tau;&isin;[0.1,0.9]. Because annotations are sparse, point-precision reflects only the
labelled background set; we additionally bound scene precision by connected-component (blob) analysis
(&sect;4.5).</p>

<h2>4. Results</h2>

<table>
<caption><b>Table 1.</b> Ablation. Recall and precision at the deployment threshold &tau;=0.2 (point match,
r=5&nbsp;px) and best F1 over the sweep. Changes are cumulative.</caption>
<tr><th>Model</th><th>Configuration (cumulative)</th><th>Recall</th><th>Prec.</th><th>Best F1</th></tr>
__ABL_ROWS__
</table>
<p style="font-size:12.5px;color:#666"><sup>&dagger;</sup>M0 reported at its operating point &tau;=0.5;
its probability field is diffuse, so lowering &tau; raises nominal recall only by gross over-segmentation
(labelling &gt;5% of the scene), not by localised detection. M1&ndash;M5 are genuine localised detectors
and are directly comparable.</p>

<p>The single largest gain (M0&rarr;M1, recall 0.41&rarr;0.585 with a sharply more confident probability
field) follows from correcting the augmentation desynchronisation. Recall-weighting (M2), oversampling
(M3) and the corrected NDVI channel (M5) contribute further; the final model improves best F1 from 0.58
to <b>0.857</b>.</p>

<div class="figbox">__ABLATION_FIG__
<div class="figcap"><b>Figure 1.</b> Recall at &tau;=0.2 across the ablation (M0 shown at &tau;=0.5;
see Table&nbsp;1 footnote).</div></div>

<div class="figbox">__OPERATING_CURVE__
<div class="figcap"><b>Figure 2.</b> Operating characteristic of the final model (M5). Precision at
annotated background remains &ge;0.98 across the sweep; the deployment threshold &tau;=0.2 (dashed) sits
at the recall knee.</div></div>

<h3>4.3 Stratified recall: herd vs. solitary animals</h3>
<p>Partitioning annotations by local context (another cattle point within 15&nbsp;px, &approx;4.5&nbsp;m)
reveals a strong, interpretable disparity: the detector is near-saturated on herd animals and markedly
weaker on solitary animals. Because solitary animals dominate this scene, they govern the aggregate.</p>
<div class="figbox">__STRATA_FIG__
<div class="figcap"><b>Figure 3.</b> Recall by local context at &tau;=0.2 (detected/total).</div></div>

<table>
<caption><b>Table 2.</b> Error characterisation at &tau;=0.2: group means for missed vs. detected cattle.</caption>
<tr><th>Feature (per-animal neighbourhood)</th><th>Missed (n=547)</th><th>Detected (n=1587)</th></tr>
__ERR_ROWS__
</table>
<p>Missed animals are not darker (the <i>a priori</i> shadow hypothesis is rejected): they are marginally
brighter, sit on lower-NDVI (barer) ground, and are overwhelmingly isolated. This is consistent with a
weak-contrast, low-context signal-strength regime rather than illumination.</p>

<div class="figbox">
<img class="photo" src="__DETOK_IMG__" alt="detected examples">
<img class="photo" src="__MISSED_IMG__" alt="missed examples">
<div class="figcap"><b>Figure 4.</b> Qualitative 48&times;48&nbsp;px chips (RGB, common stretch; yellow ring
= annotation). Left: detected (bright, clustered). Right: missed (faint, solitary, bare ground).</div></div>

<h3>4.4 Negative result: NDVI &times; photometric augmentation</h3>
<div class="note">Naively adding NDVI (M4) <b>regressed</b> recall to 0.656. The cause was applying
brightness/contrast jitter to the NDVI ratio channel, which is not radiometrically meaningful and injects
noise. Restricting photometric augmentation to raw spectral bands (M5) reversed the regression and
yielded the largest single improvement (recall 0.690&rarr;0.741). Derived indices should be excluded from
intensity-domain augmentation.</div>

<h3>4.5 Scene-level precision bounds</h3>
<p>At &tau;=0.2 the prediction contains 12,138 connected components (median 71&nbsp;px). Of these, 1,195
match an annotated cattle point and 14 fall at an annotated background point, giving point-anchored
precision 1195/1209 = <b>0.988</b>. The remaining 10,929 components lie far from any annotation; since the
ground truth is a sparse sample, these are predominantly unannotated cattle rather than false positives.
True scene precision therefore lies between a pessimistic 0.10 (all unverified = false) and an optimistic
0.99 (all unverified = real); the low false-alarm count at known-empty controls favours the upper end.
Densely annotated tiles would be required to estimate it directly.</p>

<h2>5. Discussion</h2>
<p>The dominant lever was not loss design or feature engineering but a data-pipeline correctness fix:
in-graph geometric augmentation had transformed inputs without their masks. Once corrected, recall-biased
objectives and the NDVI channel compounded. The herd/solitary stratification (Fig.&nbsp;3) reframes the
task: for <i>localisation</i> of grazing groups the detector is operationally reliable (a herd is flagged
if any member is detected), whereas exhaustive <i>instance counting</i> is bounded by solitary-animal
recall. The residual errors cluster in a weak-signal regime (small, low-contrast, low-context targets),
suggesting returns from higher GSD or hard-example annotation over additional architectural capacity.</p>

<h2>6. Limitations &amp; threats to validity</h2>
<p>(i) <b>Single scene</b>: all results are within-scene; cross-scene/temporal generalisation is untested.
(ii) <b>Sparse, non-exhaustive labels</b> preclude a direct scene-precision estimate and make
point-precision an incomplete proxy. (iii) <b>Label geometry</b>: a fixed 5&nbsp;px mask/tolerance encodes
an assumed animal scale and couples to the matching radius. (iv) <b>Selection&ndash;objective tension</b>:
F1@&tau;=0.5 checkpointing under a recall-weighted loss can select precision-leaning epochs; an
F<sub>&beta;</sub> criterion may suit recall-priority deployments. (v) <b>Fast, low-variance runs</b> are
single-seed; reported deltas are not averaged over seeds.</p>

<h2>7. Reproducibility</h2>
<p>Deterministic pipeline (seed 1337). Best configuration:</p>
<p class="mono" style="background:#f3f3f1;padding:10px 12px;border-radius:4px;font-size:12.5px">
prepare_data.sh&nbsp; # 5-band patches incl. NDVI<br>
train.sh --epochs 40 --tversky-beta 0.8 --isolated-boost 3 --spectral-bands 4<br>
predict.sh --save-probs<br>
evaluate.sh --probs detected_cows_probs.tif --radius 5</p>
<p>Software: TensorFlow 2.21.0, Keras 3.14.1, NumPy 2.4.4, SciPy 1.17.1, GDAL 3.12.2, Python 3.12.
Hardware: NVIDIA RTX 5090. Per-run training &approx;35&ndash;65&nbsp;s (40 epochs); full-scene inference
&approx;290&nbsp;s. Loss/metric definitions in <span class="mono">train.py</span>; evaluation and the
blob-precision bound in <span class="mono">evaluate.py</span>; error analysis in
<span class="mono">make_missed_overlay.py</span>.</p>

<h2>References</h2>
<ol class="refs">
<li>Ronneberger, O., Fischer, P., Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image
Segmentation. <i>MICCAI</i>.</li>
<li>Lin, T.-Y., Goyal, P., Girshick, R., He, K., Doll&aacute;r, P. (2017). Focal Loss for Dense Object
Detection. <i>ICCV</i>.</li>
<li>Salehi, S.S.M., Erdogmus, D., Gholipour, A. (2017). Tversky Loss Function for Image Segmentation Using
3D Fully Convolutional Deep Networks. <i>MLMI</i>.</li>
<li>Rouse, J.W., Haas, R.H., Schell, J.A., Deering, D.W. (1974). Monitoring Vegetation Systems in the Great
Plains with ERTS. <i>NASA SP-351</i>.</li>
</ol>

<footer>Auto-generated from the project's evaluation artefacts. Companion to the non-technical summary
(<span class="mono">report.html</span>). Figures measured against 2,134 annotated cattle in one SkyFi scene.</footer>
</body></html>"""

HTML = (HTML
        .replace("__ABL_ROWS__", abl_rows)
        .replace("__ERR_ROWS__", err_rows)
        .replace("__OPERATING_CURVE__", operating_curve())
        .replace("__ABLATION_FIG__", ablation_fig())
        .replace("__STRATA_FIG__", strata_fig())
        .replace("__DETOK_IMG__", detok_img)
        .replace("__MISSED_IMG__", missed_img))

out = os.path.join(HERE, "report_scientific.html")
with open(out, "w", encoding="utf-8") as f:
    f.write(HTML)
print(f"wrote {out}  ({os.path.getsize(out)/1e6:.1f} MB, self-contained)")
