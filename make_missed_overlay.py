"""
Visualize which annotated cows the model misses, to expose what they share.

For every ground-truth cow point (Color=1):
  - read a small probability window around it and mark it DETECTED if any
    pixel >= threshold within `radius`, else MISSED (same rule as evaluate.py)
  - read an RGB+NIR chip (windowed, cheap) for display and statistics

Outputs (under the repo root):
  - missed_cows_overlay.png   : contact sheet of missed cows (red border)
  - detected_cows_sample.png  : contact sheet of detected cows (green border)
  - missed_vs_detected.geojson : every cow point tagged status=missed|detected
                                 (drop on the imagery in QGIS)
  - a printed summary comparing missed vs detected on brightness / NDVI /
    local contrast / herd-membership

Usage (via the venv):
    ~/ml/.venv/bin/python make_missed_overlay.py --threshold 0.2 --radius 5
"""

import argparse
import json
import os

import numpy as np
from osgeo import gdal

gdal.UseExceptions()

HERE = os.path.dirname(os.path.abspath(__file__))
DEF_PROBS = os.path.join(HERE, "detected_cows_probs.tif")
DEF_IMAGES = os.environ.get("TF_COWDETECT_IMAGES", os.path.join(HERE, "source", "input_images"))
DEF_LABELS = os.environ.get(
    "TF_COWDETECT_LABELS",
    os.path.join(HERE, "source", "terrain_truth",
                 "GroundTruth_cattlepoints_30cm_20250422_with_background.geojson"),
)
BAND_SUFFIXES = ["blue", "green", "red", "nir"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--probs", default=DEF_PROBS)
    p.add_argument("--image-dir", default=DEF_IMAGES)
    p.add_argument("--labels", default=DEF_LABELS)
    p.add_argument("--threshold", type=float, default=0.2,
                   help="Operating threshold (matches the shipped binary)")
    p.add_argument("--radius", type=int, default=5, help="Detection search radius (px)")
    p.add_argument("--chip", type=int, default=48, help="Display chip size (px)")
    p.add_argument("--grid", type=int, default=60, help="Max chips per contact sheet")
    p.add_argument("--herd-dist", type=int, default=15,
                   help="A cow is 'in a herd' if another cow point is within this many px")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def find_band_files(image_dir):
    tifs = [f for f in os.listdir(image_dir) if f.endswith(".tif")]
    out = []
    for suffix in BAND_SUFFIXES:
        m = [f for f in tifs if f"_{suffix}_" in f.lower()]
        if len(m) != 1:
            raise FileNotFoundError(f"band '{suffix}': found {len(m)}")
        out.append(os.path.join(image_dir, m[0]))
    return out


def read_window(band, col, row, half, w, h):
    """Read a (2*half) square window centred on (col,row), zero-padded at edges."""
    size = 2 * half
    c0, r0 = col - half, row - half
    cc0, rr0 = max(0, c0), max(0, r0)
    cc1, rr1 = min(w, c0 + size), min(h, r0 + size)
    out = np.zeros((size, size), dtype=np.float32)
    if cc1 <= cc0 or rr1 <= rr0:
        return out
    arr = band.ReadAsArray(cc0, rr0, cc1 - cc0, rr1 - rr0).astype(np.float32)
    out[rr0 - r0:rr1 - r0, cc0 - c0:cc1 - c0] = arr
    return out


def main():
    args = parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # --- open rasters ---
    pds = gdal.Open(args.probs, gdal.GA_ReadOnly)
    pband = pds.GetRasterBand(1)
    gt = pds.GetGeoTransform()
    w, h = pds.RasterXSize, pds.RasterYSize

    band_files = find_band_files(args.image_dir)
    _band_ds = {s: gdal.Open(f, gdal.GA_ReadOnly)
                for s, f in zip(BAND_SUFFIXES, band_files)}  # keep refs alive
    bands = {s: ds.GetRasterBand(1) for s, ds in _band_ds.items()}

    # --- cow points ---
    gj = json.load(open(args.labels, encoding="utf-8"))
    cows = []
    for feat in gj["features"]:
        if feat["properties"].get("Color", 0) != 1:
            continue
        lon, lat = feat["geometry"]["coordinates"][:2]
        col = int(round((lon - gt[0]) / gt[1]))
        row = int(round((lat - gt[3]) / gt[5]))
        if 0 <= col < w and 0 <= row < h:
            cows.append((col, row, lon, lat))
    print(f"Cow points in bounds : {len(cows)}")

    cols = np.array([c[0] for c in cows])
    rows = np.array([c[1] for c in cows])

    # --- classify detected/missed + gather per-cow stats ---
    rad = args.radius
    recs = []
    for col, row, lon, lat in cows:
        pw = read_window(pband, col, row, rad + 1, w, h)
        detected = bool(np.any(pw >= args.threshold))

        # core (radius 3) for brightness / NDVI, ring (radius 12) for contrast
        red = read_window(bands["red"], col, row, 12, w, h)
        nir = read_window(bands["nir"], col, row, 12, w, h)
        grn = read_window(bands["green"], col, row, 12, w, h)
        blu = read_window(bands["blue"], col, row, 12, w, h)
        cen = slice(12 - 3, 12 + 3)
        bright = float((red[cen, cen] + grn[cen, cen] + blu[cen, cen]).mean() / 3.0)
        r_c, n_c = red[cen, cen].mean(), nir[cen, cen].mean()
        ndvi = float((n_c - r_c) / (n_c + r_c + 1e-6))
        local = (red + grn + blu) / 3.0
        contrast = float(local.std())

        # herd membership: another cow point within herd-dist px
        d2 = (cols - col) ** 2 + (rows - row) ** 2
        in_herd = bool(np.any((d2 > 0) & (d2 <= args.herd_dist ** 2)))

        recs.append(dict(col=col, row=row, lon=lon, lat=lat, detected=detected,
                         brightness=bright, ndvi=ndvi, contrast=contrast, herd=in_herd))

    missed = [r for r in recs if not r["detected"]]
    detok = [r for r in recs if r["detected"]]
    n = len(recs)
    print(f"Detected             : {len(detok)} ({100*len(detok)/n:.1f}%)")
    print(f"Missed               : {len(missed)} ({100*len(missed)/n:.1f}%)")

    # --- summary stats: missed vs detected ---
    def summarize(group, key):
        v = np.array([g[key] for g in group], dtype=np.float64)
        return v.mean(), np.median(v)

    print("\n" + "=" * 64)
    print(f"  MISSED vs DETECTED  (threshold {args.threshold}, radius {rad})")
    print("=" * 64)
    print(f"  {'metric':<14}{'missed (mean/med)':>22}{'detected (mean/med)':>22}")
    for key in ["brightness", "ndvi", "contrast"]:
        mm, mmed = summarize(missed, key)
        dm, dmed = summarize(detok, key)
        print(f"  {key:<14}{mm:>10.3f}/{mmed:<10.3f}{dm:>11.3f}/{dmed:<10.3f}")
    hm = 100 * np.mean([g["herd"] for g in missed])
    hd = 100 * np.mean([g["herd"] for g in detok])
    print(f"  {'in-herd %':<14}{hm:>21.1f}{hd:>22.1f}")
    print("=" * 64)

    # --- consistent RGB stretch from a sample of all cow chips ---
    rng = np.random.default_rng(args.seed)
    half = args.chip // 2
    sample_for_stretch = [recs[i] for i in rng.choice(n, min(300, n), replace=False)]
    stk = {s: [] for s in ["red", "green", "blue"]}
    for r in sample_for_stretch:
        for s in ["red", "green", "blue"]:
            stk[s].append(read_window(bands[s], r["col"], r["row"], half, w, h))
    lo = {s: np.percentile(np.stack(stk[s]), 2) for s in stk}
    hi = {s: np.percentile(np.stack(stk[s]), 98) for s in stk}

    def rgb_chip(r):
        chans = []
        for s in ["red", "green", "blue"]:
            a = read_window(bands[s], r["col"], r["row"], half, w, h)
            a = np.clip((a - lo[s]) / max(1.0, hi[s] - lo[s]), 0, 1)
            chans.append(a)
        return np.dstack(chans)

    def contact_sheet(group, path, border, title):
        sel = group if len(group) <= args.grid else \
            [group[i] for i in rng.choice(len(group), args.grid, replace=False)]
        ncol = 10
        nrow = int(np.ceil(len(sel) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 1.5, nrow * 1.6))
        axes = np.atleast_2d(axes)
        for ax in axes.ravel():
            ax.axis("off")
        for i, r in enumerate(sel):
            ax = axes[i // ncol, i % ncol]
            ax.imshow(rgb_chip(r))
            ax.add_patch(plt.Circle((half, half), rad, fill=False,
                                    edgecolor="yellow", linewidth=0.7))
            for sp in ax.spines.values():
                sp.set_visible(True); sp.set_color(border); sp.set_linewidth(2.5)
            ax.set_xticks([]); ax.set_yticks([])
            ax.axis("on")
        fig.suptitle(title, fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(path, dpi=110)
        plt.close(fig)
        print(f"wrote {path}  ({len(sel)} of {len(group)} chips)")

    contact_sheet(missed, os.path.join(HERE, "missed_cows_overlay.png"),
                  "red", f"MISSED cows (n={len(missed)}, showing up to {args.grid}) "
                         f"@ thr {args.threshold}")
    contact_sheet(detok, os.path.join(HERE, "detected_cows_sample.png"),
                  "lime", f"DETECTED cows (n={len(detok)}, showing up to {args.grid}) "
                          f"@ thr {args.threshold}")

    # --- GeoJSON overlay for QGIS ---
    fc = {"type": "FeatureCollection", "features": [
        {"type": "Feature",
         "geometry": {"type": "Point", "coordinates": [r["lon"], r["lat"]]},
         "properties": {"status": "detected" if r["detected"] else "missed",
                        "brightness": round(r["brightness"], 1),
                        "ndvi": round(r["ndvi"], 3),
                        "herd": r["herd"]}}
        for r in recs]}
    gpath = os.path.join(HERE, "missed_vs_detected.geojson")
    json.dump(fc, open(gpath, "w"))
    print(f"wrote {gpath}  ({len(missed)} missed / {len(detok)} detected)")


if __name__ == "__main__":
    main()
