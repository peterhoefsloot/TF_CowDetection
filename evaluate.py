"""
Evaluate cow detection results against ground truth annotations.

Compares the predicted raster with the GeoJSON point annotations and
produces a confusion matrix, per-class metrics, and a summary report.

For each ground truth cow point (Color=1):
  - Check if any predicted cow pixel exists within a search radius
  - If yes: True Positive (cow found)
  - If no:  False Negative (cow missed)

For each ground truth background point (Color=0):
  - Check if any predicted cow pixel exists within a search radius
  - If yes: False Positive (false alarm at known BG location)
  - If no:  True Negative (correctly ignored)

Also reports total predicted cow pixels and estimated false positive rate.

Usage:
    evaluate.bat
    evaluate.bat --radius 10
    evaluate.bat --predicted detected_cows.tif --threshold 0.4 --probs detected_cows_probs.tif
"""

import argparse
import json
import os
import sys
import time

import numpy as np
from osgeo import gdal

gdal.UseExceptions()

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PREDICTED = os.path.join(HERE, "detected_cows.tif")
DEFAULT_LABELS = os.environ.get("TF_COWDETECT_LABELS", os.path.join(HERE, "source", "terrain_truth", "GroundTruth_cattlepoints_30cm_20250422_with_background.geojson"))

BAND_SUFFIXES = ["blue", "green", "red", "nir"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--predicted", default=DEFAULT_PREDICTED,
                    help="Predicted classified raster (binary 0/1)")
    p.add_argument("--probs", default=None,
                    help="Probability raster (Float32). If given, --threshold is applied to this instead")
    p.add_argument("--threshold", type=float, default=0.5,
                    help="Threshold for probability raster (only used with --probs)")
    p.add_argument("--labels", default=DEFAULT_LABELS,
                    help="GeoJSON with ground truth point annotations")
    p.add_argument("--radius", type=int, default=5,
                    help="Search radius in pixels around each GT point (default=5, ~1.5m)")
    p.add_argument("--blob-match-radius", type=int, default=10,
                    help="Max pixel distance from a predicted blob centroid to a GT point "
                         "for the scene-wide precision analysis (default=10, ~3m)")
    p.add_argument("--output", default=os.path.join(HERE, "evaluation_report.txt"),
                    help="Output text report path")
    return p.parse_args()


def geo_to_pixel(gt, lon, lat):
    """Convert geographic coordinates to pixel (col, row)."""
    col = (lon - gt[0]) / gt[1]
    row = (lat - gt[3]) / gt[5]
    return int(round(col)), int(round(row))


def check_detection(raster, row, col, radius, h, w):
    """Check if any positive pixel exists within radius of (row, col)."""
    r_start = max(0, row - radius)
    r_end = min(h, row + radius + 1)
    c_start = max(0, col - radius)
    c_end = min(w, col + radius + 1)
    patch = raster[r_start:r_end, c_start:c_end]
    return np.any(patch > 0)


def _min_dist_to_points(centroids, points):
    """Min Euclidean distance (in pixels) from each centroid to a point set."""
    if len(points) == 0:
        return np.full(len(centroids), np.inf)
    pts = np.asarray(points, dtype=np.float64)         # (P, 2) as (row, col)
    out = np.empty(len(centroids), dtype=np.float64)
    for i in range(0, len(centroids), 2000):            # chunk to bound memory
        chunk = centroids[i:i + 2000]
        d = np.sqrt(((chunk[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
        out[i:i + 2000] = d.min(1)
    return out


def scene_precision(predicted, cow_points, bg_points, match_radius):
    """Estimate scene-wide precision via connected-component (blob) analysis.

    The point-based precision only inspects the labeled background points, so it
    is blind to false alarms over the rest of the scene. Here we label every
    predicted blob and classify it by the nearest ground-truth annotation:
      - explained   : a GT cow point lies within match_radius of the centroid
      - fp_at_bg     : a GT background point lies within match_radius
      - unverified   : near no annotation (real-but-unannotated cow OR false alarm)

    Because the GT is sparse (not every cow is annotated) we can only bound the
    true precision; we report the optimistic and pessimistic ends of that range.
    """
    try:
        from scipy import ndimage
    except ImportError:
        return None

    labeled, n_blobs = ndimage.label(predicted > 0)
    if n_blobs == 0:
        return {"n_blobs": 0}

    centroids = np.array(
        ndimage.center_of_mass(predicted > 0, labeled, range(1, n_blobs + 1))
    )  # (n_blobs, 2) as (row, col)
    sizes = np.array(ndimage.sum(np.ones_like(predicted), labeled,
                                 range(1, n_blobs + 1)))

    d_cow = _min_dist_to_points(centroids, cow_points)
    d_bg = _min_dist_to_points(centroids, bg_points)

    explained = d_cow <= match_radius
    fp_at_bg = (~explained) & (d_bg <= match_radius)
    unverified = (~explained) & (~fp_at_bg)

    n_exp = int(explained.sum())
    n_fp = int(fp_at_bg.sum())
    n_unv = int(unverified.sum())

    # Optimistic: treat unverified blobs as real (unannotated) cows.
    prec_opt = n_exp / (n_exp + n_fp) if (n_exp + n_fp) > 0 else 0.0
    # Pessimistic: treat every unverified blob as a false alarm.
    prec_pess = n_exp / n_blobs if n_blobs > 0 else 0.0

    return {
        "n_blobs": n_blobs,
        "median_size": float(np.median(sizes)),
        "explained": n_exp,
        "fp_at_bg": n_fp,
        "unverified": n_unv,
        "prec_opt": prec_opt,
        "prec_pess": prec_pess,
        "match_radius": match_radius,
    }


def main() -> int:
    args = parse_args()
    print("=" * 60)
    print("  TF_CowDetection  |  EVALUATION")
    print("=" * 60)

    t0 = time.time()

    # --- Load predicted raster ---
    if args.probs and os.path.exists(args.probs):
        print(f"Probs raster : {args.probs}")
        print(f"Threshold    : {args.threshold}")
        ds = gdal.Open(args.probs, gdal.GA_ReadOnly)
        prob_data = ds.GetRasterBand(1).ReadAsArray()
        predicted = (prob_data >= args.threshold).astype(np.uint8)
        gt = ds.GetGeoTransform()
        w, h = ds.RasterXSize, ds.RasterYSize
        ds = None
    else:
        print(f"Predicted    : {args.predicted}")
        if not os.path.exists(args.predicted):
            print(f"ERROR: {args.predicted} not found. Run predict.bat first.")
            return 1
        ds = gdal.Open(args.predicted, gdal.GA_ReadOnly)
        predicted = ds.GetRasterBand(1).ReadAsArray()
        gt = ds.GetGeoTransform()
        w, h = ds.RasterXSize, ds.RasterYSize
        ds = None

    print(f"Raster size  : {w}x{h}")
    print(f"Search radius: {args.radius} pixels")

    total_cow_pixels = np.sum(predicted > 0)
    print(f"Total predicted cow pixels: {total_cow_pixels:,}")

    # --- Load ground truth ---
    with open(args.labels, "r", encoding="utf-8") as f:
        gj = json.load(f)
    features = gj["features"]

    # Separate cow and background points
    cow_points = []
    bg_points = []
    skipped = 0
    for feat in features:
        lon, lat = feat["geometry"]["coordinates"][:2]
        col, row = geo_to_pixel(gt, lon, lat)
        color = feat["properties"].get("Color", 0)
        if 0 <= col < w and 0 <= row < h:
            if color == 1:
                cow_points.append((row, col))
            else:
                bg_points.append((row, col))
        else:
            skipped += 1

    print(f"\nGround truth : {args.labels}")
    print(f"  Cow points     : {len(cow_points)}")
    print(f"  Background pts : {len(bg_points)}")
    if skipped > 0:
        print(f"  Skipped (OOB)  : {skipped}")

    # --- Evaluate at cow points ---
    tp = 0  # True Positive: cow detected at cow location
    fn = 0  # False Negative: cow missed
    for row, col in cow_points:
        if check_detection(predicted, row, col, args.radius, h, w):
            tp += 1
        else:
            fn += 1

    # --- Evaluate at background points ---
    fp_at_bg = 0  # False Positive: cow detected at known background
    tn = 0        # True Negative: no cow at known background
    for row, col in bg_points:
        if check_detection(predicted, row, col, args.radius, h, w):
            fp_at_bg += 1
        else:
            tn += 1

    # --- Metrics ---
    total_cow = tp + fn
    total_bg = tn + fp_at_bg
    recall = tp / total_cow if total_cow > 0 else 0
    precision_at_gt = tp / (tp + fp_at_bg) if (tp + fp_at_bg) > 0 else 0
    f1 = 2 * precision_at_gt * recall / (precision_at_gt + recall) if (precision_at_gt + recall) > 0 else 0
    accuracy = (tp + tn) / (total_cow + total_bg) if (total_cow + total_bg) > 0 else 0

    # --- Print report ---
    report = []
    report.append("=" * 60)
    report.append("  CONFUSION MATRIX (point-based)")
    report.append("=" * 60)
    report.append("")
    report.append(f"                    Predicted COW    Predicted BG")
    report.append(f"  Actual COW        {tp:>8,} (TP)    {fn:>8,} (FN)")
    report.append(f"  Actual BG         {fp_at_bg:>8,} (FP)    {tn:>8,} (TN)")
    report.append("")
    report.append("-" * 60)
    report.append(f"  Recall (sensitivity)  : {recall:.4f}  ({tp}/{total_cow} cows detected)")
    report.append(f"  Precision (at GT pts) : {precision_at_gt:.4f}  ({tp}/{tp + fp_at_bg} detections correct)")
    report.append(f"  F1 score              : {f1:.4f}")
    report.append(f"  Accuracy              : {accuracy:.4f}")
    report.append("")
    report.append(f"  Total predicted cow pixels : {total_cow_pixels:,}")
    report.append(f"  Search radius              : {args.radius} pixels")
    if args.probs:
        report.append(f"  Probability threshold      : {args.threshold}")
    report.append("")

    # --- Multiple threshold analysis if probs available ---
    if args.probs and os.path.exists(args.probs):
        report.append("=" * 60)
        report.append("  THRESHOLD SENSITIVITY ANALYSIS")
        report.append("=" * 60)
        report.append("")
        report.append(f"  {'Threshold':>10s}  {'Recall':>8s}  {'Precision':>10s}  {'F1':>8s}  {'Cow pixels':>12s}")
        report.append(f"  {'-'*10}  {'-'*8}  {'-'*10}  {'-'*8}  {'-'*12}")

        for thr in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            pred_thr = (prob_data >= thr).astype(np.uint8)
            cow_pix = np.sum(pred_thr > 0)

            tp_t = sum(1 for r, c in cow_points
                       if check_detection(pred_thr, r, c, args.radius, h, w))
            fp_t = sum(1 for r, c in bg_points
                       if check_detection(pred_thr, r, c, args.radius, h, w))
            fn_t = total_cow - tp_t

            rec_t = tp_t / total_cow if total_cow > 0 else 0
            pre_t = tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else 0
            f1_t = 2 * pre_t * rec_t / (pre_t + rec_t) if (pre_t + rec_t) > 0 else 0
            marker = " <-- current" if abs(thr - args.threshold) < 0.01 else ""
            report.append(f"  {thr:>10.1f}  {rec_t:>8.4f}  {pre_t:>10.4f}  {f1_t:>8.4f}  {cow_pix:>12,}{marker}")

        report.append("")

    # --- Scene-wide precision (blob analysis) ---
    sp = scene_precision(predicted, cow_points, bg_points, args.blob_match_radius)
    report.append("=" * 60)
    report.append("  SCENE-WIDE PRECISION (blob analysis)")
    report.append("=" * 60)
    report.append("")
    if sp is None:
        report.append("  (skipped: scipy not available)")
    elif sp["n_blobs"] == 0:
        report.append("  No predicted blobs at this threshold.")
    else:
        report.append(f"  Predicted blobs (connected) : {sp['n_blobs']:,}")
        report.append(f"  Median blob size (px)       : {sp['median_size']:.0f}")
        report.append(f"  Match radius                : {sp['match_radius']} px")
        report.append("")
        report.append(f"  Explained by a GT cow       : {sp['explained']:,}")
        report.append(f"  At a GT background point    : {sp['fp_at_bg']:,}  (false alarms)")
        report.append(f"  Unverified (no GT nearby)   : {sp['unverified']:,}  (unannotated cow OR FP)")
        report.append("")
        report.append(f"  Precision (optimistic)      : {sp['prec_opt']:.4f}  "
                      f"(unverified counted as real)")
        report.append(f"  Precision (pessimistic)     : {sp['prec_pess']:.4f}  "
                      f"(unverified counted as false)")
        report.append(f"  --> true scene precision lies in this range; GT is sparse so")
        report.append(f"      most 'unverified' blobs are likely unannotated cows.")
    report.append("")

    dt = time.time() - t0
    report.append(f"Evaluation done in {dt:.1f}s")

    # Print and save
    report_text = "\n".join(report)
    print(report_text)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\nReport saved: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
