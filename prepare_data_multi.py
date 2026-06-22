"""
Prepare multi-scene training data for cow detection U-Net.

Scans multiple directories for SkyFi scenes (4-band stacked TIFs or
individual blue/green/red/nir bands), matches ground truth points to
scenes by spatial overlap, extracts patches, and adds diverse background
patches from ALL scenes (even unannotated ones) to reduce false positives.

Usage:
    prepare_data_multi.bat
    prepare_data_multi.bat --max-bg-per-scene 500
    prepare_data_multi.bat --cow-radius 3 --patch-size 128
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
EXTERNAL_DIR = os.environ.get("TF_COWDETECT_ROOT", HERE)
DATA_DIR = os.path.join(EXTERNAL_DIR, "data")

DEFAULT_SCENE_DIRS = os.environ.get("TF_COWDETECT_SCENES", os.path.join(EXTERNAL_DIR, "source", "scenes")).split(os.pathsep)
DEFAULT_LABELS = os.environ.get("TF_COWDETECT_LABELS", os.path.join(EXTERNAL_DIR, "source", "terrain_truth", "GroundTruth_cattlepoints_30cm_20250422_with_background.geojson"))

BAND_SUFFIXES = ["blue", "green", "red", "nir"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--scene-dirs", nargs="+", default=DEFAULT_SCENE_DIRS,
                    help="Directories to scan for scenes")
    p.add_argument("--labels", default=DEFAULT_LABELS,
                    help="GeoJSON with cattle point annotations")
    p.add_argument("--patch-size", type=int, default=64)
    p.add_argument("--cow-radius", type=int, default=5)
    p.add_argument("--max-bg-per-scene", type=int, default=200,
                    help="Max random background patches per scene (even unannotated)")
    p.add_argument("--max-bg-annotated", type=int, default=2000,
                    help="Max BG patches per annotated scene (from GT background points + random)")
    p.add_argument("--output-dir", default=DATA_DIR)
    return p.parse_args()


def find_scenes(scene_dirs: list[str]) -> list[dict]:
    """Find all scenes across directories. Prefer stacked TIFs, fall back to individual bands."""
    scenes = []
    seen = set()

    for d in scene_dirs:
        if not os.path.isdir(d):
            print(f"  WARNING: {d} not found, skipping")
            continue

        tifs = [f for f in os.listdir(d) if f.endswith(".tif")]

        # Find stacked files
        stacked = [f for f in tifs if "_stacked.tif" in f.lower() or "_epsg4326_stacked.tif" in f.lower()]
        for sf in stacked:
            name = sf.replace("_epsg4326_stacked.tif", "").replace("_stacked.tif", "")
            key = f"{d}|{name}"
            if key not in seen:
                seen.add(key)
                scenes.append({
                    "name": name,
                    "dir": d,
                    "stacked": os.path.join(d, sf),
                    "bands": None,
                })

        # Find individual band sets (for scenes without stacked files)
        blue_files = [f for f in tifs if "_blue_" in f.lower() and "_stacked" not in f.lower()]
        for bf in blue_files:
            prefix = bf.split("_blue_")[0]
            name = prefix
            key = f"{d}|{name}"
            if key in seen:
                continue

            band_files = []
            complete = True
            for suffix in BAND_SUFFIXES:
                matches = [f for f in tifs if f.startswith(prefix) and f"_{suffix}_" in f.lower()
                           and "_stacked" not in f.lower()]
                if len(matches) == 1:
                    band_files.append(os.path.join(d, matches[0]))
                else:
                    complete = False
                    break

            if complete:
                seen.add(key)
                scenes.append({
                    "name": name,
                    "dir": d,
                    "stacked": None,
                    "bands": band_files,
                })

    return scenes


def get_scene_bounds(scene: dict) -> tuple:
    """Return (gt, w, h, left, bottom, right, top) for a scene."""
    if scene["stacked"]:
        ds = gdal.Open(scene["stacked"], gdal.GA_ReadOnly)
    else:
        ds = gdal.Open(scene["bands"][0], gdal.GA_ReadOnly)

    gt = ds.GetGeoTransform()
    w, h = ds.RasterXSize, ds.RasterYSize
    left = gt[0]
    top = gt[3]
    right = left + w * gt[1]
    bottom = top + h * gt[5]  # gt[5] is negative
    if bottom > top:
        bottom, top = top, bottom
    ds = None
    return gt, w, h, left, bottom, right, top


def read_scene_data(scene: dict) -> np.ndarray:
    """Read scene as (H, W, 4) array. Uses stacked file if available."""
    if scene["stacked"]:
        ds = gdal.Open(scene["stacked"], gdal.GA_ReadOnly)
        data = ds.ReadAsArray()  # (4, H, W)
        ds = None
        return np.transpose(data, (1, 2, 0))  # (H, W, 4)
    else:
        ds0 = gdal.Open(scene["bands"][0], gdal.GA_ReadOnly)
        w, h = ds0.RasterXSize, ds0.RasterYSize
        dtype = gdal.GetDataTypeName(ds0.GetRasterBand(1).DataType)
        np_dtype = np.uint16 if "16" in dtype else np.float32
        ds0 = None

        img = np.empty((h, w, 4), dtype=np_dtype)
        for i, bf in enumerate(scene["bands"]):
            ds = gdal.Open(bf, gdal.GA_ReadOnly)
            img[:, :, i] = ds.GetRasterBand(1).ReadAsArray()
            ds = None
        return img


def geo_to_pixel(gt, lon, lat):
    col = (lon - gt[0]) / gt[1]
    row = (lat - gt[3]) / gt[5]
    return int(round(col)), int(round(row))


def make_circle_mask(radius: int) -> np.ndarray:
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    return (x * x + y * y) <= radius * radius


def main() -> int:
    args = parse_args()
    print("=" * 60)
    print("  TF_CowDetection  |  PREPARE DATA (Multi-Scene)")
    print("=" * 60)

    os.makedirs(args.output_dir, exist_ok=True)
    half = args.patch_size // 2
    t0 = time.time()

    # --- Find all scenes ---
    print(f"\nScanning {len(args.scene_dirs)} directories for scenes ...")
    scenes = find_scenes(args.scene_dirs)
    print(f"Found {len(scenes)} scenes")
    for s in scenes:
        src = "stacked" if s["stacked"] else "4 bands"
        print(f"  {s['name'][:60]:60s}  [{src}]  {os.path.basename(s['dir'])}")

    # --- Get bounds for each scene ---
    print("\nReading scene bounds ...")
    for s in scenes:
        gt, w, h, left, bottom, right, top = get_scene_bounds(s)
        s["gt"] = gt
        s["w"] = w
        s["h"] = h
        s["bounds"] = (left, bottom, right, top)
        print(f"  {s['name'][:50]:50s}  {w}x{h}  [{left:.4f},{bottom:.4f},{right:.4f},{top:.4f}]")

    # --- Load ground truth ---
    with open(args.labels, "r", encoding="utf-8") as f:
        gj = json.load(f)
    features = gj["features"]
    print(f"\nGround truth: {len(features)} points from {args.labels}")

    # --- Match GT points to scenes ---
    circle = make_circle_mask(args.cow_radius)
    all_patches = []
    all_masks = []
    total_cow_patches = 0
    total_bg_patches = 0
    rng = np.random.default_rng(42)

    for si, scene in enumerate(scenes):
        gt = scene["gt"]
        w, h = scene["w"], scene["h"]
        left, bottom, right, top = scene["bounds"]

        # Find GT points within this scene's bounds
        cow_points = []
        bg_points = []
        for feat in features:
            lon, lat = feat["geometry"]["coordinates"][:2]
            if left <= lon <= right and bottom <= lat <= top:
                col, row = geo_to_pixel(gt, lon, lat)
                if half <= col < w - half and half <= row < h - half:
                    color = feat["properties"].get("Color", 0)
                    if color == 1:
                        cow_points.append((col, row))
                    else:
                        bg_points.append((col, row))

        # Determine if we need to load this scene
        n_gt = len(cow_points) + len(bg_points)
        will_add_random = args.max_bg_per_scene > 0

        if n_gt == 0 and not will_add_random:
            continue

        short_name = scene["name"][:45]
        print(f"\n--- Scene {si + 1}/{len(scenes)}: {short_name} ---")
        print(f"  GT points: {len(cow_points)} cow, {len(bg_points)} bg")

        # Read scene data
        print(f"  Reading image ...", end="", flush=True)
        img = read_scene_data(scene)
        print(f" done ({img.shape[0]}x{img.shape[1]}, dtype={img.dtype})")

        # Create cow mask if there are cow points
        mask = np.zeros((h, w), dtype=np.uint8)
        r = args.cow_radius
        for col, row in cow_points:
            r_start = max(0, row - r)
            r_end = min(h, row + r + 1)
            c_start = max(0, col - r)
            c_end = min(w, col + r + 1)
            cr_start = r_start - (row - r)
            cr_end = cr_start + (r_end - r_start)
            cc_start = c_start - (col - r)
            cc_end = cc_start + (c_end - c_start)
            mask[r_start:r_end, c_start:c_end] |= circle[cr_start:cr_end, cc_start:cc_end]

        # Extract cow patches
        cow_extracted = 0
        for col, row in cow_points:
            p = img[row - half:row + half, col - half:col + half, :]
            m = mask[row - half:row + half, col - half:col + half]
            all_patches.append(p)
            all_masks.append(m)
            cow_extracted += 1
        total_cow_patches += cow_extracted

        # Extract annotated background patches
        bg_extracted = 0
        for col, row in bg_points:
            p = img[row - half:row + half, col - half:col + half, :]
            m = mask[row - half:row + half, col - half:col + half]
            all_patches.append(p)
            all_masks.append(m)
            bg_extracted += 1

        # Random background patches (from all scenes, annotated or not)
        n_random = args.max_bg_per_scene
        if n_gt > 0:
            # Annotated scene: more BG patches allowed
            n_random = max(n_random, args.max_bg_annotated - bg_extracted)

        random_added = 0
        attempts = 0
        while random_added < n_random and attempts < n_random * 10:
            rc = rng.integers(half, w - half)
            rr = rng.integers(half, h - half)
            # Skip if near any annotation
            too_close = False
            for ac, ar in cow_points + bg_points:
                if abs(rc - ac) < args.patch_size and abs(rr - ar) < args.patch_size:
                    too_close = True
                    break
            if not too_close:
                p = img[rr - half:rr + half, rc - half:rc + half, :]
                m = mask[rr - half:rr + half, rc - half:rc + half]
                all_patches.append(p)
                all_masks.append(m)
                random_added += 1
            attempts += 1

        total_bg_patches += bg_extracted + random_added
        print(f"  Extracted: {cow_extracted} cow, {bg_extracted} annotated BG, {random_added} random BG")

        # Free memory
        del img, mask

    # --- Stack and save ---
    n = len(all_patches)
    if n == 0:
        print("\nERROR: No patches extracted")
        return 1

    print(f"\n{'=' * 60}")
    print(f"Total patches: {n:,}")
    print(f"  With cows    : {total_cow_patches:,}")
    print(f"  Background   : {total_bg_patches:,}")

    # Stack — handle mixed dtypes by converting to the first patch's dtype
    ref_dtype = all_patches[0].dtype
    patches = np.stack([p.astype(ref_dtype) for p in all_patches], axis=0)
    masks_arr = np.stack(all_masks, axis=0)[:, :, :, np.newaxis]

    cow_pixel_pct = 100.0 * np.sum(masks_arr) / masks_arr.size
    print(f"  Cow pixel %  : {cow_pixel_pct:.2f}%")

    patches_path = os.path.join(args.output_dir, "patches.npy")
    masks_path = os.path.join(args.output_dir, "masks.npy")
    np.save(patches_path, patches)
    np.save(masks_path, masks_arr)

    dt = time.time() - t0
    print(f"\nSaved {n:,} patches to {patches_path}")
    print(f"Saved {n:,} masks   to {masks_path}")
    print(f"Patches: {patches.shape}, dtype={patches.dtype}, {patches.nbytes / 1e6:.1f} MB")
    print(f"Masks  : {masks_arr.shape}, dtype={masks_arr.dtype}, {masks_arr.nbytes / 1e6:.1f} MB")
    print(f"Done in {dt:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
