"""
Prepare training data for cow detection U-Net.

Steps:
  1. Stack 4 single-band SkyFi TIFs (blue, green, red, nir) into one array
  2. Load GeoJSON cattle point annotations
  3. Rasterize cow points into a binary mask (circle radius = 5 px)
  4. Extract 64x64 image patches + matching mask patches at each annotation
  5. Add random background patches from unannotated areas
  6. Save patches.npy (N, 64, 64, 4) and masks.npy (N, 64, 64, 1)

Usage:
    prepare_data.bat
    prepare_data.bat --cow-radius 3 --patch-size 128
    prepare_data.bat --max-bg-patches 5000
"""

import argparse
import json
import os
import sys
import time

import numpy as np
from osgeo import gdal, ogr

import ndvi_util
import run_logging

gdal.UseExceptions()

HERE = os.path.dirname(os.path.abspath(__file__))
EXTERNAL_DIR = os.environ.get("TF_COWDETECT_ROOT", HERE)
DATA_DIR = os.path.join(EXTERNAL_DIR, "data")

DEFAULT_IMAGE_DIR = os.environ.get("TF_COWDETECT_IMAGES", os.path.join(EXTERNAL_DIR, "source", "input_images"))
DEFAULT_LABELS = os.environ.get("TF_COWDETECT_LABELS", os.path.join(EXTERNAL_DIR, "source", "terrain_truth", "GroundTruth_cattlepoints_30cm_20250422_with_background.geojson"))

# Band file suffixes in stacking order: blue, green, red, nir
BAND_SUFFIXES = ["blue", "green", "red", "nir"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--image-dir", default=DEFAULT_IMAGE_DIR, help="Directory with single-band TIFs")
    p.add_argument("--labels", default=DEFAULT_LABELS, help="GeoJSON with cattle point annotations")
    p.add_argument("--patch-size", type=int, default=64, help="Patch width/height in pixels")
    p.add_argument("--cow-radius", type=int, default=5, help="Radius in pixels to burn around each cow point")
    p.add_argument("--max-bg-patches", type=int, default=3000, help="Max random background patches to add")
    p.add_argument("--no-ndvi", action="store_false", dest="add_ndvi", default=True,
                    help="Disable the derived NDVI input channel (default: NDVI added as a 5th band)")
    p.add_argument("--output-dir", default=DATA_DIR, help="Directory to save .npy files")
    return p.parse_args()


def find_band_files(image_dir: str) -> list[str]:
    """Find the 4 band TIFs in order: blue, green, red, nir."""
    tifs = [f for f in os.listdir(image_dir) if f.endswith(".tif")]
    band_files = []
    for suffix in BAND_SUFFIXES:
        matches = [f for f in tifs if f"_{suffix}_" in f.lower()]
        if len(matches) != 1:
            raise FileNotFoundError(f"Expected 1 file for band '{suffix}', found {len(matches)}: {matches}")
        band_files.append(os.path.join(image_dir, matches[0]))
    return band_files


def geo_to_pixel(gt, lon, lat):
    """Convert geographic coordinates to pixel (col, row)."""
    col = (lon - gt[0]) / gt[1]
    row = (lat - gt[3]) / gt[5]
    return int(round(col)), int(round(row))


def make_circle_mask(radius: int) -> np.ndarray:
    """Create a boolean circle mask of given radius."""
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    return (x * x + y * y) <= radius * radius


def main() -> int:
    args = parse_args()
    run_logging.setup_tee("prepare_data")
    print("=" * 60)
    print("  TF_CowDetection  |  PREPARE DATA")
    print("=" * 60)
    os.makedirs(args.output_dir, exist_ok=True)
    half = args.patch_size // 2
    t0 = time.time()

    # --- Find and open band files ---
    band_files = find_band_files(args.image_dir)
    print("Band files:")
    for i, f in enumerate(band_files):
        print(f"  {BAND_SUFFIXES[i]:5s}: {os.path.basename(f)}")

    ds0 = gdal.Open(band_files[0], gdal.GA_ReadOnly)
    w, h = ds0.RasterXSize, ds0.RasterYSize
    gt = ds0.GetGeoTransform()
    print(f"\nImage size : {w}x{h}, pixel={gt[1]:.10f} deg (~30cm)")

    # --- Read and stack all bands: (H, W, 4) [+ NDVI -> (H, W, 5)] ---
    n_bands = len(BAND_SUFFIXES) + (1 if args.add_ndvi else 0)
    print(f"Reading bands ({n_bands} channels{', incl. NDVI' if args.add_ndvi else ''}) ...",
          end="", flush=True)
    img = np.empty((h, w, n_bands), dtype=np.uint16)
    for i, bf in enumerate(band_files):
        ds = gdal.Open(bf, gdal.GA_ReadOnly)
        if ds.RasterXSize != w or ds.RasterYSize != h:
            print(f"\nERROR: Band {BAND_SUFFIXES[i]} has different dimensions")
            return 1
        img[:, :, i] = ds.GetRasterBand(1).ReadAsArray()
        ds = None
    ds0 = None
    if args.add_ndvi:
        ndvi_util.fill_ndvi_channel(img, ndvi_idx=len(BAND_SUFFIXES))
    print(f" done ({img.nbytes / 1e9:.1f} GB)")

    # --- Load GeoJSON points ---
    with open(args.labels, "r", encoding="utf-8") as f:
        gj = json.load(f)
    features = gj["features"]
    print(f"Annotations: {len(features)} points")

    cow_points = []   # (col, row) for cattle
    bg_points = []    # (col, row) for background annotations
    for feat in features:
        lon, lat = feat["geometry"]["coordinates"][:2]
        col, row = geo_to_pixel(gt, lon, lat)
        color = feat["properties"].get("Color", 0)
        if 0 <= col < w and 0 <= row < h:
            if color == 1:
                cow_points.append((col, row))
            else:
                bg_points.append((col, row))

    print(f"  Cattle points  : {len(cow_points)}")
    print(f"  Background pts : {len(bg_points)}")

    # --- Create binary mask ---
    print(f"Rasterizing cow mask (radius={args.cow_radius}px) ...", end="", flush=True)
    mask = np.zeros((h, w), dtype=np.uint8)
    circle = make_circle_mask(args.cow_radius)
    r = args.cow_radius
    for col, row in cow_points:
        r_start = max(0, row - r)
        r_end = min(h, row + r + 1)
        c_start = max(0, col - r)
        c_end = min(w, col + r + 1)
        # Corresponding slice in the circle template
        cr_start = r_start - (row - r)
        cr_end = cr_start + (r_end - r_start)
        cc_start = c_start - (col - r)
        cc_end = cc_start + (c_end - c_start)
        mask[r_start:r_end, c_start:c_end] |= circle[cr_start:cr_end, cc_start:cc_end]
    print(f" done  (cow pixels: {np.sum(mask == 1):,})")

    # --- Extract patches at annotated points ---
    all_patches = []
    all_masks = []

    def extract_patch(col, row):
        """Extract a patch if it fits within the image bounds."""
        if row - half < 0 or row + half > h or col - half < 0 or col + half > w:
            return False
        p = img[row - half:row + half, col - half:col + half, :]
        m = mask[row - half:row + half, col - half:col + half]
        all_patches.append(p)
        all_masks.append(m)
        return True

    # Patches at cow points
    cow_extracted = 0
    for col, row in cow_points:
        if extract_patch(col, row):
            cow_extracted += 1
    print(f"\nCow patches extracted     : {cow_extracted}")

    # Patches at annotated background points
    bg_extracted = 0
    for col, row in bg_points:
        if extract_patch(col, row):
            bg_extracted += 1
    print(f"Annotated BG patches      : {bg_extracted}")

    # Random additional background patches
    rng = np.random.default_rng(42)
    n_random_bg = max(0, args.max_bg_patches - bg_extracted)
    if n_random_bg > 0:
        print(f"Adding {n_random_bg} random background patches ...", end="", flush=True)
        # Avoid areas near any annotation
        ann_cols = [c for c, r in cow_points + bg_points]
        ann_rows = [r for c, r in cow_points + bg_points]
        random_added = 0
        attempts = 0
        while random_added < n_random_bg and attempts < n_random_bg * 10:
            rc = rng.integers(half, w - half)
            rr = rng.integers(half, h - half)
            # Check not too close to any annotation (min 2*patch_size away)
            min_dist = args.patch_size * 2
            too_close = False
            for ac, ar in zip(ann_cols, ann_rows):
                if abs(rc - ac) < min_dist and abs(rr - ar) < min_dist:
                    too_close = True
                    break
            if not too_close:
                if extract_patch(rc, rr):
                    random_added += 1
            attempts += 1
        print(f" {random_added} added")

    # --- Border background patches (nodata edges cause false positives) ---
    # Find the data/nodata boundary by checking where pixels are zero across all bands
    print("Adding border background patches ...", end="", flush=True)
    # Create a mask of valid (non-zero) pixels using the raw spectral bands only
    # (the derived NDVI channel is non-zero even over nodata, so exclude it).
    valid_data = np.any(img[:, :, :len(BAND_SUFFIXES)] > 0, axis=2)
    border_added = 0
    border_spacing = args.patch_size  # one patch every patch_size pixels along border
    for row in range(half, h - half, border_spacing):
        for col in range(half, w - half, border_spacing):
            patch_valid = valid_data[row - half:row + half, col - half:col + half]
            # Border patch: has BOTH valid and nodata pixels (partial coverage)
            valid_pct = np.mean(patch_valid)
            if 0.05 < valid_pct < 0.95:
                if extract_patch(col, row):
                    border_added += 1
    print(f" {border_added} added")

    # --- Stack into arrays ---
    n = len(all_patches)
    patches = np.stack(all_patches, axis=0)  # (N, 64, 64, 4)
    masks_arr = np.stack(all_masks, axis=0)   # (N, 64, 64)
    masks_arr = masks_arr[:, :, :, np.newaxis] # (N, 64, 64, 1)

    # --- Summary ---
    n_cow_patches = sum(1 for m in all_masks if np.any(m > 0))
    n_bg_patches = n - n_cow_patches
    print(f"\nTotal patches: {n}")
    print(f"  With cows    : {n_cow_patches}")
    print(f"  Background   : {n_bg_patches}")
    print(f"  Cow pixel %  : {100.0 * np.sum(masks_arr) / masks_arr.size:.2f}%")

    # --- Save ---
    patches_path = os.path.join(args.output_dir, "patches.npy")
    masks_path = os.path.join(args.output_dir, "masks.npy")
    np.save(patches_path, patches)
    np.save(masks_path, masks_arr)

    dt = time.time() - t0
    print(f"\nSaved {n} patches to {patches_path}")
    print(f"Saved {n} masks   to {masks_path}")
    print(f"Patches: {patches.shape}, dtype={patches.dtype}, {patches.nbytes / 1e6:.1f} MB")
    print(f"Masks  : {masks_arr.shape}, dtype={masks_arr.dtype}, {masks_arr.nbytes / 1e6:.1f} MB")
    print(f"Done in {dt:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
