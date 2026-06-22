"""
Pre-process SkyFi imagery + cow detections into browsable PNG tiles
for the false-positive annotation editor.

Cuts the image into 500x500 pixel tiles, overlays detected cow blobs
as red circles, and saves as PNGs. Also generates a tiles_index.json
with tile metadata (bounds, pixel offsets) for the web editor.

Usage:
    make_tiles.bat
    make_tiles.bat --probs detected_cows_v2_probs.tif --threshold 0.2
    make_tiles.bat --tile-size 1000
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
DEFAULT_IMAGE_DIR = os.environ.get("TF_COWDETECT_IMAGES", os.path.join(HERE, "source", "input_images"))
DEFAULT_PROBS = os.path.join(HERE, "detected_cows_v2_probs.tif")
DEFAULT_OUTPUT_DIR = os.path.join(HERE, "tiles")

BAND_SUFFIXES = ["red", "green", "blue"]  # RGB order for display


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--image-dir", default=DEFAULT_IMAGE_DIR,
                    help="Directory with separate band TIFs (blue/green/red/nir)")
    p.add_argument("--image", default=None,
                    help="Single multi-band TIF (overrides --image-dir). Bands 1-3 used as RGB.")
    p.add_argument("--probs", default=DEFAULT_PROBS, help="Probability raster")
    p.add_argument("--threshold", type=float, default=0.2)
    p.add_argument("--tile-size", type=int, default=500)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return p.parse_args()


def find_band_file(image_dir: str, suffix: str) -> str:
    tifs = [f for f in os.listdir(image_dir) if f.endswith(".tif")]
    matches = [f for f in tifs if f"_{suffix}_" in f.lower()]
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected 1 file for band '{suffix}', found {len(matches)}")
    return os.path.join(image_dir, matches[0])


def main() -> int:
    args = parse_args()
    print("=" * 60)
    print("  TF_CowDetection  |  MAKE TILES for annotation editor")
    print("=" * 60)

    os.makedirs(args.output_dir, exist_ok=True)
    t0 = time.time()
    ts = args.tile_size

    # --- Read RGB image ---
    if args.image:
        print(f"Reading multi-band image: {args.image}")
        ds0 = gdal.Open(args.image, gdal.GA_ReadOnly)
        gt = ds0.GetGeoTransform()
        w, h = ds0.RasterXSize, ds0.RasterYSize
        n_bands = ds0.RasterCount
        print(f"Image: {w}x{h}, {n_bands} bands")
        # Use first 3 bands as RGB
        rgb = []
        for b in range(1, 4):
            rgb.append(ds0.GetRasterBand(b).ReadAsArray().astype(np.float32))
    else:
        print("Reading separate RGB band files ...")
        rgb = []
        ds0 = None
        for suffix in BAND_SUFFIXES:
            path = find_band_file(args.image_dir, suffix)
            ds = gdal.Open(path, gdal.GA_ReadOnly)
            if ds0 is None:
                ds0 = ds
            rgb.append(ds.GetRasterBand(1).ReadAsArray().astype(np.float32))
        gt = ds0.GetGeoTransform()
        w, h = ds0.RasterXSize, ds0.RasterYSize
        print(f"Image: {w}x{h}")

    # Stack and normalize to 0-255 for display
    rgb_raw = np.stack(rgb, axis=-1)  # (H, W, 3) — keep raw for nodata check
    rgb = rgb_raw.copy()
    for c in range(3):
        p2, p98 = np.percentile(rgb[:, :, c], [2, 98])
        if p98 > p2:
            rgb[:, :, c] = np.clip((rgb[:, :, c] - p2) / (p98 - p2) * 255, 0, 255)
    rgb = rgb.astype(np.uint8)

    # --- Read probability raster ---
    probs = None
    if os.path.exists(args.probs):
        print(f"Reading probability map (threshold={args.threshold}) ...")
        pds = gdal.Open(args.probs, gdal.GA_ReadOnly)
        probs = pds.GetRasterBand(1).ReadAsArray()
        pds = None
        cow_mask = (probs >= args.threshold).astype(np.uint8)
        print(f"  Cow pixels: {np.sum(cow_mask):,}")

        # Find blob centroids for overlay
        from scipy import ndimage
        labelled, n_blobs = ndimage.label(cow_mask)
        centroids = ndimage.center_of_mass(cow_mask, labelled, range(1, n_blobs + 1))
        blob_sizes = ndimage.sum(cow_mask, labelled, range(1, n_blobs + 1))
        # Filter to blobs >= 2 pixels
        cow_centroids = [(int(r), int(c)) for (r, c), s in zip(centroids, blob_sizes) if s >= 2]
        print(f"  Cow blobs (>=2px): {len(cow_centroids):,}")
    else:
        cow_centroids = []
        print("No probability map found, tiles will show imagery only")

    # --- Generate tiles ---
    n_rows = (h + ts - 1) // ts
    n_cols = (w + ts - 1) // ts
    n_tiles = n_rows * n_cols
    print(f"Generating {n_tiles} tiles ({n_rows} rows x {n_cols} cols, {ts}x{ts}px) ...")

    from PIL import Image, ImageDraw

    tiles_index = []
    count = 0

    for row in range(n_rows):
        for col in range(n_cols):
            r_start = row * ts
            c_start = col * ts
            r_end = min(r_start + ts, h)
            c_end = min(c_start + ts, w)

            # Extract tile
            tile_rgb = rgb[r_start:r_end, c_start:c_end, :]

            # Pad if at edge
            th, tw = tile_rgb.shape[:2]
            if th < ts or tw < ts:
                padded = np.zeros((ts, ts, 3), dtype=np.uint8)
                padded[:th, :tw, :] = tile_rgb
                tile_rgb = padded

            # Create PIL image
            img = Image.fromarray(tile_rgb)
            draw = ImageDraw.Draw(img)

            # Count cow detections in this tile
            tile_cows = []
            for cr, cc in cow_centroids:
                if r_start <= cr < r_end and c_start <= cc < c_end:
                    local_r = cr - r_start
                    local_c = cc - c_start

                    # Check if detection is in a black/nodata area using raw (pre-normalized) values
                    r_check = 4  # check pixels in a small window
                    gy0 = max(0, r_start + local_r - r_check)
                    gy1 = min(h, r_start + local_r + r_check + 1)
                    gx0 = max(0, c_start + local_c - r_check)
                    gx1 = min(w, c_start + local_c + r_check + 1)
                    patch_raw = rgb_raw[gy0:gy1, gx0:gx1, :]
                    is_border = float(patch_raw.max()) == 0

                    tile_cows.append((local_c, local_r, is_border))
                    # Draw red circle
                    r_circle = 6
                    draw.ellipse(
                        [local_c - r_circle, local_r - r_circle,
                         local_c + r_circle, local_r + r_circle],
                        outline="red", width=2
                    )

            # Save tile
            tile_name = f"tile_{row:04d}_{col:04d}.png"
            img.save(os.path.join(args.output_dir, tile_name), "PNG")

            # Tile metadata
            lon_start = gt[0] + c_start * gt[1]
            lat_start = gt[3] + r_start * gt[5]
            lon_end = gt[0] + c_end * gt[1]
            lat_end = gt[3] + r_end * gt[5]

            tiles_index.append({
                "file": tile_name,
                "row": row,
                "col": col,
                "px_x": c_start,
                "px_y": r_start,
                "lon_min": min(lon_start, lon_end),
                "lat_min": min(lat_start, lat_end),
                "lon_max": max(lon_start, lon_end),
                "lat_max": max(lat_start, lat_end),
                "n_detections": len(tile_cows),
                "detections": [{"x": x, "y": y, "border": b} for x, y, b in tile_cows],
            })

            count += 1
            if count % 100 == 0 or count == n_tiles:
                print(f"  {count}/{n_tiles} tiles", flush=True)

    # Save index
    index_path = os.path.join(args.output_dir, "tiles_index.json")

    # Also save geotransform for coordinate conversion
    meta = {
        "image_width": w,
        "image_height": h,
        "tile_size": ts,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "geotransform": list(gt),
        "threshold": args.threshold,
        "tiles": tiles_index,
    }
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Count tiles with detections
    tiles_with_cows = sum(1 for t in tiles_index if t["n_detections"] > 0)
    print(f"\nTiles with detections: {tiles_with_cows} / {n_tiles}")
    print(f"Saved to {args.output_dir}")

    dt = time.time() - t0
    print(f"Done in {dt:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
