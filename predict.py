"""
Detect cattle in a multi-band raster using the trained U-Net model.

Slides a 64x64 window across the stacked image, predicts per-pixel cow
probability, averages overlapping predictions, and writes a classified
GeoTIFF (uint8, 0=background, 1=cattle).

Usage:
    predict.bat
    predict.bat --stride 32 --threshold 0.5
    predict.bat --output D:\\path\\to\\output.tif
"""

import argparse
import json
import os
import sys
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import warnings
warnings.filterwarnings("ignore")

import h5py
import numpy as np
import tensorflow as tf
from tensorflow import keras
from osgeo import gdal

import ndvi_util
import run_logging

gdal.UseExceptions()

HERE = os.path.dirname(os.path.abspath(__file__))
EXTERNAL_DIR = os.environ.get("TF_COWDETECT_ROOT", HERE)
MODELS_DIR = os.path.join(EXTERNAL_DIR, "models")
DEFAULT_MODEL = os.path.join(MODELS_DIR, "cowdetect_unet.keras")
DEFAULT_META = os.path.join(MODELS_DIR, "metadata.json")

DEFAULT_IMAGE_DIR = os.environ.get("TF_COWDETECT_IMAGES", os.path.join(EXTERNAL_DIR, "source", "input_images"))
DEFAULT_OUTPUT = os.path.join(HERE, "detected_cows.tif")

BAND_SUFFIXES = ["blue", "green", "red", "nir"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--image-dir", default=DEFAULT_IMAGE_DIR, help="Directory with single-band TIFs")
    p.add_argument("--output", default=DEFAULT_OUTPUT, help="Output classified GeoTIFF")
    p.add_argument("--model", default=DEFAULT_MODEL, help="Path to .keras model")
    p.add_argument("--metadata", default=DEFAULT_META, help="Path to metadata.json")
    p.add_argument("--stride", type=int, default=32, help="Prediction stride (default=32, 50%% overlap)")
    p.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for cow class")
    p.add_argument("--batch", type=int, default=64, help="Prediction batch size")
    p.add_argument("--save-probs", action="store_true",
                    help="Also save raw probability map (Float32 GeoTIFF) for threshold tuning in QGIS")
    return p.parse_args()


def find_band_files(image_dir: str) -> list[str]:
    tifs = [f for f in os.listdir(image_dir) if f.endswith(".tif")]
    band_files = []
    for suffix in BAND_SUFFIXES:
        matches = [f for f in tifs if f"_{suffix}_" in f.lower()]
        if len(matches) != 1:
            raise FileNotFoundError(f"Expected 1 file for band '{suffix}', found {len(matches)}")
        band_files.append(os.path.join(image_dir, matches[0]))
    return band_files


def main() -> int:
    args = parse_args()
    run_logging.setup_tee("predict")
    print("=" * 60)
    print("  TF_CowDetection  |  PREDICTION (U-Net)")
    print("=" * 60)

    # --- Load metadata ---
    if not os.path.exists(args.metadata):
        print(f"ERROR: metadata.json not found at {args.metadata}")
        return 1
    with open(args.metadata, "r", encoding="utf-8") as f:
        meta = json.load(f)

    patch_size = meta["patch_size"]
    num_bands = meta["num_bands"]
    band_mean = np.array(meta["band_mean"], dtype=np.float32)
    band_std = np.array(meta["band_std"], dtype=np.float32)

    print(f"Model      : {args.model}")
    print(f"Patch size : {patch_size}, stride={args.stride}")
    print(f"Threshold  : {args.threshold}")
    print(f"Band mean  : {band_mean}")
    print(f"Band std   : {band_std}")

    # --- Load model ---
    if not os.path.exists(args.model):
        print(f"ERROR: model not found at {args.model}")
        return 1
    model = keras.models.load_model(args.model, compile=False)

    # --- Open and stack bands ---
    band_files = find_band_files(args.image_dir)
    ds0 = gdal.Open(band_files[0], gdal.GA_ReadOnly)
    w, h = ds0.RasterXSize, ds0.RasterYSize
    gt = ds0.GetGeoTransform()
    proj = ds0.GetProjection()
    print(f"Image      : {w}x{h}, {num_bands} bands")

    # Keep the full stack as native uint16 (~half the RAM of float32). The host
    # has less RAM than the GPU has VRAM, so the full-image float32 stack +
    # accumulators used to OOM; we normalize small float32 batches instead.
    print("Reading bands (uint16, native) ...", end="", flush=True)
    img = np.empty((h, w, num_bands), dtype=np.uint16)
    for i, bf in enumerate(band_files):
        ds = gdal.Open(bf, gdal.GA_ReadOnly)
        img[:, :, i] = ds.GetRasterBand(1).ReadAsArray()
        ds = None
    ds0 = None
    # The model may expect a derived NDVI channel beyond the raw band files.
    # Rebuild it identically to prepare_data.py (must stay in sync).
    n_raw = len(band_files)
    if num_bands == n_raw + 1:
        ndvi_util.fill_ndvi_channel(img, ndvi_idx=n_raw)
        print(" + NDVI", end="")
    elif num_bands != n_raw:
        print(f"\nERROR: model expects {num_bands} bands but found {n_raw} band files "
              f"and no rule to derive the rest.")
        return 1
    print(f" done ({img.nbytes / 1e9:.1f} GB held in RAM)")

    # --- Sliding window prediction with averaging ---
    # count stays uint16 (overlap is tiny) to save another full-image array.
    prob_sum = np.zeros((h, w), dtype=np.float32)
    count = np.zeros((h, w), dtype=np.uint16)

    row_starts = list(range(0, h - patch_size + 1, args.stride))
    col_starts = list(range(0, w - patch_size + 1, args.stride))
    centres = [(r, c) for r in row_starts for c in col_starts]
    n_patches = len(centres)
    print(f"Patches    : {n_patches:,} ({len(row_starts)} rows x {len(col_starts)} cols)")

    t0 = time.time()
    processed = 0

    for batch_start in range(0, n_patches, args.batch):
        batch_centres = centres[batch_start : batch_start + args.batch]
        batch_patches = np.empty((len(batch_centres), patch_size, patch_size, num_bands),
                                  dtype=np.float32)
        for i, (r, c) in enumerate(batch_centres):
            batch_patches[i] = img[r:r + patch_size, c:c + patch_size, :]
        # Normalize this batch only (uint16 -> float32 already done by the copy above)
        batch_patches = (batch_patches - band_mean) / band_std

        preds = model.predict(batch_patches, verbose=0)  # (B, 64, 64, 1)
        preds = preds[:, :, :, 0]  # (B, 64, 64)

        for i, (r, c) in enumerate(batch_centres):
            prob_sum[r:r + patch_size, c:c + patch_size] += preds[i]
            count[r:r + patch_size, c:c + patch_size] += 1

        processed += len(batch_centres)
        if processed % (args.batch * 20) == 0 or processed == n_patches:
            elapsed = time.time() - t0
            pct = 100.0 * processed / n_patches
            print(f"  {processed:>10,}/{n_patches:,} ({pct:5.1f}%)  [{elapsed:.0f}s]", flush=True)

    dt = time.time() - t0
    print(f"Prediction done in {dt:.1f}s")
    del img  # free the ~4.6 GB image stack before writing outputs

    # Average probabilities where overlapping, in place (reuse prob_sum to
    # avoid allocating another full-image float32 array).
    np.divide(prob_sum, count, out=prob_sum, where=count > 0)
    prob_avg = prob_sum  # alias: prob_sum now holds the averaged probability

    # Threshold
    result = (prob_avg >= args.threshold).astype(np.uint8)

    # --- Write output GeoTIFF ---
    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(
        args.output, w, h, 1, gdal.GDT_Byte,
        options=["COMPRESS=DEFLATE", "TILED=YES"]
    )
    out_ds.SetGeoTransform(gt)
    out_ds.SetProjection(proj)
    out_band = out_ds.GetRasterBand(1)
    out_band.SetNoDataValue(255)
    out_band.WriteArray(result)
    out_band.FlushCache()
    out_ds = None

    cow_pixels = np.sum(result == 1)
    print(f"\nClassified raster : {args.output}")
    print(f"Cow pixels        : {cow_pixels:,} ({100.0 * cow_pixels / result.size:.4f}%)")
    print(f"Background pixels : {np.sum(result == 0):,}")

    # --- Optionally save probability map ---
    if args.save_probs:
        probs_path = args.output.replace(".tif", "_probs.tif")
        prob_ds = driver.Create(
            probs_path, w, h, 1, gdal.GDT_Float32,
            options=["COMPRESS=DEFLATE", "TILED=YES"]
        )
        prob_ds.SetGeoTransform(gt)
        prob_ds.SetProjection(proj)
        prob_band = prob_ds.GetRasterBand(1)
        prob_band.WriteArray(prob_avg)
        prob_band.FlushCache()
        prob_ds = None
        print(f"Probability map   : {probs_path}")
        print(f"  Use in QGIS: Raster Calculator > \"cow_probs@1\" > 0.4")

    return 0


if __name__ == "__main__":
    sys.exit(main())
