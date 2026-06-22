"""
Detect cattle herds by clustering individual cow point detections using DBSCAN.

Definition: A herd is a cluster of 10+ cow points within 50m of each other.
Uses DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
to group nearby points into herds automatically.

Output: GeoJSON with one convex hull polygon per herd, including:
  - herd_id, cow_count, area_m2, centroid_lon, centroid_lat
  - density (cows per hectare)

Usage:
    detect_herds.bat
    detect_herds.bat --input detected_cows_05_probs.tif --threshold 0.2
    detect_herds.bat --input GroundTruth.geojson
    detect_herds.bat --eps 50 --min-cows 5
"""

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

DEFAULT_GT = os.environ.get("TF_COWDETECT_LABELS", os.path.join(HERE, "source", "terrain_truth", "GroundTruth_cattlepoints_30cm_20250422_with_background.geojson"))
DEFAULT_OUTPUT = os.path.join(HERE, "herds.geojson")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--input", default=DEFAULT_GT,
                    help="Input: GeoJSON with cow points, or probability raster (.tif)")
    p.add_argument("--threshold", type=float, default=0.2,
                    help="Threshold for probability raster (only used with .tif input)")
    p.add_argument("--eps", type=float, default=50.0,
                    help="DBSCAN: max distance (metres) between cows in same herd")
    p.add_argument("--min-cows", type=int, default=10,
                    help="Minimum cows to form a herd")
    p.add_argument("--max-cows", type=int, default=200,
                    help="Discard herds with more cows than this (likely false positives)")
    p.add_argument("--max-density", type=float, default=500.0,
                    help="Discard herds with density above this (cows/ha, physically impossible)")
    p.add_argument("--output", default=DEFAULT_OUTPUT, help="Output GeoJSON path")
    return p.parse_args()


def load_cow_points_geojson(path: str) -> list[tuple[float, float]]:
    """Load cow points (Color=1) from GeoJSON. Returns [(lon, lat), ...]."""
    with open(path, "r", encoding="utf-8") as f:
        gj = json.load(f)
    points = []
    for feat in gj["features"]:
        color = feat["properties"].get("Color", 1)
        if color == 1:
            lon, lat = feat["geometry"]["coordinates"][:2]
            points.append((lon, lat))
    return points


def load_cow_points_raster(path: str, threshold: float) -> list[tuple[float, float]]:
    """Extract cow blob centroids from a probability raster. Returns [(lon, lat), ...].

    Instead of returning every cow pixel (millions), finds connected components
    (blobs) and returns one centroid per blob. This reduces 4M pixels to a few
    thousand points, making DBSCAN feasible.
    """
    from osgeo import gdal
    from scipy import ndimage
    gdal.UseExceptions()

    ds = gdal.Open(path, gdal.GA_ReadOnly)
    gt = ds.GetGeoTransform()
    data = ds.GetRasterBand(1).ReadAsArray()
    ds = None

    # Threshold to binary
    binary = (data >= threshold).astype(np.uint8)
    n_cow_pixels = np.sum(binary)
    print(f"  Cow pixels above threshold {threshold}: {n_cow_pixels:,}")

    # Find connected components (blobs)
    labelled, n_blobs = ndimage.label(binary)
    print(f"  Connected blobs found: {n_blobs:,}")

    # Get centroid of each blob
    centroids = ndimage.center_of_mass(binary, labelled, range(1, n_blobs + 1))
    print(f"  Blob centroids: {len(centroids):,} (one per blob)")

    # Optional: also get blob sizes to filter tiny noise
    blob_sizes = ndimage.sum(binary, labelled, range(1, n_blobs + 1))

    # Convert pixel centroids to geographic coordinates
    # Filter: only keep blobs with at least 2 pixels (skip single-pixel noise)
    points = []
    for i, (r, c) in enumerate(centroids):
        if blob_sizes[i] >= 2:
            lon = gt[0] + (c + 0.5) * gt[1]
            lat = gt[3] + (r + 0.5) * gt[5]
            points.append((lon, lat))

    print(f"  After filtering (>=2 pixels): {len(points):,} cow detections")
    return points


def estimate_metres_per_degree(lat: float) -> tuple[float, float]:
    """Approximate metres per degree at a given latitude."""
    lat_rad = np.radians(lat)
    m_per_deg_lat = 111132.92 - 559.82 * np.cos(2 * lat_rad)
    m_per_deg_lon = 111412.84 * np.cos(lat_rad)
    return m_per_deg_lon, m_per_deg_lat


def convex_hull_2d(points: np.ndarray) -> np.ndarray:
    """Compute convex hull of 2D points. Returns ordered hull vertices."""
    from scipy.spatial import ConvexHull
    if len(points) < 3:
        return points
    try:
        hull = ConvexHull(points)
        return points[hull.vertices]
    except Exception:
        return points


def polygon_area_m2(vertices: np.ndarray, m_per_deg_lon: float, m_per_deg_lat: float) -> float:
    """Compute area of a polygon in square metres using the shoelace formula."""
    # Convert to metres
    x = vertices[:, 0] * m_per_deg_lon
    y = vertices[:, 1] * m_per_deg_lat
    # Shoelace formula
    n = len(x)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += x[i] * y[j]
        area -= x[j] * y[i]
    return abs(area) / 2.0


def main() -> int:
    args = parse_args()
    print("=" * 60)
    print("  TF_CowDetection  |  HERD DETECTION (DBSCAN)")
    print("=" * 60)

    t0 = time.time()

    # --- Load cow points ---
    print(f"\nInput: {args.input}")
    if args.input.lower().endswith(".tif"):
        points = load_cow_points_raster(args.input, args.threshold)
    else:
        points = load_cow_points_geojson(args.input)

    print(f"Total cow points: {len(points):,}")

    if len(points) < args.min_cows:
        print(f"Not enough points for a herd (need {args.min_cows})")
        return 1

    coords = np.array(points)  # (N, 2) — lon, lat

    # --- Convert eps from metres to degrees ---
    mean_lat = np.mean(coords[:, 1])
    m_per_deg_lon, m_per_deg_lat = estimate_metres_per_degree(mean_lat)
    eps_deg = args.eps / min(m_per_deg_lon, m_per_deg_lat)  # conservative
    print(f"DBSCAN: eps={args.eps}m (~{eps_deg:.6f} deg), min_cows={args.min_cows}")

    # --- Run DBSCAN ---
    from sklearn.cluster import DBSCAN

    clustering = DBSCAN(eps=eps_deg, min_samples=args.min_cows, metric="euclidean")
    labels = clustering.fit_predict(coords)

    n_herds = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = np.sum(labels == -1)
    print(f"\nResults:")
    print(f"  Herds found   : {n_herds}")
    print(f"  Cows in herds : {np.sum(labels >= 0):,}")
    print(f"  Isolated cows : {n_noise:,} (not in any herd)")

    if n_herds == 0:
        print("No herds found. Try lowering --eps or --min-cows.")
        return 0

    # --- Build herd polygons ---
    features = []
    print(f"\n  {'Herd':>5s}  {'Cows':>6s}  {'Area (ha)':>10s}  {'Density':>10s}  {'Centroid':>25s}")
    print(f"  {'-'*5}  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*25}")

    for herd_id in range(n_herds):
        herd_mask = labels == herd_id
        herd_points = coords[herd_mask]
        n_cows = len(herd_points)

        centroid_lon = float(np.mean(herd_points[:, 0]))
        centroid_lat = float(np.mean(herd_points[:, 1]))

        # Convex hull
        hull_vertices = convex_hull_2d(herd_points)

        # Close the polygon (first point = last point)
        hull_closed = np.vstack([hull_vertices, hull_vertices[0:1]])

        # Area
        area_m2 = polygon_area_m2(hull_vertices, m_per_deg_lon, m_per_deg_lat)
        area_ha = area_m2 / 10000.0
        density = n_cows / area_ha if area_ha > 0 else 0

        print(f"  {herd_id:>5d}  {n_cows:>6d}  {area_ha:>10.2f}  {density:>8.1f}/ha  "
              f"({centroid_lon:.5f}, {centroid_lat:.5f})")

        # GeoJSON feature
        feature = {
            "type": "Feature",
            "properties": {
                "herd_id": int(herd_id),
                "cow_count": int(n_cows),
                "area_m2": round(area_m2, 1),
                "area_ha": round(area_ha, 2),
                "density_per_ha": round(density, 1),
                "centroid_lon": round(centroid_lon, 6),
                "centroid_lat": round(centroid_lat, 6),
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[round(float(lon), 8), round(float(lat), 8)]
                                 for lon, lat in hull_closed]]
            }
        }
        features.append(feature)

    # --- Filter herds ---
    n_before = len(features)
    filtered = []
    removed_size = 0
    removed_density = 0
    for feat in features:
        props = feat["properties"]
        if props["cow_count"] > args.max_cows:
            removed_size += 1
            continue
        if props["density_per_ha"] > args.max_density:
            removed_density += 1
            continue
        filtered.append(feat)

    if removed_size + removed_density > 0:
        print(f"\nFiltering:")
        print(f"  Removed (>{args.max_cows} cows)       : {removed_size}")
        print(f"  Removed (>{args.max_density} cows/ha)  : {removed_density}")
        print(f"  Remaining herds        : {len(filtered)} (from {n_before})")

    # Renumber herd IDs
    for i, feat in enumerate(filtered):
        feat["properties"]["herd_id"] = i
    features = filtered

    # --- Save GeoJSON ---
    geojson = {
        "type": "FeatureCollection",
        "features": features
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(geojson, f, indent=2)

    dt = time.time() - t0
    n_final = len(features)
    print(f"\nSaved {n_final} herds to {args.output}")
    print(f"Done in {dt:.1f}s")

    # Summary stats
    cow_counts = [feat["properties"]["cow_count"] for feat in features]
    areas = [feat["properties"]["area_ha"] for feat in features]
    print(f"\nSummary:")
    print(f"  Total herds     : {n_final}")
    print(f"  Total cows      : {sum(cow_counts):,}")
    print(f"  Avg herd size   : {np.mean(cow_counts):.1f} cows")
    print(f"  Largest herd    : {max(cow_counts)} cows")
    print(f"  Smallest herd   : {min(cow_counts)} cows")
    print(f"  Total area      : {sum(areas):.2f} ha")

    return 0


if __name__ == "__main__":
    sys.exit(main())
