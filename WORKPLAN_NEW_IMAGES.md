# Workplan — Analysing a New Image Set

How to take a fresh delivery of SkyFi scenes from intake through to an improved,
validated model. A delivery is normally **several scenes** at once.

## Two-machine split

| Machine | Role |
|---|---|
| **Sperwer** (Windows) | Holds a copy of the GeoTIFFs. The OpenLayers/HTML editor lives here. Operators **manually digitize** cattle into a GeoJSON per image — this is the **ground-truthing** phase. Each point is tagged `Color=1` (cattle) or `Color=0` (background). Reachable over **Tailscale at `100.122.176.20`**. |
| **This box** (PeterAI, Ubuntu, RTX 5090) | Inference + evaluation + retraining. No editor/viewer here anymore. |

The editor's GeoJSONs are **input** to this box, not output. Cattle points train and
score the model; background points suppress false positives.

---

## Phase 0 — Intake (this box + Sperwer)

1. Receive the new scenes. Each scene is either four SkyFi bands
   (blue / green / red / nir, ~30 cm) or one `*_stacked.tif`.
2. On **this box**, give each scene its own directory under `source/scenes/`:
   ```
   source/scenes/<scene_name>/   # band TIFs or <scene>_stacked.tif
   ```
   (`source/scenes/` is the default `TF_COWDETECT_SCENES`.)
3. Copy the **same** imagery to Sperwer for ground-truthing.
4. Sanity-check each scene: **4 bands, CRS EPSG:4326, ~30 cm**. CRS matters —
   ground-truth points are matched to scenes by geographic overlap, so a CRS
   mismatch silently drops every point. (The editor's GeoJSON output is already
   EPSG:4326 lon/lat — verified against the Sperwer samples — so no reprojection
   is needed on the label side.)

## Phase 1 — Ground-truthing (Sperwer)

1. Operators digitize cattle (`Color=1`) and a sample of clear background
   (`Color=0`) for each image in the HTML editor.
2. One GeoJSON per image, named after the scene, e.g.
   `SkyFi_Sector3_2_2513WR37-2_2023-12-02_epsg4326.geojson`. **Highest-value
   annotations: solitary cows on bare / low-NDVI ground** — that is the current
   model's known weak spot, so prioritise labelling those.
3. Return all per-image GeoJSONs to this box.

**Editor output schema** (confirmed from Sperwer samples copied locally to
`examples/ground_truth/` — gitignored, not pushed):
each feature is a `Point` with properties
`{id, Color (1=cow / 0=non-cow), Class ("cow"/"non-cow"), Source ("point_digitizer"), Image (<scene name>)}`,
coordinates in **EPSG:4326 lon/lat** (no `crs` member → GeoJSON default WGS84).
The pipeline keys on `Color`. Note a freshly-opened image can come back **empty**
(0 features) if it wasn't digitized — the merge step below just skips those.

## Phase 2 — Consolidate ground truth (this box)

`prepare_data_multi` and `evaluate` each take **one** `--labels` file, matched to
scenes by spatial overlap. So merge the per-image GeoJSONs into one master file.

1. Pull the per-image GeoJSONs straight off Sperwer into
   `source/terrain_truth/incoming/`:
   ```bash
   SMB_PASS=... ./pull_groundtruth.sh          # or omit SMB_PASS to be prompted
   ./pull_groundtruth.sh --list                # preview what's on Sperwer first
   ```
   It grabs `SkyFi_*_epsg4326.geojson` by default (override with `--pattern`).
   The editor names files `<scene>_epsg4326.geojson`, and each point also carries
   its scene in the `Image` property, so attribution doesn't depend on filenames.
2. Merge them into one combined ground-truth file (empty files are skipped):
   ```bash
   cd ~/TF_CowDetection
   ~/ml/.venv/bin/python - <<'PY'
   import json, glob, os
   out = {"type": "FeatureCollection", "features": []}
   for f in sorted(glob.glob("source/terrain_truth/incoming/*.geojson")):
       fc = json.load(open(f))
       feats = fc.get("features", [])
       print(f"{os.path.basename(f)}: {len(feats)} pts" + (" (empty, skipped)" if not feats else ""))
       out["features"] += feats
   dst = "source/terrain_truth/GroundTruth_combined.geojson"
   os.makedirs(os.path.dirname(dst), exist_ok=True)
   json.dump(out, open(dst, "w"))
   print(f"-> {dst}: {len(out['features'])} total")
   PY
   ```
   Dry-run it now against the local samples by pointing the glob at
   `examples/ground_truth/*.geojson` (verified: 76 features merged, the 1 empty
   scene skipped automatically).
3. The merged points are EPSG:4326 lon/lat (same CRS as the imagery) — confirmed
   for the editor's output, so no reprojection is needed. If a future source ever
   emits a different CRS, reproject before merging.

## Phase 3 — Evaluate the **current** model on the new scenes (baseline)

Score the existing model on the new imagery *before* retraining, so you have a
before/after baseline. Run per scene:

```bash
cd ~/TF_CowDetection
for s in source/scenes/*/; do
  name=$(basename "$s")
  ./predict.sh  --image-dir "$s" --output "predictions/${name}_cows.tif" --save-probs
  ./evaluate.sh --predicted "predictions/${name}_cows.tif" \
                --probs    "predictions/${name}_cows_probs.tif" \
                --labels   "${s}truth.geojson" \
                --output   "evaluation_report_${name}_oldmodel.txt"
done
```

Record scene-wide F1 / recall / precision per scene. (`predict.py` rebuilds the
NDVI 5th channel automatically — only the 4 raw bands need to be on disk.)

## Phase 4 — Retrain with the new data folded in

1. **Back up the current model first** so a regression is reversible:
   ```bash
   cp models/cowdetect_unet.keras models/backups/cowdetect_unet_$(...date...).keras
   cp models/metadata.json        models/backups/
   ```
   (stamp the filename with the delivery date manually.)
2. Build 5-band patches from **all** scenes (old + new) against the merged labels:
   ```bash
   ./prepare_data_multi.sh \
     --scene-dirs source/scenes/* \
     --labels source/terrain_truth/GroundTruth_combined.geojson
   ```
   → `data/patches.npy` + `data/masks.npy` (NDVI added as the 5th channel;
   `--no-ndvi` to disable, but the production model expects it).
3. Train with the production recipe:
   ```bash
   ./train.sh --epochs 40 --tversky-beta 0.8 --isolated-boost 3 --spectral-bands 4
   ./plot_history.sh
   ```
   - `--spectral-bands 4` is **mandatory**: it keeps photometric augmentation off
     the NDVI ratio channel (jittering NDVI corrupts it and regresses recall).
   - Leave `--patience` at its default of **8** — raising it pushes the model
     precision-ward and *lowers* scene-wide recall.
   - Watch live: `tail -f logs/train_*.log`, and `tensorboard --logdir runs`
     → http://localhost:6006.

## Phase 5 — Evaluate the new model & gate the swap

1. Re-run Phase 3's loop with the new model, writing
   `evaluation_report_${name}_newmodel.txt`.
2. Also re-evaluate **previously-held scenes** to catch regression on old data.
3. **Keep the new model only if scene-wide F1/recall improves without tanking
   precision.** Otherwise restore the backup from `models/backups/`.
4. Confirm the operating point: `detected_cows.tif` is written at **threshold 0.2**
   (recall ≈ 0.74, precision ≈ 0.99 on the reference scene).

## Phase 6 — Deliverables

- **Per-scene herd polygons** (clusters of ≥10 cows within 50 m, with counts /
  density), for reporting:
  ```bash
  for s in source/scenes/*/; do
    name=$(basename "$s")
    ./detect_herds.sh --input "predictions/${name}_cows_probs.tif" \
                      --threshold 0.2 \
                      --output "herds_${name}.geojson"
  done
  ```
- Ship rasters (`*_cows.tif`) and/or herd GeoJSONs back to wherever they're consumed.
- Regenerate the HTML report(s) if needed: `~/ml/.venv/bin/python make_report*.py`.

---

## Quick reference

| Step | Command |
|---|---|
| Pull ground truth from Sperwer | `SMB_PASS=... ./pull_groundtruth.sh` (`--list` to preview) |
| Predict (per scene) | `./predict.sh --image-dir <scene> --output <out>.tif --save-probs` |
| Evaluate | `./evaluate.sh --predicted <out>.tif --probs <out>_probs.tif --labels <truth>.geojson` |
| Build training patches | `./prepare_data_multi.sh --scene-dirs source/scenes/* --labels <combined>.geojson` |
| Train | `./train.sh --epochs 40 --tversky-beta 0.8 --isolated-boost 3 --spectral-bands 4` |
| Herds | `./detect_herds.sh --input <out>_probs.tif --threshold 0.2` |

## Gotchas

- **CRS**: incoming GeoJSONs must be EPSG:4326 lon/lat (the editor's output already is), or spatial matching drops points.
- **Empty labels**: an un-digitized image returns a valid GeoJSON with 0 features — the merge skips it; don't mistake it for an error.
- **Sperwer access**: `100.122.176.20` over Tailscale, SMB admin share `d$` (editor + samples live in `d:\tf_cowdetection\tiles`). `curl`'s SMBv1 is refused by Windows; use an SMB2/3 client (`uv run --with smbprotocol …`, or `gio mount`).
- **NDVI**: model is 5-band; always train with `--spectral-bands 4`. `predict` rebuilds NDVI itself.
- **Don't raise `--patience`**; per-patch `val_f1` is anti-correlated with the scene-wide recall that matters.
- **Threshold 0.2** is the production operating point.
- **VRAM (32 GB)** bounds batch size; host RAM is 64 GB.
- The model is a strong **herd/group** locator but a **moderate individual counter** — residual misses are isolated cows near the resolution limit. More solitary-cow ground truth is the real lever.
- The editors (`make_tiles`/`serve_tiles` and the COG `serve_editor`) were removed — viewing/digitizing happens on Sperwer.
