# TF_CowDetection (Linux / GPU)

U-Net cattle detection on SkyFi 4-band imagery (TensorFlow 2.21 / Keras 3).
Ported from the original Windows/OSGeo4W project to PeterAI (Ubuntu, RTX 5090).

## Layout

```
TF_CowDetection/
├── train.py / predict.py / prepare_data.py / prepare_data_multi.py
├── evaluate.py / detect_herds.py / make_tiles.py / serve_tiles.py / plot_history.py
├── *.sh                     # one launcher per script (activate venv + set paths)
├── requirements.txt / requirements-geo.txt
├── tiles/                   # editor.html + false_positives_background.geojson (+ generated PNGs)
├── data/                    # patches.npy, masks.npy
├── models/                  # cowdetect_unet.keras, metadata.json
└── source/
    ├── input_images/        # SkyFi blue/green/red/nir TIFs (one scene)
    ├── terrain_truth/       # GroundTruth_*.geojson
    └── scenes/              # (empty) drop multi-scene dirs here for prepare_data_multi
```

## Environment

The `.sh` launchers activate `~/ml/.venv` (TF 2.21 + CUDA 12.8, GPU-verified;
keep the `LD_LIBRARY_PATH` block in its `activate` script — it's the TF-GPU fix).
Override the venv with `TFLC_VENV=/path/to/venv`.

Paths are configurable via env vars (defaults shown):
- `TF_COWDETECT_ROOT`   — holds `data/` + `models/` (default: code dir)
- `TF_COWDETECT_IMAGES` — band TIF dir (default: `$ROOT/source/input_images`)
- `TF_COWDETECT_LABELS` — ground-truth geojson (default: `$ROOT/source/terrain_truth/GroundTruth_...geojson`)
- `TF_COWDETECT_SCENES` — multi-scene dirs, `os.pathsep`-separated (default: `$ROOT/source/scenes`)

## Usage

```bash
./train.sh                       # 30 epochs (GPU); --epochs 2 for a smoke test
./plot_history.sh                # training_curves.png
./predict.sh                     # classify source/input_images -> detected_cows.tif
./predict.sh --save-probs        # also write detected_cows_probs.tif
./evaluate.sh                    # confusion matrix vs ground truth
./detect_herds.sh --input detected_cows_probs.tif --threshold 0.2
./make_tiles.sh --probs detected_cows_probs.tif   # PNG tiles for the editor
./serve_tiles.sh                 # http://localhost:8090/editor.html
```

## Notes / migration changes vs Windows

- Hardcoded `D:\` paths replaced by the env vars above; Windows h5py-DLL hack removed.
- **`predict.py` was made RAM-safe**: PeterAI has only ~30 GiB RAM vs 32 GB VRAM.
  The image stack is kept as native uint16 and normalized per-batch (not as a
  full-image float32 copy), and the overlap-average is done in place. Peak host
  RAM ≈ 9 GB for the 29925×19072 scene instead of ~25–30 GB.
- The 940 GB `D:\TF_PROCESSING` multi-scene dirs were **not** copied — put scene
  folders under `source/scenes/` (or pass `--scene-dirs`) to use `prepare_data_multi`.
- The Achini / Google-Maps download tooling was not migrated (stayed on Windows).
- `make_tiles.py` still loads the RGB scene at full size (~16 GB peak); run it
  alone (no training/predict in parallel). Reduce with a smaller scene if needed.
