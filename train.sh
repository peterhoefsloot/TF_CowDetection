#!/usr/bin/env bash
# Launch train.py for TF_CowDetection on Linux (GPU).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# Virtualenv with TF 2.21 + CUDA (keeps the LD_LIBRARY_PATH fix in activate).
VENV="${TFLC_VENV:-$HOME/ml/.venv}"
# shellcheck disable=SC1091
source "$VENV/bin/activate"

export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0
# data/ and models/ live under the root; source/ holds imagery, labels, scenes.
export TF_COWDETECT_ROOT="${TF_COWDETECT_ROOT:-$HERE}"
export TF_COWDETECT_IMAGES="${TF_COWDETECT_IMAGES:-$HERE/source/input_images}"
export TF_COWDETECT_LABELS="${TF_COWDETECT_LABELS:-$HERE/source/terrain_truth/GroundTruth_cattlepoints_30cm_20250422_with_background.geojson}"
export TF_COWDETECT_SCENES="${TF_COWDETECT_SCENES:-$HERE/source/scenes}"

python train.py "$@"
