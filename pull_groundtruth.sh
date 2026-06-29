#!/usr/bin/env bash
# Pull per-image ground-truth GeoJSONs from Sperwer (SMB2/3 over Tailscale).
# smbprotocol is not in the TF venv (no pip there), so run it via uv in an
# ephemeral env. Password is read from $SMB_PASS or prompted -- never hardcoded.
#   SMB_PASS=... ./pull_groundtruth.sh            # non-interactive
#   ./pull_groundtruth.sh --list                  # prompts for the password
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

UV="${UV:-$HOME/.local/bin/uv}"
command -v "$UV" >/dev/null 2>&1 || UV="$(command -v uv || true)"
if [ -z "${UV:-}" ]; then
  echo "uv not found (expected ~/.local/bin/uv). Install uv or set \$UV." >&2
  exit 127
fi

exec "$UV" run --quiet --with smbprotocol python pull_groundtruth.py "$@"
