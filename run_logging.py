"""
Shared run-logging helper for the TF_CowDetection pipeline.

setup_tee() mirrors everything a script prints to stdout/stderr into a
timestamped logfile under <root>/logs/, so a run can be followed live from
another terminal with `tail -f` (handy over RDP/SSH) and inspected afterwards.
Existing print(..., end="", flush=True) progress lines keep working unchanged —
the tee wraps write()/flush(), so carriage-return progress bars tee too.
"""

import os
import sys
from datetime import datetime


class _Tee:
    """Forward writes to the real stream and a logfile."""

    def __init__(self, stream, fh):
        self._stream = stream
        self._fh = fh

    def write(self, data):
        self._stream.write(data)
        self._fh.write(data)
        return len(data)

    def flush(self):
        self._stream.flush()
        self._fh.flush()

    def __getattr__(self, name):  # isatty(), encoding, fileno, ...
        return getattr(self._stream, name)


def _root() -> str:
    return os.environ.get(
        "TF_COWDETECT_ROOT", os.path.dirname(os.path.abspath(__file__))
    )


def setup_tee(script_name: str, root: str | None = None) -> str:
    """Tee stdout+stderr to <root>/logs/<script_name>_<timestamp>.log.

    Returns the log path. Call once near the top of main().
    """
    log_dir = os.path.join(root or _root(), "logs")
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"{script_name}_{ts}.log")
    fh = open(log_path, "a", buffering=1, encoding="utf-8")  # line-buffered
    sys.stdout = _Tee(sys.stdout, fh)
    sys.stderr = _Tee(sys.stderr, fh)
    print(f"[log] tee -> {log_path}  (tail -f to follow)", flush=True)
    return log_path
