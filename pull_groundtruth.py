"""
Pull per-image ground-truth GeoJSONs from the Sperwer digitization station
(Windows, over Tailscale/SMB) into the local incoming/ folder, ready for the
Phase 2 merge in WORKPLAN_NEW_IMAGES.md.

Credentials come from the environment and are NEVER hardcoded:
    SMB_USER   SMB username           (default: administrator)
    SMB_PASS   SMB password           (required; if unset you are prompted)
    SMB_HOST   host / Tailscale IP     (default: 100.122.176.20)
    SMB_SHARE  share name              (default: d$)

Sperwer's SMB is SMB2/3 only (Windows refuses curl's SMBv1), so run this through
uv with the smbprotocol package -- the pull_groundtruth.sh launcher does that.

Usage:
    bash pull_groundtruth.sh                       # pull SkyFi_*_epsg4326.geojson
    bash pull_groundtruth.sh --list                # just list remote .geojson files
    bash pull_groundtruth.sh --pattern '*.geojson' # pull all geojson
    bash pull_groundtruth.sh --dest /some/where    # override destination
"""

import argparse
import fnmatch
import getpass
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_HOST = os.environ.get("SMB_HOST", "100.122.176.20")
DEFAULT_SHARE = os.environ.get("SMB_SHARE", "d$")
DEFAULT_REMOTE = os.environ.get("SMB_REMOTE_DIR", r"tf_cowdetection\tiles")
DEFAULT_DEST = os.path.join(HERE, "source", "terrain_truth", "incoming")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--host", default=DEFAULT_HOST, help="host / Tailscale IP")
    p.add_argument("--share", default=DEFAULT_SHARE, help="SMB share (e.g. d$)")
    p.add_argument("--remote-dir", default=DEFAULT_REMOTE,
                   help=r"path under the share (default: tf_cowdetection\tiles)")
    p.add_argument("--pattern", default="SkyFi_*_epsg4326.geojson",
                   help="filename glob to pull (default: per-image ground truth)")
    p.add_argument("--dest", default=DEFAULT_DEST, help="local destination dir")
    p.add_argument("--list", action="store_true", help="list remote geojson, do not download")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    try:
        import smbclient
    except ImportError:
        print("smbprotocol not available -- run via ./pull_groundtruth.sh "
              "(uv run --with smbprotocol).", file=sys.stderr)
        return 2

    user = os.environ.get("SMB_USER", "administrator")
    password = os.environ.get("SMB_PASS")
    if not password:
        password = getpass.getpass(f"SMB password for {user}@{args.host}: ")

    try:
        smbclient.register_session(args.host, username=user, password=password)
    except Exception as e:
        print(f"Could not connect to {args.host}: {e}", file=sys.stderr)
        return 1

    base = rf"\\{args.host}\{args.share}\{args.remote_dir}"
    try:
        names = [e.name for e in smbclient.scandir(base) if not e.is_dir()]
    except Exception as e:
        print(f"Could not list {base}: {e}", file=sys.stderr)
        return 1

    geojson = sorted(n for n in names if n.lower().endswith(".geojson"))
    matched = [n for n in geojson if fnmatch.fnmatch(n, args.pattern)]

    if args.list:
        print(f"{base}\n  {len(geojson)} .geojson, {len(matched)} match '{args.pattern}':")
        for n in geojson:
            print(("  * " if n in matched else "    ") + n)
        return 0

    if not matched:
        print(f"No files match '{args.pattern}' in {base}", file=sys.stderr)
        return 1

    os.makedirs(args.dest, exist_ok=True)
    for n in matched:
        with smbclient.open_file(base + "\\" + n, mode="rb") as f:
            data = f.read()
        out = os.path.join(args.dest, n)
        with open(out, "wb") as o:
            o.write(data)
        print(f"pulled {n} ({len(data):,} bytes) -> {out}")

    print(f"\n{len(matched)} file(s) -> {args.dest}")
    print("next: run the Phase 2 merge (WORKPLAN_NEW_IMAGES.md) over that folder.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
