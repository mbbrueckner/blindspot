"""Downloads a small, reproducible SANPO subset (WP 1.2) over HTTPS, without gcloud.

Per session only ``description.json`` and the images and depth of the left chest camera are
downloaded, and of those only every ``--stride``-th frame, at most ``--max-frames`` per session.
Sessions are drawn with a fixed seed from the official split lists. Without ``--yes`` the script
only counts and downloads nothing.

Example::

    python scripts/download_sanpo.py --kind real --split train --sessions 5 --stride 30 --max-frames 20
"""

import argparse
import json
import random
import sys
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from blindspot.sanpo import CHEST, SANPO_URL  # noqa: E402

API = "https://storage.googleapis.com/storage/v1/b/gresearch/o"
PREFIX = "sanpo_dataset/v0"


def _get(url: str) -> bytes:
    """Fetches a URL and returns the content as bytes."""
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read()


def list_objects(prefix: str) -> list[tuple[str, int]]:
    """Lists all objects under a bucket prefix.

    Args:
        prefix: Prefix relative to ``sanpo_dataset/v0/``.

    Returns:
        List of (object name relative to ``sanpo_dataset/v0/``, size in bytes).
    """
    out, token = [], None
    while True:
        q = {"prefix": f"{PREFIX}/{prefix}", "fields": "items(name,size),nextPageToken"}
        if token:
            q["pageToken"] = token
        data = json.loads(_get(f"{API}?{urllib.parse.urlencode(q)}"))
        out += [(i["name"][len(PREFIX) + 1 :], int(i["size"])) for i in data.get("items", [])]
        token = data.get("nextPageToken")
        if not token:
            return out


def plan_session(kind: str, sid: str, stride: int, max_frames: int) -> list[tuple[str, int]]:
    """Selects the files of a session (empty if it has no chest camera).

    Args:
        kind: "real" or "synthetic".
        sid: Session ID.
        stride: Take only every n-th frame.
        max_frames: Maximum number of frames per session.

    Returns:
        List of (path relative to ``v0/``, size in bytes; 0 if unknown).
    """
    base = f"sanpo-{kind}/{sid}"
    left = f"{base}/{CHEST}/left"
    frames = sorted(o for o in list_objects(f"{left}/video_frames/") if o[0].endswith(".png"))
    if not frames:
        return []
    chosen = frames[::stride][:max_frames]
    depth = [(f"{left}/depth_maps/{Path(n).stem}.float16.gz", 0) for n, _ in chosen]
    return [(f"{base}/description.json", 0)] + chosen + depth


def main() -> None:
    """Parses the arguments, selects sessions and downloads the files (with ``--yes``)."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--kind", choices=["real", "synthetic"], required=True)
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--sessions", type=int, default=5, help="number of sessions")
    ap.add_argument("--stride", type=int, default=30, help="take only every n-th frame")
    ap.add_argument("--max-frames", type=int, default=20, help="frames per session")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("data/sanpo"))
    ap.add_argument("--yes", action="store_true", help="actually download")
    args = ap.parse_args()

    ids = _get(f"{SANPO_URL}/sanpo-{args.kind}/splits/{args.split}_session_ids.txt").decode().split()
    random.Random(args.seed).shuffle(ids)

    selected, plans = [], {}
    for sid in ids:
        files = plan_session(args.kind, sid, args.stride, args.max_frames)
        if files:
            selected.append(sid)
            plans[sid] = files
        if len(selected) == args.sessions:
            break

    # The size of the depth maps is unknown (0); the PNGs are about 3.5 MB per frame.
    n_files = sum(len(f) for f in plans.values())
    mb = sum(s for f in plans.values() for _, s in f) / 1e6
    print(f"{len(selected)} sessions, {n_files} files, at least {mb:.0f} MB (without depth maps)")
    if not args.yes:
        print("Plan only. Use --yes to download.")
        return

    for sid in selected:
        for name, _ in plans[sid]:
            dest = args.out / args.kind / name.split("/", 1)[1]
            if dest.exists():
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(_get(f"{SANPO_URL}/{urllib.parse.quote(name)}"))
    (args.out / args.kind / f"{args.split}_sessions.txt").write_text("\n".join(selected) + "\n")
    print(f"Done: {args.out / args.kind}")


if __name__ == "__main__":
    main()
