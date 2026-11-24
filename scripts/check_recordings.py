"""Checks the iPhone recordings: names, manifest, frame counts and the video-to-depth frame offset.

Example::

    uv run python scripts/check_recordings.py data/iPhone
    uv run python scripts/check_recordings.py data/iPhone --motion   # slow: confirms the offset from motion
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from blindspot.data import recordings, stray  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    """Prints one line per recording and exits with 1 if anything is wrong.

    Args:
        argv: Command line arguments; ``None`` uses ``sys.argv``.
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("root", type=Path, nargs="?", default=Path("data/iPhone"))
    ap.add_argument("--motion", action="store_true", help="estimate the frame offset from motion (slow)")
    args = ap.parse_args(argv)

    ok = True
    manifest_path = args.root / "recordings.csv"
    manifest = {}
    if manifest_path.is_file():
        try:
            manifest = recordings.load_manifest(manifest_path)
            print(f"manifest: {len(manifest)} entries")
        except ValueError as e:
            print(f"manifest INVALID: {e}")
            ok = False
    else:
        print("manifest: none (optional)")
    valid, invalid = recordings.find_recordings(args.root)
    for name in invalid:
        ok = False
        try:
            recordings.parse_recording_name(name)
        except ValueError as e:
            print(f"INVALID NAME {name}: {e}")
    for name in sorted(set(manifest) - set(valid)):
        print(f"manifest lists {name}, but there is no such recording folder")
        ok = False
    for name in valid:
        rec = args.root / name
        n_depth = len(stray.depth_indices(rec))
        n_video = sum(1 for _ in stray.iter_video_frames(rec / "rgb.mp4"))
        try:
            offset = stray.frame_offset(n_video, n_depth)
            line = f"{name}: {n_video} video / {n_depth} depth frames, offset {offset}"
        except ValueError as e:
            line, ok = f"{name}: {e}", False
        if args.motion:
            best, corr = stray.estimate_frame_offset(rec)
            line += f", motion says offset {best} (r={corr[best]:.2f})"
        split = manifest[name].split if name in manifest else "-"
        print(f"{line}, split {split}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
