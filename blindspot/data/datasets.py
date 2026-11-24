"""Loads the evaluation datasets (SANPO and our iPhone recordings) with frame subsampling and tags."""

from pathlib import Path
from typing import Optional, Sequence

from blindspot.core import Sample
from blindspot.data import sanpo, stray
from blindspot.data.recordings import find_recordings, load_manifest

MANIFEST_NAME = "recordings.csv"


def load_sanpo(
    root: Path,
    sessions: Optional[Sequence[str]] = None,
    stride: int = 1,
    max_frames: Optional[int] = None,
    camera: str = sanpo.CHEST,
) -> list[Sample]:
    """Loads downloaded SANPO frames.

    Args:
        root: Directory with the session folders (e.g. ``data/sanpo/real``).
        sessions: Session IDs; ``None`` reads ``test_sessions.txt`` in ``root`` if it exists, otherwise
            all session folders.
        stride: Take every n-th downloaded frame per session.
        max_frames: Upper limit of frames over all sessions, ``None`` for all.
        camera: SANPO camera, default chest camera.

    Returns:
        Samples with the tags ``dataset`` ("sanpo") and ``session``.

    Raises:
        FileNotFoundError: If ``root`` or a listed session folder is missing.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(root)
    if sessions is None:
        listed = root / "test_sessions.txt"
        if listed.is_file():
            sessions = sanpo.read_session_ids(listed)
        else:
            sessions = sorted(p.name for p in root.iterdir() if (p / "description.json").is_file())
    samples = []
    for sid in sessions:
        session_dir = root / sid
        if not session_dir.is_dir():
            raise FileNotFoundError(session_dir)
        for index in sanpo.frame_indices(session_dir, camera)[::stride]:
            if max_frames is not None and len(samples) >= max_frames:
                return samples
            sample = sanpo.load_sample(session_dir, index, camera)
            sample.tags = {"dataset": "sanpo", "session": sid}
            samples.append(sample)
    return samples


def load_iphone(
    root: Path,
    recordings: Optional[Sequence[str]] = None,
    split: Optional[str] = "test",
    stride: int = 10,
    max_frames_per_recording: Optional[int] = None,
    scale: int = 3,
    min_confidence: int = 2,
    max_recordings: Optional[int] = None,
) -> tuple[list[Sample], list[str]]:
    """Loads our own iPhone recordings (Stray Scanner).

    Args:
        root: Directory with one folder per recording (e.g. ``data/iPhone``), optionally with
            ``recordings.csv``.
        recordings: Recording names; ``None`` takes all validly named folders.
        split: Only recordings with this split in the manifest; ``None`` or a missing manifest loads all.
            Recordings missing from an existing manifest are skipped if a split is requested.
        stride: Take every n-th frame per recording.
        max_frames_per_recording: Upper limit per recording, ``None`` for all.
        scale: Grid size as a multiple of the depth resolution, see :func:`stray.load_recording`.
        min_confidence: Lowest ARKit confidence that is kept.
        max_recordings: Without explicit ``recordings``: use only the first n (after the split filter).

    Returns:
        (samples, folders skipped because their name does not follow the naming convention).

    Raises:
        FileNotFoundError: If ``root`` or a requested recording is missing.
        ValueError: If a requested recording name or the manifest is invalid.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(root)
    manifest = load_manifest(root / MANIFEST_NAME) if (root / MANIFEST_NAME).is_file() else {}
    invalid: list[str] = []
    if recordings is None:
        recordings, invalid = find_recordings(root)
        if split is not None and manifest:
            recordings = [r for r in recordings if r in manifest and manifest[r].split == split]
        if max_recordings is not None:
            recordings = recordings[:max_recordings]
    samples = []
    for name in recordings:
        if not (root / name).is_dir():
            raise FileNotFoundError(root / name)
        samples += stray.load_recording(
            root / name,
            stride=stride,
            max_frames=max_frames_per_recording,
            scale=scale,
            min_confidence=min_confidence,
            manifest=manifest.get(name),
        )
    return samples, invalid
