"""Loader for Stray Scanner recordings from the iPhone 12 Pro (WP 1.3).

Layout of one recording (format: https://github.com/strayrobots/scanner/blob/main/docs/format.md)::

    <recording>/rgb.mp4             HEVC video, 1920x1440
    <recording>/depth/000000.png    LiDAR depth, uint16 millimeters, 256x192
    <recording>/confidence/000000.png  ARKit confidence 0 (low), 1 (medium), 2 (high)
    <recording>/odometry.csv        pose and intrinsics (fx, fy, cx, cy at video resolution) per frame
    <recording>/camera_matrix.csv   one 3x3 intrinsics matrix at video resolution (fallback)

Frame alignment: the video decoders (OpenCV, PyAV, ffmpeg) return one frame fewer than there are depth
maps; the container lists the missing frame, but it is not decodable. Correlating camera rotation
(odometry) with image motion showed that the *first* frame is the missing one: decoded video frame ``k``
belongs to ``depth/<k+1>.png``. The loader detects this from the frame counts (offset 1 if the video has
one frame fewer, offset 0 if the counts match) and raises an error otherwise.
:func:`estimate_frame_offset` repeats the motion check for a recording.

Image, depth and intrinsics are put on one pixel grid: the depth resolution times ``scale`` (default 3,
i.e. 768x576). The image is downsampled with area interpolation; depth and confidence are upsampled
with nearest neighbor, so every LiDAR pixel becomes a ``scale x scale`` block and no depth values are
invented. The ground truth is unreliable on glass, at night and beyond about 5 m.
"""

import csv
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np

from blindspot.core import Intrinsics, Sample
from blindspot.data.calibration import scale_intrinsics
from blindspot.data.recordings import ManifestEntry, parse_recording_name

VIDEO_SIZE = (1920, 1440)  # (width, height) of rgb.mp4, the resolution of the intrinsics


def read_depth_png(path: Path) -> np.ndarray:
    """Reads a Stray Scanner depth map.

    Args:
        path: 16-bit PNG in millimeters.

    Returns:
        Depth in meters, float32, NaN where the value is 0.

    Raises:
        FileNotFoundError: If the file cannot be read.
    """
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(path)
    depth = raw.astype(np.float32) / 1000.0
    depth[raw == 0] = np.nan
    return depth


def read_intrinsics_table(recording: Path) -> dict[int, Intrinsics]:
    """Reads the per-frame intrinsics from ``odometry.csv`` (at video resolution).

    Args:
        recording: Recording folder.

    Returns:
        Dict mapping frame number to intrinsics; empty if the file or the columns are missing.
    """
    path = Path(recording) / "odometry.csv"
    if not path.is_file():
        return {}
    table = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        if not {"frame", "fx", "fy", "cx", "cy"} <= set(reader.fieldnames or []):
            return {}
        for row in reader:
            table[int(row["frame"])] = Intrinsics(
                float(row["fx"]), float(row["fy"]), float(row["cx"]), float(row["cy"])
            )
    return table


def read_camera_matrix(recording: Path) -> Intrinsics:
    """Reads the fixed intrinsics from ``camera_matrix.csv`` (at video resolution).

    Args:
        recording: Recording folder.

    Raises:
        FileNotFoundError: If the file is missing.
        ValueError: If the file is not a 3x3 matrix.
    """
    text = (Path(recording) / "camera_matrix.csv").read_text()
    rows = [[float(v) for v in line.split(",")] for line in text.splitlines() if line.strip()]
    if len(rows) != 3 or any(len(r) != 3 for r in rows):
        raise ValueError(f"{recording}/camera_matrix.csv is not a 3x3 matrix")
    return Intrinsics(rows[0][0], rows[1][1], rows[0][2], rows[1][2])


def depth_indices(recording: Path) -> list[int]:
    """Sorted frame numbers of all depth maps of a recording.

    Args:
        recording: Recording folder.
    """
    return sorted(int(p.stem) for p in (Path(recording) / "depth").glob("*.png"))


def iter_video_frames(path: Path, backend: str = "auto") -> Iterator[np.ndarray]:
    """Decodes a video frame by frame.

    Args:
        path: Video file.
        backend: "opencv", "pyav" or "auto" (OpenCV, PyAV if OpenCV cannot open the file or decodes no
            frame, e.g. an OpenCV build without HEVC support).

    Yields:
        Frames as BGR uint8 arrays (H, W, 3).

    Raises:
        RuntimeError: If no backend can decode the video.
    """
    if backend in ("auto", "opencv"):
        cap = cv2.VideoCapture(str(path))
        ok, frame = cap.read() if cap.isOpened() else (False, None)
        if ok:
            while ok:
                yield frame
                ok, frame = cap.read()
            cap.release()
            return
        cap.release()
        if backend == "opencv":
            raise RuntimeError(f"OpenCV cannot decode {path}")
    try:
        import av
    except ImportError:
        raise RuntimeError(
            f"OpenCV cannot decode {path} and PyAV is not installed (uv sync --extra video)"
        ) from None
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            yield frame.to_ndarray(format="bgr24")


def frame_offset(n_video: int, n_depth: int) -> int:
    """Number of leading depth frames without a video frame (see module description).

    Args:
        n_video: Number of decoded video frames.
        n_depth: Number of depth maps.

    Returns:
        0 if the counts match, 1 if the video has exactly one frame fewer.

    Raises:
        ValueError: For any other difference, because the alignment would be guesswork.
    """
    if n_video == n_depth:
        return 0
    if n_video == n_depth - 1:
        return 1
    raise ValueError(f"{n_video} video frames but {n_depth} depth maps; cannot align image and depth")


def load_recording(
    recording: Path,
    stride: int = 10,
    max_frames: Optional[int] = None,
    scale: int = 3,
    min_confidence: int = 2,
    manifest: Optional[ManifestEntry] = None,
    backend: str = "auto",
) -> list[Sample]:
    """Loads every ``stride``-th frame of a recording as samples on one pixel grid.

    Args:
        recording: Recording folder; its name must follow the naming convention.
        stride: Take every n-th depth frame (frame numbers 0, n, 2n, ... that have a video frame).
        max_frames: Upper limit of loaded frames, ``None`` for all.
        scale: Grid size as a multiple of the depth resolution (256x192); 3 gives 768x576.
        min_confidence: Depth pixels with a lower ARKit confidence become NaN (2 keeps only "high").
        manifest: Manifest entry of the recording, adds ``split`` and ``weather`` tags.
        backend: Video decoder, see :func:`iter_video_frames`.

    Returns:
        Samples with ``frame_id`` ``"<recording>/<depth frame number:06d>"`` and the tags ``dataset``,
        ``recording``, ``env``, ``scene``, ``light``, ``obstacles`` (plus ``split``, ``weather`` and
        ``camera_height_m`` from the manifest).

    Raises:
        ValueError: If the name is invalid, the frame counts cannot be aligned or the video has an
            unexpected resolution.
    """
    recording = Path(recording)
    name = parse_recording_name(recording.name)
    if stride < 1 or scale < 1:
        raise ValueError(f"stride and scale must be >= 1, got {stride}, {scale}")
    depth_ids = depth_indices(recording)
    if not depth_ids or depth_ids != list(range(len(depth_ids))):
        raise ValueError(f"{recording}/depth must contain consecutive frames 000000.png, 000001.png, ...")
    k_table = read_intrinsics_table(recording)
    k_fixed: Optional[Intrinsics] = None
    depth_h, depth_w = read_depth_png(recording / "depth" / "000000.png").shape
    grid_w, grid_h = depth_w * scale, depth_h * scale

    # The offset is only known after decoding (it depends on the frame count), so keep the resized
    # frames for both possible offsets and pick afterwards.
    wanted = {d for d in depth_ids[::stride]}
    kept: dict[int, np.ndarray] = {}
    n_video = 0
    for v, bgr in enumerate(iter_video_frames(recording / "rgb.mp4", backend)):
        if v == 0 and (bgr.shape[1], bgr.shape[0]) != VIDEO_SIZE:
            raise ValueError(f"{recording}/rgb.mp4 is {bgr.shape[1]}x{bgr.shape[0]}, expected {VIDEO_SIZE}")
        if v in wanted or v + 1 in wanted:
            kept[v] = cv2.cvtColor(
                cv2.resize(bgr, (grid_w, grid_h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB
            )
        n_video += 1
    offset = frame_offset(n_video, len(depth_ids))

    tags = {"dataset": "iphone", "recording": recording.name, **name.tags}
    if manifest is not None:
        tags |= {"split": manifest.split, **({"weather": manifest.weather} if manifest.weather else {})}
        if manifest.camera_height_m is not None:
            tags["camera_height_m"] = f"{manifest.camera_height_m:g}"
    sx, sy = grid_w / VIDEO_SIZE[0], grid_h / VIDEO_SIZE[1]
    samples = []
    for d in sorted(wanted):
        if d - offset not in kept:
            continue  # first depth frame without video frame
        depth = read_depth_png(recording / "depth" / f"{d:06d}.png")
        conf = cv2.imread(str(recording / "confidence" / f"{d:06d}.png"), cv2.IMREAD_UNCHANGED)
        if conf is not None:
            depth[conf < min_confidence] = np.nan
        depth = cv2.resize(depth, (grid_w, grid_h), interpolation=cv2.INTER_NEAREST)
        k = k_table.get(d)
        if k is None:
            k_fixed = k_fixed or read_camera_matrix(recording)
            k = k_fixed
        samples.append(
            Sample(
                frame_id=f"{recording.name}/{d:06d}",
                rgb=kept[d - offset],
                depth=depth,
                intrinsics=scale_intrinsics(k, sx, sy),
                tags=dict(tags),
            )
        )
        if max_frames is not None and len(samples) >= max_frames:
            break
    return samples


def estimate_frame_offset(recording: Path, max_lag: int = 2) -> tuple[int, dict[int, float]]:
    """Estimates the video-to-depth frame offset from motion (diagnostic, slow: decodes the whole video).

    Correlates the rotation angle between consecutive odometry poses with the median optical flow
    between consecutive video frames for several lags. The lag with the highest correlation is the
    offset: video frame ``k`` belongs to depth frame ``k + lag``. Recordings without much rotation
    give low correlations for every lag and no reliable answer.

    Args:
        recording: Recording folder.
        max_lag: Lags from ``-max_lag`` to ``max_lag`` are tested.

    Returns:
        (best lag, dict lag -> correlation coefficient).
    """
    with open(Path(recording) / "odometry.csv", newline="") as f:
        rows = list(csv.DictReader(f, skipinitialspace=True))
    q = np.array([[float(r[k]) for k in ("qx", "qy", "qz", "qw")] for r in rows])
    rot = 2 * np.arccos(np.clip(np.abs(np.sum(q[1:] * q[:-1], axis=1)), 0, 1))
    motion, prev = [], None
    for bgr in iter_video_frames(Path(recording) / "rgb.mp4"):
        gray = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (160, 120)).astype(np.float32)
        if prev is not None:
            flow = cv2.calcOpticalFlowFarneback(prev, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            motion.append(float(np.median(np.linalg.norm(flow, axis=2))))
        prev = gray
    corr = {}
    for lag in range(-max_lag, max_lag + 1):
        pairs = [(m, rot[i + lag]) for i, m in enumerate(motion) if 0 <= i + lag < len(rot)]
        a, b = np.array(pairs).T
        corr[lag] = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")
    best = max(corr, key=lambda k: -np.inf if np.isnan(corr[k]) else corr[k])
    return best, corr
