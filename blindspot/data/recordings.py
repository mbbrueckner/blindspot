"""Names and manifest of our own iPhone recordings (WP 1.3).

Every recording folder is named ``<env>_<scene>_<light>_<obstacles>_<nn>``, e.g.
``out_sidewalk_day_bollard-bike_01``. The parts become tags of every frame, so reports can group the
metrics by environment, scene, lighting and obstacle type. An optional manifest
``recordings.csv`` next to the folders adds split, camera height, weather and notes.
"""

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

ENVS = ("in", "out", "trans")
SCENES = (
    "corridor",
    "room",
    "stairs",
    "door",
    "sidewalk",
    "street",
    "square",
    "park",
    "platform",
    "underpass",
    "garage",
    "terrace",
)
LIGHTS = ("day", "dusk", "night", "artificial", "backlight")
OBSTACLES = (
    "pole",
    "bollard",
    "bike",
    "bin",
    "sign",
    "person",
    "car",
    "glass",
    "step",
    "branch",
    "furniture",
    "wall",
    "plant",
    "door",
    "none",
)
SPLITS = ("test", "train")
MANIFEST_COLUMNS = ("recording", "split", "camera_height_m", "weather", "original_name", "notes")

_NN = re.compile(r"\d{2}")


@dataclass(frozen=True)
class RecordingName:
    """Parsed recording name.

    Attributes:
        env: "in", "out" or "trans" (transition between indoors and outdoors).
        scene: Scene type, see ``SCENES``.
        light: Lighting, see ``LIGHTS``.
        obstacles: Obstacle types in the recording, see ``OBSTACLES``; ``("none",)`` if there are none.
        index: Running number ``nn``.
    """

    env: str
    scene: str
    light: str
    obstacles: tuple[str, ...]
    index: int

    @property
    def name(self) -> str:
        """The canonical folder name."""
        return f"{self.env}_{self.scene}_{self.light}_{'-'.join(self.obstacles)}_{self.index:02d}"

    @property
    def tags(self) -> dict[str, str]:
        """Tags for :attr:`blindspot.core.Sample.tags` (obstacles joined with "-")."""
        return {
            "env": self.env,
            "scene": self.scene,
            "light": self.light,
            "obstacles": "-".join(self.obstacles),
        }


def _check(value: str, allowed: tuple[str, ...], part: str, name: str) -> None:
    """Raises a ValueError naming the allowed values if ``value`` is not one of them."""
    if value not in allowed:
        raise ValueError(
            f"Recording {name!r}: {part} {value!r} is not allowed. Allowed: {', '.join(allowed)}"
        )


def parse_recording_name(name: str) -> RecordingName:
    """Parses and validates a recording folder name.

    Args:
        name: Folder name, e.g. ``out_sidewalk_day_bollard-bike_01``.

    Returns:
        The parsed :class:`RecordingName`.

    Raises:
        ValueError: If the name does not follow ``<env>_<scene>_<light>_<obstacles>_<nn>``; the message
            lists the allowed values of the offending part.
    """
    parts = name.split("_")
    if len(parts) != 5:
        raise ValueError(
            f"Recording {name!r} must have 5 parts <env>_<scene>_<light>_<obstacles>_<nn> separated by '_', "
            f"got {len(parts)}. Example: out_sidewalk_day_bollard-bike_01"
        )
    env, scene, light, obstacles, nn = parts
    _check(env, ENVS, "env", name)
    _check(scene, SCENES, "scene", name)
    _check(light, LIGHTS, "light", name)
    obs = tuple(obstacles.split("-"))
    for o in obs:
        _check(o, OBSTACLES, "obstacle", name)
    if len(set(obs)) != len(obs):
        raise ValueError(f"Recording {name!r}: obstacle listed twice in {obstacles!r}")
    if "none" in obs and len(obs) > 1:
        raise ValueError(f"Recording {name!r}: 'none' cannot be combined with other obstacles")
    if not _NN.fullmatch(nn):
        raise ValueError(f"Recording {name!r}: number {nn!r} must have two digits, e.g. 01")
    return RecordingName(env, scene, light, obs, int(nn))


@dataclass(frozen=True)
class ManifestEntry:
    """One row of ``recordings.csv``.

    Attributes:
        recording: Folder name (valid recording name).
        split: "test" or "train".
        camera_height_m: Height of the camera above the ground in meters, ``None`` if not measured.
        weather: Free text, e.g. "cloudy".
        original_name: Folder name before renaming (Stray Scanner hash).
        notes: Free text.
    """

    recording: str
    split: str
    camera_height_m: Optional[float] = None
    weather: str = ""
    original_name: str = ""
    notes: str = ""


def load_manifest(path: Path) -> dict[str, ManifestEntry]:
    """Reads and validates ``recordings.csv``.

    Args:
        path: CSV file with the header ``recording,split,camera_height_m,weather,original_name,notes``
            (``recording`` and ``split`` are required, the other columns are optional).

    Returns:
        Dict mapping recording name to its entry.

    Raises:
        ValueError: If a column is unknown or missing, a name is invalid or duplicated, the split is
            not "test" or "train", or the camera height is not a positive number.
    """
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        columns = [c.strip() for c in reader.fieldnames or []]
        unknown = set(columns) - set(MANIFEST_COLUMNS)
        if unknown or not {"recording", "split"} <= set(columns):
            raise ValueError(
                f"{path}: columns must be a subset of {', '.join(MANIFEST_COLUMNS)} including recording and "
                f"split, got {columns}"
            )
        entries: dict[str, ManifestEntry] = {}
        for line, raw in enumerate(reader, start=2):
            row = {k.strip(): (v or "").strip() for k, v in raw.items() if k is not None}
            name = row["recording"]
            try:
                parse_recording_name(name)
            except ValueError as e:
                raise ValueError(f"{path}:{line}: {e}") from None
            if name in entries:
                raise ValueError(f"{path}:{line}: recording {name!r} listed twice")
            if row["split"] not in SPLITS:
                raise ValueError(f"{path}:{line}: split {row['split']!r} must be one of {', '.join(SPLITS)}")
            height = row.get("camera_height_m", "")
            try:
                height_m = float(height) if height else None
            except ValueError:
                raise ValueError(f"{path}:{line}: camera_height_m {height!r} is not a number") from None
            if height_m is not None and height_m <= 0:
                raise ValueError(f"{path}:{line}: camera_height_m must be positive, got {height_m}")
            entries[name] = ManifestEntry(
                name,
                row["split"],
                height_m,
                row.get("weather", ""),
                row.get("original_name", ""),
                row.get("notes", ""),
            )
    return entries


def find_recordings(root: Path) -> tuple[list[str], list[str]]:
    """Lists the recording folders below ``root``.

    A folder counts as a recording if it contains ``rgb.mp4`` and ``depth/``.

    Args:
        root: Directory with one folder per recording (e.g. ``data/iPhone``).

    Returns:
        (valid names, invalid names), both sorted. Invalid names do not follow the naming convention.
    """
    valid, invalid = [], []
    for d in sorted(Path(root).iterdir()):
        if not (d.is_dir() and (d / "rgb.mp4").is_file() and (d / "depth").is_dir()):
            continue
        try:
            parse_recording_name(d.name)
            valid.append(d.name)
        except ValueError:
            invalid.append(d.name)
    return valid, invalid
