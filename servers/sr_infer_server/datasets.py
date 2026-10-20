"""Hard-coded LR -> HR mappings for the project datasets.

Only the directories listed here are paired.  Adding a new dataset later is a
one-line change: append a ``DatasetMapping`` entry.

Naming conventions in this project:

* DIV2K:  HR ``0001.jpeg``  <->  LR ``0001x4.png``
* Flickr2K: HR ``0001.png`` <->  LR ``0001x4.png``
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .config import Settings


@dataclass(frozen=True, slots=True)
class DatasetMapping:
    label: str
    lr_root: str  # relative to project root
    hr_root: str  # relative to project root


DATASET_MAPPINGS: tuple[DatasetMapping, ...] = (
    DatasetMapping(
        label="DIV2K 训练集",
        lr_root="data/DIV2K/DIV2K_train_LR_bicubic",
        hr_root="data/DIV2K/DIV2K_train_HR",
    ),
    DatasetMapping(
        label="DIV2K 训练集",
        lr_root="data/DIV2K/DIV2K_train_LR_unknown",
        hr_root="data/DIV2K/DIV2K_train_HR",
    ),
    DatasetMapping(
        label="DIV2K 验证集",
        lr_root="data/DIV2K/DIV2K_valid_LR_bicubic",
        hr_root="data/DIV2K/DIV2K_valid_HR",
    ),
    DatasetMapping(
        label="DIV2K 验证集",
        lr_root="data/DIV2K/DIV2K_valid_LR_unknown",
        hr_root="data/DIV2K/DIV2K_valid_HR",
    ),
    DatasetMapping(
        label="Flickr2K",
        lr_root="data/Flickr2K/Flickr2K_LR_bicubic",
        hr_root="data/Flickr2K/Flickr2K_HR",
    ),
    DatasetMapping(
        label="Flickr2K",
        lr_root="data/Flickr2K/Flickr2K_LR_unknown",
        hr_root="data/Flickr2K/Flickr2K_HR",
    ),
)

_SCALE_DIR_RE = re.compile(r"^X(?P<scale>\d+)$", re.IGNORECASE)
_LR_SUFFIX_RE = re.compile(r"x\d+$", re.IGNORECASE)


def _display(settings: Settings, path: Path) -> str:
    try:
        rel = path.resolve().relative_to(settings.project_root.resolve())
    except ValueError:
        return path.resolve().as_posix()
    return rel.as_posix()


def _find_hr_file(hr_root: Path, stem: str) -> Path | None:
    # Prefer the two extensions used by DIV2K/Flickr2K.
    for ext in (".png", ".jpg", ".jpeg"):
        candidate = hr_root / f"{stem}{ext}"
        if candidate.is_file():
            return candidate
    # Fallback: case-insensitive suffix match (some datasets use mixed case).
    if hr_root.is_dir():
        lowered = stem.lower()
        for candidate in hr_root.iterdir():
            if candidate.is_file() and candidate.stem.lower() == lowered:
                if candidate.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                    return candidate
    return None


def find_hr_for_lr(settings: Settings, lr_path: Path) -> dict[str, object] | None:
    """Return HR metadata when ``lr_path`` lives in a known dataset LR folder."""
    lr_resolved = lr_path.resolve()
    for mapping in DATASET_MAPPINGS:
        lr_root = (settings.project_root / mapping.lr_root).resolve()
        try:
            rel = lr_resolved.relative_to(lr_root)
        except ValueError:
            continue

        parts = rel.parts
        if len(parts) != 2:
            continue

        scale_match = _SCALE_DIR_RE.match(parts[0])
        if scale_match is None:
            continue
        scale = int(scale_match.group("scale"))

        lr_stem = Path(parts[1]).stem
        hr_stem = _LR_SUFFIX_RE.sub("", lr_stem)

        hr_root = (settings.project_root / mapping.hr_root).resolve()
        hr_path = _find_hr_file(hr_root, hr_stem)
        if hr_path is None:
            continue

        return {
            "dataset": mapping.label,
            "lr_scale": scale,
            "lr_dir": _display(settings, lr_resolved.parent),
            "hr_path": _display(settings, hr_path),
            "hr_abs_path": str(hr_path),
            "hr_name": hr_path.name,
        }
    return None
