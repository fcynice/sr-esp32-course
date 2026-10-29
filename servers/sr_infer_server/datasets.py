"""Automatic LR -> HR pairing for all DIV2K / Flickr2K degradation variants.

The HR layout is fixed per split::

    data/DIV2K/DIV2K_train_HR
    data/DIV2K/DIV2K_valid_HR
    data/Flickr2K/Flickr2K_HR

Any image inside a directory named ``<base>_LR_<variant>`` (or ``<base>_LR``)
under one of the dataset roots is paired with the sibling ``<base>_HR``
directory.  New degradation variants therefore require no code changes::

    data/DIV2K/DIV2K_train_LR_esp32_v1/X4/v0/0001x4.png
        -> data/DIV2K/DIV2K_train_HR/0001.jpeg

    data/DIV2K/DIV2K_valid_LR_esp32_v2/X4/v1/0801x4.png
        -> data/DIV2K/DIV2K_valid_HR/0801.jpeg

    data/Flickr2K/Flickr2K_LR_esp32_v1/X4/v0/000001x4.png
        -> data/Flickr2K/Flickr2K_HR/000001.png

Variant folders before or after ``X<scale>`` are also supported; the extra
folder names are appended to the displayed ``lr_type``, e.g.
``..._LR_esp32_v1/X4/v0/0001x4.png`` -> ``esp32_v1/v0``.

Known splits (``DIV2K_train`` / ``DIV2K_valid`` / ``Flickr2K``) also accept
variant directories without ``_LR`` in the name, e.g.
``DIV2K_train_degraded_v3/X4/...``.
"""

from __future__ import annotations

import re
from pathlib import Path

from .config import Settings

# Only these dataset roots are scanned.  The HR directory is derived from the
# LR directory name (``<base>_LR_<variant>`` -> ``<base>_HR``).
DATASET_ROOTS: tuple[str, ...] = ("data/DIV2K", "data/Flickr2K")

_LR_DIR_RE = re.compile(
    r"^(?P<base>.+?)_LR(?:_(?P<variant>.+))?$", re.IGNORECASE
)
_SCALE_DIR_RE = re.compile(r"^X(?P<scale>\d+)$", re.IGNORECASE)
_LR_SUFFIX_RE = re.compile(r"x\d+$", re.IGNORECASE)

_DATASET_LABELS = {
    "DIV2K_train": "DIV2K 训练集",
    "DIV2K_valid": "DIV2K 验证集",
    "Flickr2K": "Flickr2K",
}

# Fallback for LR directories that do not contain ``_LR`` in the name, e.g.
# ``DIV2K_train_degraded_v3``.  These still map to the fixed ``*_HR`` split.
_SPLIT_BASES = ("DIV2K_train", "DIV2K_valid", "Flickr2K")


def _display(settings: Settings, path: Path) -> str:
    try:
        rel = path.resolve().relative_to(settings.project_root.resolve())
    except ValueError:
        return path.resolve().as_posix()
    return rel.as_posix()


def _find_hr_file(hr_root: Path, stem: str) -> Path | None:
    # Prefer the extensions used by DIV2K/Flickr2K.
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


def _split_base_and_variant(split_name: str) -> tuple[str, str, str] | None:
    """Return ``(hr_base, lr_type, label)`` for a split directory.

    ``DIV2K_train_LR_esp32_v1`` -> ``("DIV2K_train", "esp32_v1", "DIV2K 训练集")``
    ``DIV2K_train_degraded_v3`` -> ``("DIV2K_train", "degraded_v3", "DIV2K 训练集")``
    ``DIV2K_train_HR`` -> ``None`` (this is an HR directory, not LR)
    """
    lr_match = _LR_DIR_RE.match(split_name)
    if lr_match is not None:
        base = lr_match.group("base")
        variant = lr_match.group("variant") or ""
        return base, variant, _DATASET_LABELS.get(base, base)

    # Fallback for variant folders that do not include ``_LR``.  Only known
    # split prefixes are accepted, and the fixed HR directory is excluded.
    lowered = split_name.lower()
    for base in _SPLIT_BASES:
        if not lowered.startswith(base.lower()):
            continue
        remainder = split_name[len(base) :].strip("_-")
        if remainder[:2].lower() == "lr":
            remainder = remainder[2:].strip("_-")
        if remainder.lower() == "hr":
            return None
        return base, remainder, _DATASET_LABELS.get(base, base)
    return None


def _scale_and_index(parts: tuple[str, ...]) -> tuple[int, int] | None:
    """Return the last ``X<scale>`` directory and its index in ``parts``."""
    scale: int | None = None
    index: int | None = None
    for position, part in enumerate(parts[:-1]):
        match = _SCALE_DIR_RE.fullmatch(part)
        if match is not None:
            scale = int(match.group("scale"))
            index = position
    if scale is None or index is None:
        return None
    return scale, index


def find_hr_for_lr(settings: Settings, lr_path: Path) -> dict[str, object] | None:
    """Find the fixed HR image for an LR image from DIV2K/Flickr2K.

    Returns ``None`` for images outside the supported dataset roots, images
    that are not below an ``X<scale>`` directory, or when the derived HR file
    does not exist.
    """
    lr_resolved = lr_path.resolve()
    project_root = settings.project_root.resolve()

    for root_rel in DATASET_ROOTS:
        dataset_root = (project_root / root_rel).resolve()
        try:
            rel = lr_resolved.relative_to(dataset_root)
        except ValueError:
            continue

        # The first component is the split directory, e.g.
        # ``DIV2K_train_LR_esp32`` or ``Flickr2K_LR_esp32``.  A real image path
        # needs at least split / X<scale> / filename.
        if len(rel.parts) < 3:
            continue

        split_name = rel.parts[0]
        split_info = _split_base_and_variant(split_name)
        if split_info is None:
            continue
        base, variant, label = split_info

        hr_root = (dataset_root / f"{base}_HR").resolve()
        if not hr_root.is_dir():
            continue

        scale_info = _scale_and_index(rel.parts)
        if scale_info is None:
            continue
        scale, scale_index = scale_info

        # The image itself must be below the X<scale> directory.
        if scale_index >= len(rel.parts) - 1:
            continue

        lr_dir = lr_resolved.parent
        # Folder names between the LR root and X<scale>, and below X<scale>,
        # are both part of the variant, e.g.
        # ``..._LR_esp32_v1/preA/X4/v0/...`` -> ``esp32_v1/preA/v0``.
        before_scale = rel.parts[1:scale_index]
        after_scale = rel.parts[scale_index + 1 : -1]
        lr_type = "/".join(
            part for part in (variant, *before_scale, *after_scale) if part
        )

        hr_stem = _LR_SUFFIX_RE.sub("", lr_resolved.stem)
        hr_path = _find_hr_file(hr_root, hr_stem)
        if hr_path is None:
            continue

        return {
            "dataset": label,
            "lr_type": lr_type,
            "lr_scale": scale,
            "lr_dir": _display(settings, lr_dir),
            "hr_path": _display(settings, hr_path),
            "hr_abs_path": str(hr_path),
            "hr_name": hr_path.name,
        }
    return None
