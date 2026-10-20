"""Paired HR/LR datasets with *aligned* cropping.

The LR image is assumed to be the HR image downscaled by an integer factor
``scale``.  When a patch is taken from the HR image, the corresponding LR patch
is taken from the same spatial location::

    HR patch origin  = (y * scale, x * scale)
    HR patch size    = (patch_size, patch_size)      # patch_size % scale == 0
    LR patch origin  = (y, x)
    LR patch size    = (patch_size // scale, patch_size // scale)

This guarantees HR/LR pixels stay aligned, which is essential for a
sub-pixel (PixelShuffle) model.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def collect_pairs(
    hr_dir: str | Path,
    lr_dir: str | Path,
    scale: int,
    exts: Iterable[str] = IMG_EXTS,
) -> list[tuple[Path, Path]]:
    """Match HR files in ``hr_dir`` with ``<stem>x{scale}.png`` in ``lr_dir``."""
    hr_dir = Path(hr_dir)
    lr_dir = Path(lr_dir)
    exts = {e.lower() for e in exts}

    pairs: list[tuple[Path, Path]] = []
    for hr in sorted(hr_dir.iterdir()):
        if not hr.is_file() or hr.suffix.lower() not in exts:
            continue
        candidate = lr_dir / f"{hr.stem}x{scale}.png"
        if not candidate.exists():
            # Fall back to any extension.
            matches = list(lr_dir.glob(f"{hr.stem}x{scale}.*"))
            if not matches:
                continue
            candidate = matches[0]
        pairs.append((hr, candidate))
    return pairs


def _load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def lr_type_of(path: str | Path) -> str:
    """Infer the LR variant (``bicubic`` / ``unknown``) from a LR file path."""
    for parent in Path(path).parents:
        name = parent.name.lower()
        if "bicubic" in name:
            return "bicubic"
        if "unknown" in name:
            return "unknown"
    return ""


def collect_hr_items(
    hr_dir: str | Path,
    lr_dirs: dict[str, str | Path],
    scale: int,
    exts: Iterable[str] = IMG_EXTS,
) -> list[tuple[Path, dict[str, Path]]]:
    """Group each HR file with all of its available LR variants.

    ``lr_dirs`` maps a LR type (e.g. ``"bicubic"``) to its directory.  Returns
    ``(hr_path, {lr_type: lr_path})`` entries; HR files without any matching LR
    are skipped.  A missing variant is simply omitted from the inner dict.
    """
    hr_dir = Path(hr_dir)
    lr_dirs = {k: Path(v) for k, v in lr_dirs.items()}
    exts = {e.lower() for e in exts}

    items: list[tuple[Path, dict[str, Path]]] = []
    for hr in sorted(hr_dir.iterdir()):
        if not hr.is_file() or hr.suffix.lower() not in exts:
            continue
        lrs: dict[str, Path] = {}
        for lr_type, lr_dir in lr_dirs.items():
            candidate = lr_dir / f"{hr.stem}x{scale}.png"
            if candidate.exists():
                lrs[lr_type] = candidate
            else:
                matches = list(lr_dir.glob(f"{hr.stem}x{scale}.*"))
                if matches:
                    lrs[lr_type] = matches[0]
        if lrs:
            items.append((hr, lrs))
    return items


def collect_hr_files(
    hr_dir: str | Path,
    exts: Iterable[str] = IMG_EXTS,
) -> list[Path]:
    """Return image files in ``hr_dir`` (no LR pairing required)."""
    hr_dir = Path(hr_dir)
    exts = {e.lower() for e in exts}
    if not hr_dir.is_dir():
        return []
    return [
        p
        for p in sorted(hr_dir.iterdir())
        if p.is_file() and p.suffix.lower() in exts
    ]


class SRTrainDataset(Dataset):
    """Random aligned patch pairs for training."""

    def __init__(
        self,
        pairs: Sequence[tuple[Path, Path]],
        scale: int = 4,
        patch_size: int = 128,
        augment: bool = True,
        seed: int | None = None,
    ) -> None:
        if patch_size % scale != 0:
            raise ValueError(
                f"patch_size ({patch_size}) must be divisible by scale ({scale})"
            )
        self.pairs = list(pairs)
        self.scale = scale
        self.patch_size = patch_size
        self.lr_patch_size = patch_size // scale
        self.augment = augment
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.pairs)

    def _crop(self, hr: Image.Image, lr: Image.Image) -> tuple[Image.Image, Image.Image]:
        lw, lh = lr.size
        p = self.lr_patch_size

        # If the image is too small, resize it up so a full patch fits.
        if lw < p or lh < p:
            new_w = max(lw, p)
            new_h = max(lh, p)
            lr = lr.resize((new_w, new_h), Image.BICUBIC)
            hr = hr.resize((new_w * self.scale, new_h * self.scale), Image.BICUBIC)
            lw, lh = new_w, new_h

        x = self.rng.randint(0, lw - p)
        y = self.rng.randint(0, lh - p)

        lr_patch = lr.crop((x, y, x + p, y + p))
        s = self.scale
        hr_patch = hr.crop((x * s, y * s, (x + p) * s, (y + p) * s))
        return hr_patch, lr_patch

    def _augment(
        self, hr: Image.Image, lr: Image.Image
    ) -> tuple[Image.Image, Image.Image]:
        if self.rng.random() < 0.5:
            hr = hr.transpose(Image.FLIP_LEFT_RIGHT)
            lr = lr.transpose(Image.FLIP_LEFT_RIGHT)
        if self.rng.random() < 0.5:
            hr = hr.transpose(Image.FLIP_TOP_BOTTOM)
            lr = lr.transpose(Image.FLIP_TOP_BOTTOM)
        k = self.rng.randint(0, 3)
        if k:
            # PIL rotates counter-clockwise; 90-degree rotations keep alignment.
            hr = hr.rotate(90 * k, expand=True)
            lr = lr.rotate(90 * k, expand=True)
        return hr, lr

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        hr_path, lr_path = self.pairs[idx]
        hr = _load_rgb(hr_path)
        lr = _load_rgb(lr_path)

        # Keep the region of HR that the LR actually covers (alignment guard).
        lw, lh = lr.size
        ew, eh = lw * self.scale, lh * self.scale
        if hr.size != (ew, eh):
            hr = hr.crop((0, 0, ew, eh))

        hr_patch, lr_patch = self._crop(hr, lr)
        if self.augment:
            hr_patch, lr_patch = self._augment(hr_patch, lr_patch)

        lr_t = torch.from_numpy(_to_numpy(lr_patch)).permute(2, 0, 1).float() / 255.0
        hr_t = torch.from_numpy(_to_numpy(hr_patch)).permute(2, 0, 1).float() / 255.0
        return lr_t, hr_t


class SRMixTrainDataset(SRTrainDataset):
    """Training dataset with random LR-variant selection per access.

    ``groups`` is a list of ``(hr_path, {lr_type: lr_path})``.  Each HR is
    exposed ``repeats`` times per epoch, and every access picks one of its
    available LR variants at random.  The same HR can therefore be seen with
    two different LR types within one epoch, but this is *not guaranteed*.
    """

    def __init__(
        self,
        groups,
        scale: int = 4,
        patch_size: int = 192,
        augment: bool = True,
        seed: int | None = None,
        repeats: int = 2,
    ) -> None:
        super().__init__(
            [],
            scale=scale,
            patch_size=patch_size,
            augment=augment,
            seed=seed,
        )
        if repeats < 1:
            raise ValueError("repeats must be >= 1")
        self.groups = list(groups)
        self.repeats = repeats

    def __len__(self) -> int:
        return len(self.groups) * self.repeats

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        hr_path, lrs = self.groups[idx % len(self.groups)]
        lr_path = lrs[self.rng.choice(tuple(lrs.keys()))]

        hr = _load_rgb(hr_path)
        lr = _load_rgb(lr_path)

        # Keep the region of HR that the LR actually covers (alignment guard).
        lw, lh = lr.size
        ew, eh = lw * self.scale, lh * self.scale
        if hr.size != (ew, eh):
            hr = hr.crop((0, 0, ew, eh))

        hr_patch, lr_patch = self._crop(hr, lr)
        if self.augment:
            hr_patch, lr_patch = self._augment(hr_patch, lr_patch)

        lr_t = torch.from_numpy(_to_numpy(lr_patch)).permute(2, 0, 1).float() / 255.0
        hr_t = torch.from_numpy(_to_numpy(hr_patch)).permute(2, 0, 1).float() / 255.0
        return lr_t, hr_t


class SRValDataset(Dataset):
    """Full-image pairs for validation.

    Full-resolution images are returned so PSNR is measured on the whole image,
    which is the standard SR evaluation protocol.  Coordinate alignment is
    preserved because the model is fully convolutional.
    """

    def __init__(self, pairs: Sequence[tuple[Path, Path]], scale: int = 4) -> None:
        self.pairs = list(pairs)
        self.scale = scale

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        hr_path, lr_path = self.pairs[idx]
        hr = _load_rgb(hr_path)
        lr = _load_rgb(lr_path)
        lw, lh = lr.size
        ew, eh = lw * self.scale, lh * self.scale
        if hr.size != (ew, eh):
            hr = hr.crop((0, 0, ew, eh))

        lr_t = torch.from_numpy(_to_numpy(lr)).permute(2, 0, 1).float() / 255.0
        hr_t = torch.from_numpy(_to_numpy(hr)).permute(2, 0, 1).float() / 255.0
        lr_type = lr_type_of(lr_path)
        name = f"{lr_type}/{hr_path.name}" if lr_type else hr_path.name
        return lr_t, hr_t, name


def _to_numpy(img: Image.Image):
    import numpy as np

    # np.array copies, avoiding the non-writable-array warning from torch.
    return np.array(img, dtype="uint8")


class ESP32SRTrainDataset(Dataset):
    """On-the-fly ESP32 degradation for training patches.

    A larger HR region is cropped first (``margin`` LR pixels of context on
    each side), degraded as a whole, and then centre-cropped back to the
    requested LR patch size.  This reduces blur/ghost/vignette boundary
    artifacts compared with degrading each patch in isolation.

    Every access draws a fresh profile and seed, so the same HR patch is seen
    with different realistic degradations in different epochs.
    """

    def __init__(
        self,
        hr_paths: Sequence[Path],
        scale: int = 4,
        patch_size: int = 192,
        margin: int = 8,
        version: str = "v1",
        severity: float = 1.0,
        augment: bool = True,
        seed: int | None = None,
        profiles: Sequence[str] | None = None,
    ) -> None:
        if patch_size % scale != 0:
            raise ValueError(
                f"patch_size ({patch_size}) must be divisible by scale ({scale})"
            )
        self.hr_paths = list(hr_paths)
        self.scale = scale
        self.patch_size = patch_size
        self.lr_patch_size = patch_size // scale
        self.margin = max(0, int(margin))
        self.version = version
        self.severity = float(severity)
        self.augment = augment
        self.seed = int(seed if seed is not None else random.randrange(0, 2**31 - 1))
        self.profiles = tuple(profiles) if profiles else None
        self._rng = random.Random(self.seed + 7919)
        self._worker_id = -1

    def __len__(self) -> int:
        return len(self.hr_paths)

    def _worker_rng(self) -> random.Random:
        info = torch.utils.data.get_worker_info()
        worker_id = info.id if info is not None else 0
        if worker_id != self._worker_id:
            self._rng = random.Random(self.seed + 104729 * (worker_id + 1))
            self._worker_id = worker_id
        return self._rng

    def _augment_pair(self, hr: Image.Image, lr: Image.Image) -> tuple[Image.Image, Image.Image]:
        rng = self._worker_rng()
        if rng.random() < 0.5:
            hr = hr.transpose(Image.FLIP_LEFT_RIGHT)
            lr = lr.transpose(Image.FLIP_LEFT_RIGHT)
        if rng.random() < 0.5:
            hr = hr.transpose(Image.FLIP_TOP_BOTTOM)
            lr = lr.transpose(Image.FLIP_TOP_BOTTOM)
        k = rng.randint(0, 3)
        if k:
            hr = hr.rotate(90 * k, expand=True)
            lr = lr.rotate(90 * k, expand=True)
        return hr, lr

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        from .degrade import degrade_patch

        rng = self._worker_rng()
        path = self.hr_paths[idx]
        hr = _load_rgb(path)

        margin_hr = self.margin * self.scale
        region_size = self.patch_size + 2 * margin_hr
        if hr.width < region_size or hr.height < region_size:
            hr = hr.resize(
                (max(hr.width, region_size), max(hr.height, region_size)),
                Image.BICUBIC,
            )

        x = rng.randint(0, hr.width - region_size)
        y = rng.randint(0, hr.height - region_size)
        region = hr.crop((x, y, x + region_size, y + region_size))

        profile = rng.choice(self.profiles) if self.profiles else None
        seed = rng.randrange(0, 2**31 - 1)
        lr_region, _ = degrade_patch(
            region,
            version=self.version,
            seed=seed,
            profile=profile,
            scale=self.scale,
            severity=self.severity,
            output_lr_size=(
                self.lr_patch_size + 2 * self.margin,
                self.lr_patch_size + 2 * self.margin,
            ),
        )
        lr_patch = lr_region.crop(
            (
                self.margin,
                self.margin,
                self.margin + self.lr_patch_size,
                self.margin + self.lr_patch_size,
            )
        )
        hr_patch = region.crop(
            (
                margin_hr,
                margin_hr,
                margin_hr + self.patch_size,
                margin_hr + self.patch_size,
            )
        )

        if self.augment:
            hr_patch, lr_patch = self._augment_pair(hr_patch, lr_patch)

        lr_t = torch.from_numpy(_to_numpy(lr_patch)).permute(2, 0, 1).float() / 255.0
        hr_t = torch.from_numpy(_to_numpy(hr_patch)).permute(2, 0, 1).float() / 255.0
        return lr_t, hr_t


class ESP32SRValDataset(Dataset):
    """Deterministic full-image ESP32 degradation for validation.

    The profile cycles through the version's profile list and the seed is
    derived from the image index, so PSNR is comparable across epochs.
    """

    def __init__(
        self,
        hr_paths: Sequence[Path],
        scale: int = 4,
        version: str = "v1",
        severity: float = 1.0,
        seed: int = 42,
        profiles: Sequence[str] | None = None,
    ) -> None:
        from .degrade import profile_names

        self.hr_paths = list(hr_paths)
        self.scale = scale
        self.version = version
        self.severity = float(severity)
        self.seed = int(seed)
        names = tuple(profiles) if profiles else profile_names(version)
        if not names:
            raise ValueError("no degradation profiles available")
        self.profiles = names

    def __len__(self) -> int:
        return len(self.hr_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        from .degrade import degrade_image

        path = self.hr_paths[idx]
        hr = _load_rgb(path)
        profile = self.profiles[idx % len(self.profiles)]
        seed = self.seed + idx * 1000003
        lr, _ = degrade_image(
            hr,
            version=self.version,
            seed=seed,
            profile=profile,
            scale=self.scale,
            severity=self.severity,
        )

        lw, lh = lr.size
        ew, eh = lw * self.scale, lh * self.scale
        if hr.size != (ew, eh):
            hr = hr.crop((0, 0, ew, eh))

        lr_t = torch.from_numpy(_to_numpy(lr)).permute(2, 0, 1).float() / 255.0
        hr_t = torch.from_numpy(_to_numpy(hr)).permute(2, 0, 1).float() / 255.0
        return lr_t, hr_t, f"esp32/{profile}/{path.name}"
