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
        return lr_t, hr_t, hr_path.name


def _to_numpy(img: Image.Image):
    import numpy as np

    # np.array copies, avoiding the non-writable-array warning from torch.
    return np.array(img, dtype="uint8")
