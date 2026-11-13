#!/usr/bin/env python3
"""Measure the bicubic-upsampling PSNR baseline on the validation split.

The training/validation pipeline compares an SR model against HR.  A useful
sanity check is the classical bicubic baseline: resize the LR image back to the
HR size with bicubic interpolation and measure PSNR.  This script uses exactly
the same validation pairs as ``superresolution.train`` (DIV2K valid + the
Flickr2K 5% hold-out, fixed seed), so its numbers are directly comparable to
``train_log.csv``.

Usage::

    .venv/bin/python scripts/eval_baseline.py --lr-type both
    .venv/bin/python scripts/eval_baseline.py --lr-type esp32_v2
    .venv/bin/python scripts/eval_baseline.py --lr-type bicubic --limit 20
"""

from __future__ import annotations

import argparse
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from superresolution.dataset import lr_type_of
from superresolution.train import build_datasets, psnr


def evaluate(data_root: str, scale: int, lr_type: str, seed: int, device: str, limit: int):
    args = SimpleNamespace(data_root=data_root, scale=scale, lr_type=lr_type, seed=seed)
    _, val_pairs = build_datasets(args)
    if limit > 0:
        val_pairs = val_pairs[:limit]

    by_type: dict[str, list[tuple[str, str]]] = {}
    for hr_path, lr_path in val_pairs:
        by_type.setdefault(lr_type_of(lr_path), []).append((str(hr_path), str(lr_path)))

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    results: dict[str, float] = {}
    counts: dict[str, int] = {}
    for name, pairs in by_type.items():
        values = []
        for hr_path, lr_path in pairs:
            hr = Image.open(hr_path).convert("RGB")
            lr = Image.open(lr_path).convert("RGB")
            ew, eh = lr.width * scale, lr.height * scale
            if hr.size != (ew, eh):
                hr = hr.crop((0, 0, ew, eh))

            lr_t = torch.from_numpy(np.asarray(lr, dtype=np.float32) / 255.0)
            hr_t = torch.from_numpy(np.asarray(hr, dtype=np.float32) / 255.0)
            lr_t = lr_t.permute(2, 0, 1).unsqueeze(0).to(dev)
            hr_t = hr_t.permute(2, 0, 1).unsqueeze(0).to(dev)

            sr = F.interpolate(
                lr_t, size=hr_t.shape[-2:], mode="bicubic", align_corners=False
            ).clamp(0.0, 1.0)
            values.append(psnr(sr, hr_t))

        avg = sum(values) / len(values)
        results[name] = avg
        counts[name] = len(values)
        print(f"{name:12s} n={len(values):3d}  bicubic PSNR = {avg:.3f} dB")

    if len(results) > 1:
        total = sum(results[k] * counts[k] for k in results) / sum(counts.values())
        print(f"{'combined':12s} n={sum(counts.values()):3d}  bicubic PSNR = {total:.3f} dB")
    return results


def main() -> None:
    p = argparse.ArgumentParser(description="Bicubic PSNR baseline on the SR validation split")
    p.add_argument("--data-root", default="data")
    p.add_argument("--scale", type=int, default=4, choices=[2, 3, 4, 8])
    p.add_argument(
        "--lr-type",
        default="both",
        help="bicubic / unknown / both / esp32_v1 / esp32_v2 (same as train.py)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int, default=0, help="debug: only use the first N pairs")
    args = p.parse_args()
    evaluate(args.data_root, args.scale, args.lr_type, args.seed, args.device, args.limit)


if __name__ == "__main__":
    main()
