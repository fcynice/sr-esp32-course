#!/usr/bin/env python3
"""Export inference-only checkpoints for GitHub Releases.

``runs/resNN/best.pth`` stores the model, optimizer, scheduler and AMP scaler.
For inference we only need ``model`` / ``model_name`` / ``model_kwargs`` (plus
``args`` / ``epoch`` / ``psnr`` for metadata), so this script strips the
optimizer state and writes ~43 MB files with release-friendly names.

Usage::

    .venv/bin/python scripts/export_weights.py
    .venv/bin/python scripts/export_weights.py --run res03:edsr_x4_bicubic

The output directory (default ``release_assets/``) is git-ignored; upload its
files to a GitHub Release instead of committing them.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import torch

# Default set used by docs/RESULTS.md and scripts/fetch_weights.py.
DEFAULT_EXPORTS = (
    ("res03", "edsr_x4_bicubic.pth"),
    ("res08", "edsr_x4_esp32_v2.pth"),
)

KEEP_KEYS = ("model", "model_name", "model_kwargs", "args", "epoch", "psnr")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_one(runs_dir: Path, output_dir: Path, run: str, name: str) -> Path:
    src = runs_dir / run / "best.pth"
    if not src.is_file():
        raise FileNotFoundError(f"checkpoint not found: {src}")

    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    if "model" not in ckpt:
        raise ValueError(f"{src} has no 'model' state dict")

    slim = {key: ckpt[key] for key in KEEP_KEYS if key in ckpt}
    out = output_dir / name
    torch.save(slim, out)
    print(f"{out}  {out.stat().st_size / 1e6:.1f} MB  sha256={sha256(out)}")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Export inference-only .pth files for Releases")
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--output-dir", default="release_assets")
    p.add_argument(
        "--run",
        action="append",
        metavar="RUN:NAME",
        help="export one run (e.g. res03:edsr_x4_bicubic); repeatable. "
        "Default: res03 + res08.",
    )
    args = p.parse_args()

    runs_dir = Path(args.runs_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = DEFAULT_EXPORTS
    if args.run:
        pairs = []
        for item in args.run:
            if ":" not in item:
                raise SystemExit(f"--run must be RUN:NAME, got {item!r}")
            run, name = item.split(":", 1)
            pairs.append((run, name))

    for run, name in pairs:
        export_one(runs_dir, output_dir, run, name)
    print(f"\nUpload {output_dir}/*.pth to a GitHub Release (do not commit them).")


if __name__ == "__main__":
    main()
