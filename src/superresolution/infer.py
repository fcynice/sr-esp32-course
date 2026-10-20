"""Inference script for a trained ESPCN super-resolution model.

Examples
--------
Single image::

    python -m superresolution.infer --checkpoint runs/xxx/best.pth \
        --input lr.png --output sr.png

Whole directory::

    python -m superresolution.infer --checkpoint runs/xxx/best.pth \
        --input lr_dir --output sr_dir
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .model import build_model, model_kwargs_for

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def load_model(checkpoint: str | Path, device: torch.device):
    """Rebuild a model from a self-describing checkpoint.

    New checkpoints store ``model_name`` / ``model_kwargs``.  Older checkpoints
    (before the registry existed) are treated as ESPCN.
    """
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model_name = ckpt.get("model_name")
    model_kwargs = ckpt.get("model_kwargs")
    if model_name is None:
        model_name = "espcn"
        model_kwargs = model_kwargs_for("espcn", ckpt.get("args", {}))
    elif model_kwargs is None:
        model_kwargs = model_kwargs_for(model_name, ckpt.get("args", {}))

    model = build_model(model_name, **model_kwargs)
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)
    return model


def _tile_starts(size: int, tile: int, step: int) -> list[int]:
    """Return tile origins covering ``size`` with the last tile flush at the edge."""
    if size <= tile:
        return [0]
    starts = list(range(0, size - tile + 1, step))
    tail = size - tile
    if starts[-1] != tail:
        # If the tail is within the overlap distance of the previous tile, the
        # last regular tile can be moved to the edge instead of adding another
        # (mostly redundant) tile.
        overlap = tile - step
        if len(starts) >= 2 and tail - starts[-1] <= overlap:
            starts[-1] = tail
        else:
            starts.append(tail)
    return starts


@torch.inference_mode()
def super_resolve(
    model: torch.nn.Module,
    img: Image.Image,
    device: torch.device,
    tile: int = 0,
    tile_overlap: int = 8,
    progress_cb: Callable[[float, str], None] | None = None,
) -> Image.Image:
    """Run the model on an RGB PIL image, optionally in tiles (saves memory).

    ``progress_cb`` receives ``(fraction, message)`` where fraction is in
    ``[0, 1]``.  Tiled inference reports after every tile; full-image inference
    reports only start and finish.
    """

    def report(fraction: float, message: str) -> None:
        if progress_cb is not None:
            progress_cb(max(0.0, min(1.0, fraction)), message)

    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)

    if tile <= 0 or (tensor.shape[-2] <= tile and tensor.shape[-1] <= tile):
        report(0.05, "整图推理")
        out = model(tensor)
        report(0.9, "转换输出")
        out = out.clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
        result = Image.fromarray((out * 255.0 + 0.5).astype(np.uint8))
        report(1.0, "完成")
        return result

    # Tiled inference with overlap blending.
    scale = model.scale
    _, _, h, w = tensor.shape
    out = torch.zeros((1, 3, h * scale, w * scale), device=device)
    weight = torch.zeros((1, 1, h * scale, w * scale), device=device)
    step = max(1, tile - max(0, min(tile_overlap, tile - 1)))
    ys = _tile_starts(h, tile, step)
    xs = _tile_starts(w, tile, step)
    total = len(ys) * len(xs)
    done = 0
    report(0.0, f"0/{total} 块")
    for y in ys:
        for x in xs:
            patch = tensor[:, :, y : y + tile, x : x + tile]
            sr = model(patch)
            py, px = y * scale, x * scale
            ph, pw = sr.shape[-2], sr.shape[-1]
            out[:, :, py : py + ph, px : px + pw] += sr
            weight[:, :, py : py + ph, px : px + pw] += 1.0
            done += 1
            report(done / total, f"{done}/{total} 块")
    report(1.0, "拼接分块")
    out = (out / weight.clamp_min(1e-8)).clamp(0, 1).squeeze(0).permute(1, 2, 0)
    result = Image.fromarray((out.cpu().numpy() * 255.0 + 0.5).astype(np.uint8))
    report(1.0, "完成")
    return result


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ESPCN inference")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--input", type=str, required=True, help="image or directory")
    p.add_argument("--output", type=str, required=True, help="image or directory")
    p.add_argument("--tile", type=int, default=0, help="tile size, 0 = full image")
    p.add_argument("--tile-overlap", type=int, default=8)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)

    src = Path(args.input)
    dst = Path(args.output)

    if src.is_dir():
        dst.mkdir(parents=True, exist_ok=True)
        files = sorted(f for f in src.iterdir() if f.suffix.lower() in IMG_EXTS)
        for f in files:
            img = Image.open(f)
            sr = super_resolve(model, img, device, args.tile, args.tile_overlap)
            out_path = dst / f"{f.stem}_x{model.scale}.png"
            sr.save(out_path)
            print(f"{f.name} {img.size} -> {out_path.name} {sr.size}")
    else:
        img = Image.open(src)
        sr = super_resolve(model, img, device, args.tile, args.tile_overlap)
        dst.parent.mkdir(parents=True, exist_ok=True)
        sr.save(dst)
        print(f"{src.name} {img.size} -> {dst.name} {sr.size}")


if __name__ == "__main__":
    main()
