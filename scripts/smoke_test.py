#!/usr/bin/env python3
"""Data-free smoke test: build both models, run inference, round-trip a checkpoint.

This verifies the Python environment, CUDA/CPU device selection, the model
forward pass, tiled inference and self-describing checkpoints without needing
DIV2K/Flickr2K or any trained weights.

Usage::

    .venv/bin/python scripts/smoke_test.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch
from PIL import Image

from superresolution.infer import load_model, super_resolve
from superresolution.model import build_model, model_kwargs_for

SCALE = 4


def check_models() -> None:
    x = torch.randn(1, 3, 32, 32)
    for name in ("espcn", "edsr"):
        kwargs = model_kwargs_for(name, {"scale": SCALE})
        model = build_model(name, **kwargs).eval()
        with torch.inference_mode():
            y = model(x)
        assert tuple(y.shape) == (1, 3, 32 * SCALE, 32 * SCALE), (name, y.shape)
        params = sum(p.numel() for p in model.parameters())
        print(f"ok  {name:5s} params={params:,} output={tuple(y.shape)}")


def check_inference_and_checkpoint() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model("espcn", scale=SCALE).eval().to(device)
    img = Image.fromarray(
        (torch.rand(48, 64, 3).numpy() * 255).astype("uint8"), mode="RGB"
    )

    sr = super_resolve(model, img, device, tile=32, tile_overlap=8)
    assert sr.size == (64 * SCALE, 48 * SCALE), sr.size
    print(f"ok  tiled inference {img.size} -> {sr.size} on {device.type}")

    with tempfile.TemporaryDirectory() as tmp:
        ckpt = Path(tmp) / "best.pth"
        torch.save(
            {
                "model": model.state_dict(),
                "model_name": "espcn",
                "model_kwargs": model_kwargs_for("espcn", {"scale": SCALE}),
            },
            ckpt,
        )
        restored = load_model(ckpt, device)
        assert isinstance(restored, type(model))
        print("ok  self-describing checkpoint round-trip")


def main() -> None:
    check_models()
    check_inference_and_checkpoint()
    print("smoke test passed")


if __name__ == "__main__":
    main()
