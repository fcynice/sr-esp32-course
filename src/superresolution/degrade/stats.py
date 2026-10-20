"""Small image statistics used to compare simulated LR with ESP32 references."""

from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageFilter

RGB_WEIGHTS = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def image_stats(img: Image.Image) -> dict[str, float | int]:
    im = img.convert("RGB")
    arr = np.asarray(im, dtype=np.float32)
    y = arr @ RGB_WEIGHTS
    mean = arr.reshape(-1, 3).mean(axis=0)
    sat = float((arr.max(axis=2) - arr.min(axis=2)).mean())

    blurred = np.asarray(im.filter(ImageFilter.GaussianBlur(1.2)), dtype=np.float32)
    residual = (arr - blurred)[:-1, :-1]
    gy = np.abs(np.diff(y, axis=0))[:, :-1]
    gx = np.abs(np.diff(y, axis=1))[:-1, :]
    flat = (gy + gx) < 3.0
    noise = float(np.std(residual[flat])) if flat.sum() > 100 else float("nan")

    lap = (
        -4.0 * y[1:-1, 1:-1]
        + y[:-2, 1:-1]
        + y[2:, 1:-1]
        + y[1:-1, :-2]
        + y[1:-1, 2:]
    )

    return {
        "width": int(im.width),
        "height": int(im.height),
        "mean_r": float(mean[0]),
        "mean_g": float(mean[1]),
        "mean_b": float(mean[2]),
        "green_cast": float(mean[1] - 0.5 * (mean[0] + mean[2])),
        "saturation": sat,
        "y_mean": float(y.mean()),
        "y_std": float(y.std()),
        "noise_sigma": noise,
        "laplacian_var": float(lap.var()),
        "clip_low": float((arr <= 2).mean()),
        "clip_high": float((arr >= 253).mean()),
    }


def summarize(records: list[dict[str, float | int]]) -> dict[str, dict[str, float]]:
    keys = [
        "mean_r",
        "mean_g",
        "mean_b",
        "green_cast",
        "saturation",
        "y_mean",
        "y_std",
        "noise_sigma",
        "laplacian_var",
        "clip_low",
        "clip_high",
    ]
    out: dict[str, dict[str, float]] = {}
    for key in keys:
        vals = np.array(
            [float(r[key]) for r in records if math.isfinite(float(r.get(key, "nan")))],
            dtype=np.float64,
        )
        if vals.size == 0:
            continue
        out[key] = {
            "mean": float(vals.mean()),
            "median": float(np.median(vals)),
            "p10": float(np.percentile(vals, 10)),
            "p90": float(np.percentile(vals, 90)),
            "min": float(vals.min()),
            "max": float(vals.max()),
        }
    return out
