"""Frozen ESP32 camera degradation model, version 1.

IMPORTANT
---------
After a version has been used for review, do **not** edit its profile ranges or
pipeline.  Create ``v2.py`` instead and register it in ``__init__.py``.  This
keeps every reviewed dataset reproducible and makes A/B comparison honest.

The parameters below are calibrated against ``data/esp32_camera/data01`` (low
light, green cast, low saturation, noisy, soft) with lower weight given to the
other camera-test directories.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from . import ops

VERSION = "v1"

# ---------------------------------------------------------------------------
# Profile definitions
# ---------------------------------------------------------------------------
# A spec value is either:
#   * (lo, hi)  -> uniform / integer uniform draw
#   * [a, b, c] -> random choice
#   * scalar    -> fixed value
# Unspecified keys inherit DEFAULTS.
DEFAULTS: dict[str, object] = {
    "resample": "bicubic",
    "lens_blur": 0.0,
    "defocus_radius": 0,
    "motion_length": 0,
    "motion_angle": 0.0,
    "chroma_shift": 0,
    "vignette": 0.0,
    "ghost_count": 0,
    "ghost_alpha": 0.0,
    "ghost_shift": 0,
    "ghost_blur": 0.0,
    "smear_length": 0,
    "band_jitter": 0,
    "banding_amp": 0.0,
    "banding_freq": 0.0,
    "exposure": 1.0,
    "gamma": 1.0,
    "contrast": 1.0,
    "black_lift": 0.0,
    "highlight_knee": 0.98,
    "bloom": 0.0,
    "wb_r": 1.0,
    "wb_g": 1.0,
    "wb_b": 1.0,
    "green_offset": 0.0,
    "saturation": 1.0,
    "shot_noise": 0.0,
    "read_noise": 0.0,
    "chroma_noise": 0.0,
    "row_noise": 0.0,
    "col_noise": 0.0,
    "hot_pixel_rate": 0.0,
    "posterize": 0,
    "jpeg_quality": 95,
}

# Values are tuned to the ESP32 reference: mostly low-light, greenish, soft,
# low-saturation images.  The day_soft profile keeps some clean examples so the
# model also sees easier data.
PROFILES: dict[str, dict[str, object]] = {
    "day_soft": {
        "resample": ["lanczos", "bicubic"],
        "lens_blur": (0.45, 1.10),
        "exposure": (0.82, 1.10),
        "gamma": (0.94, 1.06),
        "contrast": (0.88, 1.00),
        "highlight_knee": (0.94, 0.99),
        "wb_r": (0.96, 1.04),
        "wb_g": (0.98, 1.04),
        "wb_b": (0.94, 1.03),
        "green_offset": (0.0, 0.006),
        "saturation": (0.55, 0.85),
        "read_noise": (0.0015, 0.005),
        "chroma_noise": (0.001, 0.0035),
        "vignette": (0.0, 0.08),
        "jpeg_quality": (68, 91),
    },
    "green_indoor": {
        "resample": ["bilinear", "bicubic"],
        "lens_blur": (0.80, 1.70),
        "defocus_radius": (0, 1),
        "exposure": (0.55, 0.92),
        "gamma": (0.92, 1.10),
        "contrast": (0.68, 0.90),
        "black_lift": (0.006, 0.030),
        "highlight_knee": (0.92, 0.98),
        "wb_r": (0.90, 0.99),
        "wb_g": (1.02, 1.09),
        "wb_b": (0.90, 0.99),
        "green_offset": (0.002, 0.012),
        "saturation": (0.35, 0.65),
        "shot_noise": (0.0, 0.008),
        "read_noise": (0.003, 0.010),
        "chroma_noise": (0.002, 0.008),
        "row_noise": (0.001, 0.004),
        "vignette": (0.02, 0.14),
        "jpeg_quality": (50, 82),
    },
    "lowlight_noise": {
        "resample": ["bilinear", "bicubic"],
        "lens_blur": (0.80, 1.80),
        "exposure": (0.42, 0.78),
        "gamma": (1.00, 1.22),
        "contrast": (0.62, 0.90),
        "black_lift": (0.002, 0.020),
        "highlight_knee": (0.88, 0.97),
        "wb_r": (0.90, 1.02),
        "wb_g": (1.00, 1.08),
        "wb_b": (0.92, 1.06),
        "green_offset": (0.0, 0.010),
        "saturation": (0.35, 0.72),
        "shot_noise": (0.006, 0.020),
        "read_noise": (0.005, 0.016),
        "chroma_noise": (0.005, 0.014),
        "row_noise": (0.002, 0.008),
        "col_noise": (0.001, 0.005),
        "hot_pixel_rate": (0.0, 0.00008),
        "vignette": (0.06, 0.24),
        "jpeg_quality": (44, 76),
    },
    "motion_blur": {
        "resample": ["bilinear", "bicubic"],
        "lens_blur": (0.45, 1.05),
        "motion_length": (3, 9),
        "exposure": (0.62, 1.02),
        "gamma": (0.95, 1.12),
        "contrast": (0.78, 0.96),
        "wb_r": (0.93, 1.03),
        "wb_g": (1.00, 1.07),
        "wb_b": (0.92, 1.03),
        "green_offset": (0.0, 0.008),
        "saturation": (0.38, 0.75),
        "read_noise": (0.003, 0.012),
        "chroma_noise": (0.002, 0.008),
        "vignette": (0.02, 0.16),
        "jpeg_quality": (52, 84),
    },
    "ghosting": {
        "resample": ["bilinear", "bicubic"],
        "lens_blur": (0.70, 1.40),
        "motion_length": (0, 3),
        "ghost_count": (1, 3),
        "ghost_alpha": (0.10, 0.34),
        "ghost_shift": (1, 7),
        "ghost_blur": (0.4, 1.3),
        "smear_length": (0, 7),
        "band_jitter": (0, 1),
        "banding_amp": (0.0, 0.022),
        "banding_freq": (0.6, 4.0),
        "exposure": (0.55, 0.95),
        "gamma": (0.96, 1.14),
        "contrast": (0.72, 0.92),
        "wb_r": (0.92, 1.02),
        "wb_g": (1.00, 1.08),
        "wb_b": (0.92, 1.03),
        "green_offset": (0.0, 0.010),
        "saturation": (0.35, 0.72),
        "read_noise": (0.004, 0.014),
        "chroma_noise": (0.003, 0.010),
        "row_noise": (0.001, 0.006),
        "vignette": (0.04, 0.18),
        "jpeg_quality": (48, 80),
    },
    "defocus": {
        "resample": ["bilinear", "bicubic"],
        "lens_blur": (1.00, 2.00),
        "defocus_radius": (1, 3),
        "exposure": (0.68, 1.04),
        "gamma": (0.94, 1.10),
        "contrast": (0.70, 0.90),
        "black_lift": (0.0, 0.016),
        "wb_r": (0.93, 1.02),
        "wb_g": (1.00, 1.07),
        "wb_b": (0.92, 1.02),
        "green_offset": (0.0, 0.008),
        "saturation": (0.38, 0.75),
        "read_noise": (0.003, 0.013),
        "chroma_noise": (0.002, 0.008),
        "vignette": (0.02, 0.14),
        "jpeg_quality": (54, 86),
    },
    "overexposed": {
        "resample": ["bicubic", "lanczos"],
        "lens_blur": (0.60, 1.30),
        "exposure": (1.08, 1.48),
        "gamma": (0.82, 1.02),
        "contrast": (0.82, 1.00),
        "highlight_knee": (0.78, 0.92),
        "bloom": (0.15, 0.55),
        "wb_r": (0.95, 1.05),
        "wb_g": (1.00, 1.05),
        "wb_b": (0.94, 1.04),
        "green_offset": (0.0, 0.006),
        "saturation": (0.38, 0.75),
        "read_noise": (0.003, 0.012),
        "chroma_noise": (0.002, 0.008),
        "vignette": (0.0, 0.10),
        "jpeg_quality": (50, 84),
    },
    "jpeg_hard": {
        "resample": ["bilinear", "bicubic"],
        "lens_blur": (0.60, 1.20),
        "exposure": (0.72, 1.05),
        "gamma": (0.94, 1.10),
        "contrast": (0.82, 0.98),
        "wb_r": (0.93, 1.03),
        "wb_g": (1.00, 1.07),
        "wb_b": (0.92, 1.03),
        "green_offset": (0.0, 0.008),
        "saturation": (0.38, 0.75),
        "read_noise": (0.002, 0.010),
        "chroma_noise": (0.002, 0.008),
        "vignette": (0.0, 0.10),
        "posterize": [0, 0, 0, 64, 96],
        "jpeg_quality": (34, 55),
    },
    "mixed_hard": {
        "resample": ["bilinear", "bicubic"],
        "lens_blur": (1.00, 2.00),
        "defocus_radius": (0, 2),
        "motion_length": (2, 7),
        "ghost_count": (1, 2),
        "ghost_alpha": (0.10, 0.30),
        "ghost_shift": (2, 7),
        "ghost_blur": (0.5, 1.4),
        "smear_length": (0, 6),
        "band_jitter": (0, 1),
        "banding_amp": (0.0, 0.020),
        "banding_freq": (0.8, 4.5),
        "exposure": (0.45, 0.85),
        "gamma": (0.98, 1.18),
        "contrast": (0.62, 0.88),
        "black_lift": (0.004, 0.022),
        "wb_r": (0.88, 1.00),
        "wb_g": (1.01, 1.09),
        "wb_b": (0.88, 1.00),
        "green_offset": (0.002, 0.014),
        "saturation": (0.32, 0.65),
        "shot_noise": (0.006, 0.020),
        "read_noise": (0.005, 0.016),
        "chroma_noise": (0.005, 0.014),
        "row_noise": (0.002, 0.008),
        "col_noise": (0.001, 0.004),
        "hot_pixel_rate": (0.0, 0.00005),
        "vignette": (0.08, 0.26),
        "jpeg_quality": (38, 64),
    },
}

PROFILE_WEIGHTS: dict[str, float] = {
    "day_soft": 0.10,
    "green_indoor": 0.20,
    "lowlight_noise": 0.20,
    "motion_blur": 0.12,
    "ghosting": 0.12,
    "defocus": 0.12,
    "overexposed": 0.07,
    "jpeg_hard": 0.05,
    "mixed_hard": 0.02,
}


def profile_names() -> tuple[str, ...]:
    return tuple(PROFILES)


def choose_profile(
    rng: np.random.Generator,
    profiles: list[str] | tuple[str, ...] | None = None,
) -> str:
    names = list(profiles) if profiles else list(PROFILES)
    weights = np.array([PROFILE_WEIGHTS[n] for n in names], dtype=np.float64)
    weights /= weights.sum()
    return str(rng.choice(names, p=weights))


def _draw(rng: np.random.Generator, value: object, key: str) -> object:
    if isinstance(value, tuple):
        lo, hi = value
        if key in ops.INT_FIELDS:
            return int(rng.integers(int(lo), int(hi) + 1))
        return float(rng.uniform(float(lo), float(hi)))
    if isinstance(value, list):
        return value[int(rng.integers(0, len(value)))]
    if key in ops.INT_FIELDS:
        return int(value)
    if key == "resample":
        return str(value)
    return float(value)


def sample_spec(
    profile: str,
    rng: np.random.Generator,
    severity: float = 1.0,
) -> dict[str, object]:
    """Sample one concrete degradation spec for a profile."""
    if profile not in PROFILES:
        raise KeyError(f"unknown profile: {profile} (available: {', '.join(PROFILES)})")
    raw = dict(DEFAULTS)
    raw.update(PROFILES[profile])

    spec: dict[str, object] = {"profile": profile, "version": VERSION}
    for key, value in raw.items():
        spec[key] = _draw(rng, value, key)

    severity = float(np.clip(severity, 0.0, 2.0))
    multiplicative = (
        "lens_blur",
        "defocus_radius",
        "motion_length",
        "chroma_shift",
        "vignette",
        "ghost_alpha",
        "ghost_shift",
        "ghost_blur",
        "smear_length",
        "band_jitter",
        "banding_amp",
        "bloom",
        "black_lift",
        "green_offset",
        "shot_noise",
        "read_noise",
        "chroma_noise",
        "row_noise",
        "col_noise",
        "hot_pixel_rate",
    )
    for key in multiplicative:
        spec[key] = type(spec[key])(float(spec[key]) * severity) if key in ops.INT_FIELDS else (
            float(spec[key]) * severity
        )

    for key in ("exposure", "contrast", "gamma", "saturation", "wb_r", "wb_g", "wb_b"):
        spec[key] = 1.0 + (float(spec[key]) - 1.0) * severity
    spec["highlight_knee"] = 1.0 - (1.0 - float(spec["highlight_knee"])) * severity
    spec["jpeg_quality"] = int(round(95 - (95 - int(spec["jpeg_quality"])) * severity))
    spec["jpeg_quality"] = int(np.clip(spec["jpeg_quality"], 1, 100))
    spec["jpeg_subsampling"] = 2 if spec["jpeg_quality"] < 75 else (1 if spec["jpeg_quality"] < 90 else 0)
    spec["motion_angle"] = float(rng.uniform(0.0, 180.0))
    return spec


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def run(
    img: Image.Image,
    spec: dict[str, object],
    rng: np.random.Generator,
    scale: int = 4,
) -> Image.Image:
    arr = ops.to_float(img)
    h, w = arr.shape[:2]
    target = (max(1, w // scale), max(1, h // scale))

    arr = ops.resize_rgb(arr, target, str(spec["resample"]))
    arr = ops.chromatic_aberration(arr, int(spec["chroma_shift"]))

    if float(spec["lens_blur"]) > 0:
        arr = ops.gaussian_blur(arr, float(spec["lens_blur"]))
    if int(spec["defocus_radius"]) > 0:
        arr = ops.convolve2d(arr, ops.disk_kernel(int(spec["defocus_radius"])))
    if int(spec["motion_length"]) > 1:
        arr = ops.convolve2d(
            arr,
            ops.motion_kernel(int(spec["motion_length"]), float(spec["motion_angle"])),
        )

    if int(spec["ghost_count"]) > 0 and float(spec["ghost_alpha"]) > 0:
        arr = ops.ghosting(
            arr,
            rng,
            int(spec["ghost_count"]),
            float(spec["ghost_alpha"]),
            max(1, int(spec["ghost_shift"])),
            float(spec["ghost_blur"]),
        )
    if int(spec["smear_length"]) > 1:
        arr = ops.smear(arr, rng, int(spec["smear_length"]))
    if int(spec["band_jitter"]) > 0:
        arr = ops.band_jitter(arr, rng, int(spec["band_jitter"]))

    arr = ops.apply_tone(arr, spec)
    arr = ops.apply_white_balance(arr, spec)
    arr = ops.soft_highlight(arr, float(spec["highlight_knee"]))
    arr = ops.bloom(arr, rng, float(spec["bloom"]))
    arr = ops.banding(arr, rng, float(spec["banding_amp"]), float(spec["banding_freq"]))
    arr = ops.vignette(arr, float(spec["vignette"]), rng)
    arr = ops.add_sensor_noise(arr, spec, rng)

    arr = ops.clip01(arr)
    if int(spec["posterize"]) > 0:
        arr = ops.posterize(arr, int(spec["posterize"]))
    arr = ops.jpeg_roundtrip(
        arr,
        int(spec["jpeg_quality"]),
        int(spec.get("jpeg_subsampling", 2)),
    )
    return ops.to_pil(arr)
