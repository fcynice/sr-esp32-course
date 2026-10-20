"""Low-level image operations shared by versioned ESP32 degraders.

This module intentionally contains only deterministic, parameter-free building
blocks.  A degrader version (``v1``, ``v2``, ...) owns its own profile ranges
and pipeline so that an already-reviewed version never changes behaviour.
"""

from __future__ import annotations

import io
import math

import numpy as np
from PIL import Image

RGB_WEIGHTS = np.array([0.299, 0.587, 0.114], dtype=np.float32)

# Keys in a degradation spec that should be stored/treated as integers.
INT_FIELDS = frozenset(
    {
        "defocus_radius",
        "motion_length",
        "chroma_shift",
        "ghost_count",
        "ghost_shift",
        "smear_length",
        "band_jitter",
        "jpeg_quality",
        "posterize",
    }
)


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------
def to_float(img: Image.Image) -> np.ndarray:
    """RGB PIL image -> float32 HWC array in [0, 1]."""
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    return arr / 255.0


def to_uint8(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr * 255.0 + 0.5, 0, 255).astype(np.uint8)


def to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(to_uint8(arr), mode="RGB")


def from_uint8(arr: np.ndarray) -> np.ndarray:
    return arr.astype(np.float32) / 255.0


def clip01(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr, 0.0, 1.0)


def resize_rgb(
    arr: np.ndarray,
    size: tuple[int, int],
    method: str | int = "bicubic",
) -> np.ndarray:
    """Resize float RGB array. ``size`` is ``(width, height)``."""
    if size[0] <= 0 or size[1] <= 0:
        raise ValueError(f"invalid resize target: {size}")
    if arr.shape[1] == size[0] and arr.shape[0] == size[1]:
        return arr
    method_map = {
        "nearest": Image.Resampling.NEAREST,
        "box": Image.Resampling.BOX,
        "bilinear": Image.Resampling.BILINEAR,
        "bicubic": Image.Resampling.BICUBIC,
        "lanczos": Image.Resampling.LANCZOS,
    }
    resample = method_map.get(method, method) if isinstance(method, str) else method
    im = Image.fromarray(to_uint8(arr), mode="RGB")
    im = im.resize((int(size[0]), int(size[1])), resample=resample)
    return from_uint8(np.asarray(im))


def resize_float_plane(plane: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize a single-channel float plane with bilinear interpolation."""
    if plane.shape[1] == size[0] and plane.shape[0] == size[1]:
        return plane.astype(np.float32, copy=False)
    im = Image.fromarray(plane.astype(np.float32), mode="F")
    im = im.resize((int(size[0]), int(size[1])), Image.Resampling.BILINEAR)
    return np.asarray(im, dtype=np.float32)


# ---------------------------------------------------------------------------
# Convolution / blur
# ---------------------------------------------------------------------------
def _conv1d(arr: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    k = np.asarray(kernel, dtype=np.float32)
    if k.size <= 1:
        return arr
    pad = k.size // 2
    pads = [(0, 0)] * arr.ndim
    pads[axis] = (pad, pad)
    padded = np.pad(arr, pads, mode="reflect")
    windows = np.lib.stride_tricks.sliding_window_view(padded, k.size, axis=axis)
    return np.tensordot(windows, k, axes=([-1], [0])).astype(np.float32, copy=False)


def gaussian_kernel(sigma: float) -> np.ndarray:
    if sigma <= 0:
        return np.array([1.0], dtype=np.float32)
    radius = max(1, int(math.ceil(sigma * 3.0)))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-(x * x) / (2.0 * sigma * sigma))
    return k / k.sum()


def gaussian_blur(arr: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return arr
    k = gaussian_kernel(sigma)
    return _conv1d(_conv1d(arr, k, axis=0), k, axis=1)


def disk_kernel(radius: int) -> np.ndarray:
    radius = max(0, int(radius))
    if radius <= 0:
        return np.ones((1, 1), dtype=np.float32)
    y, x = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    mask = (x * x + y * y) <= (radius + 0.35) ** 2
    k = mask.astype(np.float32)
    return k / k.sum()


def motion_kernel(length: int, angle_deg: float) -> np.ndarray:
    """A thin line kernel approximating shutter motion blur."""
    length = max(1, int(length))
    if length <= 1:
        return np.ones((1, 1), dtype=np.float32)
    radius = length // 2
    size = radius * 2 + 1
    y, x = np.mgrid[-radius : radius + 1, -radius : radius + 1].astype(np.float32)
    theta = math.radians(angle_deg)
    # Distance from the line through the origin with direction theta.
    dist = np.abs(-math.sin(theta) * x + math.cos(theta) * y)
    along = x * math.cos(theta) + y * math.sin(theta)
    mask = (dist <= 0.75) & (np.abs(along) <= radius)
    k = mask.astype(np.float32)
    if k.sum() <= 0:
        return np.ones((1, 1), dtype=np.float32)
    return k / k.sum()


def convolve2d(arr: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    k = np.asarray(kernel, dtype=np.float32)
    if k.shape == (1, 1):
        return arr
    kh, kw = k.shape
    pad_y, pad_x = kh // 2, kw // 2
    padded = np.pad(arr, ((pad_y, pad_y), (pad_x, pad_x), (0, 0)), mode="reflect")
    windows = np.lib.stride_tricks.sliding_window_view(padded, (kh, kw), axis=(0, 1))
    # (H, W, C, kh, kw) @ (kh, kw)
    return np.tensordot(windows, k, axes=([3, 4], [0, 1])).astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Geometric / temporal artifacts
# ---------------------------------------------------------------------------
def translate(arr: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Translate with reflect padding (positive dx/dy moves content right/down)."""
    dx, dy = int(dx), int(dy)
    h, w = arr.shape[:2]
    if dx == 0 and dy == 0:
        return arr
    py, px = abs(dy), abs(dx)
    padded = np.pad(
        arr,
        ((py, py), (px, px), (0, 0)),
        mode="reflect",
    )
    y0 = py - dy
    x0 = px - dx
    return padded[y0 : y0 + h, x0 : x0 + w]


def chromatic_aberration(arr: np.ndarray, shift: int) -> np.ndarray:
    shift = int(shift)
    if shift <= 0:
        return arr
    out = arr.copy()
    out[..., 0] = translate(arr[..., 0:1], shift, 0)[..., 0]
    out[..., 2] = translate(arr[..., 2:3], -shift, 0)[..., 0]
    return out


def ghosting(
    arr: np.ndarray,
    rng: np.random.Generator,
    count: int,
    alpha: float,
    shift: int,
    blur: float,
) -> np.ndarray:
    """Blend shifted/blurred echoes of the current frame into itself."""
    if count <= 0 or alpha <= 0:
        return arr
    base = arr.copy()
    out = arr.copy()
    for i in range(count):
        dx = int(rng.integers(-shift, shift + 1))
        dy = int(rng.integers(-shift, shift + 1))
        if dx == 0 and dy == 0:
            dx = shift if rng.random() < 0.5 else -shift
        echo = translate(base, dx, dy)
        if blur > 0:
            echo = gaussian_blur(echo, blur)
        a = alpha * (0.62**i)
        out = out * (1.0 - a) + echo * a
    return out


def smear(arr: np.ndarray, rng: np.random.Generator, length: int) -> np.ndarray:
    """Slow-shutter smear using several shifted copies."""
    length = int(length)
    if length <= 1:
        return arr
    horizontal = bool(rng.integers(0, 2))
    sign = 1 if rng.random() < 0.5 else -1
    acc = np.zeros_like(arr)
    weight = 0.0
    for i in range(length):
        w = 0.5 ** i
        d = sign * i
        shifted = translate(arr, d, 0) if horizontal else translate(arr, 0, d)
        acc += shifted * w
        weight += w
    return acc / max(weight, 1e-6)


def band_jitter(arr: np.ndarray, rng: np.random.Generator, amplitude: int) -> np.ndarray:
    """Rolling-shutter style horizontal band displacement (integer pixels)."""
    amplitude = int(amplitude)
    if amplitude <= 0:
        return arr
    h = arr.shape[0]
    n_bands = int(rng.integers(2, 7))
    edges = np.linspace(0, h, n_bands + 1, dtype=int)
    out = arr.copy()
    for a, b in zip(edges[:-1], edges[1:]):
        if b <= a:
            continue
        dx = int(rng.integers(-amplitude, amplitude + 1))
        if dx:
            out[a:b] = np.roll(arr[a:b], dx, axis=1)
    return out


# ---------------------------------------------------------------------------
# Exposure / colour / sensor
# ---------------------------------------------------------------------------
def apply_tone(arr: np.ndarray, spec: dict) -> np.ndarray:
    out = arr * float(spec["exposure"])
    lift = float(spec["black_lift"])
    if lift:
        out = out + lift * (1.0 - np.clip(out, 0.0, 1.0))
    contrast = float(spec["contrast"])
    if contrast != 1.0:
        out = (out - 0.5) * contrast + 0.5
    gamma = float(spec["gamma"])
    if gamma != 1.0:
        out = np.sign(out) * (np.abs(out) ** gamma)
    return out


def apply_white_balance(arr: np.ndarray, spec: dict) -> np.ndarray:
    out = arr.copy()
    out[..., 0] *= float(spec["wb_r"])
    out[..., 1] *= float(spec["wb_g"])
    out[..., 2] *= float(spec["wb_b"])
    green_offset = float(spec["green_offset"])
    if green_offset:
        out[..., 1] += green_offset
    sat = float(spec["saturation"])
    if sat != 1.0:
        luma = out @ RGB_WEIGHTS
        out = luma[..., None] + sat * (out - luma[..., None])
    return out


def soft_highlight(arr: np.ndarray, knee: float) -> np.ndarray:
    knee = float(np.clip(knee, 0.5, 0.999))
    if knee >= 0.999:
        return arr
    hi = np.maximum(arr - knee, 0.0)
    compressed = knee + (1.0 - knee) * np.tanh(hi / (1.0 - knee))
    return np.where(arr > knee, compressed, arr)


def bloom(arr: np.ndarray, rng: np.random.Generator, strength: float) -> np.ndarray:
    if strength <= 0:
        return arr
    luma = clip01(arr @ RGB_WEIGHTS)
    threshold = float(rng.uniform(0.72, 0.92))
    mask = np.clip((luma - threshold) / (1.0 - threshold), 0.0, 1.0)
    if mask.max() <= 0:
        return arr
    sigma = float(rng.uniform(2.0, 7.0))
    glow = gaussian_blur(mask[..., None], sigma)[..., 0]
    tint = np.array([1.0, 0.97, 0.90], dtype=np.float32)
    return arr + strength * glow[..., None] * tint * (1.0 - arr)


def vignette(arr: np.ndarray, strength: float, rng: np.random.Generator) -> np.ndarray:
    if strength <= 0:
        return arr
    h, w = arr.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx = w * (0.5 + float(rng.uniform(-0.08, 0.08)))
    cy = h * (0.5 + float(rng.uniform(-0.08, 0.08)))
    r2 = ((xx - cx) / (w * 0.5)) ** 2 + ((yy - cy) / (h * 0.5)) ** 2
    mask = 1.0 - strength * np.clip(r2 / 2.0, 0.0, 1.0) ** 1.25
    return arr * mask[..., None]


def banding(arr: np.ndarray, rng: np.random.Generator, amplitude: float, freq: float) -> np.ndarray:
    if amplitude <= 0:
        return arr
    h = arr.shape[0]
    phase = float(rng.uniform(0, 2 * math.pi))
    if freq <= 0:
        freq = float(rng.uniform(0.5, 5.0))
    y = np.arange(h, dtype=np.float32)
    wave = np.sin(2 * math.pi * freq * y / max(h, 1) + phase)
    return arr * (1.0 + amplitude * wave[:, None, None])


def add_sensor_noise(arr: np.ndarray, spec: dict, rng: np.random.Generator) -> np.ndarray:
    out = arr.astype(np.float32, copy=True)
    h, w = out.shape[:2]

    shot = float(spec["shot_noise"])
    if shot > 0:
        sigma = np.sqrt(shot * np.clip(out, 0.0, 1.0))
        out += rng.normal(0.0, 1.0, size=out.shape).astype(np.float32) * sigma

    read = float(spec["read_noise"])
    if read > 0:
        out += rng.normal(0.0, read, size=out.shape).astype(np.float32)

    chroma = float(spec["chroma_noise"])
    if chroma > 0:
        lh, lw = max(2, h // 6), max(2, w // 6)
        field = rng.normal(0.0, chroma, size=(lh, lw, 2)).astype(np.float32)
        c0 = resize_float_plane(field[..., 0], (w, h))
        c1 = resize_float_plane(field[..., 1], (w, h))
        out[..., 0] += c0
        out[..., 1] -= 0.5 * c0
        out[..., 2] += c1

    row_sigma = float(spec["row_noise"])
    if row_sigma > 0:
        rows = rng.normal(0.0, row_sigma, size=(h, 1, 1)).astype(np.float32)
        rows = gaussian_blur(rows, 0.8)
        out += rows

    col_sigma = float(spec["col_noise"])
    if col_sigma > 0:
        cols = rng.normal(0.0, col_sigma, size=(1, w, 1)).astype(np.float32)
        cols = gaussian_blur(cols, 0.8)
        out += cols

    hot_rate = float(spec["hot_pixel_rate"])
    if hot_rate > 0:
        n = int(hot_rate * h * w)
        if n > 0:
            ys = rng.integers(0, h, size=n)
            xs = rng.integers(0, w, size=n)
            ch = rng.integers(0, 3, size=n)
            vals = rng.uniform(0.55, 1.0, size=n).astype(np.float32)
            out[ys, xs, ch] = vals

    return out


# ---------------------------------------------------------------------------
# Compression / finishing
# ---------------------------------------------------------------------------
def jpeg_roundtrip(arr: np.ndarray, quality: int, subsampling: int | None = None) -> np.ndarray:
    quality = int(np.clip(quality, 1, 100))
    if subsampling is None:
        subsampling = 2 if quality < 75 else (1 if quality < 90 else 0)
    im = Image.fromarray(to_uint8(arr), mode="RGB")
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality, subsampling=int(subsampling))
    buf.seek(0)
    with Image.open(buf) as decoded:
        return to_float(decoded)


def posterize(arr: np.ndarray, levels: int) -> np.ndarray:
    levels = int(levels)
    if levels <= 0 or levels >= 256:
        return arr
    return np.round(arr * (levels - 1)) / (levels - 1)
