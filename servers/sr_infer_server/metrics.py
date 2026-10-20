"""PSNR / SSIM without adding a scikit-image dependency."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def _to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def _gaussian_window(
    window_size: int = 11,
    sigma: float = 1.5,
    channels: int = 3,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords**2) / (2.0 * sigma**2))
    g = g / g.sum()
    window = torch.outer(g, g)
    return window.expand(channels, 1, window_size, window_size).contiguous()


def psnr_ssim(sr: Image.Image, hr: Image.Image) -> dict[str, float]:
    """Compute PSNR/SSIM for two images with identical dimensions."""
    if sr.size != hr.size:
        raise ValueError("PSNR/SSIM require images with identical dimensions")

    x = _to_tensor(sr)
    y = _to_tensor(hr)

    mse = torch.mean((x - y) ** 2).item()
    psnr = 100.0 if mse == 0 else 10.0 * np.log10(1.0 / mse)

    window_size = 11
    pad = window_size // 2
    window = _gaussian_window(window_size=window_size, channels=3)

    mu_x = F.conv2d(x, window, padding=pad, groups=3)
    mu_y = F.conv2d(y, window, padding=pad, groups=3)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(x * x, window, padding=pad, groups=3) - mu_x2
    sigma_y2 = F.conv2d(y * y, window, padding=pad, groups=3) - mu_y2
    sigma_xy = F.conv2d(x * y, window, padding=pad, groups=3) - mu_xy

    c1 = 0.01**2
    c2 = 0.03**2
    ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    )
    ssim = float(ssim_map.mean().item())

    return {"psnr": float(psnr), "ssim": ssim}
