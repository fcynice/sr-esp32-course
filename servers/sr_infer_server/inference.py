"""Inference wrapper with a small checkpoint/model cache."""

from __future__ import annotations

import logging
import re
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image, ImageOps

from superresolution.infer import load_model, super_resolve

from .config import Settings
from .datasets import find_hr_for_lr
from .metrics import psnr_ssim
from .paths import display_path

logger = logging.getLogger("sr_infer.inference")

_model_cache: OrderedDict[
    tuple[str, int, str], tuple[torch.nn.Module, torch.device]
] = OrderedDict()
_model_lock = threading.Lock()
_MODEL_CACHE_LIMIT = 4

_SAFE_STEM_RE = re.compile(r"[^\w\-\u4e00-\u9fff]+", re.UNICODE)


def resolve_device(settings: Settings) -> torch.device:
    requested = (settings.device or "auto").lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("requested %s but CUDA is unavailable, falling back to CPU", requested)
        return torch.device("cpu")
    return torch.device(requested)


def _get_model(
    settings: Settings, checkpoint: Path
) -> tuple[torch.nn.Module, torch.device]:
    device = resolve_device(settings)
    stat = checkpoint.stat()
    key = (str(checkpoint.resolve()), stat.st_mtime_ns, str(device))

    with _model_lock:
        cached = _model_cache.get(key)
        if cached is not None:
            _model_cache.move_to_end(key)
            return cached

    logger.info("loading checkpoint %s on %s", checkpoint, device)
    model = load_model(checkpoint, device)
    model.eval()

    with _model_lock:
        _model_cache[key] = (model, device)
        _model_cache.move_to_end(key)
        while len(_model_cache) > _MODEL_CACHE_LIMIT:
            _model_cache.popitem(last=False)
    return model, device


def _safe_stem(stem: str) -> str:
    cleaned = _SAFE_STEM_RE.sub("_", stem).strip("_")
    return (cleaned or "image")[:80]


def _result_path(settings: Settings, image_path: Path, scale: int) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return settings.results_dir / f"{stamp}_{_safe_stem(image_path.stem)}_x{scale}.png"


def run_inference(
    settings: Settings,
    image_path: Path,
    checkpoint_path: Path,
    progress_cb: Callable[[float, str], None],
) -> dict[str, object]:
    """Run one inference job and return a JSON-serialisable result."""
    started = time.time()
    image_path = image_path.resolve()
    checkpoint_path = checkpoint_path.resolve()

    progress_cb(0.01, "读取图片")
    with Image.open(image_path) as raw:
        lr = ImageOps.exif_transpose(raw).convert("RGB")

    pixels = lr.width * lr.height
    if pixels > settings.max_input_pixels:
        raise ValueError(
            f"图片像素数 {pixels:,} 超过上限 {settings.max_input_pixels:,}，"
            "请先裁剪或缩小图片"
        )

    progress_cb(0.05, "加载模型")
    model, device = _get_model(settings, checkpoint_path)
    scale = int(getattr(model, "scale", 1))

    output_pixels = lr.width * lr.height * scale * scale
    if output_pixels > settings.max_output_pixels:
        raise ValueError(
            f"推理输出约 {output_pixels:,} 像素，超过上限 "
            f"{settings.max_output_pixels:,}；请先缩小输入图或调整 "
            "SR_INFER_MAX_OUTPUT_PIXELS"
        )

    if (
        settings.auto_tile > 0
        and max(lr.width, lr.height) > settings.auto_tile_threshold
    ):
        tile = settings.auto_tile
        overlap = max(0, min(settings.tile_overlap, max(0, tile // 4)))
    else:
        tile = 0
        overlap = 0

    progress_cb(0.10, f"推理中（{device}）")

    def on_progress(fraction: float, message: str) -> None:
        progress_cb(0.10 + 0.80 * fraction, message)

    sr = super_resolve(
        model,
        lr,
        device,
        tile=tile,
        tile_overlap=overlap,
        progress_cb=on_progress,
    )

    progress_cb(0.92, "保存结果")
    settings.ensure_dirs()
    output_path = _result_path(settings, image_path, scale)
    sr.save(output_path)

    progress_cb(0.95, "检查 HR 配对")
    metrics: dict[str, float] | None = None
    metrics_note: str | None = None
    hr_info: dict[str, object] | None = None
    pair = find_hr_for_lr(settings, image_path)

    if pair is not None:
        hr_path = Path(str(pair["hr_abs_path"]))
        try:
            with Image.open(hr_path) as raw_hr:
                hr = ImageOps.exif_transpose(raw_hr).convert("RGB")
            hr_info = {
                "path": pair["hr_path"],
                "abs_path": pair["hr_abs_path"],
                "name": pair["hr_name"],
                "width": hr.width,
                "height": hr.height,
                "dataset": pair["dataset"],
                "lr_scale": pair["lr_scale"],
            }
            if hr.size == sr.size:
                progress_cb(0.96, "计算 PSNR / SSIM")
                metrics = psnr_ssim(sr, hr)
            else:
                metrics_note = (
                    f"HR 尺寸 {hr.width}×{hr.height} 与 SR 尺寸 "
                    f"{sr.width}×{sr.height} 不一致，未计算 PSNR/SSIM"
                )
        except Exception as exc:  # noqa: BLE001 - metrics are best-effort
            logger.exception("failed to compute metrics for %s", image_path)
            metrics_note = f"HR 读取或指标计算失败：{exc}"

    elapsed = time.time() - started
    result: dict[str, object] = {
        "lr_path": display_path(settings, image_path),
        "lr_abs_path": str(image_path),
        "lr_name": image_path.name,
        "lr_width": lr.width,
        "lr_height": lr.height,
        "sr_path": display_path(settings, output_path),
        "sr_abs_path": str(output_path),
        "sr_name": output_path.name,
        "sr_width": sr.width,
        "sr_height": sr.height,
        "checkpoint_path": display_path(settings, checkpoint_path),
        "run": checkpoint_path.parent.name,
        "checkpoint": checkpoint_path.stem,
        "model": type(model).__name__.lower(),
        "scale": scale,
        "device": str(device),
        "tile": tile,
        "tile_overlap": overlap,
        "elapsed_seconds": round(elapsed, 3),
        "metrics": metrics,
        "metrics_note": metrics_note,
        "hr": hr_info,
        "dataset": pair,
    }

    logger.info(
        "inference done: %s -> %s (%dx%d -> %dx%d) model=%s scale=%d device=%s %.2fs",
        image_path.name,
        output_path.name,
        lr.width,
        lr.height,
        sr.width,
        sr.height,
        result["model"],
        scale,
        device,
        elapsed,
    )
    progress_cb(1.0, "完成")
    return result
