"""Scan ``runs/resNN`` directories and expose their metadata."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from .config import Settings

RUN_DIR_RE = re.compile(r"^res(?P<idx>\d+)$")
CHECKPOINT_NAMES = ("best", "last")


def _read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _read_train_log(path: Path) -> dict[str, object]:
    result: dict[str, object] = {
        "best_psnr": None,
        "best_epoch": None,
        "last_epoch": None,
        "last_psnr": None,
        "epochs_logged": 0,
    }
    if not path.is_file():
        return result
    try:
        with path.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
    except OSError:
        return result

    result["epochs_logged"] = len(rows)
    best_psnr = float("-inf")
    for row in rows:
        epoch_raw = (row.get("epoch") or "").strip()
        psnr_raw = (row.get("val_psnr") or "").strip()
        if epoch_raw:
            try:
                result["last_epoch"] = int(float(epoch_raw))
            except ValueError:
                pass
        if not psnr_raw:
            continue
        try:
            value = float(psnr_raw)
            epoch = int(float(epoch_raw)) if epoch_raw else None
        except ValueError:
            continue
        result["last_psnr"] = value
        if value > best_psnr:
            best_psnr = value
            result["best_psnr"] = value
            result["best_epoch"] = epoch
    return result


def _display(settings: Settings, path: Path) -> str:
    try:
        return path.resolve().relative_to(settings.project_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _summary(run_info: dict, args: dict) -> dict[str, object]:
    model = run_info.get("model") or args.get("model") or "espcn"
    scale = run_info.get("scale") or args.get("scale") or 4
    lr_type = run_info.get("lr_type") or args.get("lr_type") or "unknown"
    model_kwargs = run_info.get("model_kwargs") or {}

    if model == "edsr":
        channels = model_kwargs.get("channels", args.get("channels"))
        n_blocks = model_kwargs.get("n_blocks", args.get("n_blocks"))
        arch_text = (
            f"{channels} 通道 / {n_blocks} 残差块"
            if channels is not None and n_blocks is not None
            else ""
        )
    elif model == "espcn":
        mid1 = model_kwargs.get("mid1", args.get("mid1"))
        mid2 = model_kwargs.get("mid2", args.get("mid2"))
        arch_text = (
            f"中间通道 {mid1} / {mid2}"
            if mid1 is not None and mid2 is not None
            else ""
        )
    else:
        arch_text = ""

    return {
        "model": model,
        "scale": scale,
        "lr_type": lr_type,
        "arch_text": arch_text,
        "channels": (
            model_kwargs.get("channels", args.get("channels"))
            if model == "edsr"
            else None
        ),
        "n_blocks": (
            model_kwargs.get("n_blocks", args.get("n_blocks"))
            if model == "edsr"
            else None
        ),
        "mid1": (
            model_kwargs.get("mid1", args.get("mid1"))
            if model == "espcn"
            else None
        ),
        "mid2": (
            model_kwargs.get("mid2", args.get("mid2"))
            if model == "espcn"
            else None
        ),
        "patch_size": args.get("patch_size"),
        "batch_size": args.get("batch_size"),
        "epochs": args.get("epochs"),
        "lr": args.get("lr"),
        "seed": args.get("seed"),
        "no_augment": args.get("no_augment"),
        "no_amp": args.get("no_amp"),
        "train_device": args.get("device"),
        "timestamp": run_info.get("timestamp"),
        "index": run_info.get("index"),
    }


def scan_runs(settings: Settings) -> list[dict[str, object]]:
    base = settings.runs_dir
    if not base.is_dir():
        return []

    runs: list[tuple[int, dict[str, object]]] = []
    for path in base.iterdir():
        if not path.is_dir():
            continue
        match = RUN_DIR_RE.match(path.name)
        if match is None:
            continue

        run_info = _read_json(path / "run_info.json")
        args = _read_json(path / "args.json")
        log = _read_train_log(path / "train_log.csv")
        summary = _summary(run_info, args)

        index = run_info.get("index")
        if not isinstance(index, int):
            index = int(match.group("idx"))

        checkpoints: list[dict[str, object]] = []
        for name in CHECKPOINT_NAMES:
            ckpt = path / f"{name}.pth"
            if not ckpt.is_file():
                continue
            if name == "best":
                epoch = log.get("best_epoch")
                psnr = log.get("best_psnr")
            else:
                epoch = log.get("last_epoch")
                psnr = log.get("last_psnr")
            checkpoints.append(
                {
                    "name": name,
                    "label": f"{name}.pth",
                    "path": _display(settings, ckpt),
                    "epoch": epoch,
                    "psnr": psnr,
                }
            )

        runs.append(
            (
                index,
                {
                    "name": path.name,
                    "index": index,
                    "path": _display(settings, path),
                    "info": summary,
                    "run_info": run_info,
                    "args": args,
                    "checkpoints": checkpoints,
                    "log": log,
                },
            )
        )

    runs.sort(key=lambda item: item[0], reverse=True)
    return [item[1] for item in runs]
