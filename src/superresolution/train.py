"""Training entry point for the ESPCN super-resolution model.

Example
-------
    python -m superresolution.train \
        --scale 4 --epochs 100 --batch-size 16 --patch-size 128 \
        --data-root data --output-dir runs
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import (
    SRMixTrainDataset,
    SRValDataset,
    collect_hr_items,
    lr_type_of,
)
from .model import build_model, model_kwargs_for

# ---------------------------------------------------------------------------
# Data layout
# ---------------------------------------------------------------------------
TRAIN_SOURCES = [
    # (HR dir, LR dir template, reserved-for-val fraction)
    ("DIV2K/DIV2K_train_HR", "DIV2K/DIV2K_train_LR_{lr_type}/X{scale}", 0.0),
    ("Flickr2K/Flickr2K_HR", "Flickr2K/Flickr2K_LR_{lr_type}/X{scale}", 0.05),
]
VAL_SOURCES = [
    ("DIV2K/DIV2K_valid_HR", "DIV2K/DIV2K_valid_LR_{lr_type}/X{scale}", 1.0),
]


def _fmt(path: str, scale: int, lr_type: str) -> str:
    return path.format(scale=scale, lr_type=lr_type)


def requested_types(lr_type: str) -> tuple[str, ...]:
    """LR variants available for a run; ``both`` means bicubic + unknown."""
    return ("bicubic", "unknown") if lr_type == "both" else (lr_type,)


# Run directories are named ``res{index:02d}`` (e.g. ``res01``), assigned in
# chronological order (oldest run gets the smallest number).  Details such as
# the model, scale and lr_type live inside the run in ``run_info.json``.
RUN_DIR_RE = re.compile(r"^res(?P<idx>\d+)$")


def next_run_index(output_dir: str | Path) -> int:
    """Return the next free run index under ``output_dir``."""
    base = Path(output_dir)
    max_idx = 0
    if base.is_dir():
        for p in base.iterdir():
            m = RUN_DIR_RE.match(p.name)
            if m and p.is_dir():
                max_idx = max(max_idx, int(m.group("idx")))
    return max_idx + 1


def build_datasets(args: argparse.Namespace):
    root = Path(args.data_root)
    rng = random.Random(args.seed)

    train_types = requested_types(args.lr_type)
    val_types = train_types
    all_types = tuple(sorted(set(train_types) | set(val_types)))

    train_groups: list[tuple[Path, dict[str, Path]]] = []
    val_groups: list[tuple[Path, dict[str, Path]]] = []

    # Training sources (hold out a fraction of Flickr's HR images for validation).
    for hr_rel, lr_rel, val_frac in TRAIN_SOURCES:
        lr_dirs = {lt: root / _fmt(lr_rel, args.scale, lt) for lt in all_types}
        items = collect_hr_items(root / hr_rel, lr_dirs, args.scale)
        if not items:
            raise FileNotFoundError(
                f"No HR/LR pairs found for {hr_rel} ({', '.join(all_types)})"
            )
        rng.shuffle(items)
        n_val = int(len(items) * val_frac)
        val_groups.extend(items[:n_val])
        train_groups.extend(items[n_val:])

    # Dedicated validation sources (DIV2K valid).
    for hr_rel, lr_rel, _ in VAL_SOURCES:
        lr_dirs = {lt: root / _fmt(lr_rel, args.scale, lt) for lt in all_types}
        items = collect_hr_items(root / hr_rel, lr_dirs, args.scale)
        if not items:
            raise FileNotFoundError(
                f"No HR/LR pairs found for {hr_rel} ({', '.join(all_types)})"
            )
        val_groups.extend(items)

    def flatten(groups, types):
        pairs = []
        for hr, lrs in groups:
            for lt in types:
                if lt in lrs:
                    pairs.append((hr, lrs[lt]))
        return pairs

    return train_groups, flatten(val_groups, val_types)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
@torch.no_grad()
def psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> float:
    mse = torch.mean((pred - target) ** 2).item()
    if mse == 0:
        return float("inf")
    import math

    return 10.0 * math.log10((data_range**2) / mse)


# ---------------------------------------------------------------------------
# Train / validate
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_amp, epoch, epochs):
    model.train()
    running = 0.0
    pbar = tqdm(loader, desc=f"epoch {epoch}/{epochs}", leave=False)
    for lr, hr in pbar:
        lr = lr.to(device, non_blocking=True)
        hr = hr.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            sr = model(lr)
            loss = criterion(sr, hr)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running += loss.item() * lr.size(0)
        pbar.set_postfix(loss=f"{running / max(1, pbar.n * lr.size(0)):.5f}")
    return running / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, device, use_amp):
    model.eval()
    total_psnr = 0.0
    per_image = []
    for lr, hr, names in tqdm(loader, desc="val", leave=False):
        lr = lr.to(device, non_blocking=True)
        hr = hr.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            sr = model(lr)
        sr = sr.clamp(0.0, 1.0).float()
        p = psnr(sr, hr.float())
        total_psnr += p * lr.size(0)
        per_image.append((names[0], p))
    return total_psnr / max(1, len(loader.dataset)), per_image


@torch.no_grad()
def validate_all(model, loaders: dict, device, use_amp):
    """Validate on one loader per LR type and return the combined PSNR."""
    per_type: dict[str, float] = {}
    per_image = []
    weighted = 0.0
    count = 0
    for lr_type, loader in loaders.items():
        p, imgs = validate(model, loader, device, use_amp)
        per_type[lr_type] = p
        per_image.extend(imgs)
        n = len(loader.dataset)
        weighted += p * n
        count += n
    return weighted / max(1, count), per_type, per_image


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a super-resolution model")
    p.add_argument("--data-root", type=str, default="data")
    p.add_argument("--output-dir", type=str, default="runs")
    p.add_argument("--scale", type=int, default=4, choices=[2, 3, 4, 8])
    p.add_argument(
        "--model",
        type=str,
        default="edsr",
        choices=["espcn", "edsr"],
        help="espcn (small, 3 conv + PixelShuffle) or edsr (residual, larger)",
    )
    p.add_argument(
        "--lr-type",
        type=str,
        default="both",
        choices=["bicubic", "unknown", "both", "esp32"],
        help=(
            "bicubic = clean downscale; unknown = realistic degradation; "
            "both = mix bicubic and unknown; esp32 = pre-generated ESP32 LR "
            "under DIV2K_train_LR_esp32 / Flickr2K_LR_esp32"
        ),
    )
    p.add_argument("--patch-size", type=int, default=192, help="HR patch size")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument(
        "--lr",
        type=float,
        default=None,
        help="initial learning rate (default: 1e-3 for espcn, 2e-4 for edsr)",
    )
    p.add_argument("--channels", type=int, default=128, help="EDSR body channels")
    p.add_argument("--n-blocks", type=int, default=32, help="EDSR residual blocks")
    p.add_argument("--mid1", type=int, default=64, help="ESPCN conv1 channels")
    p.add_argument("--mid2", type=int, default=32, help="ESPCN conv2 channels")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--val-every", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--max-train-items", type=int, default=0, help="debug: cap train set")
    p.add_argument("--max-val-items", type=int, default=0, help="debug: cap val set")
    p.add_argument(
        "--init-from",
        type=str,
        default=None,
        help=(
            "initialize model weights from a checkpoint or run directory "
            "(e.g. runs/res05/best.pth); optimizer/scheduler/epoch are not loaded"
        ),
    )
    p.add_argument(
        "--resume",
        type=str,
        default=None,
        help="run directory or checkpoint to resume in-place (uses last.pth)",
    )
    return p.parse_args(argv)


def resolve_resume(path_str: str, output_dir: str | Path) -> tuple[Path, Path]:
    """Resolve ``--resume`` to ``(run_dir, checkpoint_path)``."""
    p = Path(path_str)
    if not p.exists():
        alt = Path(output_dir) / path_str
        if alt.exists():
            p = alt
        else:
            raise FileNotFoundError(f"resume target not found: {path_str}")
    if p.is_dir():
        return p, p / "last.pth"
    return p.parent, p


def resolve_init_checkpoint(path_str: str, output_dir: str | Path) -> Path:
    """Resolve ``--init-from`` to a checkpoint file.

    A run directory is accepted and resolves to ``best.pth`` first, then
    ``last.pth``.  Only model weights are loaded from this checkpoint.
    """
    p = Path(path_str)
    if not p.exists():
        alt = Path(output_dir) / path_str
        if alt.exists():
            p = alt
        else:
            raise FileNotFoundError(f"init checkpoint not found: {path_str}")
    if p.is_dir():
        for name in ("best.pth", "last.pth"):
            candidate = p / name
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(f"no best.pth or last.pth in {p}")
    if not p.is_file() or p.suffix.lower() != ".pth":
        raise ValueError(f"init checkpoint must be a .pth file or run directory: {p}")
    return p


def main(argv=None) -> None:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parsed = parse_args(raw_argv)

    if parsed.resume and parsed.init_from:
        raise SystemExit("--resume and --init-from cannot be used together")

    resume_ckpt = None
    init_ckpt = None
    init_path: Path | None = None
    run_dir: Path | None = None
    if parsed.init_from:
        init_path = resolve_init_checkpoint(parsed.init_from, parsed.output_dir)
        init_ckpt = torch.load(init_path, map_location="cpu", weights_only=False)

    if parsed.resume:
        run_dir, ckpt_path = resolve_resume(parsed.resume, parsed.output_dir)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
        resume_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        # Restore the original run configuration. Only runtime options and an
        # explicitly-provided --epochs (to extend training) are overridden.
        saved = dict(resume_ckpt.get("args", {}))
        for k, v in vars(parsed).items():
            saved.setdefault(k, v)
        if "model" not in saved:
            saved["model"] = resume_ckpt.get("model_name", "espcn")
        model_kw = resume_ckpt.get("model_kwargs") or {}
        for k in ("scale", "mid1", "mid2", "channels", "n_blocks"):
            if k in model_kw:
                saved.setdefault(k, model_kw[k])
        explicit_epochs = any(
            a == "--epochs" or a.startswith("--epochs=") for a in raw_argv
        )
        if explicit_epochs:
            saved["epochs"] = parsed.epochs
        saved["device"] = parsed.device
        saved["num_workers"] = parsed.num_workers
        args = argparse.Namespace(**saved)
    else:
        args = parsed

    if args.lr is None:
        args.lr = 1e-3 if args.model == "espcn" else 2e-4

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    use_amp = (not args.no_amp) and device.type == "cuda"

    train_groups, val_pairs = build_datasets(args)
    if args.max_train_items > 0:
        train_groups = train_groups[: args.max_train_items]
    if args.max_val_items > 0:
        val_pairs = val_pairs[: args.max_val_items]

    repeats = len(requested_types(args.lr_type))
    print(
        f"lr type: {args.lr_type} | train HRs: {len(train_groups)} "
        f"(x{repeats} = {len(train_groups) * repeats} samples) | "
        f"val pairs: {len(val_pairs)}"
    )

    train_ds = SRMixTrainDataset(
        train_groups,
        scale=args.scale,
        patch_size=args.patch_size,
        augment=not args.no_augment,
        seed=args.seed,
        repeats=repeats,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )

    val_by_type: dict[str, list[tuple[Path, Path]]] = {}
    for hr, lr in val_pairs:
        val_by_type.setdefault(lr_type_of(lr), []).append((hr, lr))
    val_loaders = {
        lr_type: DataLoader(
            SRValDataset(pairs, scale=args.scale),
            batch_size=1,
            shuffle=False,
            num_workers=min(2, args.num_workers),
            pin_memory=True,
        )
        for lr_type, pairs in val_by_type.items()
    }

    arch_kwargs = model_kwargs_for(args.model, args)
    model = build_model(args.model, **arch_kwargs)
    if init_ckpt is not None:
        ckpt_model_name = init_ckpt.get("model_name", "espcn")
        if ckpt_model_name != args.model:
            raise ValueError(
                f"init checkpoint model is {ckpt_model_name!r}, but --model is "
                f"{args.model!r}; use a matching model or change --model"
            )
        state_dict = init_ckpt.get("model", init_ckpt)
        if not isinstance(state_dict, dict):
            raise ValueError(f"init checkpoint has no model state dict: {init_path}")
        state_dict = {
            (k[7:] if k.startswith("module.") else k): v for k, v in state_dict.items()
        }
        model.load_state_dict(state_dict, strict=True)
        print(
            f"init model from {init_path} "
            f"(epoch={init_ckpt.get('epoch', '?')}, "
            f"psnr={init_ckpt.get('psnr', '?')}); "
            "optimizer/scheduler/epoch are fresh"
        )
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {args.model} {arch_kwargs}")
    print(f"model params: {n_params:,}")

    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Output directory + config snapshot.
    now = datetime.now()
    if run_dir is None:
        index = next_run_index(args.output_dir)
        run_dir = Path(args.output_dir) / f"res{index:02d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "run_info.json").write_text(
            json.dumps(
                {
                    "index": index,
                    "timestamp": now.isoformat(timespec="seconds"),
                    "lr_type": args.lr_type,
                    "scale": args.scale,
                    "model": args.model,
                    "model_kwargs": arch_kwargs,
                    "init_from": str(init_path) if init_path is not None else None,
                    "init_epoch": (
                        init_ckpt.get("epoch") if init_ckpt is not None else None
                    ),
                    "init_psnr": (
                        init_ckpt.get("psnr") if init_ckpt is not None else None
                    ),
                },
                indent=2,
            )
        )
    else:
        run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "args.json").write_text(json.dumps(vars(args), indent=2))
    print(f"run dir: {run_dir}")

    log_path = run_dir / "train_log.csv"
    if resume_ckpt is None or not log_path.exists():
        with log_path.open("w", newline="") as f:
            csv.writer(f).writerow(["epoch", "lr", "train_l1", "val_psnr", "seconds"])

    best_psnr = -1.0
    start_epoch = 1
    if resume_ckpt is not None:
        model.load_state_dict(resume_ckpt["model"])
        if "optimizer" in resume_ckpt:
            optimizer.load_state_dict(resume_ckpt["optimizer"])
        else:
            print("warning: no optimizer state in checkpoint; optimizer restarted")
        if "scaler" in resume_ckpt:
            scaler.load_state_dict(resume_ckpt["scaler"])

        start_epoch = int(resume_ckpt.get("epoch", 0)) + 1
        best_psnr = float(resume_ckpt.get("psnr", -1.0))
        # CosineAnnealingLR is incremental, so jumping last_epoch directly
        # would produce a wrong LR. Place the scheduler at the last completed
        # epoch and set the upcoming epoch's LR from the cosine formula.
        scheduler.last_epoch = start_epoch - 1
        eta_min = args.lr * 0.01
        lr_now = eta_min + (args.lr - eta_min) * 0.5 * (
            1.0 + math.cos(math.pi * (start_epoch - 1) / max(1, args.epochs))
        )
        for group in optimizer.param_groups:
            group["lr"] = lr_now
        print(
            f"resuming {run_dir.name} from epoch {start_epoch} "
            f"(best PSNR {best_psnr:.3f}, lr {lr_now:.2e}, "
            f"total epochs {args.epochs})"
        )

    def make_checkpoint(epoch: int, psnr_value: float, partial: bool = False) -> dict:
        return {
            "model": model.state_dict(),
            "model_name": args.model,
            "model_kwargs": arch_kwargs,
            "args": vars(args),
            "epoch": epoch,
            "psnr": psnr_value,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "partial": partial,
        }

    epoch = start_epoch - 1
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            t0 = time.time()
            train_loss = train_one_epoch(
                model, train_loader, criterion, optimizer, scaler, device, use_amp, epoch, args.epochs
            )
            scheduler.step()

            val_psnr = None
            per_type_psnr: dict[str, float] = {}
            if epoch % args.val_every == 0 or epoch == args.epochs:
                val_psnr, per_type_psnr, per_image = validate_all(
                    model, val_loaders, device, use_amp
                )
                if val_psnr > best_psnr:
                    best_psnr = val_psnr
                    torch.save(make_checkpoint(epoch, best_psnr), run_dir / "best.pth")
                (run_dir / "last_val.txt").write_text(
                    "\n".join(f"{n}\t{p:.4f}" for n, p in per_image)
                )

            torch.save(make_checkpoint(epoch, best_psnr), run_dir / "last.pth")

            dt = time.time() - t0
            cur_lr = scheduler.get_last_lr()[0]
            msg = f"[{epoch:3d}/{args.epochs}] l1={train_loss:.5f} lr={cur_lr:.2e}"
            if val_psnr is not None:
                msg += f" val_psnr={val_psnr:.3f}"
                if len(per_type_psnr) > 1:
                    msg += " (" + ", ".join(
                        f"{k}={v:.3f}" for k, v in per_type_psnr.items()
                    ) + ")"
                msg += f" (best {best_psnr:.3f})"
            msg += f" {dt:.1f}s"
            print(msg)
            with log_path.open("a", newline="") as f:
                csv.writer(f).writerow(
                    [epoch, f"{cur_lr:.6g}", f"{train_loss:.6f}",
                     "" if val_psnr is None else f"{val_psnr:.4f}", f"{dt:.2f}"]
                )
    except KeyboardInterrupt:
        torch.save(make_checkpoint(epoch, best_psnr, partial=True), run_dir / "last.pth")
        print(f"\ninterrupted at epoch {epoch}; saved {run_dir / 'last.pth'}")
        print(f"resume with: python -m superresolution.train --resume {run_dir}")
        return

    print(f"done. run dir: {run_dir} | best PSNR: {best_psnr:.3f}")


if __name__ == "__main__":
    main()
