"""Generate pre-computed ESP32 LR datasets for paired training.

The output follows the existing data layout, so ``train.py --lr-type esp32``
uses the normal aligned HR/LR pipeline::

    data/DIV2K/DIV2K_train_LR_esp32/X4/0001x4.png
    data/DIV2K/DIV2K_valid_LR_esp32/X4/0801x4.png
    data/Flickr2K/Flickr2K_LR_esp32/X4/000001x4.png

Examples
--------
Generate all three splits with the frozen v1 model::

    python -m superresolution.degrade --version v1 --scale 4

Generate a small test batch::

    python -m superresolution.degrade --splits div2k_valid --max-items 10

Generate JPEG instead of the default PNG::

    python -m superresolution.degrade --output-format jpg
"""

from __future__ import annotations

import argparse
import json
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from . import (
    DEFAULT_VERSION,
    degrade_image,
    fingerprint,
    get_version,
    list_versions,
    profile_names,
    profile_weights,
)
from .stats import image_stats, summarize

PROJECT_ROOT = Path(__file__).resolve().parents[3]
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

# HR directory -> LR directory (without the X{scale} suffix).
SPLITS: dict[str, dict[str, str]] = {
    "div2k_train": {
        "hr": "DIV2K/DIV2K_train_HR",
        "lr": "DIV2K/DIV2K_train_LR_esp32",
        "profile_mode": "weighted",
    },
    "div2k_valid": {
        "hr": "DIV2K/DIV2K_valid_HR",
        "lr": "DIV2K/DIV2K_valid_LR_esp32",
        "profile_mode": "cycle",
    },
    "flickr2k": {
        # All 2650 HR images.  train.py keeps its existing 95/5 HR-level split.
        "hr": "Flickr2K/Flickr2K_HR",
        "lr": "Flickr2K/Flickr2K_LR_esp32",
        "profile_mode": "weighted",
    },
}


@dataclass(frozen=True)
class Task:
    source: Path
    target: Path
    source_rel: str
    target_rel: str
    split: str
    profile: str
    seed: int
    version: str
    scale: int
    severity: float
    output_format: str


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _resolve_project_path(raw: str | Path) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def collect_hr_files(hr_dir: Path) -> list[Path]:
    if not hr_dir.is_dir():
        return []
    return [
        p
        for p in sorted(hr_dir.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]


def _seed_for(base_seed: int, source_rel: str) -> int:
    # Stable across runs and independent of worker scheduling.
    return (int(base_seed) * 1_000_003 + zlib.crc32(source_rel.encode("utf-8"))) & 0x7FFFFFFF


def _sample_profile(
    rng: np.random.Generator,
    profiles: list[str],
    weights: dict[str, float],
) -> str:
    values = np.array([weights[p] for p in profiles], dtype=np.float64)
    values /= values.sum()
    return str(rng.choice(profiles, p=values))


def _process_one(task: Task) -> dict[str, object]:
    try:
        with Image.open(task.source) as im:
            hr = im.convert("RGB")
        lr, spec = degrade_image(
            hr,
            version=task.version,
            seed=task.seed,
            profile=task.profile,
            scale=task.scale,
            severity=task.severity,
        )
        task.target.parent.mkdir(parents=True, exist_ok=True)
        if task.output_format == "jpg":
            # The degradation already includes a JPEG round-trip; use a high
            # quality here to avoid adding strong second-generation artifacts.
            lr.save(task.target, format="JPEG", quality=95, subsampling=0)
        else:
            lr.save(task.target, format="PNG")

        stats = image_stats(lr)
        record = {
            "split": task.split,
            "source": task.source_rel,
            "lr": task.target_rel,
            "profile": task.profile,
            "version": task.version,
            "scale": task.scale,
            "severity": task.severity,
            "seed": task.seed,
            "spec": spec,
            "stats": stats,
        }
        return {"ok": True, "record": record}
    except Exception as exc:  # noqa: BLE001 - report per-file failures
        return {"ok": False, "source": str(task.source), "error": repr(exc)}


def _build_tasks(
    args: argparse.Namespace,
    data_root: Path,
    profiles: list[str],
    weights: dict[str, float],
) -> tuple[list[Task], dict[str, dict[str, int]]]:
    splits = [x.strip() for x in args.splits.split(",") if x.strip()]
    unknown = [x for x in splits if x not in SPLITS]
    if unknown:
        raise SystemExit(f"unknown splits: {unknown}; available: {list(SPLITS)}")

    tasks: list[Task] = []
    counts: dict[str, dict[str, int]] = {}
    for split in splits:
        meta = SPLITS[split]
        hr_dir = data_root / meta["hr"]
        lr_dir = data_root / meta["lr"] / f"X{args.scale}"
        files = collect_hr_files(hr_dir)
        if not files:
            print(f"warning: no HR files in {hr_dir}", file=sys.stderr)
            counts[split] = {"total": 0, "queued": 0, "skipped": 0}
            continue
        if args.max_items > 0:
            files = files[: args.max_items]

        mode = args.profile_mode if args.profile_mode != "auto" else meta["profile_mode"]
        queued = 0
        skipped = 0
        for index, source in enumerate(files):
            source_rel = _relative(source)
            seed = _seed_for(args.seed, source_rel)
            if mode == "cycle":
                profile = profiles[index % len(profiles)]
            else:
                profile = _sample_profile(np.random.default_rng(seed), profiles, weights)

            target = lr_dir / f"{source.stem}x{args.scale}.{args.output_format}"
            if target.exists() and not args.overwrite:
                skipped += 1
                continue
            tasks.append(
                Task(
                    source=source,
                    target=target,
                    source_rel=source_rel,
                    target_rel=_relative(target),
                    split=split,
                    profile=profile,
                    seed=seed,
                    version=args.version,
                    scale=args.scale,
                    severity=args.severity,
                    output_format=args.output_format,
                )
            )
            queued += 1
        counts[split] = {"total": len(files), "queued": queued, "skipped": skipped}
    return tasks, counts


def _write_manifest(
    manifest_dir: Path,
    args: argparse.Namespace,
    profiles: list[str],
    records: list[dict[str, object]],
    counts: dict[str, dict[str, int]],
    errors: list[dict[str, object]],
) -> None:
    manifest_dir.mkdir(parents=True, exist_ok=True)
    header = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "version": args.version,
        "fingerprint": fingerprint(args.version),
        "scale": args.scale,
        "severity": args.severity,
        "seed": args.seed,
        "profiles": profiles,
        "profile_mode": args.profile_mode,
        "output_format": args.output_format,
        "counts": counts,
        "generated": len(records),
        "errors": len(errors),
        "outputs": {
            split: f"{meta['lr']}/X{args.scale}" for split, meta in SPLITS.items()
        },
    }
    (manifest_dir / "manifest.json").write_text(
        json.dumps(header, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (manifest_dir / "manifest.jsonl").open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    by_profile = {
        profile: summarize(
            [r["stats"] for r in records if r["profile"] == profile]  # type: ignore[index]
        )
        for profile in profiles
    }
    by_split = {
        split: summarize(
            [r["stats"] for r in records if r["split"] == split]  # type: ignore[index]
        )
        for split in counts
    }
    summary = {
        "version": args.version,
        "fingerprint": header["fingerprint"],
        "generated": len(records),
        "overall": summarize([r["stats"] for r in records]),  # type: ignore[index]
        "by_profile": by_profile,
        "by_split": by_split,
        "counts": counts,
        "errors": errors,
    }
    (manifest_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate pre-computed ESP32 LR images for paired training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--scale", type=int, default=4, choices=[2, 3, 4, 8])
    p.add_argument("--version", default=DEFAULT_VERSION, choices=list(list_versions()))
    p.add_argument("--severity", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--splits",
        type=str,
        default="div2k_train,div2k_valid,flickr2k",
        help="comma-separated: div2k_train, div2k_valid, flickr2k",
    )
    p.add_argument(
        "--profiles",
        type=str,
        default="",
        help="comma-separated profile subset (default: all)",
    )
    p.add_argument(
        "--profile-mode",
        choices=["auto", "weighted", "cycle"],
        default="auto",
        help="auto = weighted for training splits, cycle for DIV2K valid",
    )
    p.add_argument(
        "--output-format",
        choices=["png", "jpg"],
        default="png",
        help="PNG keeps the baked-in JPEG artifacts without another lossy encode",
    )
    p.add_argument("--max-items", type=int, default=0, help="debug: cap each split")
    p.add_argument("--workers", type=int, default=4, help="parallel worker processes")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--manifest-dir",
        type=Path,
        default=Path("data/esp32_degrade_dataset"),
        help="dataset manifest / summary output directory",
    )
    p.add_argument("--no-manifest", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--list-versions", action="store_true")
    p.add_argument("--list-profiles", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if args.list_versions:
        print("\n".join(list_versions()))
        return
    if args.list_profiles:
        print("\n".join(profile_names(args.version)))
        return

    version = get_version(args.version)
    available = list(version.profile_names())
    requested = [x.strip() for x in args.profiles.split(",") if x.strip()]
    if requested:
        missing = [x for x in requested if x not in available]
        if missing:
            raise SystemExit(f"unknown profiles: {missing}; available: {available}")
        profiles = requested
    else:
        profiles = available
    weights = profile_weights(args.version)

    data_root = _resolve_project_path(args.data_root)
    manifest_dir = _resolve_project_path(args.manifest_dir)

    tasks, counts = _build_tasks(args, data_root, profiles, weights)
    total = sum(c["total"] for c in counts.values())
    queued = sum(c["queued"] for c in counts.values())
    skipped = sum(c["skipped"] for c in counts.values())
    print(f"version: {args.version} | fingerprint: {fingerprint(args.version)}")
    for split, c in counts.items():
        out_dir = data_root / SPLITS[split]["lr"] / f"X{args.scale}"
        print(
            f"  {split:14s} total={c['total']:5d} queued={c['queued']:5d} "
            f"skipped={c['skipped']:5d} -> {_relative(out_dir)}"
        )
    print(f"total={total} queued={queued} skipped={skipped}")

    if args.dry_run or not tasks:
        if skipped and not args.overwrite:
            print("existing files were skipped; use --overwrite to regenerate them")
        return

    records: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []
    workers = max(1, int(args.workers))
    if workers == 1:
        for task in tqdm(tasks, desc="esp32-lr", unit="img"):
            result = _process_one(task)
            if result["ok"]:
                records.append(result["record"])  # type: ignore[arg-type]
            else:
                errors.append(result)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_process_one, task): task for task in tasks}
            for future in tqdm(as_completed(futures), total=len(futures), desc="esp32-lr", unit="img"):
                result = future.result()
                if result["ok"]:
                    records.append(result["record"])  # type: ignore[arg-type]
                else:
                    errors.append(result)

    records.sort(key=lambda r: str(r["lr"]))
    if not args.no_manifest:
        _write_manifest(manifest_dir, args, profiles, records, counts, errors)
        print(f"manifest: {_relative(manifest_dir / 'manifest.jsonl')}")
        print(f"summary:  {_relative(manifest_dir / 'summary.json')}")

    print(f"generated {len(records)} images, {len(errors)} errors")
    if errors:
        for err in errors[:10]:
            print(f"  failed: {err.get('source')}: {err.get('error')}", file=sys.stderr)
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
