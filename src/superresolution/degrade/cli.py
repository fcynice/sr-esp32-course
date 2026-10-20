"""Review-image generator for ESP32 degradation models.

Examples
--------
List frozen profiles::

    python -m superresolution.degrade --list-profiles

Generate a stratified review session (2 images per profile per dataset)::

    python -m superresolution.degrade --per-profile 2 --seed 42

The output is intentionally *not* used by training.  Training uses the same
frozen version on the fly; these files exist only for human review.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

from . import DEFAULT_VERSION, degrade_image, fingerprint, get_version, list_versions, profile_names
from .stats import image_stats, summarize

PROJECT_ROOT = Path(__file__).resolve().parents[3]
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

SOURCES = {
    "div2k_train": {
        "group": "div2k",
        "split": "train",
        "hr_dir": "DIV2K/DIV2K_train_HR",
    },
    "div2k_valid": {
        "group": "div2k",
        "split": "valid",
        "hr_dir": "DIV2K/DIV2K_valid_HR",
    },
    "flickr2k_train": {
        "group": "flickr2k",
        "split": "train",
        "hr_dir": "Flickr2K/Flickr2K_HR",
    },
}


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


def _collect_pools(data_root: Path, dataset_names: list[str]) -> dict[str, list[dict[str, object]]]:
    pools: dict[str, list[dict[str, object]]] = {}
    for source_name, meta in SOURCES.items():
        group = str(meta["group"])
        if group not in dataset_names:
            continue
        hr_dir = data_root / str(meta["hr_dir"])
        files = collect_hr_files(hr_dir)
        if not files:
            print(f"warning: no HR files in {hr_dir}", file=sys.stderr)
            continue
        for path in files:
            pools.setdefault(group, []).append(
                {
                    "source_name": source_name,
                    "group": group,
                    "split": str(meta["split"]),
                    "path": path,
                }
            )
    return pools


def _pick_records(
    pools: dict[str, list[dict[str, object]]],
    profiles: list[str],
    per_profile: int,
    rng: random.Random,
) -> list[dict[str, object]]:
    # Shuffle copies so every profile gets different HR images across the run.
    remaining = {group: list(items) for group, items in pools.items()}
    for items in remaining.values():
        rng.shuffle(items)

    records: list[dict[str, object]] = []
    groups = [g for g in ("div2k", "flickr2k") if g in remaining]
    for profile in profiles:
        for group in groups:
            items = remaining.get(group, [])
            if len(items) < per_profile:
                raise RuntimeError(
                    f"not enough unused HR images for group={group}: "
                    f"need {per_profile}, have {len(items)}"
                )
            chosen = [items.pop() for _ in range(per_profile)]
            for item in chosen:
                seed = rng.randrange(0, 2**31 - 1)
                records.append(
                    {
                        **item,
                        "profile": profile,
                        "seed": seed,
                    }
                )
    return records


def _output_name(record: dict[str, object]) -> str:
    path = Path(str(record["path"]))
    return f"{path.stem}__{record['profile']}__s{record['seed']}.png"


def _make_contact_sheet(
    items: list[tuple[Path, str]],
    output: Path,
    *,
    columns: int = 6,
    tile_size: tuple[int, int] = (260, 190),
) -> None:
    if not items:
        return
    tw, th = tile_size
    label_h = 26
    rows = (len(items) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tw, rows * (th + label_h)), (18, 18, 20))
    draw = ImageDraw.Draw(sheet)
    for idx, (path, label) in enumerate(items):
        row, col = divmod(idx, columns)
        x, y = col * tw, row * (th + label_h)
        try:
            with Image.open(path) as im:
                im = im.convert("RGB")
                im.thumbnail((tw - 8, th - 8), Image.Resampling.LANCZOS)
                ox = x + (tw - im.width) // 2
                oy = y + (th - im.height) // 2
                sheet.paste(im, (ox, oy))
        except Exception as exc:  # noqa: BLE001
            draw.text((x + 6, y + 6), f"read error: {exc}", fill=(255, 120, 120))
        draw.text((x + 5, y + th + 4), label[:42], fill=(210, 210, 210))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=92)


def _reference_files(reference_dir: Path) -> list[Path]:
    return sorted(
        p
        for p in reference_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def _compute_reference_stats(reference_dir: Path) -> dict[str, object]:
    files = _reference_files(reference_dir)
    if not files:
        return {}
    rows: list[dict[str, float | int]] = []
    for path in files:
        try:
            with Image.open(path) as im:
                rows.append(image_stats(im))
        except Exception:  # noqa: BLE001
            continue
    return {
        "dir": _relative(reference_dir),
        "count": len(rows),
        "stats": summarize(rows),
    }


def _new_session_dir(root: Path, name: str | None, overwrite: bool = False, prefix: str = "") -> Path:
    if name:
        if Path(name).name != name or name in {".", ".."}:
            raise ValueError(f"invalid session name: {name!r}")
        session = root / name
        if session.exists():
            if not overwrite:
                raise FileExistsError(f"session already exists: {session}")
            shutil.rmtree(session)
        return session
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"{prefix}_{stamp}" if prefix else stamp
    session = root / base
    idx = 1
    while session.exists():
        session = root / f"{base}_{idx:02d}"
        idx += 1
    return session


def _write_manifest_header(
    session: Path,
    args: argparse.Namespace,
    profiles: list[str],
    records: list[dict[str, object]],
) -> None:
    header = {
        "session": session.name,
        "created": datetime.now().isoformat(timespec="seconds"),
        "version": args.version,
        "fingerprint": fingerprint(args.version),
        "scale": args.scale,
        "severity": args.severity,
        "seed": args.seed,
        "per_profile": args.per_profile,
        "profiles": profiles,
        "record_count": len(records),
        "source_counts": {},
        "note": (
            "Human-review images only. Training uses the same frozen degradation "
            "version on the fly and does not read this folder."
        ),
    }
    counts: dict[str, int] = {}
    for r in records:
        key = f"{r['group']}/{r['split']}"
        counts[key] = counts.get(key, 0) + 1
    header["source_counts"] = counts
    (session / "manifest.json").write_text(json.dumps(header, indent=2, ensure_ascii=False))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate stratified ESP32-style LR images for human review.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument(
        "--review-root",
        type=Path,
        default=Path("data/esp32_degrade_review"),
        help="root directory for review sessions",
    )
    p.add_argument("--session", type=str, default=None, help="session folder name")
    p.add_argument("--version", default=DEFAULT_VERSION, help="frozen degrader version")
    p.add_argument("--scale", type=int, default=4)
    p.add_argument("--severity", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--per-profile",
        type=int,
        default=2,
        help="number of images per profile per dataset group",
    )
    p.add_argument(
        "--profiles",
        type=str,
        default="",
        help="comma-separated subset of profiles (default: all)",
    )
    p.add_argument(
        "--datasets",
        type=str,
        default="div2k,flickr2k",
        help="comma-separated groups: div2k, flickr2k",
    )
    p.add_argument(
        "--reference-dir",
        type=Path,
        default=Path("data/esp32_camera/data01"),
        help="real ESP32 images used for a comparison contact sheet",
    )
    p.add_argument("--preview-count", type=int, default=36)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--list-profiles", action="store_true")
    p.add_argument("--list-versions", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if args.list_versions:
        print("\n".join(list_versions()))
        return
    if args.list_profiles:
        for name in profile_names(args.version):
            print(name)
        return

    version_module = get_version(args.version)
    requested = [x.strip() for x in args.profiles.split(",") if x.strip()]
    profiles = list(version_module.profile_names())
    if requested:
        missing = [x for x in requested if x not in profiles]
        if missing:
            raise SystemExit(f"unknown profiles: {missing}; available: {profiles}")
        profiles = requested

    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    unknown = [x for x in datasets if x not in {"div2k", "flickr2k"}]
    if unknown:
        raise SystemExit(f"unknown datasets: {unknown}")

    data_root = _resolve_project_path(args.data_root)
    review_root = _resolve_project_path(args.review_root)
    reference_dir = _resolve_project_path(args.reference_dir)

    pools = _collect_pools(data_root, datasets)
    if not pools:
        raise SystemExit(f"no HR files found under {data_root}")

    rng = random.Random(args.seed)
    records = _pick_records(pools, profiles, args.per_profile, rng)

    session = _new_session_dir(
        review_root, args.session, overwrite=args.overwrite, prefix=args.version
    )
    session.mkdir(parents=True, exist_ok=False)
    images_dir = session / "images"
    preview_dir = session / "previews"
    images_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    # Write the header before generation so a partially finished session is
    # still inspectable.
    _write_manifest_header(session, args, profiles, records)

    print(
        f"session: {session}\n"
        f"version: {args.version} | scale: {args.scale} | profiles: {len(profiles)} | "
        f"records: {len(records)}"
    )

    manifest_rows: list[dict[str, object]] = []
    stats_rows: list[dict[str, float | int]] = []
    preview_items: list[tuple[Path, str]] = []

    with (session / "manifest.jsonl").open("w", encoding="utf-8") as manifest:
        for record in tqdm(records, desc="degrade", unit="img"):
            source_path = Path(str(record["path"]))
            with Image.open(source_path) as im:
                hr = im.convert("RGB")
            hr_size = hr.size
            lr, spec = degrade_image(
                hr,
                version=args.version,
                seed=int(record["seed"]),
                profile=str(record["profile"]),
                scale=args.scale,
                severity=args.severity,
            )
            out_dir = images_dir / str(record["group"]) / str(record["split"])
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / _output_name(record)
            lr.save(out_path, format="PNG", optimize=True)

            stats = image_stats(lr)
            stats_rows.append(stats)
            row = {
                "id": (
                    f"{record['group']}_{record['split']}_{source_path.stem}_"
                    f"{record['profile']}_{record['seed']}"
                ),
                "source_group": record["group"],
                "split": record["split"],
                "source": _relative(source_path),
                "hr_size": list(hr_size),
                "lr_size": list(lr.size),
                "lr": out_path.relative_to(session).as_posix(),
                "profile": record["profile"],
                "version": args.version,
                "scale": args.scale,
                "severity": args.severity,
                "seed": int(record["seed"]),
                "spec": spec,
                "stats": stats,
            }
            manifest_rows.append(row)
            manifest.write(json.dumps(row, ensure_ascii=False) + "\n")
            label = f"{record['group']}/{record['split']} {source_path.stem} | {record['profile']}"
            preview_items.append((out_path, label))

    summary = {
        "session": session.name,
        "version": args.version,
        "fingerprint": fingerprint(args.version),
        "count": len(manifest_rows),
        "overall": summarize(stats_rows),
        "by_profile": {
            profile: summarize([s for s, r in zip(stats_rows, manifest_rows) if r["profile"] == profile])
            for profile in profiles
        },
    }
    reference = _compute_reference_stats(reference_dir)
    summary["reference"] = reference
    (session / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    # Regenerate the header with the final counts.
    _write_manifest_header(session, args, profiles, records)

    _make_contact_sheet(preview_items[: args.preview_count], preview_dir / "contact_sheet.jpg")
    ref_files = _reference_files(reference_dir)
    if ref_files:
        random.Random(args.seed + 1).shuffle(ref_files)
        ref_items = [(p, "ESP32 data01 reference") for p in ref_files[: min(18, len(ref_files))]]
        _make_contact_sheet(ref_items, preview_dir / "reference_sheet.jpg")

    print(f"\nwrote {len(manifest_rows)} review images")
    print(f"manifest: {session / 'manifest.jsonl'}")
    print(f"summary:  {session / 'summary.json'}")
    print(f"preview:  {preview_dir / 'contact_sheet.jpg'}")
    if reference:
        print(f"reference stats: {reference['dir']} ({reference['count']} images)")


if __name__ == "__main__":
    main()
