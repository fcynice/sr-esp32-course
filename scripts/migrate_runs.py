#!/usr/bin/env python3
"""Migrate ``runs/`` directories to the indexed naming scheme.

Naming scheme
-------------
* New:    ``espcn_x{scale}_{lr_type}_{index:02d}``  (e.g. ``espcn_x4_unknown_01``)
* Legacy: ``espcn_x{scale}_{YYYYmmdd-HHMMSS}``

Directory names are assigned a chronological index per scale (oldest run gets
the smallest number), and the wall-clock timestamp is moved *inside* the run as
``run_info.json``.  Legacy ``args.json`` files are backfilled with ``lr_type``
when missing.

The script is idempotent: already-migrated runs are detected and left in place.

Usage::

    python scripts/migrate_runs.py [runs_dir] [--default-lr-type unknown] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

LEGACY_RE = re.compile(r"^espcn_x(?P<scale>\d+)_(?P<ts>\d{8}-\d{6})$")
NEW_RE = re.compile(r"^espcn_x(?P<scale>\d+)_(?P<lr>[a-z]+)_(?P<idx>\d+)$")


def read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def legacy_timestamp(name: str) -> datetime | None:
    m = LEGACY_RE.match(name)
    if not m:
        return None
    try:
        return datetime.strptime(m.group("ts"), "%Y%m%d-%H%M%S")
    except ValueError:
        return None


def collect(runs_dir: Path, default_lr_type: str):
    records = []
    for p in sorted(runs_dir.iterdir()):
        if not p.is_dir():
            continue
        args = read_json(p / "args.json")
        info = read_json(p / "run_info.json")

        new_m = NEW_RE.match(p.name)
        legacy_m = LEGACY_RE.match(p.name)
        if new_m:
            scale = int(new_m.group("scale"))
            lr = args.get("lr_type") or info.get("lr_type") or new_m.group("lr")
            ts_raw = info.get("timestamp")
            ts = (
                datetime.fromisoformat(ts_raw)
                if ts_raw
                else datetime.fromtimestamp(p.stat().st_mtime)
            )
        elif legacy_m:
            scale = int(legacy_m.group("scale"))
            lr = args.get("lr_type") or info.get("lr_type") or default_lr_type
            ts = legacy_timestamp(p.name) or datetime.fromtimestamp(p.stat().st_mtime)
        else:
            print(f"skip (unrecognized): {p.name}")
            continue

        records.append({"path": p, "scale": scale, "lr": lr, "ts": ts})
    return records


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs_dir", nargs="?", default="runs")
    ap.add_argument(
        "--default-lr-type",
        default="unknown",
        help="lr_type to assume for legacy runs whose args.json has none",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = Path(args.runs_dir)
    if not root.is_dir():
        print(f"no such directory: {root}")
        return

    records = collect(root, args.default_lr_type)
    if not records:
        print("nothing to migrate")
        return

    # Assign chronological indices per scale.
    targets = []
    for scale in sorted({r["scale"] for r in records}):
        group = sorted((r for r in records if r["scale"] == scale), key=lambda r: r["ts"])
        for i, r in enumerate(group, start=1):
            r["index"] = i
            r["new_path"] = root / f"espcn_x{scale}_{r['lr']}_{i:02d}"
            targets.append(r)

    # Two-phase rename to avoid collisions between old and new names.
    staged = []
    for r in targets:
        if r["path"] == r["new_path"]:
            continue
        print(f"{r['path'].name}  ->  {r['new_path'].name}")
        if args.dry_run:
            continue
        tmp = r["path"].with_name(r["path"].name + ".migrating")
        r["path"].rename(tmp)
        staged.append((tmp, r))
    for tmp, r in staged:
        tmp.rename(r["new_path"])

    # (Re)write metadata inside every run.
    for r in targets:
        if args.dry_run:
            continue
        r["new_path"].mkdir(exist_ok=True)
        (r["new_path"] / "run_info.json").write_text(
            json.dumps(
                {
                    "index": r["index"],
                    "timestamp": r["ts"].isoformat(timespec="seconds"),
                    "lr_type": r["lr"],
                    "scale": r["scale"],
                    "model": "espcn",
                },
                indent=2,
            )
        )
        args_path = r["new_path"] / "args.json"
        if args_path.is_file():
            a = read_json(args_path)
            a.setdefault("lr_type", r["lr"])
            args_path.write_text(json.dumps(a, indent=2))

    print("dry run complete" if args.dry_run else "migration complete")


if __name__ == "__main__":
    main()
