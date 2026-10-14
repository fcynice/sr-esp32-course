#!/usr/bin/env python3
"""Migrate ``runs/`` directories to the plain indexed naming scheme.

Naming scheme
-------------
* New:      ``res{index:02d}``                  (e.g. ``res01``)
* Interim:  ``espcn_x{scale}_{lr_type}_{index:02d}``
* Legacy:   ``espcn_x{scale}_{YYYYmmdd-HHMMSS}``

Directories are named ``res01``, ``res02``, ... and assigned a *chronological*
index (oldest run gets the smallest number).  Details such as the model, scale
and ``lr_type`` are stored inside each run in ``run_info.json`` (legacy
``args.json`` files are backfilled with ``lr_type`` when missing).

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
INTERIM_RE = re.compile(r"^espcn_x(?P<scale>\d+)_(?P<lr>[a-z]+)_(?P<idx>\d+)$")
NEW_RE = re.compile(r"^res(?P<idx>\d+)$")


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

        legacy_m = LEGACY_RE.match(p.name)
        interim_m = INTERIM_RE.match(p.name)
        new_m = NEW_RE.match(p.name)
        if not (legacy_m or interim_m or new_m):
            print(f"skip (unrecognized): {p.name}")
            continue

        # Scale (unknown for the plain resNN format).
        if legacy_m or interim_m:
            scale = int((legacy_m or interim_m).group("scale"))
        else:
            scale = args.get("scale") or info.get("scale")

        # lr_type.
        if legacy_m:
            lr = args.get("lr_type") or info.get("lr_type") or default_lr_type
        elif interim_m:
            lr = args.get("lr_type") or info.get("lr_type") or interim_m.group("lr")
        else:
            lr = args.get("lr_type") or info.get("lr_type") or default_lr_type

        # Timestamp: run_info first, then legacy name, then mtime.
        ts_raw = info.get("timestamp")
        if ts_raw:
            ts = datetime.fromisoformat(ts_raw)
        else:
            ts = (
                legacy_timestamp(p.name)
                if legacy_m
                else datetime.fromtimestamp(p.stat().st_mtime)
            )

        records.append({"path": p, "scale": scale, "lr": lr, "ts": ts, "info": info})
    return records


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs_dir", nargs="?", default="runs")
    ap.add_argument(
        "--default-lr-type",
        default="unknown",
        help="lr_type to assume for runs whose metadata has none",
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

    # Assign chronological indices across all runs.
    targets = []
    for i, r in enumerate(sorted(records, key=lambda r: r["ts"]), start=1):
        r["index"] = i
        r["new_path"] = root / f"res{i:02d}"
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
        info = dict(r["info"])
        info.setdefault("model", "espcn")
        info["index"] = r["index"]
        info["timestamp"] = r["ts"].isoformat(timespec="seconds")
        info["lr_type"] = r["lr"]
        if r["scale"] is not None:
            info["scale"] = r["scale"]
        (r["new_path"] / "run_info.json").write_text(json.dumps(info, indent=2))

        args_path = r["new_path"] / "args.json"
        if args_path.is_file():
            a = read_json(args_path)
            a.setdefault("lr_type", r["lr"])
            args_path.write_text(json.dumps(a, indent=2))

    print("dry run complete" if args.dry_run else "migration complete")


if __name__ == "__main__":
    main()
