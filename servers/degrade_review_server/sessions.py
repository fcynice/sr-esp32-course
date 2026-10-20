"""Session/manifest loading and reference image listing."""

from __future__ import annotations

import json
from pathlib import Path

from .config import IMAGE_EXTS, Settings
from .paths import PathError, display_path, image_url, resolve_session_dir

MANIFEST = "manifest.json"
MANIFEST_ROWS = "manifest.jsonl"
SUMMARY = "summary.json"


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise PathError(f"无法读取 {path.name}: {exc}") from exc


def list_sessions(settings: Settings) -> list[dict[str, object]]:
    if not settings.review_dir.is_dir():
        return []
    sessions: list[dict[str, object]] = []
    for child in settings.review_dir.iterdir():
        if not child.is_dir() or not (child / MANIFEST).is_file():
            continue
        try:
            header = _read_json(child / MANIFEST)
        except PathError:
            continue
        sessions.append(
            {
                "id": child.name,
                "created": header.get("created"),
                "version": header.get("version"),
                "fingerprint": header.get("fingerprint"),
                "scale": header.get("scale"),
                "severity": header.get("severity"),
                "seed": header.get("seed"),
                "record_count": header.get("record_count"),
                "profiles": header.get("profiles", []),
                "source_counts": header.get("source_counts", {}),
            }
        )
    sessions.sort(key=lambda item: str(item.get("created") or item["id"]), reverse=True)
    return sessions


def _attach_urls(settings: Settings, session_dir: Path, record: dict) -> dict:
    out = dict(record)
    lr_rel = record.get("lr")
    if lr_rel:
        lr_path = (session_dir / str(lr_rel)).resolve()
        out["lr_path"] = display_path(settings, lr_path)
        out["lr_url"] = image_url(settings, lr_path, variant="full")
        out["lr_thumb_url"] = image_url(settings, lr_path, variant="thumb")

    source = record.get("source")
    if source:
        src_path = Path(str(source))
        if not src_path.is_absolute():
            src_path = settings.project_root / src_path
        src_path = src_path.resolve()
        if src_path.is_file() and settings.is_inside_allowed(src_path):
            out["hr_available"] = True
            out["hr_path"] = display_path(settings, src_path)
            out["hr_url"] = image_url(settings, src_path, variant="full")
            out["hr_thumb_url"] = image_url(settings, src_path, variant="thumb")
        else:
            out["hr_available"] = False
    return out


def load_session(settings: Settings, session_id: str) -> dict[str, object]:
    session_dir = resolve_session_dir(settings, session_id)
    header = _read_json(session_dir / MANIFEST)
    records: list[dict] = []
    rows_path = session_dir / MANIFEST_ROWS
    if rows_path.is_file():
        for line in rows_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                records.append(_attach_urls(settings, session_dir, json.loads(line)))
            except json.JSONDecodeError:
                continue

    summary: dict = {}
    if (session_dir / SUMMARY).is_file():
        try:
            summary = _read_json(session_dir / SUMMARY)
        except PathError:
            summary = {}

    profiles: dict[str, int] = {}
    sources: dict[str, int] = {}
    for record in records:
        profiles[str(record.get("profile", "?"))] = profiles.get(str(record.get("profile", "?")), 0) + 1
        key = f"{record.get('source_group', '?')}/{record.get('split', '?')}"
        sources[key] = sources.get(key, 0) + 1

    return {
        "id": session_id,
        "dir": display_path(settings, session_dir),
        "header": header,
        "summary": summary,
        "records": records,
        "profile_counts": profiles,
        "source_counts": sources,
        "preview_urls": {
            "contact_sheet": image_url(settings, session_dir / "previews" / "contact_sheet.jpg")
            if (session_dir / "previews" / "contact_sheet.jpg").is_file()
            else None,
            "reference_sheet": image_url(settings, session_dir / "previews" / "reference_sheet.jpg")
            if (session_dir / "previews" / "reference_sheet.jpg").is_file()
            else None,
        },
    }


def list_reference_images(settings: Settings, limit: int = 0) -> list[dict[str, object]]:
    root = settings.reference_dir
    if not root.is_dir():
        return []
    files = sorted(
        p
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    if limit > 0:
        files = files[:limit]
    out = []
    for path in files:
        out.append(
            {
                "name": path.name,
                "path": display_path(settings, path),
                "url": image_url(settings, path, variant="full"),
                "thumb_url": image_url(settings, path, variant="thumb"),
            }
        )
    return out
