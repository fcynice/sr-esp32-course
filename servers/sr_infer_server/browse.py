"""Directory browsing, image metadata and thumbnail caching."""

from __future__ import annotations

import io
import re
import threading
from collections import OrderedDict
from pathlib import Path

from PIL import Image, ImageOps

from .config import EXCLUDED_DIRS, IMAGE_EXTS, Settings
from .datasets import find_hr_for_lr
from .paths import display_path, resolve_dir, resolve_image

_NATURAL_RE = re.compile(r"(\d+)")


def _natural_key(name: str) -> list[object]:
    return [
        int(part) if part.isdigit() else part.lower()
        for part in _NATURAL_RE.split(name)
    ]


def _breadcrumbs(settings: Settings, path: Path) -> list[dict[str, str]]:
    project_root = settings.project_root.resolve()
    resolved = path.resolve()
    try:
        rel = resolved.relative_to(project_root)
    except ValueError:
        # External allowed root: start at the matching root.
        for root in settings.resolved_allowed_roots():
            try:
                rel = resolved.relative_to(root)
            except ValueError:
                continue
            crumbs = [{"name": root.name or root.as_posix(), "path": root.as_posix()}]
            current = root
            for part in rel.parts:
                current = current / part
                crumbs.append({"name": part, "path": current.as_posix()})
            return crumbs
        return [{"name": resolved.as_posix(), "path": resolved.as_posix()}]

    crumbs = [{"name": "项目根", "path": ""}]
    current = project_root
    for part in rel.parts:
        current = current / part
        crumbs.append({"name": part, "path": display_path(settings, current)})
    return crumbs


def _quick_roots(settings: Settings, current: Path) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    seen: set[str] = set()

    def add(label: str, target: Path) -> None:
        target = target.resolve()
        key = str(target)
        if key in seen or not target.is_dir():
            return
        if not settings.is_inside_allowed(target):
            return
        seen.add(key)
        entries.append(
            {
                "name": label,
                "path": display_path(settings, target),
                "active": target == current.resolve(),
            }
        )

    for label, rel in settings.quick_roots:
        target = (
            (settings.project_root / rel).resolve()
            if rel
            else settings.project_root.resolve()
        )
        add(label, target)

    # Results directory is configurable; show it wherever it lives.
    add("推理结果", settings.results_dir)
    return entries


def _thumb_url(settings: Settings, path: Path) -> str:
    from urllib.parse import quote

    return f"/api/image?path={quote(display_path(settings, path), safe='')}&variant=thumb"


def list_directory(settings: Settings, raw_path: str | None) -> dict[str, object]:
    path = resolve_dir(settings, raw_path)
    dirs: list[dict[str, str]] = []
    images: list[dict[str, str]] = []

    for child in path.iterdir():
        name = child.name
        if name.startswith(".") or name in EXCLUDED_DIRS:
            continue
        try:
            if child.is_dir():
                dirs.append(
                    {
                        "name": name,
                        "path": display_path(settings, child),
                    }
                )
            elif child.is_file() and child.suffix.lower() in IMAGE_EXTS:
                images.append(
                    {
                        "name": name,
                        "path": display_path(settings, child),
                        "thumb_url": _thumb_url(settings, child),
                    }
                )
        except OSError:
            continue

    dirs.sort(key=lambda item: _natural_key(item["name"]))
    images.sort(key=lambda item: _natural_key(item["name"]))

    roots = settings.resolved_allowed_roots()
    parent: str | None = None
    resolved = path.resolve()
    for root in roots:
        try:
            resolved.relative_to(root)
        except ValueError:
            continue
        if resolved != root:
            parent = display_path(settings, resolved.parent)
        break

    return {
        "path": display_path(settings, path),
        "abs_path": str(path),
        "parent": parent,
        "breadcrumbs": _breadcrumbs(settings, path),
        "quick_roots": _quick_roots(settings, path),
        "dirs": dirs,
        "images": images,
        "image_count": len(images),
        "dir_count": len(dirs),
    }


def image_info(settings: Settings, raw_path: str | None) -> dict[str, object]:
    path = resolve_image(settings, raw_path)
    try:
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im)
            width, height = im.size
            mode = im.mode
    except Exception as exc:  # noqa: BLE001 - Pillow can raise many types
        raise ValueError(f"无法读取图片：{exc}") from exc

    pair = find_hr_for_lr(settings, path)
    hr: dict[str, object] | None = None
    if pair is not None:
        hr_path = Path(str(pair["hr_abs_path"]))
        try:
            with Image.open(hr_path) as im:
                im = ImageOps.exif_transpose(im)
                hr_width, hr_height = im.size
        except Exception:  # noqa: BLE001
            hr_width = hr_height = None
        hr = {
            "path": pair["hr_path"],
            "abs_path": pair["hr_abs_path"],
            "name": pair["hr_name"],
            "width": hr_width,
            "height": hr_height,
        }

    return {
        "path": display_path(settings, path),
        "abs_path": str(path),
        "name": path.name,
        "width": width,
        "height": height,
        "mode": mode,
        "dataset": pair,
        "hr": hr,
    }


# ---------------------------------------------------------------------------
# Thumbnail cache
# ---------------------------------------------------------------------------

_thumb_cache: OrderedDict[tuple[str, int, int, int], bytes] = OrderedDict()
_thumb_lock = threading.Lock()
_THUMB_CACHE_LIMIT = 512


def clear_thumbnail_cache() -> None:
    with _thumb_lock:
        _thumb_cache.clear()


def thumbnail_bytes(settings: Settings, raw_path: str | None) -> tuple[bytes, str]:
    path = resolve_image(settings, raw_path)
    stat = path.stat()
    size = max(64, int(settings.thumbnail_size))
    key = (str(path.resolve()), stat.st_mtime_ns, stat.st_size, size)

    with _thumb_lock:
        cached = _thumb_cache.get(key)
        if cached is not None:
            _thumb_cache.move_to_end(key)
            return cached, "image/jpeg"

    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        im.thumbnail((size, size), Image.Resampling.LANCZOS)
        if im.mode in {"RGBA", "LA"} or (im.mode == "P" and "transparency" in im.info):
            rgba = im.convert("RGBA")
            background = Image.new("RGB", rgba.size, (18, 20, 24))
            background.paste(rgba, mask=rgba.getchannel("A"))
            im = background
        else:
            im = im.convert("RGB")
        buffer = io.BytesIO()
        im.save(buffer, format="JPEG", quality=86, optimize=True)
        data = buffer.getvalue()

    with _thumb_lock:
        _thumb_cache[key] = data
        _thumb_cache.move_to_end(key)
        while len(_thumb_cache) > _THUMB_CACHE_LIMIT:
            _thumb_cache.popitem(last=False)
    return data, "image/jpeg"
