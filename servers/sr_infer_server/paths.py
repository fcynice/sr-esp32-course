"""Path validation and display helpers."""

from __future__ import annotations

from pathlib import Path

from .config import IMAGE_EXTS, Settings


class PathError(ValueError):
    """Raised when a path is missing, unsupported or outside the sandbox."""


def _candidate(settings: Settings, raw: str | None) -> Path:
    if raw is None or not str(raw).strip() or str(raw).strip() == ".":
        return settings.project_root.resolve()
    p = Path(str(raw)).expanduser()
    if not p.is_absolute():
        p = settings.project_root / p
    return p.resolve()


def display_path(settings: Settings, path: Path) -> str:
    """Return a stable token for the frontend.

    Paths inside the project are relative (with the project root represented by
    an empty string); external allowed roots keep their absolute path.
    """
    resolved = path.resolve()
    try:
        rel = resolved.relative_to(settings.project_root.resolve())
    except ValueError:
        return resolved.as_posix()
    return "" if not rel.parts else rel.as_posix()


def resolve_dir(settings: Settings, raw: str | None) -> Path:
    path = _candidate(settings, raw)
    if not settings.is_inside_allowed(path):
        raise PathError("路径不在允许浏览的范围内")
    if not path.exists():
        raise PathError(f"目录不存在：{path}")
    if not path.is_dir():
        raise PathError(f"不是目录：{path}")
    return path


def resolve_image(settings: Settings, raw: str | None) -> Path:
    path = _candidate(settings, raw)
    if not settings.is_inside_allowed(path):
        raise PathError("路径不在允许浏览的范围内")
    if not path.exists():
        raise PathError(f"文件不存在：{path}")
    if not path.is_file():
        raise PathError(f"不是文件：{path}")
    if path.suffix.lower() not in IMAGE_EXTS:
        raise PathError(f"不支持的图片格式：{path.suffix or '(无后缀)'}")
    return path


def resolve_checkpoint(settings: Settings, raw: str | None) -> Path:
    path = _candidate(settings, raw)
    runs_root = settings.runs_dir.resolve()
    try:
        path.relative_to(runs_root)
    except ValueError:
        raise PathError("checkpoint 必须位于 runs 目录内") from None
    if not path.exists():
        raise PathError(f"checkpoint 不存在：{path}")
    if not path.is_file() or path.suffix.lower() != ".pth":
        raise PathError(f"不是 .pth checkpoint：{path}")
    return path
