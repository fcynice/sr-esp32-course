"""Path validation and display helpers for the review server."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import quote

from .config import IMAGE_EXTS, Settings


class PathError(ValueError):
    """Raised when a path is missing, unsupported or outside the sandbox."""


def _candidate(settings: Settings, raw: str | Path | None) -> Path:
    if raw is None or not str(raw).strip() or str(raw).strip() == ".":
        return settings.project_root.resolve()
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = settings.project_root / path
    return path.resolve()


def display_path(settings: Settings, path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        rel = resolved.relative_to(settings.project_root.resolve())
    except ValueError:
        return resolved.as_posix()
    return rel.as_posix()


def resolve_file(settings: Settings, raw: str | Path | None) -> Path:
    path = _candidate(settings, raw)
    if not settings.is_inside_allowed(path):
        raise PathError("路径不在允许访问的范围内")
    if not path.exists():
        raise PathError(f"文件不存在：{path}")
    if not path.is_file():
        raise PathError(f"不是文件：{path}")
    if path.suffix.lower() not in IMAGE_EXTS:
        raise PathError(f"不支持的图片格式：{path.suffix or '(无后缀)'}")
    return path


def resolve_session_dir(settings: Settings, session_id: str) -> Path:
    if not session_id or "/" in session_id or "\\" in session_id or session_id in {".", ".."}:
        raise PathError("非法 session id")
    root = settings.review_dir.resolve()
    path = (root / session_id).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        raise PathError("session 不在审查目录内") from None
    if not path.is_dir():
        raise PathError(f"session 不存在：{session_id}")
    if not (path / "manifest.json").is_file():
        raise PathError(f"session 缺少 manifest.json：{session_id}")
    return path


def image_url(settings: Settings, path: Path, *, variant: str = "full") -> str:
    token = display_path(settings, path)
    return f"/api/image?path={quote(token, safe='')}&variant={variant}"


def asset_url(path: str) -> str:
    return "/static/" + path.lstrip("/")
