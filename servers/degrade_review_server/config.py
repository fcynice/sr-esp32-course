"""Configuration for the ESP32 degradation review visualiser.

Mirrors ``servers/sr_infer_server``: paths are anchored at the project root and
can be overridden with environment variables or CLI flags in ``__main__``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGE_EXTS = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp"})


def _env_str(name: str, default: str) -> str:
    raw = os.environ.get(name)
    return default if raw is None or not raw.strip() else raw.strip()


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return int(raw)


@dataclass(slots=True)
class Settings:
    host: str = field(default_factory=lambda: _env_str("DEGRADE_REVIEW_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: _env_int("DEGRADE_REVIEW_PORT", 8002))

    project_root: Path = PROJECT_ROOT
    review_dir: Path = field(
        default_factory=lambda: Path(
            _env_str(
                "DEGRADE_REVIEW_ROOT",
                str(PROJECT_ROOT / "data" / "esp32_degrade_review"),
            )
        ).resolve()
    )
    reference_dir: Path = field(
        default_factory=lambda: Path(
            _env_str(
                "DEGRADE_REVIEW_REFERENCE",
                str(PROJECT_ROOT / "data" / "esp32_camera" / "data01"),
            )
        ).resolve()
    )
    thumbnail_size: int = field(
        default_factory=lambda: _env_int("DEGRADE_REVIEW_THUMB_SIZE", 420)
    )

    def ensure_dirs(self) -> None:
        self.review_dir.mkdir(parents=True, exist_ok=True)

    def allowed_roots(self) -> list[Path]:
        roots: list[Path] = []
        for path in (self.project_root, self.review_dir, self.reference_dir):
            resolved = Path(path).expanduser().resolve()
            if resolved not in roots:
                roots.append(resolved)
        return roots

    def is_inside_allowed(self, path: Path) -> bool:
        resolved = Path(path).resolve()
        for root in self.allowed_roots():
            try:
                resolved.relative_to(root)
                return True
            except ValueError:
                continue
        return False


settings = Settings()
