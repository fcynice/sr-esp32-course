"""Configuration for the super-resolution inference visualiser.

The server is intended to run on the same machine as the project, so it reads
images directly from disk instead of accepting uploads.  All paths are anchored
at the project root and can be overridden with environment variables or CLI
flags in ``__main__``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# .../superresolution/servers/sr_infer_server/config.py -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# The service only deals with the formats used by the project.  ``.jpeg`` is
# included because DIV2K HR files use it.
IMAGE_EXTS = frozenset({".png", ".jpg", ".jpeg"})

# Hidden / tooling directories that would only add noise to the browser.
EXCLUDED_DIRS = frozenset(
    {
        ".git",
        ".venv",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "__pycache__",
        "node_modules",
    }
)


def _env_str(name: str, default: str) -> str:
    raw = os.environ.get(name)
    return default if raw is None or not raw.strip() else raw.strip()


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return int(raw)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on", "y"}


def _env_path_list(name: str, default: list[Path]) -> list[Path]:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    paths: list[Path] = []
    for part in raw.split(os.pathsep):
        part = part.strip()
        if not part:
            continue
        p = Path(part).expanduser()
        paths.append((p if p.is_absolute() else PROJECT_ROOT / p).resolve())
    return paths or default


@dataclass(slots=True)
class Settings:
    """Runtime settings for the inference visualiser."""

    host: str = field(default_factory=lambda: _env_str("SR_INFER_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: _env_int("SR_INFER_PORT", 8001))

    project_root: Path = PROJECT_ROOT
    runs_dir: Path = field(
        default_factory=lambda: Path(
            _env_str("SR_INFER_RUNS_DIR", str(PROJECT_ROOT / "runs"))
        ).resolve()
    )
    data_dir: Path = field(
        default_factory=lambda: Path(
            _env_str("SR_INFER_DATA_DIR", str(PROJECT_ROOT / "data"))
        ).resolve()
    )
    results_dir: Path = field(
        default_factory=lambda: Path(
            _env_str(
                "SR_INFER_RESULTS_DIR",
                str(PROJECT_ROOT / "data" / "sr_infer" / "results"),
            )
        ).resolve()
    )

    # The browser is sandboxed to these roots.  The project root is always
    # present by default; extra roots can be supplied with a colon-separated
    # ``SR_INFER_ALLOWED_ROOTS`` (e.g. another image collection).
    allowed_roots: list[Path] = field(
        default_factory=lambda: _env_path_list(
            "SR_INFER_ALLOWED_ROOTS", [PROJECT_ROOT]
        )
    )

    default_browse_path: str = field(
        default_factory=lambda: _env_str("SR_INFER_DEFAULT_BROWSE", "data")
    )

    # Inference.
    device: str = field(default_factory=lambda: _env_str("SR_INFER_DEVICE", "auto"))
    auto_tile: int = field(default_factory=lambda: _env_int("SR_INFER_TILE", 512))
    auto_tile_threshold: int = field(
        default_factory=lambda: _env_int("SR_INFER_TILE_THRESHOLD", 512)
    )
    tile_overlap: int = field(
        default_factory=lambda: _env_int("SR_INFER_TILE_OVERLAP", 16)
    )
    max_input_pixels: int = field(
        default_factory=lambda: _env_int("SR_INFER_MAX_PIXELS", 80_000_000)
    )
    max_output_pixels: int = field(
        default_factory=lambda: _env_int("SR_INFER_MAX_OUTPUT_PIXELS", 100_000_000)
    )

    # Gallery.
    thumbnail_size: int = field(
        default_factory=lambda: _env_int("SR_INFER_THUMB_SIZE", 320)
    )

    # Quick navigation entries (label, path relative to project root).
    quick_roots: list[tuple[str, str]] = field(
        default_factory=lambda: [
            ("项目根", ""),
            ("data", "data"),
            ("DIV2K", "data/DIV2K"),
            ("Flickr2K", "data/Flickr2K"),
            ("ESP32 相机", "data/esp32_camera"),
            ("大学教室", "data/大学教室"),
            ("大学校园", "data/大学校园"),
        ]
    )

    def ensure_dirs(self) -> None:
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def resolved_allowed_roots(self) -> list[Path]:
        roots: list[Path] = []
        for root in self.allowed_roots:
            resolved = Path(root).expanduser().resolve()
            if resolved not in roots:
                roots.append(resolved)
        # The results directory and data root must always be reachable by the
        # image endpoints, even when they are configured outside the project.
        for extra in (self.data_dir, self.results_dir):
            resolved = Path(extra).expanduser().resolve()
            if resolved not in roots:
                roots.append(resolved)
        project_root = self.project_root.resolve()
        if project_root not in roots:
            roots.append(project_root)
        return roots

    def is_inside_allowed(self, path: Path) -> bool:
        resolved = path.resolve()
        for root in self.resolved_allowed_roots():
            try:
                resolved.relative_to(root)
                return True
            except ValueError:
                continue
        return False


settings = Settings()
