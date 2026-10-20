"""FastAPI visualiser for ESP32 degradation review sessions.

Start it with::

    uv run python -m servers.degrade_review_server

The UI lives at ``/`` and talks to JSON endpoints under ``/api``.
"""

from __future__ import annotations

import argparse
import io
import logging
import mimetypes
import os
import threading
from collections import OrderedDict
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps

from . import __version__
from .config import settings
from .paths import PathError, display_path, resolve_file
from .sessions import list_reference_images, list_sessions, load_session

logger = logging.getLogger("degrade_review")

STATIC_DIR = Path(__file__).resolve().parent / "static"
INDEX_PATH = STATIC_DIR / "index.html"


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings.ensure_dirs()
    logger.info("project root: %s", settings.project_root)
    logger.info("review dir: %s", settings.review_dir)
    logger.info("reference dir: %s", settings.reference_dir)
    yield


app = FastAPI(
    title="ESP32 Degradation Review",
    description="Compare on-the-fly ESP32-style degraded LR images with their HR source and real camera references.",
    version=__version__,
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.exception_handler(PathError)
async def path_error_handler(_, exc: PathError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(ValueError)
async def value_error_handler(_, exc: ValueError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(INDEX_PATH, headers={"Cache-Control": "no-cache"})


@app.get("/api/health", response_class=JSONResponse)
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "time": datetime.now().isoformat(timespec="seconds"),
        "version": __version__,
        "project_root": str(settings.project_root),
        "review_dir": str(settings.review_dir),
        "reference_dir": str(settings.reference_dir),
        "sessions": len(list_sessions(settings)),
    }


@app.get("/api/config", response_class=JSONResponse)
def runtime_config() -> dict[str, object]:
    return {
        "project_root": str(settings.project_root),
        "review_dir": display_path(settings, settings.review_dir),
        "reference_dir": display_path(settings, settings.reference_dir),
        "thumbnail_size": settings.thumbnail_size,
    }


@app.get("/api/sessions", response_class=JSONResponse)
def sessions() -> dict[str, object]:
    return {"sessions": list_sessions(settings)}


@app.get("/api/session/{session_id}", response_class=JSONResponse)
def session(session_id: str) -> dict[str, object]:
    return load_session(settings, session_id)


@app.get("/api/reference", response_class=JSONResponse)
def reference(limit: int = Query(default=0, ge=0, le=500)) -> dict[str, object]:
    return {
        "dir": display_path(settings, settings.reference_dir),
        "images": list_reference_images(settings, limit=limit),
    }


# ---------------------------------------------------------------------------
# Thumbnail cache
# ---------------------------------------------------------------------------
_thumb_cache: OrderedDict[tuple[str, int, int, int], bytes] = OrderedDict()
_thumb_lock = threading.Lock()
_THUMB_CACHE_LIMIT = 512


def _thumb_bytes(path: Path) -> bytes:
    stat = path.stat()
    size = max(64, int(settings.thumbnail_size))
    key = (str(path), stat.st_mtime_ns, stat.st_size, size)
    with _thumb_lock:
        cached = _thumb_cache.get(key)
        if cached is not None:
            _thumb_cache.move_to_end(key)
            return cached

    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail((size, size), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=88)
        data = buf.getvalue()

    with _thumb_lock:
        _thumb_cache[key] = data
        _thumb_cache.move_to_end(key)
        while len(_thumb_cache) > _THUMB_CACHE_LIMIT:
            _thumb_cache.popitem(last=False)
    return data


@app.get("/api/image")
def image(
    path: str = Query(...),
    variant: Literal["full", "thumb"] = Query(default="full"),
    download: bool = Query(default=False),
) -> Response:
    resolved = resolve_file(settings, path)
    if variant == "thumb":
        return Response(
            content=_thumb_bytes(resolved),
            media_type="image/jpeg",
            headers={"Cache-Control": "public, max-age=3600"},
        )
    guessed, _ = mimetypes.guess_type(resolved.name)
    return FileResponse(
        resolved,
        media_type=guessed or "application/octet-stream",
        filename=resolved.name if download else None,
        headers={"Cache-Control": "public, max-age=60"},
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="ESP32 degradation review visualiser")
    parser.add_argument("--host", default=settings.host)
    parser.add_argument("--port", type=int, default=settings.port)
    parser.add_argument("--review-dir", type=Path, default=settings.review_dir)
    parser.add_argument("--reference-dir", type=Path, default=settings.reference_dir)
    parser.add_argument("--thumb-size", type=int, default=settings.thumbnail_size)
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    settings.host = args.host
    settings.port = args.port
    settings.review_dir = args.review_dir.expanduser().resolve()
    settings.reference_dir = args.reference_dir.expanduser().resolve()
    settings.thumbnail_size = args.thumb_size
    settings.ensure_dirs()

    os.environ["DEGRADE_REVIEW_HOST"] = settings.host
    os.environ["DEGRADE_REVIEW_PORT"] = str(settings.port)
    os.environ["DEGRADE_REVIEW_ROOT"] = str(settings.review_dir)
    os.environ["DEGRADE_REVIEW_REFERENCE"] = str(settings.reference_dir)
    os.environ["DEGRADE_REVIEW_THUMB_SIZE"] = str(settings.thumbnail_size)

    logger.info("listening on http://%s:%d", settings.host, settings.port)
    logger.info("review dir: %s", settings.review_dir)
    logger.info("reference dir: %s", settings.reference_dir)

    import uvicorn

    uvicorn.run(
        "servers.degrade_review_server.main:app",
        host=settings.host,
        port=settings.port,
        reload=args.reload,
        log_level="debug" if args.verbose else "info",
    )


if __name__ == "__main__":
    main()
