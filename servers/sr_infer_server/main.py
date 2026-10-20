"""FastAPI app for the local super-resolution inference visualiser.

Start it with::

    uv run python -m servers.sr_infer_server

The UI lives at ``/`` and talks to the JSON endpoints under ``/api``.
"""

from __future__ import annotations

import argparse
import logging
import mimetypes
import os
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Literal

import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import __version__
from .browse import image_info, list_directory, thumbnail_bytes
from .config import IMAGE_EXTS, settings
from .jobs import JobManager
from .paths import PathError, display_path, resolve_checkpoint, resolve_image
from .runs import scan_runs

logger = logging.getLogger("sr_infer")

STATIC_DIR = Path(__file__).resolve().parent / "static"
INDEX_PATH = STATIC_DIR / "index.html"

# The manager is a module-level singleton, like the storage object in the ESP32
# server.  ``main()`` mutates ``settings`` before uvicorn imports/starts the app.
jobs = JobManager(settings)


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings.ensure_dirs()
    logger.info("project root: %s", settings.project_root)
    logger.info("runs dir: %s", settings.runs_dir)
    logger.info("results dir: %s", settings.results_dir)
    yield
    jobs.shutdown()


app = FastAPI(
    title="SR Inference Visualiser",
    description="Local super-resolution inference with side-by-side and draggable-line comparison.",
    version=__version__,
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class InferRequest(BaseModel):
    image_path: str
    checkpoint_path: str


def _media_type(path: Path) -> str:
    guessed, _ = mimetypes.guess_type(path.name)
    return guessed or "application/octet-stream"


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
    cuda = torch.cuda.is_available()
    return {
        "status": "ok",
        "time": datetime.now().isoformat(timespec="seconds"),
        "version": __version__,
        "device": "cuda" if cuda else "cpu",
        "cuda": cuda,
        "gpu_name": torch.cuda.get_device_name(0) if cuda else None,
        "runs": len(scan_runs(settings)),
        "project_root": str(settings.project_root),
        "runs_dir": str(settings.runs_dir),
        "results_dir": str(settings.results_dir),
        "default_browse_path": settings.default_browse_path,
        "image_exts": sorted(IMAGE_EXTS),
    }


@app.get("/api/config", response_class=JSONResponse)
def runtime_config() -> dict[str, object]:
    return {
        "default_browse_path": settings.default_browse_path,
        "runs_dir": str(settings.runs_dir),
        "results_dir": str(settings.results_dir),
        "thumbnail_size": settings.thumbnail_size,
        "image_exts": sorted(IMAGE_EXTS),
        "auto_tile": settings.auto_tile,
        "auto_tile_threshold": settings.auto_tile_threshold,
        "tile_overlap": settings.tile_overlap,
        "max_output_pixels": settings.max_output_pixels,
    }


@app.get("/api/runs", response_class=JSONResponse)
def runs() -> dict[str, object]:
    return {"runs": scan_runs(settings)}


@app.get("/api/browse", response_class=JSONResponse)
def browse(path: str = Query(default="")) -> dict[str, object]:
    return list_directory(settings, path)


@app.get("/api/image-info", response_class=JSONResponse)
def image_meta(path: str = Query(...)) -> dict[str, object]:
    return image_info(settings, path)


@app.get("/api/image")
def image(
    path: str = Query(...),
    variant: Literal["full", "thumb"] = Query(default="full"),
    download: bool = Query(default=False),
) -> Response:
    if variant == "thumb":
        data, media_type = thumbnail_bytes(settings, path)
        return Response(
            content=data,
            media_type=media_type,
            headers={"Cache-Control": "public, max-age=3600"},
        )

    resolved = resolve_image(settings, path)
    filename = resolved.name if download else None
    return FileResponse(
        resolved,
        media_type=_media_type(resolved),
        filename=filename,
        headers={"Cache-Control": "public, max-age=60"},
    )


@app.post("/api/infer", response_class=JSONResponse)
def infer(request: InferRequest) -> dict[str, object]:
    image_path = resolve_image(settings, request.image_path)
    checkpoint_path = resolve_checkpoint(settings, request.checkpoint_path)

    # Fail fast for unreadable images instead of discovering it in the worker.
    image_info(settings, request.image_path)

    job = jobs.submit(image_path, checkpoint_path)
    logger.info(
        "queued job %s: %s + %s",
        job.id,
        display_path(settings, image_path),
        display_path(settings, checkpoint_path),
    )
    return {"job_id": job.id, "status": job.status}


@app.get("/api/jobs/{job_id}", response_class=JSONResponse)
def job_status(job_id: str) -> dict[str, object]:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    return job


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Super-resolution inference visualiser")
    parser.add_argument("--host", default=settings.host)
    parser.add_argument("--port", type=int, default=settings.port)
    parser.add_argument("--runs-dir", type=Path, default=settings.runs_dir)
    parser.add_argument("--results-dir", type=Path, default=settings.results_dir)
    parser.add_argument(
        "--device",
        default=settings.device,
        help="auto | cpu | cuda | cuda:0",
    )
    parser.add_argument("--auto-tile", type=int, default=settings.auto_tile)
    parser.add_argument("--tile-overlap", type=int, default=settings.tile_overlap)
    parser.add_argument(
        "--allowed-root",
        action="append",
        default=None,
        help="extra directory the browser may access (repeatable)",
    )
    parser.add_argument("--default-browse", default=settings.default_browse_path)
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    settings.host = args.host
    settings.port = args.port
    settings.runs_dir = args.runs_dir.expanduser().resolve()
    settings.results_dir = args.results_dir.expanduser().resolve()
    settings.device = args.device
    settings.auto_tile = args.auto_tile
    settings.tile_overlap = args.tile_overlap
    settings.default_browse_path = args.default_browse

    if args.allowed_root:
        roots = [settings.project_root.resolve()]
        for raw in args.allowed_root:
            p = Path(raw).expanduser()
            roots.append((p if p.is_absolute() else settings.project_root / p).resolve())
        settings.allowed_roots = roots

    settings.ensure_dirs()

    # Export the effective settings so uvicorn's reload worker (which imports
    # this module in a fresh process) sees the same values.
    os.environ["SR_INFER_HOST"] = settings.host
    os.environ["SR_INFER_PORT"] = str(settings.port)
    os.environ["SR_INFER_RUNS_DIR"] = str(settings.runs_dir)
    os.environ["SR_INFER_RESULTS_DIR"] = str(settings.results_dir)
    os.environ["SR_INFER_DEVICE"] = settings.device
    os.environ["SR_INFER_TILE"] = str(settings.auto_tile)
    os.environ["SR_INFER_TILE_OVERLAP"] = str(settings.tile_overlap)
    os.environ["SR_INFER_DEFAULT_BROWSE"] = settings.default_browse_path
    os.environ["SR_INFER_ALLOWED_ROOTS"] = os.pathsep.join(
        str(path) for path in settings.resolved_allowed_roots()
    )

    logger.info("listening on http://%s:%d", settings.host, settings.port)
    logger.info("runs: %s", settings.runs_dir)
    logger.info("results: %s", settings.results_dir)
    logger.info("allowed roots: %s", [str(p) for p in settings.resolved_allowed_roots()])

    import uvicorn

    uvicorn.run(
        "servers.sr_infer_server.main:app",
        host=settings.host,
        port=settings.port,
        reload=args.reload,
        log_level="debug" if args.verbose else "info",
    )


if __name__ == "__main__":
    main()
