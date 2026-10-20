"""In-process background job manager for inference requests."""

from __future__ import annotations

import logging
import threading
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .config import Settings
from .inference import run_inference

logger = logging.getLogger("sr_infer.jobs")

MAX_JOBS = 100


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


@dataclass(slots=True)
class Job:
    id: str
    status: str = "queued"  # queued | running | success | error
    progress: float = 0.0
    message: str = "排队中"
    created_at: str = field(default_factory=_now)
    started_at: str | None = None
    finished_at: str | None = None
    image_path: str = ""
    checkpoint_path: str = ""
    result: dict[str, object] | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "status": self.status,
            "progress": round(self.progress, 4),
            "message": self.message,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "image_path": self.image_path,
            "checkpoint_path": self.checkpoint_path,
            "result": self.result,
            "error": self.error,
        }


class JobManager:
    """Serialises GPU inference and exposes polling-friendly state."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="sr-infer"
        )
        self._jobs: OrderedDict[str, Job] = OrderedDict()
        self._lock = threading.Lock()

    def submit(self, image_path: Path, checkpoint_path: Path) -> Job:
        job = Job(
            id=uuid.uuid4().hex,
            image_path=str(image_path),
            checkpoint_path=str(checkpoint_path),
        )
        with self._lock:
            self._jobs[job.id] = job
            self._trim_locked()
        self._executor.submit(self._run, job.id)
        return job

    def _trim_locked(self) -> None:
        while len(self._jobs) > MAX_JOBS:
            for job_id, job in list(self._jobs.items()):
                if job.status in {"success", "error"}:
                    del self._jobs[job_id]
                    break
            else:
                break

    def _update(self, job_id: str, **changes: object) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            for key, value in changes.items():
                setattr(job, key, value)

    def _run(self, job_id: str) -> None:
        self._update(job_id, status="running", started_at=_now(), message="开始推理")

        def progress(fraction: float, message: str) -> None:
            with self._lock:
                job = self._jobs.get(job_id)
                if job is None:
                    return
                # Callbacks can report overlapping stages; never move backwards.
                job.progress = max(job.progress, fraction)
                job.message = message

        try:
            # Re-read paths from the job so the worker always uses the validated
            # absolute paths stored at submit time.
            with self._lock:
                job = self._jobs.get(job_id)
                image_path = Path(job.image_path) if job else None
                checkpoint_path = Path(job.checkpoint_path) if job else None
            if image_path is None or checkpoint_path is None:
                raise RuntimeError("job disappeared")

            result = run_inference(
                self.settings, image_path, checkpoint_path, progress
            )
        except Exception as exc:  # noqa: BLE001 - surfaced to the UI
            logger.exception("inference job %s failed", job_id)
            self._update(
                job_id,
                status="error",
                message="推理失败",
                error=str(exc) or exc.__class__.__name__,
                finished_at=_now(),
            )
        else:
            self._update(
                job_id,
                status="success",
                progress=1.0,
                message="完成",
                result=result,
                finished_at=_now(),
            )

    def get(self, job_id: str) -> dict[str, object] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return job.to_dict() if job is not None else None

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
