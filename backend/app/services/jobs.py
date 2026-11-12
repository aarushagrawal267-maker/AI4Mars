"""Small in-process background job runner (dataset scans/downloads, evaluations)."""
from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

log = logging.getLogger(__name__)


@dataclass
class Job:
    kind: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    status: str = "queued"            # queued | running | completed | failed | cancelled
    progress: float = 0.0
    stage: str = ""
    message: str = ""
    error: str | None = None
    result: Any = None
    params: dict = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)

    @property
    def active(self) -> bool:
        return self.status in ("queued", "running")

    def update(self, progress: float | None = None, message: str | None = None, stage: str | None = None) -> None:
        if progress is not None:
            self.progress = float(max(0.0, min(1.0, progress)))
        if message is not None:
            self.message = message
        if stage is not None:
            self.stage = stage

    def to_dict(self) -> dict:
        return {"id": self.id, "kind": self.kind, "status": self.status, "progress": round(self.progress, 4),
                "stage": self.stage, "message": self.message, "error": self.error, "params": self.params,
                "created_at": self.created_at, "finished_at": self.finished_at,
                "result": self.result if isinstance(self.result, (dict, list, str, int, float, type(None))) else None}


class JobRunner:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def submit(self, kind: str, fn: Callable[[Job], Any], params: dict | None = None,
               exclusive: bool = True) -> Job:
        """Start ``fn(job)`` on a thread. With ``exclusive`` only one active job per kind is allowed."""
        with self._lock:
            if exclusive and any(j.kind == kind and j.active for j in self._jobs.values()):
                raise RuntimeError(f"A {kind} job is already running")
            job = Job(kind=kind, params=params or {})
            self._jobs[job.id] = job

        def runner() -> None:
            job.status = "running"
            try:
                job.result = fn(job)
                job.status = "cancelled" if job.cancel_event.is_set() else "completed"
                if job.status == "completed":
                    job.progress = 1.0
            except Exception as exc:  # noqa: BLE001 - surfaced to the UI through the job record
                if job.cancel_event.is_set():
                    job.status, job.message = "cancelled", "Cancelled"
                else:
                    log.exception("%s job failed", kind)
                    job.status, job.error = "failed", f"{type(exc).__name__}: {exc}"
            finally:
                job.finished_at = time.time()

        threading.Thread(target=runner, name=f"job-{kind}-{job.id}", daemon=True).start()
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def latest(self, kind: str) -> Job | None:
        jobs = [j for j in self._jobs.values() if j.kind == kind]
        return max(jobs, key=lambda j: j.created_at) if jobs else None

    def active(self, kind: str) -> Job | None:
        for j in self._jobs.values():
            if j.kind == kind and j.active:
                return j
        return None
