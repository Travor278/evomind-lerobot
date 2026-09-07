"""Single-owner operation state for robots and cameras."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from threading import Lock
from uuid import uuid4

from evomind_lerobot.events import EventBroker, Operation, Phase


class HardwareBusyError(RuntimeError):
    pass


@dataclass(frozen=True)
class JobSnapshot:
    id: str
    operation: Operation
    phase: Phase
    started_at: str
    message: str

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


class JobManager:
    """Guarantee that only one operation owns physical hardware at a time."""

    def __init__(self, events: EventBroker) -> None:
        self._events = events
        self._lock = Lock()
        self._current: JobSnapshot | None = None

    @property
    def current(self) -> JobSnapshot | None:
        with self._lock:
            return self._current

    def acquire(self, operation: Operation, message: str) -> JobSnapshot:
        with self._lock:
            if self._current is not None:
                raise HardwareBusyError(
                    f"Hardware is owned by {self._current.operation} job {self._current.id}"
                )
            job = JobSnapshot(
                id=str(uuid4()),
                operation=operation,
                phase=Phase.STARTING,
                started_at=datetime.now(UTC).isoformat(),
                message=message,
            )
            self._current = job
        self._events.publish(operation, Phase.STARTING, message, job_id=job.id)
        return job

    def release(
        self,
        job_id: str,
        *,
        failed: bool = False,
        message: str = "Operation finished",
        data: dict[str, object] | None = None,
    ) -> None:
        with self._lock:
            if self._current is None or self._current.id != job_id:
                raise ValueError(f"Job {job_id} does not own the hardware")
            operation = self._current.operation
            self._current = None
        phase = Phase.FAILED if failed else Phase.COMPLETED
        self._events.publish(operation, phase, message, job_id=job_id, data=data)
