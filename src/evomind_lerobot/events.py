"""Structured runtime events shared by the API and hardware workers."""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


class Operation(StrEnum):
    IDLE = "idle"
    DISCOVERY = "discovery"
    CALIBRATION = "calibration"
    DIAGNOSTICS = "diagnostics"
    TELEOPERATION = "teleoperation"
    RECORDING = "recording"
    ROLLOUT = "rollout"
    REPLAY = "replay"


class Phase(StrEnum):
    IDLE = "idle"
    STARTING = "starting"
    CONNECTING = "connecting"
    PREPARING = "preparing"
    PROBING = "probing"
    RUNNING = "running"
    RESETTING = "resetting"
    SAVING = "saving"
    STOPPING = "stopping"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True)
class RuntimeEvent:
    sequence: int
    operation: Operation
    phase: Phase
    message: str
    job_id: str | None = None
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class EventBroker:
    """Fan structured events out to every connected local WebSocket."""

    def __init__(self) -> None:
        self._sequence = 0
        self._subscribers: set[asyncio.Queue[RuntimeEvent]] = set()
        self._latest = RuntimeEvent(
            sequence=0,
            operation=Operation.IDLE,
            phase=Phase.IDLE,
            message="Runtime ready",
        )

    @property
    def latest(self) -> RuntimeEvent:
        return self._latest

    def publish(
        self,
        operation: Operation,
        phase: Phase,
        message: str,
        *,
        job_id: str | None = None,
        data: dict[str, Any] | None = None,
    ) -> RuntimeEvent:
        self._sequence += 1
        event = RuntimeEvent(
            sequence=self._sequence,
            operation=operation,
            phase=phase,
            message=message,
            job_id=job_id,
            data=data or {},
        )
        self._latest = event
        for queue in self._subscribers:
            if queue.full():
                queue.get_nowait()
            queue.put_nowait(event)
        return event

    def subscribe(self) -> asyncio.Queue[RuntimeEvent]:
        queue: asyncio.Queue[RuntimeEvent] = asyncio.Queue(maxsize=100)
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[RuntimeEvent]) -> None:
        self._subscribers.discard(queue)
