"""Bounded SO motor/camera recovery at a complete control-frame boundary.

Only decorated SO I/O inside ``hardware_frame`` participates. No cached values,
synthetic observations or retry-time commands are returned to the recording loop.
"""

from __future__ import annotations

import logging
import math
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps

from lerobot.utils.errors import DeviceNotConnectedError
from lerobot.utils.runtime_bridge import emit_runtime_event

logger = logging.getLogger(__name__)
_active: ContextVar[bool] = ContextVar("hardware_frame_active", default=False)
RECOVERY_ATTEMPTS = 3


def hardware_frame_active() -> bool:
    return _active.get()


class HardwareRecoveryFailed(ConnectionError):
    """The bounded recovery budget was exhausted; do not resume this run."""


class _HardwareIOFailed(Exception):
    def __init__(self, device, error: Exception):
        self.device = device
        self.error = error
        super().__init__(str(error))


def _communication_error(error: Exception) -> bool:
    if isinstance(error, (ConnectionError, OSError)):
        return True
    # OpenCV currently reports transport/read-thread failures as RuntimeError.
    # Do not retry programming errors, shape errors or invalid camera settings.
    return isinstance(error, RuntimeError) and any(
        text in str(error)
        for text in ("read thread is not running", "has not captured any frames", "read failed (status=")
    )


def _positions(device):
    values = device.bus.sync_read("Present_Position", num_retry=device.config.num_read_retries)
    if set(values) != set(device.bus.motors) or not all(math.isfinite(v) for v in values.values()):
        raise ConnectionError(f"{device}: incomplete or non-finite motor feedback")
    return values


def hardware_io(method):
    """Identify hardware failures without catching policy/network/storage errors."""

    @wraps(method)
    def call(self, *args, **kwargs):
        if not hardware_frame_active():
            return method(self, *args, **kwargs)
        try:
            result = method(self, *args, **kwargs)
            if method.__name__ in {"get_observation", "get_action"} and hasattr(self, "bus"):
                keys = [f"{name}.pos" for name in self.bus.motors]
                if any(key not in result or not math.isfinite(result[key]) for key in keys):
                    raise ConnectionError(f"{self}: incomplete or non-finite observation")
            # Feetech group writes are unacknowledged. Require fresh feedback
            # before allowing an action frame to reach Dataset.add_frame.
            if method.__name__ == "send_action" and hasattr(self, "bus"):
                _positions(self)
            return result
        except Exception as error:
            if not _communication_error(error):
                raise
            raise _HardwareIOFailed(self, error) from error

    return call


def _probe_or_reopen(device) -> None:
    if hasattr(device, "left_arm") and hasattr(device, "right_arm"):
        _probe_or_reopen(device.left_arm)
        _probe_or_reopen(device.right_arm)
        return

    try:
        _positions(device)
    except (ConnectionError, OSError):
        if device.bus.is_connected:
            device.bus.disconnect(False)  # Reopen the transport; do not change torque or pose.
        device.bus.connect()
        _positions(device)

    for camera in getattr(device, "cameras", {}).values():
        try:
            camera.read_latest()
        except Exception as error:
            if not _communication_error(error):
                raise
            try:
                camera.disconnect()
            except DeviceNotConnectedError:
                pass
            if camera.__class__.__name__ == "OpenCVCamera":
                camera.connect(warmup=False)  # One open per outer recovery attempt.
                camera.async_read(timeout_ms=1000)
            else:
                camera.connect()
            camera.read_latest()


@dataclass
class HardwareFrame:
    recovered: bool = False


@contextmanager
def hardware_frame(operation: str, on_recovered=None, on_interrupted=None):
    """Discard a failed whole frame, recover at most three times, then start fresh.

    The body must contain observation -> action -> send -> dataset/ring-buffer append
    in that order. Suppressing the private I/O exception skips every remaining step
    in that body, including image/video encoding and dataset insertion.
    """
    state = HardwareFrame()
    token = _active.set(True)
    try:
        yield state
    except _HardwareIOFailed as failure:
        if on_interrupted is not None:
            on_interrupted()
        last_error = failure.error
        for attempt in range(1, RECOVERY_ATTEMPTS + 1):
            data = dict(
                hardware_recovery="retrying",
                attempt=attempt,
                max_attempts=RECOVERY_ATTEMPTS,
                device=str(failure.device),
                error=str(last_error),
                frame_discarded=True,
            )
            logger.warning(
                "Hardware frame discarded; recovering %s (%d/%d): %s",
                failure.device,
                attempt,
                RECOVERY_ATTEMPTS,
                last_error,
            )
            emit_runtime_event(operation, "connecting", **data)
            time.sleep(0.5 * attempt)
            try:
                _probe_or_reopen(failure.device)
            except (ConnectionError, OSError, RuntimeError) as error:
                last_error = error
                continue
            if on_recovered is not None:
                on_recovered()
            state.recovered = True
            emit_runtime_event(operation, "running", **{**data, "hardware_recovery": "recovered"})
            break
        else:
            emit_runtime_event(
                operation, "failed", **{**data, "hardware_recovery": "failed", "error": str(last_error)}
            )
            raise HardwareRecoveryFailed(
                f"{failure.device}: connection did not recover after {RECOVERY_ATTEMPTS} attempts; "
                "failed frame was discarded"
            ) from last_error
    finally:
        _active.reset(token)
