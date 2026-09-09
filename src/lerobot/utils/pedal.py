# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generic foot pedal listener using evdev.

Callers supply a callback receiving the pressed key code (e.g. ``"KEY_A"``)
and a dedicated device path. The workflow owns the listener and stops it
at teardown. Connection failures are available as structured status.
Strategy-specific key mapping logic lives in the caller.
"""

from __future__ import annotations

import logging
import os
import select
import threading
import time
from collections.abc import Callable
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_PEDAL_DEVICE = "/dev/input/by-id/usb-PCsensor_FootSwitch-event-kbd"


def resolve_pedal_device(device_path: str | None = None) -> str | None:
    """Prefer explicit settings; never automatically capture a generic keyboard."""
    configured = device_path or os.getenv("EVOMIND_PEDAL_DEVICE") or os.getenv("EVOMIND_DAGGER_PEDAL_DEVICE")
    if configured:
        return configured
    for candidate in ("/dev/input/evomind-pedal", DEFAULT_PEDAL_DEVICE):
        if Path(candidate).exists():
            return candidate
    return None


def pedal_key() -> str:
    return os.getenv("EVOMIND_PEDAL_KEY", os.getenv("EVOMIND_DAGGER_PEDAL_KEY", "*"))


class PedalPressFilter:
    """One press per down/up cycle, excluding autorepeat and contact bounce."""

    def __init__(self, debounce_s: float = 0.25):
        self.debounce_s = debounce_s
        self.held: set[str | int] = set()
        self.last_press = float("-inf")

    def accept(self, code: str | int, value: int, now: float) -> bool:
        if value == 0:
            self.held.discard(code)
            return False
        if value != 1 or code in self.held:
            return False
        self.held.add(code)
        if now - self.last_press < self.debounce_s:
            return False
        self.last_press = now
        return True


class PedalListener(threading.Thread):
    """A workflow-owned listener which can be stopped before the next session."""

    def __init__(self, on_press: Callable[[str], None], device_path: str):
        super().__init__(daemon=True, name="PedalListener")
        self.on_press = on_press
        self.device_path = device_path
        self.stop_event = threading.Event()
        self._status = {"state": "connecting", "device": device_path}
        self._status_lock = threading.Lock()
        self.device = None

    @property
    def status(self) -> dict:
        with self._status_lock:
            return dict(self._status)

    def set_status(self, state: str, **details) -> None:
        with self._status_lock:
            self._status = {"state": state, "device": self.device_path, **details}

    def run(self) -> None:
        from evdev import categorize, ecodes

        try:
            while not self.stop_event.is_set():
                try:
                    if self.device is None:
                        self._connect()
                    press_filter = PedalPressFilter()
                    # A held pedal at connect/reconnect is not a new press.
                    press_filter.held.update(self.device.active_keys())
                    while not self.stop_event.is_set():
                        if not select.select([self.device], [], [], 0.1)[0]:
                            continue
                        try:
                            events = list(self.device.read())
                        except BlockingIOError:
                            continue
                        for event in events:
                            if self.stop_event.is_set():
                                break
                            if event.type != ecodes.EV_KEY:
                                continue
                            key = categorize(event)
                            code = key.keycode
                            if isinstance(code, list | tuple):
                                code = code[0]
                            # Filter numeric IDs so active_keys() uses the same identity.
                            if press_filter.accept(event.code, event.value, time.monotonic()):
                                try:
                                    self.on_press(code)
                                except Exception:
                                    logger.exception("Pedal callback failed")
                except (OSError, ValueError) as error:
                    self.set_status("reconnecting", error=str(error))
                    logger.warning("Pedal reconnecting (%s): %s", self.device_path, error)
                    self._close()
                    if self.stop_event.wait(0.5):
                        break
        finally:
            self._close()
            if self.stop_event.is_set():
                self.set_status("stopped")

    def _connect(self) -> None:
        from evdev import InputDevice

        self.device = InputDevice(self.device_path)
        try:
            # Preserve exclusive capture: a space-emulating pedal must not also
            # activate the browser's currently focused button.
            self.device.grab()
        except OSError:
            self._close()
            raise
        self.set_status("connected", name=self.device.name)
        logger.info("Pedal connected: %s (%s)", self.device.name, self.device_path)

    def _close(self) -> None:
        if self.device is not None:
            try:
                self.device.close()
            finally:
                self.device = None

    def stop(self) -> None:
        self.stop_event.set()
        if self.is_alive() and threading.current_thread() is not self:
            self.join(timeout=1.0)


def pedal_status(listener: PedalListener | None) -> dict:
    return listener.status if listener is not None else {"state": "not_configured"}


def start_pedal_listener(
    on_press: Callable[[str], None],
    device_path: str = DEFAULT_PEDAL_DEVICE,
) -> PedalListener:
    """Spawn a daemon thread that forwards pedal key-press codes to ``on_press``.

    Parameters
    ----------
    on_press:
        Callback invoked with the pressed key code string (e.g. ``"KEY_A"``)
        on each pedal press event.  The callback runs in the listener thread
        and must be thread-safe.
    device_path:
        Linux input device path (e.g. ``/dev/input/by-id/...``).

    Returns
    -------
    A stoppable daemon listener. Device failures retry the configured path.
    A missing evdev dependency is reported without starting a thread.
    """
    listener = PedalListener(on_press, device_path)
    try:
        listener._connect()
    except ImportError as error:
        listener.set_status("unavailable", error=str(error))
        logger.warning("Pedal unavailable (%s): %s", device_path, error)
        return listener
    except OSError as error:
        listener.set_status("reconnecting", error=str(error))
        logger.warning("Pedal reconnecting (%s): %s", device_path, error)
    listener.start()
    return listener
