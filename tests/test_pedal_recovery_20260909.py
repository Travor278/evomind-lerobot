import os
import queue
import sys
import time
from types import SimpleNamespace

from lerobot.utils import pedal


def eventually(predicate):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("Timed out")


class Device:
    name = "Test pedal"

    def __init__(self, held=()):
        self.fd, self.write_fd = os.pipe()
        self.events = queue.Queue()
        self.held = held
        self.closed = False
        self.grabs = 0

    def fileno(self):
        return self.fd

    def grab(self):
        self.grabs += 1

    def active_keys(self):
        return self.held

    def emit(self, value):
        self.events.put(value)
        os.write(self.write_fd, b"x")

    def read(self):
        os.read(self.fd, 1)
        value = self.events.get_nowait()
        if value == "disconnect":
            raise OSError("Device disconnected")
        return [SimpleNamespace(type=1, code=57, keycode="KEY_SPACE", value=value)]

    def close(self):
        if not self.closed:
            os.close(self.fd)
            os.close(self.write_fd)
            self.closed = True


def start(monkeypatch, factory, callback):
    monkeypatch.setitem(sys.modules, "evdev", SimpleNamespace(
        InputDevice=factory, categorize=lambda e: e, ecodes=SimpleNamespace(EV_KEY=1)))
    return pedal.start_pedal_listener(callback, "/test/pedal")


def test_unplug_reconnect_preserves_grab_without_press(monkeypatch):
    first, second = Device(), Device(held=(57,))
    devices = iter((first, second))
    events = []
    listener = start(monkeypatch, lambda _: next(devices), events.append)
    try:
        first.emit("disconnect")
        eventually(lambda: first.closed and second.grabs == 1)
        for value in (1, 2, 0, 1, 1, 2):
            second.emit(value)
        eventually(lambda: len(events) == 1)
        time.sleep(0.1)
        assert events == ["KEY_SPACE"]
        assert first.grabs == second.grabs == 1
    finally:
        listener.stop()
    assert second.closed
    assert not listener.is_alive()


def test_missing_device_can_be_plugged_in(monkeypatch):
    device = Device()
    attempts = []
    def factory(_):
        attempts.append(1)
        if len(attempts) < 3:
            raise FileNotFoundError("Unplugged")
        return device
    received = []
    listener = start(monkeypatch, factory, received.append)
    try:
        eventually(lambda: listener.status["state"] == "connected")
        assert received == []
        device.emit(1)
        eventually(lambda: received == ["KEY_SPACE"])
    finally:
        listener.stop()


def test_callback_exception_isolated(monkeypatch):
    device = Device()
    received = []
    def callback(code):
        received.append(code)
        raise RuntimeError("Test callback failure")
    listener = start(monkeypatch, lambda _: device, callback)
    try:
        device.emit(1)
        eventually(lambda: len(received) == 1)
        time.sleep(0.3)
        device.emit(0)
        device.emit(1)
        eventually(lambda: len(received) == 2)
        assert listener.is_alive()
    finally:
        listener.stop()
