"""Structured operator I/O for terminal and embedded LeRobot runtimes."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Protocol


class RuntimeBridge(Protocol):
    """Connect a LeRobot operation to its operator interface."""

    def emit(self, operation: str, phase: str, data: dict[str, Any]) -> None: ...

    def take_commands(self) -> set[str]: ...

    def prompt(self, prompt_id: str, message: str) -> str: ...


class TerminalRuntimeBridge:
    """Preserve the native CLI behavior when no embedded runtime is active."""

    def emit(self, operation: str, phase: str, data: dict[str, Any]) -> None:
        pass

    def take_commands(self) -> set[str]:
        return set()

    def prompt(self, prompt_id: str, message: str) -> str:
        return input(message)


_terminal_bridge = TerminalRuntimeBridge()
_bridge: ContextVar[RuntimeBridge | None] = ContextVar("lerobot_runtime_bridge", default=None)


def _current_bridge() -> RuntimeBridge:
    return _bridge.get() or _terminal_bridge


@contextmanager
def use_runtime_bridge(bridge: RuntimeBridge) -> Iterator[None]:
    """Install a bridge for one operation without changing global CLI behavior."""

    token = _bridge.set(bridge)
    try:
        yield
    finally:
        _bridge.reset(token)


def emit_runtime_event(operation: str, phase: str, **data: Any) -> None:
    _current_bridge().emit(operation, phase, data)


def take_runtime_commands() -> set[str]:
    return _current_bridge().take_commands()


def runtime_bridge_active() -> bool:
    return _bridge.get() is not None


def runtime_prompt(prompt_id: str, message: str) -> str:
    return _current_bridge().prompt(prompt_id, message)
