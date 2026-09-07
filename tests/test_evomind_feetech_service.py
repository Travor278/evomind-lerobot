#!/usr/bin/env python

from unittest.mock import Mock, call

import pytest

from evomind_lerobot.feetech_service import (
    FeetechActionRequest,
    _apply_action,
    _validate_id_change,
    _verify_id_change,
)


def test_set_id_disables_torque_before_writing_eeprom() -> None:
    bus = Mock()
    request = FeetechActionRequest(
        device_id="controller",
        motor_id=6,
        action="set_id",
        value=5,
        confirmed=True,
    )

    _apply_action(bus, "motor_6", request)

    assert bus.method_calls == [
        call.disable_torque("motor_6", num_retry=1),
        call.write("ID", "motor_6", 5, normalize=False, num_retry=1),
    ]


def test_validate_id_change_rejects_current_id() -> None:
    with pytest.raises(ValueError, match="必须与当前 ID 不同"):
        _validate_id_change({6: 777}, 6, 6)


def test_validate_id_change_rejects_occupied_id() -> None:
    with pytest.raises(ValueError, match="已被总线上的其他舵机使用"):
        _validate_id_change({1: 777, 6: 777}, 6, 1)


def test_verify_id_change_accepts_new_id_and_removed_old_id() -> None:
    _verify_id_change({1: 777, 5: 777}, 6, 5, 777)


def test_verify_id_change_requires_expected_model_on_new_id() -> None:
    with pytest.raises(RuntimeError, match="未检测到新 ID 5"):
        _verify_id_change({5: 123}, 6, 5, 777)


def test_verify_id_change_requires_old_id_to_disappear() -> None:
    with pytest.raises(RuntimeError, match="旧 ID 6 仍然存在"):
        _verify_id_change({5: 777, 6: 777}, 6, 5, 777)
