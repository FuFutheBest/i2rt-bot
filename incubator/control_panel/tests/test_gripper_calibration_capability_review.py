from __future__ import annotations

from dataclasses import replace

from incubator.control_panel import gripper_control
from incubator.control_panel.tests.test_gripper_concurrency_safety_review import (
    _FakeContext,
    _FakeProcess,
    _manager,
    _verified_artifact,
)


def test_uncalibrated_grippers_can_calibrate_but_cannot_move() -> None:
    manager, _ = _manager()
    try:
        manager._profiles = {
            side: replace(
                profile,
                min_encoder=None,
                max_encoder=None,
                min_angle_deg=None,
                max_angle_deg=None,
            )
            for side, profile in manager._profiles.items()
        }

        state = manager.state()

        assert state['sides']['left']['calibrated'] is False
        assert state['sides']['right']['calibrated'] is False
        assert state['sides']['left']['can_calibrate'] is True
        assert state['sides']['right']['can_calibrate'] is True
        assert state['can_calibrate'] is True
        assert state['sides']['left']['can_position'] is False
        assert state['sides']['right']['can_position'] is False
    finally:
        manager.close()


def test_one_side_permission_error_does_not_block_healthy_side(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        gripper_control,
        "_calibration_artifact_state",
        _verified_artifact,
    )
    manager, _ = _manager()
    try:
        manager._permission_policy_check = lambda port: (
            'injected permission denial' if port == '/dev/ttyUSB1' else None
        )

        state = manager.state()

        assert state['sides']['left']['can_position'] is False
        assert 'permission' in ' '.join(state['sides']['left']['blockers']).lower()
        assert state['sides']['right']['can_position'] is True
    finally:
        manager.close()


class _StartFailureProcess(_FakeProcess):
    def start(self) -> None:
        raise OSError('injected process.start failure')


class _StartFailureContext(_FakeContext):
    def Process(self, **kwargs):
        process = _StartFailureProcess(**kwargs)
        self.processes.append(process)
        return process


class _PartialStartFailureProcess(_FakeProcess):
    def start(self) -> None:
        self.alive = True
        raise OSError('injected failure after child creation')


class _PartialStartFailureContext(_FakeContext):
    def Process(self, **kwargs):
        process = _PartialStartFailureProcess(**kwargs)
        self.processes.append(process)
        return process


def test_position_spawn_failure_is_contained_as_side_error() -> None:
    manager, _ = _manager()
    manager._context = _StartFailureContext()
    try:
        manager.position(
            'left', 25.0, 'left-spawn-failure-0001', expected_side_revision=0
        )

        manager._dispatch_locked()

        assert manager._workers == {}
        assert manager._pending_positions == {}
        assert manager._sides['left']['status'] == 'error'
        assert 'process.start' in manager._sides['left']['last_error']
        assert manager._sides['right']['status'] == 'idle'
    finally:
        manager.close()


def test_partial_spawn_failure_terminates_and_joins_unregistered_child() -> None:
    manager, _ = _manager()
    context = _PartialStartFailureContext()
    manager._context = context
    try:
        manager.position(
            'left', 25.0, 'left-partial-spawn-failure-0001', expected_side_revision=0
        )

        manager._dispatch_locked()

        assert manager._workers == {}
        assert len(context.processes) == 1
        assert context.processes[0].alive is False
        assert context.processes[0].join_calls >= 1
        assert manager._sides['left']['status'] == 'error'
    finally:
        manager.close()
