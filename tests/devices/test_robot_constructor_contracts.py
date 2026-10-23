"""Public robot constructors consume structured stream contracts only."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

from incubator.r1.r1_rizon import RizonRobotLKForceGated
from incubator.teleoperation.x1.x1_robot import X1Robot


FORBIDDEN_STREAM_PARAMETERS = {
    "data_shape",
    "data_dtype",
    "fps",
    "buffer_size",
    "control_shape",
    "control_dtype",
    "control_buffer_size",
    "control_type",
    "ctrl_type",
    "control_fps",
}

ROBOT_ENTRYPOINTS = {
    "devices/robots/base.py": {"BaseRobot"},
    "devices/robots/i2rt.py": {"I2RTRobot"},
    "devices/robots/rizon/robot.py": {"RizonRobot"},
    "devices/robots/robotwin.py": {"RoboTwinSimRobot"},
    "devices/robots/sim.py": {"SimRobot", "DualArmSimRobot"},
    "incubator/teleoperation/x1/x1_robot.py": {"X1Robot"},
    "incubator/r1/r1_rizon.py": {"RizonRobotLKForceGated", "YamHeadRobot"},
}


def test_registered_robot_entrypoints_have_structured_signatures(
    repo_root: Path,
) -> None:
    checked: set[str] = set()
    for relative_path, class_names in ROBOT_ENTRYPOINTS.items():
        tree = ast.parse((repo_root / relative_path).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef) or node.name not in class_names:
                continue
            initializer = next(
                child
                for child in node.body
                if isinstance(child, ast.FunctionDef) and child.name == "__init__"
            )
            parameter_names = {
                argument.arg
                for argument in (
                    *initializer.args.posonlyargs,
                    *initializer.args.args,
                    *initializer.args.kwonlyargs,
                )
            }
            assert not FORBIDDEN_STREAM_PARAMETERS & parameter_names
            assert {"state_feedback", "control"} <= parameter_names or (
                node.name == "YamHeadRobot" and "state_feedback" in parameter_names
            )
            checked.add(node.name)

    assert checked == set().union(*ROBOT_ENTRYPOINTS.values())


def test_x1_components_are_derived_from_control_primitives(monkeypatch) -> None:
    monkeypatch.setattr(X1Robot, "_connect_hardware", lambda _self: None)

    robot = X1Robot(
        state_feedback={
            "fps": 60.0,
            "buffer_size": 100,
            "hardware_latency_ms": 0.0,
            "format": {
                "shape": [30],
                "dtype": "float64",
                "type": [
                    "j1-7",
                    "xyz",
                    "qwqxqyqz",
                    "j1-7",
                    "xyz",
                    "qwqxqyqz",
                    "j1-2",
                ],
            },
        },
        control={
            "type": "eef_abs",
            "format": {
                "shape": [14],
                "dtype": "float64",
                "type": ["xyz", "qxqyqzqw", "xyz", "qxqyqzqw"],
            },
        },
    )

    assert robot.gripper_enabled == (False, False)
    assert robot.utilize_gripper is False
    assert robot.enable_head is False


def test_r1_force_gated_adapter_translates_structured_contract(monkeypatch) -> None:
    assert "airexo_type" not in inspect.signature(
        RizonRobotLKForceGated.__init__
    ).parameters
    monkeypatch.setattr(
        "devices.robots.rizon.robot.FlexivAdapter",
        lambda *_args, **_kwargs: object(),
    )

    robot = RizonRobotLKForceGated(
        device_class_id=1,
        robot_sn=("left", "right"),
        state_feedback={
            "fps": 120.0,
            "buffer_size": 1000,
            "hardware_latency_ms": 0.0,
            "format": {
                "shape": [44],
                "dtype": "float64",
                "type": [
                    "j1-7",
                    "dq1-7",
                    "xyz",
                    "qwqxqyqz",
                    "gripper",
                    "j1-7",
                    "dq1-7",
                    "xyz",
                    "qwqxqyqz",
                    "gripper",
                ],
            },
        },
        control={
            "type": "joints",
            "format": {
                "shape": [14],
                "dtype": "float64",
                "type": ["j1-7", "j1-7"],
            },
        },
        motion_limits={
            "velocity": {"joints_rad_s": {"left": [0.3] * 7, "right": [0.4] * 7}},
            "acceleration": {
                "joints_rad_s2": {"left": [2.0] * 7, "right": [3.0] * 7}
            },
        },
        gripper={"backend": "flexiv", "name": "test_gripper"},
        signal_files={},
    )

    assert robot.data_shape == (44,)
    assert robot.control_shape == (14,)
    assert robot.control_type == "joints"
    assert robot.max_velocity == [0.3] * 7 + [0.4] * 7
