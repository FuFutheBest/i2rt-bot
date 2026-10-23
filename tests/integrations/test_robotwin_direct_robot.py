"""Unit tests for the direct-drive RoboTwin robot shim without SAPIEN."""

from __future__ import annotations

import sys
import types
from typing import Any

import numpy as np

import integrations.robotwin.runtime as runtime_module
from integrations.robotwin.direct_robot import DirectDriveRobot
from integrations.robotwin.runtime import RoboTwinDirectRuntime


class _Pose:
    def __init__(self, p: Any, q: Any) -> None:
        self.p = np.asarray(p, dtype=np.float64)
        self.q = np.asarray(q, dtype=np.float64)


class _Link:
    def __init__(self, name: str) -> None:
        self._name = name

    def get_name(self) -> str:
        return self._name


class _Joint:
    def __init__(self, name: str) -> None:
        self._name = name
        self.child_link = _Link(f"{name}_link")
        self.global_pose = _Pose([0.0, 0.0, 0.8], [1.0, 0.0, 0.0, 0.0])
        self.target = 0.0
        self.velocity_target = 0.0
        self.properties: list[tuple[float, float]] = []

    def get_name(self) -> str:
        return self._name

    def get_drive_target(self) -> np.ndarray:
        return np.asarray([self.target])

    def set_drive_target(self, value: float) -> None:
        self.target = float(value)

    def set_drive_velocity_target(self, value: float) -> None:
        self.velocity_target = float(value)

    def set_drive_property(self, *, stiffness: float, damping: float) -> None:
        self.properties.append((stiffness, damping))


class _Entity:
    def __init__(self, joints: dict[str, _Joint]) -> None:
        self._joints = joints
        self._links = [_Link("left_camera"), _Link("right_camera")]
        self.passive_force_calls = 0
        self.root_pose: _Pose | None = None

    def find_joint_by_name(self, name: str) -> _Joint | None:
        return self._joints.get(name)

    def find_link_by_name(self, name: str) -> _Link | None:
        return next((link for link in self._links if link.get_name() == name), None)

    def get_links(self) -> list[_Link]:
        return self._links

    def get_active_joints(self) -> list[_Joint]:
        return list(self._joints.values())

    def set_root_pose(self, pose: _Pose) -> None:
        self.root_pose = pose

    def compute_passive_force(
        self, *, gravity: bool, coriolis_and_centrifugal: bool
    ) -> np.ndarray:
        assert gravity and coriolis_and_centrifugal
        self.passive_force_calls += 1
        return np.zeros(len(self._joints))

    def set_qf(self, _forces: np.ndarray) -> None:
        return None


class _Loader:
    def __init__(self, entity: _Entity) -> None:
        self.entity = entity
        self.fix_root_link = False

    def load(self, _path: str) -> _Entity:
        return self.entity


class _Scene:
    def __init__(self, entity: _Entity) -> None:
        self.entity = entity

    def create_urdf_loader(self) -> _Loader:
        return _Loader(self.entity)


def _embodiment_config() -> dict[str, object]:
    return {
        "urdf_path": "robot.urdf",
        "arm_joints_name": [
            [f"left_j{index}" for index in range(6)],
            [f"right_j{index}" for index in range(6)],
        ],
        "ee_joints": ["left_ee", "right_ee"],
        "gripper_name": [
            {"base": "left_gripper", "mimic": []},
            {"base": "right_gripper", "mimic": []},
        ],
        "gripper_bias": 0.12,
        "gripper_scale": [0.0, 1.0],
        "homestate": [[0.1] * 6, [-0.1] * 6],
        "robot_pose": [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]],
        "static_camera_list": [],
    }


def test_direct_robot_sets_sapien_drives_without_importing_planners(
    monkeypatch,
) -> None:
    core = types.ModuleType("sapien.core")
    core.Pose = _Pose
    sapien = types.ModuleType("sapien")
    sapien.core = core
    monkeypatch.setitem(sys.modules, "sapien", sapien)
    monkeypatch.setitem(sys.modules, "sapien.core", core)

    names = [f"left_j{index}" for index in range(6)] + [
        f"right_j{index}" for index in range(6)
    ]
    names += ["left_ee", "right_ee", "left_gripper", "right_gripper"]
    joints = {name: _Joint(name) for name in names}
    entity = _Entity(joints)
    left_config = _embodiment_config()
    right_config = _embodiment_config()
    left_config["gripper_stiffness"] = 111.0
    left_config["gripper_damping"] = 11.0
    left_config["joint_stiffness"] = 333.0
    left_config["joint_damping"] = 33.0
    right_config["gripper_stiffness"] = 222.0
    right_config["gripper_damping"] = 22.0
    right_config["joint_stiffness"] = 444.0
    right_config["joint_damping"] = 44.0
    robot = DirectDriveRobot(
        _Scene(entity),
        left_embodiment_config=left_config,
        right_embodiment_config=right_config,
        left_robot_file="/tmp/left",
        right_robot_file="/tmp/right",
        dual_arm_embodied=True,
    )
    robot.init_joints()
    robot.move_to_homestate()
    robot.set_arm_joints(np.arange(6), np.arange(6) * 0.5, "left")
    robot.set_gripper(1.0, "right", gripper_eps=0)

    assert [joint.target for joint in robot.left_arm_joints] == [
        float(index) for index in range(6)
    ]
    assert [joint.velocity_target for joint in robot.left_arm_joints] == [
        float(index) * 0.5 for index in range(6)
    ]
    assert joints["right_gripper"].target == 1.0
    assert joints["left_gripper"].properties[-1] == (111.0, 11.0)
    assert joints["right_gripper"].properties[-1] == (222.0, 22.0)
    assert joints["left_j0"].properties[-1] == (333.0, 33.0)
    assert joints["right_j0"].properties[-1] == (444.0, 44.0)
    assert entity.passive_force_calls >= 2
    assert "mplib" not in sys.modules
    assert "curobo" not in sys.modules


def test_runtime_apply_joint14_target_passes_arm_velocity() -> None:
    class _Robot:
        def __init__(self) -> None:
            self.left_arm_joints = [object()] * 6
            self.right_arm_joints = [object()] * 6
            self.arm_calls: list[tuple[np.ndarray, np.ndarray, str]] = []
            self.gripper_calls: list[tuple[float, str, float, float | None]] = []

        def set_arm_joints(
            self, target_position: np.ndarray, target_velocity: np.ndarray, arm_tag: str
        ) -> None:
            self.arm_calls.append(
                (
                    np.asarray(target_position, dtype=np.float64).copy(),
                    np.asarray(target_velocity, dtype=np.float64).copy(),
                    arm_tag,
                )
            )

        def set_gripper(
            self,
            gripper_val: float,
            arm_tag: str,
            gripper_eps: float = 0.1,
            gripper_velocity: float | None = None,
        ) -> None:
            self.gripper_calls.append(
                (float(gripper_val), arm_tag, gripper_eps, gripper_velocity)
            )

    robot = _Robot()
    runtime = RoboTwinDirectRuntime.__new__(RoboTwinDirectRuntime)
    runtime.task = types.SimpleNamespace(robot=robot)
    target = np.arange(14, dtype=np.float64)
    velocity = np.arange(14, dtype=np.float64) * 0.1

    runtime.apply_joint14_target(target, velocity)

    np.testing.assert_allclose(robot.arm_calls[0][0], target[:6])
    np.testing.assert_allclose(robot.arm_calls[0][1], velocity[:6])
    assert robot.arm_calls[0][2] == "left"
    np.testing.assert_allclose(robot.arm_calls[1][0], target[7:13])
    np.testing.assert_allclose(robot.arm_calls[1][1], velocity[7:13])
    assert robot.arm_calls[1][2] == "right"
    assert robot.gripper_calls == [
        (6.0, "left", 0, 0.6000000000000001),
        (13.0, "right", 0, 1.3),
    ]


def test_runtime_creates_gui_viewer_after_headless_task_setup(monkeypatch) -> None:
    class _Viewer:
        def __init__(self, renderer: object, *, resolutions: tuple[int, int]) -> None:
            self.renderer = renderer
            self.resolutions = resolutions
            self.scene = None
            self.xyz = None
            self.rpy = None

        def set_scene(self, scene: object) -> None:
            self.scene = scene

        def set_camera_xyz(self, **kwargs: float) -> None:
            self.xyz = kwargs

        def set_camera_rpy(self, **kwargs: float) -> None:
            self.rpy = kwargs

    viewer_module = types.ModuleType("sapien.utils.viewer")
    viewer_module.Viewer = _Viewer
    monkeypatch.setitem(sys.modules, "sapien.utils.viewer", viewer_module)
    task = types.SimpleNamespace(renderer=object(), scene=object())

    RoboTwinDirectRuntime._create_gui_viewer(
        task,
        {"camera_xyz_x": 1.25, "camera_rpy_y": 0.75},
        (640, 480),
    )

    assert isinstance(task.viewer, _Viewer)
    assert task.viewer.resolutions == (640, 480)
    assert task.viewer.xyz == {"x": 1.25, "y": 0.22, "z": 1.5}
    assert task.viewer.rpy == {"r": 0.0, "p": -0.8, "y": 0.75}


def test_runtime_captures_only_required_policy_cameras() -> None:
    class _Camera:
        def __init__(self, value: float) -> None:
            self.value = value
            self.take_picture_calls = 0

        def take_picture(self) -> None:
            self.take_picture_calls += 1

        def get_picture(self, name: str) -> np.ndarray:
            assert name == "Color"
            return np.full((2, 3, 4), self.value, dtype=np.float32)

    left = _Camera(0.1)
    right = _Camera(0.2)
    head = _Camera(0.3)
    front = _Camera(0.4)

    class _Task:
        def __init__(self) -> None:
            self.update_render_calls = 0
            self.cameras = types.SimpleNamespace(
                left_camera=left,
                right_camera=right,
                static_camera_name=["head_camera", "front_camera"],
                static_camera_list=[head, front],
            )

        def _update_render(self) -> None:
            self.update_render_calls += 1

    runtime = RoboTwinDirectRuntime(
        robotwin_root=".",
        task_name="beat_block_hammer",
        task_config="demo_clean",
        seed=0,
        physics_fps=250.0,
        enable_gui=False,
        gui_render_fps=30.0,
    )
    task = _Task()
    runtime.task = task

    frames = runtime.capture_rgb()

    assert task.update_render_calls == 1
    assert set(frames) == {"head_camera", "left_camera", "right_camera"}
    assert all(
        frame.shape == (2, 3, 3) and frame.dtype == np.uint8
        for frame in frames.values()
    )
    assert (
        left.take_picture_calls
        == right.take_picture_calls
        == head.take_picture_calls
        == 1
    )
    assert front.take_picture_calls == 0


def test_runtime_installs_direct_shim_before_task_import_and_sets_requested_timestep(
    tmp_path,
    monkeypatch,
) -> None:
    class _RuntimeLink:
        def set_mass(self, _mass: float) -> None:
            return None

    class _RuntimeEntity:
        def get_links(self) -> list[_RuntimeLink]:
            return [_RuntimeLink()]

    class _RuntimeRobot:
        def __init__(
            self, _scene: object, *, need_topp: bool, **_kwargs: object
        ) -> None:
            assert need_topp is False
            self.left_entity = _RuntimeEntity()
            self.right_entity = _RuntimeEntity()
            self.gripper_calls: list[tuple[float, str, float]] = []

        def init_joints(self) -> None:
            return None

        def set_gripper(self, value: float, arm: str, *, gripper_eps: float) -> None:
            self.gripper_calls.append((value, arm, gripper_eps))

    class _RuntimeScene:
        def __init__(self) -> None:
            self.timestep = None

        def set_timestep(self, value: float) -> None:
            self.timestep = value

    class _Render:
        def __init__(self) -> None:
            self.camera_shader_dirs: list[str] = []
            self.viewer_shader_dirs: list[str] = []

        def set_camera_shader_dir(self, value: str) -> None:
            self.camera_shader_dirs.append(value)

        def set_viewer_shader_dir(self, value: str) -> None:
            self.viewer_shader_dirs.append(value)

    class _Task:
        def setup_scene(self, **kwargs: object) -> None:
            sapien.render.set_camera_shader_dir(str(kwargs["camera_shader_dir"]))
            self.renderer_capacities = (
                kwargs["max_num_materials"],
                kwargs["max_num_textures"],
            )
            self.scene = _RuntimeScene()
            self.scene.set_timestep(float(kwargs["timestep"]))
            self.renderer = object()

        def setup_demo(self, **kwargs: object) -> None:
            self.setup_scene()
            self.load_robot(**kwargs)
            self.together_open_gripper()

    task_module = types.ModuleType("envs.beat_block_hammer")
    task_module.beat_block_hammer = _Task
    render = _Render()
    sapien = types.ModuleType("sapien")
    sapien.render = render
    root = tmp_path / "robotwin"
    (root / "envs").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.delitem(sys.modules, "envs._base_task", raising=False)
    monkeypatch.setitem(sys.modules, "envs.beat_block_hammer", task_module)
    monkeypatch.setitem(sys.modules, "sapien", sapien)
    monkeypatch.setattr(runtime_module, "DirectDriveRobot", _RuntimeRobot)
    monkeypatch.setattr(RoboTwinDirectRuntime, "_load_task_args", lambda _self: {})

    runtime = RoboTwinDirectRuntime(
        robotwin_root=root,
        task_name="beat_block_hammer",
        task_config="demo_clean",
        seed=0,
        physics_fps=400.0,
        enable_gui=False,
        gui_render_fps=30.0,
    )
    runtime.start()

    assert sys.modules["envs.robot"].Robot is _RuntimeRobot
    assert runtime.task is not None
    assert runtime.task.scene.timestep == 1.0 / 400.0
    assert runtime.task.robot.gripper_calls == [(1.0, "left", 0), (1.0, "right", 0)]
    assert render.camera_shader_dirs == ["default"]
    assert render.viewer_shader_dirs == ["minimal"]
    assert runtime.task.renderer_capacities == (128, 512)
