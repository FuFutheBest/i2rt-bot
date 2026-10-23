"""Direct-drive RoboTwin runtime used by the SuperInference bridge.

This module imports RoboTwin and SAPIEN only inside :meth:`start`.  Importing
the normal SuperInference policy and test stack therefore does not require a
graphics stack, CuRobo, or a local RoboTwin asset checkout.
"""

from __future__ import annotations

import importlib
import os
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import validate_action
from .direct_robot import DirectDriveRobot


class RoboTwinDirectRuntime:
    """Own one RoboTwin task and expose fixed-tick direct joint drives.

    RoboTwin's stock task stack imports planner modules before its own
    ``load_robot`` hook runs.  The bridge installs a narrow ``envs.robot`` shim
    before importing the task, then overrides initial gripper setup.  It keeps
    the official scene/task/camera setup while making planner import and
    execution impossible in this process.
    """

    def __init__(
        self,
        *,
        robotwin_root: str | os.PathLike[str],
        task_name: str,
        task_config: str,
        seed: int,
        physics_fps: float,
        enable_gui: bool,
        gui_render_fps: float,
        renderer_shader_dir: str = "default",
        viewer_shader_dir: str = "minimal",
        viewer_resolution: tuple[int, int] | list[int] = (640, 480),
        renderer_max_num_materials: int = 128,
        renderer_max_num_textures: int = 512,
        joint_stiffness: float | None = None,
        joint_damping: float | None = None,
        gripper_stiffness: float | None = None,
        gripper_damping: float | None = None,
    ) -> None:
        self.root = Path(robotwin_root).expanduser().resolve()
        self.task_name = str(task_name)
        self.task_config = str(task_config)
        self.seed = int(seed)
        self.physics_fps = float(physics_fps)
        self.enable_gui = bool(enable_gui)
        self.gui_render_fps = float(gui_render_fps)
        self.renderer_shader_dir = str(renderer_shader_dir).strip()
        if not self.renderer_shader_dir:
            raise ValueError("RoboTwin renderer_shader_dir must not be empty")
        self.viewer_shader_dir = str(viewer_shader_dir).strip()
        if not self.viewer_shader_dir:
            raise ValueError("RoboTwin viewer_shader_dir must not be empty")
        self.viewer_resolution = tuple(int(value) for value in viewer_resolution)
        if len(self.viewer_resolution) != 2 or min(self.viewer_resolution) <= 0:
            raise ValueError(
                "RoboTwin viewer_resolution must contain two positive values"
            )
        self.renderer_max_num_materials = int(renderer_max_num_materials)
        self.renderer_max_num_textures = int(renderer_max_num_textures)
        if self.renderer_max_num_materials <= 0 or self.renderer_max_num_textures <= 0:
            raise ValueError(
                "RoboTwin renderer material and texture capacities must be positive"
            )
        self.joint_stiffness = joint_stiffness
        self.joint_damping = joint_damping
        self.gripper_stiffness = gripper_stiffness
        self.gripper_damping = gripper_damping
        self.task: Any = None
        self._last_gui_render_step = -1

    def start(self) -> None:
        """Create a fresh task through RoboTwin's official task configuration path."""
        if not self.root.is_dir():
            raise FileNotFoundError(f"RoboTwin checkout not found: {self.root}")
        if not (self.root / "envs").is_dir():
            raise FileNotFoundError(
                f"RoboTwin envs directory missing: {self.root / 'envs'}"
            )

        # RoboTwin's task config contains relative embodiment paths.  The bridge
        # process intentionally adopts its checkout as CWD before task creation.
        os.chdir(self.root)
        for path in (
            self.root,
            self.root / "scripts",
            self.root / "description" / "utils",
        ):
            path_string = str(path)
            if path_string not in sys.path:
                sys.path.insert(0, path_string)

        args = self._load_task_args()
        self._install_direct_robot_shim()
        task_module = importlib.import_module(f"envs.{self.task_name}")
        task_class = getattr(task_module, self.task_name)

        class DirectDriveTask(task_class):
            def setup_scene(task_self, **kwargs: Any) -> None:
                # Base_Task._init_task_env_ invokes setup_scene without the
                # task arguments, so enforce the bridge's actual physics dt
                # and raster/RT shader choice before its setup-time
                # stabilization steps as well.
                kwargs["timestep"] = 1.0 / self.physics_fps
                kwargs["camera_shader_dir"] = self.renderer_shader_dir
                kwargs["max_num_materials"] = self.renderer_max_num_materials
                kwargs["max_num_textures"] = self.renderer_max_num_textures
                sapien = importlib.import_module("sapien")
                sapien.render.set_viewer_shader_dir(self.viewer_shader_dir)
                super().setup_scene(**kwargs)

            def load_robot(task_self, **kwargs: Any) -> None:
                task_self.robot = DirectDriveRobot(
                    task_self.scene, need_topp=False, **kwargs
                )
                task_self.robot.init_joints()
                for link in task_self.robot.left_entity.get_links():
                    link.set_mass(1)
                for link in task_self.robot.right_entity.get_links():
                    link.set_mass(1)

            def together_open_gripper(
                task_self,
                save_freq: int | None = -1,
                left_pos: float = 1.0,
                right_pos: float = 1.0,
            ) -> None:
                # Base_Task's version plans a gripper trajectory.  Task
                # initialization only needs open drive targets; check_stable()
                # advances the scene immediately afterwards.
                del save_freq
                task_self.robot.set_gripper(left_pos, "left", gripper_eps=0)
                task_self.robot.set_gripper(right_pos, "right", gripper_eps=0)

        args.update(
            collect_data=False,
            data_type={"rgb": True, "qpos": True, "endpose": True},
            eval_mode=False,
            eval_video_log=False,
            eval_video_save_dir=None,
            need_plan=False,
            # Keep task stabilization headless, then create one Viewer below.
            # The bridge owns GUI cadence separately from setup-time physics.
            render_freq=0,
            save_data=False,
        )
        task = DirectDriveTask()
        task.setup_demo(now_ep_num=0, seed=self.seed, is_test=True, **args)
        if self.enable_gui:
            self._create_gui_viewer(task, args, self.viewer_resolution)
        self.task = task

    def _load_task_args(self) -> dict[str, Any]:
        """Load RoboTwin task/embodiment YAML without its evaluation frontend.

        RoboTwin's ``eval_policy_xpolicylab`` happens to expose a convenient
        loader, but importing that evaluation entry point couples this bridge
        to XPolicyLab terminology and unrelated client-side dependencies.  The
        direct bridge needs only this small, stable configuration expansion.
        """
        yaml = importlib.import_module("yaml")
        config_root = self.root / "env_cfg" / "task_config"
        task_config_path = config_root / f"{self.task_config}.yml"
        if not task_config_path.is_file():
            raise FileNotFoundError(
                f"RoboTwin task config not found: {task_config_path}"
            )
        with task_config_path.open("r", encoding="utf-8") as stream:
            args = dict(yaml.safe_load(stream) or {})

        embodiment_config_path = config_root / "_embodiment_config.yml"
        with embodiment_config_path.open("r", encoding="utf-8") as stream:
            embodiments = dict(yaml.safe_load(stream) or {})
        embodiment = list(args.get("embodiment") or ())
        if len(embodiment) == 1:
            left_name = right_name = str(embodiment[0])
            args["dual_arm_embodied"] = True
        elif len(embodiment) == 3:
            left_name, right_name = (str(embodiment[0]), str(embodiment[1]))
            args["embodiment_dis"] = float(embodiment[2])
            args["dual_arm_embodied"] = False
        else:
            raise ValueError(
                "RoboTwin task embodiment must contain one shared arm or two arms plus a spacing"
            )

        def embodiment_dir(name: str) -> Path:
            try:
                configured = Path(str(embodiments[name]["file_path"])).expanduser()
            except (KeyError, TypeError) as exc:
                raise KeyError(
                    f"RoboTwin task references unknown embodiment {name!r}"
                ) from exc
            return (
                configured
                if configured.is_absolute()
                else (self.root / configured).resolve()
            )

        left_robot_dir = embodiment_dir(left_name)
        right_robot_dir = embodiment_dir(right_name)
        for robot_dir in (left_robot_dir, right_robot_dir):
            if not (robot_dir / "config.yml").is_file():
                raise FileNotFoundError(
                    f"RoboTwin embodiment config not found: {robot_dir / 'config.yml'}"
                )
        with (left_robot_dir / "config.yml").open("r", encoding="utf-8") as stream:
            args["left_embodiment_config"] = dict(yaml.safe_load(stream) or {})
        with (right_robot_dir / "config.yml").open("r", encoding="utf-8") as stream:
            args["right_embodiment_config"] = dict(yaml.safe_load(stream) or {})
        self._apply_drive_overrides(args)
        args["left_robot_file"] = str(left_robot_dir)
        args["right_robot_file"] = str(right_robot_dir)

        camera = dict(args.get("camera") or {})
        camera_type = camera.get("head_camera_type")
        with (config_root / "_camera_config.yml").open("r", encoding="utf-8") as stream:
            camera_profiles = dict(yaml.safe_load(stream) or {})
        if camera_type not in camera_profiles:
            raise KeyError(
                f"RoboTwin task references unknown head camera profile {camera_type!r}"
            )
        args["head_camera_h"] = int(camera_profiles[camera_type]["h"])
        args["head_camera_w"] = int(camera_profiles[camera_type]["w"])
        data_type = dict(args.get("data_type") or {})
        data_type.update(rgb=True, qpos=True, endpose=True)
        args.update(
            data_type=data_type,
            policy_name="superinference",
            task_name=self.task_name,
            task_config=self.task_config,
            ckpt_setting="superinference",
        )
        return args

    def _apply_drive_overrides(self, args: dict[str, Any]) -> None:
        for key, value in (
            ("joint_stiffness", self.joint_stiffness),
            ("joint_damping", self.joint_damping),
            ("gripper_stiffness", self.gripper_stiffness),
            ("gripper_damping", self.gripper_damping),
        ):
            if value is not None:
                args[key] = float(value)

    @staticmethod
    def _install_direct_robot_shim() -> None:
        """Make ``Base_Task`` bind the direct robot before it imports ``envs.robot``."""
        if "envs._base_task" in sys.modules:
            raise RuntimeError(
                "RoboTwin task modules were imported before the direct-drive bridge; "
                "start RoboTwinSimRobot in its own fresh process"
            )
        robot_module = types.ModuleType("envs.robot")
        robot_module.Robot = DirectDriveRobot
        sys.modules["envs.robot"] = robot_module

    @staticmethod
    def _create_gui_viewer(
        task: Any, args: dict[str, Any], resolution: tuple[int, int]
    ) -> None:
        """Create the SAPIEN viewer only after RoboTwin's headless task setup."""
        viewer_class = getattr(importlib.import_module("sapien.utils.viewer"), "Viewer")
        viewer = viewer_class(task.renderer, resolutions=resolution)
        viewer.set_scene(task.scene)
        viewer.set_camera_xyz(
            x=float(args.get("camera_xyz_x", 0.4)),
            y=float(args.get("camera_xyz_y", 0.22)),
            z=float(args.get("camera_xyz_z", 1.5)),
        )
        viewer.set_camera_rpy(
            r=float(args.get("camera_rpy_r", 0.0)),
            p=float(args.get("camera_rpy_p", -0.8)),
            y=float(args.get("camera_rpy_y", 2.45)),
        )
        task.viewer = viewer

    def apply_joint14_target(
        self, target_joint14: Any, velocity_joint14: Any | None = None
    ) -> None:
        """Update SAPIEN PD drive targets once; this never waits for arrival."""
        target = validate_action(target_joint14, "joint14")
        velocity = (
            np.zeros(14, dtype=np.float64)
            if velocity_joint14 is None
            else validate_action(velocity_joint14, "joint14")
        )
        robot = self._robot()
        left_dim = len(robot.left_arm_joints)
        right_dim = len(robot.right_arm_joints)
        if left_dim != 6 or right_dim != 6:
            raise RuntimeError(
                "joint14 requires a 6-DoF dual-arm embodiment; got "
                f"left={left_dim}, right={right_dim}"
            )
        robot.set_arm_joints(target[:left_dim], velocity[:left_dim], "left")
        robot.set_gripper(
            float(target[left_dim]),
            "left",
            gripper_eps=0,
            gripper_velocity=float(velocity[left_dim]),
        )
        right_offset = left_dim + 1
        robot.set_arm_joints(
            target[right_offset : right_offset + right_dim],
            velocity[right_offset : right_offset + right_dim],
            "right",
        )
        robot.set_gripper(
            float(target[right_offset + right_dim]),
            "right",
            gripper_eps=0,
            gripper_velocity=float(velocity[right_offset + right_dim]),
        )

    def step(self) -> None:
        self._task().scene.step()

    def read_joint14_state(self) -> tuple[np.ndarray, np.ndarray]:
        """Read articulation qpos/qvel, including physical gripper positions."""
        robot = self._robot()
        left_qpos, left_qvel = self._arm_state(robot.left_entity, robot.left_arm_joints)
        right_qpos, right_qvel = self._arm_state(
            robot.right_entity, robot.right_arm_joints
        )
        left_gripper, left_gripper_vel = self._normalized_gripper_state(
            robot.left_entity,
            robot.left_gripper,
            robot.left_gripper_scale,
        )
        right_gripper, right_gripper_vel = self._normalized_gripper_state(
            robot.right_entity,
            robot.right_gripper,
            robot.right_gripper_scale,
        )
        qpos = np.concatenate(
            (left_qpos, [left_gripper], right_qpos, [right_gripper])
        ).astype(np.float64)
        qvel = np.concatenate(
            (left_qvel, [left_gripper_vel], right_qvel, [right_gripper_vel])
        ).astype(np.float64)
        return validate_action(qpos, "joint14"), validate_action(qvel, "joint14")

    def capture_rgb(self, *, render_gui: bool = False) -> dict[str, np.ndarray]:
        """Return ``head_camera``, ``left_camera``, and ``right_camera`` RGB frames."""
        task = self._task()
        task._update_render()
        if render_gui and self.enable_gui:
            viewer = getattr(task, "viewer", None)
            if viewer is not None:
                viewer.render()
        # RoboTwin's Camera.update_picture() always renders every static
        # camera, including front_camera. The bridge publishes and its ACT
        # adapter consumes only these three cameras. Rendering the unused
        # target costs substantial Vulkan memory on small GPUs.
        return {
            camera_name: self._read_camera_rgb(
                self._camera_by_name(task.cameras, camera_name), camera_name
            )
            for camera_name in ("head_camera", "left_camera", "right_camera")
        }

    @staticmethod
    def _camera_by_name(cameras: Any, camera_name: str) -> Any:
        if camera_name in ("left_camera", "right_camera"):
            camera = getattr(cameras, camera_name, None)
            if camera is not None:
                return camera

        static_names = getattr(cameras, "static_camera_name", ())
        static_cameras = getattr(cameras, "static_camera_list", ())
        for name, camera in zip(static_names, static_cameras):
            if name == camera_name:
                return camera
        raise RuntimeError(
            f"RoboTwin task did not provide required camera {camera_name!r}"
        )

    @staticmethod
    def _read_camera_rgb(camera: Any, camera_name: str) -> np.ndarray:
        camera.take_picture()
        rgba = np.asarray(camera.get_picture("Color"))
        if rgba.ndim != 3 or rgba.shape[-1] < 3:
            raise ValueError(
                f"RoboTwin camera {camera_name!r} returned invalid Color shape {rgba.shape}"
            )
        return (rgba[:, :, :3] * 255).clip(0, 255).astype(np.uint8)

    def render_gui_if_due(self, sim_step: int) -> bool:
        if not self.should_render_gui(sim_step):
            return False
        task = self._task()
        viewer = getattr(task, "viewer", None)
        if viewer is None:
            return False
        viewer.render()
        return True

    def should_render_gui(self, sim_step: int) -> bool:
        if not self.enable_gui:
            return False
        render_interval = max(
            1, round(self.physics_fps / max(self.gui_render_fps, 1.0))
        )
        if sim_step - self._last_gui_render_step < render_interval:
            return False
        self._last_gui_render_step = sim_step
        return True

    def close(self) -> None:
        task = self.task
        self.task = None
        if task is None:
            return
        viewer = getattr(task, "viewer", None)
        if viewer is not None:
            try:
                viewer.close()
            except Exception:
                pass
        try:
            task.close_env(clear_cache=True)
        except Exception:
            pass

    def _task(self) -> Any:
        if self.task is None:
            raise RuntimeError("RoboTwin runtime is not started")
        return self.task

    def _robot(self) -> Any:
        return self._task().robot

    @staticmethod
    def _arm_state(entity: Any, joints: list[Any]) -> tuple[np.ndarray, np.ndarray]:
        active_joints = entity.get_active_joints()
        qpos = np.asarray(entity.get_qpos(), dtype=np.float64)
        qvel = np.asarray(entity.get_qvel(), dtype=np.float64)
        indexes = [active_joints.index(joint) for joint in joints]
        return qpos[indexes], qvel[indexes]

    @staticmethod
    def _normalized_gripper_state(
        entity: Any,
        gripper_joints: list[tuple[Any, float, float]],
        gripper_scale: Any,
    ) -> tuple[float, float]:
        if not gripper_joints:
            return 0.0, 0.0
        joint, multiplier, offset = gripper_joints[0]
        if multiplier == 0:
            return 0.0, 0.0
        active_joints = entity.get_active_joints()
        index = active_joints.index(joint)
        qpos = float(np.asarray(entity.get_qpos(), dtype=np.float64)[index])
        qvel = float(np.asarray(entity.get_qvel(), dtype=np.float64)[index])
        physical = (qpos - float(offset)) / float(multiplier)
        physical_velocity = qvel / float(multiplier)
        low, high = (float(value) for value in gripper_scale)
        span = high - low
        if abs(span) < 1e-12:
            return 0.0, 0.0
        return float(
            np.clip((physical - low) / span, 0.0, 1.0)
        ), physical_velocity / span
