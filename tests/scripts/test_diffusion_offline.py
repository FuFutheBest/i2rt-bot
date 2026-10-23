"""Hardware-free contracts for the dual-arm Diffusion offline evaluator."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.analysis import evaluate_diffusion_offline as evaluator


def _dual_policy_config(policy_class: str = "DiffusionPolicy") -> dict[str, object]:
    return {
        "class": policy_class,
        "params": {
            "num_inference_steps": 7,
            "obs_horizon": 2,
            "downsample": 3,
            "obs_time_interval": 0.04,
        },
        "chunk_length": 3,
        "obs": {
            "devices": [
                "OpenCVCameraDevice_0",
                "OpenCVCameraDevice_2",
                "RizonRobot_1",
            ]
        },
        "controls": [{"shape": [16], "dtype": "float64"}],
    }


def test_explicit_policy_allowlist_and_dual_control_contract() -> None:
    policy_config = _dual_policy_config()

    assert evaluator._policy_class(policy_config) == "DiffusionPolicy"
    evaluator._validate_dual_arm_contract(policy_config)

    with pytest.raises(ValueError, match="supports only"):
        evaluator._policy_class({"class": "TinyMLPPolicy"})
    with pytest.raises(ValueError, match="16-dimensional"):
        evaluator._validate_dual_arm_contract(
            {
                **policy_config,
                "controls": [{"shape": [8], "dtype": "float64"}],
            }
        )


@pytest.mark.parametrize(
    "config_name, policy_class",
    [
        ("policy_infer/diffusion/config_infer_rizon_eef_dual_abs_mode_umi", "DiffusionPolicy"),
        ("policy_infer/diffusion/config_infer_iphone_umi_dual", "DiffusionPolicyIPhoneUMI"),
    ],
)
def test_native_dual_diffusion_configs_compose_policy_contract(
    config_name: str,
    policy_class: str,
) -> None:
    _root_config, policy_config = evaluator._config_for_offline_evaluation(config_name)

    assert evaluator._policy_class(policy_config) == policy_class
    evaluator._validate_dual_arm_contract(policy_config)
    cameras, robot = evaluator._selected_devices(policy_config)
    assert len(cameras) == 2
    assert robot.startswith("RizonRobot_")


def test_policy_builder_forces_visualization_off_and_keeps_iphone_options(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    class FakePolicy:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(evaluator, "_load_policy_class", lambda _name: FakePolicy)
    policy_config = _dual_policy_config("DiffusionPolicyIPhoneUMI")
    params = policy_config["params"]
    assert isinstance(params, dict)
    params.update(
        {
            "left_robot_base_to_world_transform": "left.json",
            "right_robot_base_to_world_transform": "right.json",
            "use_relative_action": True,
            "use_inter_gripper": "false",
        }
    )

    evaluator._build_policy(
        policy_config,
        checkpoint=tmp_path / "checkpoint",
        device="cpu",
        camera_devices=["OpenCVCameraDevice_0", "OpenCVCameraDevice_2"],
        robot_device="RizonRobot_1",
    )

    assert captured["ckpt_dir"] == str(tmp_path / "checkpoint")
    assert captured["device"] == "cpu"
    assert captured["enable_visualization"] is False
    assert captured["camera_devices"] == ["OpenCVCameraDevice_0", "OpenCVCameraDevice_2"]
    assert captured["robot_device"] == "RizonRobot_1"
    assert captured["left_robot_base_to_world_transform"] == "left.json"
    assert captured["use_relative_action"] is True
    assert captured["use_inter_gripper"] is False


def test_main_writes_canonical_dual_arm_action_from_rebuilt_recorder_history(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    policy_config = _dual_policy_config()
    dataset_calls: list[dict[str, object]] = []
    policy_calls: list[dict[str, object]] = []

    class FakeDataset:
        inferences = [object()]

        def __init__(self, _path: Path) -> None:
            pass

        def policy_observation_history_at(
            self,
            inference_index: int,
            *,
            history_frames: int,
            history_stride: int,
            observation_interval_s: float,
            camera_devices: list[str],
            robot_device: str,
        ) -> tuple[dict[str, list[np.ndarray]], list[int], int]:
            dataset_calls.append(
                {
                    "inference_index": inference_index,
                    "history_frames": history_frames,
                    "history_stride": history_stride,
                    "observation_interval_s": observation_interval_s,
                    "camera_devices": camera_devices,
                    "robot_device": robot_device,
                }
            )
            return {
                camera_devices[0]: [
                    np.zeros((4, 5, 3), dtype=np.uint8),
                    np.zeros((4, 5, 3), dtype=np.uint8),
                ],
                camera_devices[1]: [
                    np.ones((4, 5, 3), dtype=np.uint8),
                    np.ones((4, 5, 3), dtype=np.uint8),
                ],
                robot_device: [
                    np.zeros(44, dtype=np.float64),
                    np.zeros(44, dtype=np.float64),
                ],
            }, [83, 123], 123

    class FakePolicy:
        action_step_interval_ns = 20_000_000
        obs_horizon = 2
        downsample = 3
        obs_time_interval = 0.04

        def predict(self, observation: object, **kwargs: object) -> dict[str, np.ndarray]:
            policy_calls.append({"observation": observation, **kwargs})
            action_config = kwargs["action_configs"]
            assert isinstance(action_config, list)
            robot_device = action_config[0]["device_name"]
            return {robot_device: np.full((3, 16), 1.25, dtype=np.float64)}

    monkeypatch.setattr(
        evaluator,
        "_config_for_offline_evaluation",
        lambda _name: ({}, policy_config),
    )
    monkeypatch.setattr(evaluator, "ObservationDataset", FakeDataset)
    monkeypatch.setattr(
        evaluator,
        "_build_policy",
        lambda _policy_config, **_kwargs: FakePolicy(),
    )
    monkeypatch.setattr(evaluator, "_set_seed", lambda _seed: None)

    output_dir = tmp_path / "result"
    evaluator.main(
        [
            "--checkpoint",
            str(tmp_path / "checkpoint"),
            "--observation-dir",
            str(tmp_path / "observation"),
            "--prompt",
            "stack blocks",
            "--seed",
            "19",
            "--device",
            "cpu",
            "--output-dir",
            str(output_dir),
        ]
    )

    with np.load(output_dir / "offline_result.npz", allow_pickle=False) as artifact:
        assert artifact["action"].shape == (3, 16)
        assert np.all(artifact["action"] == 1.25)
        assert artifact["history_timestamps_ns"].tolist() == [83, 123]
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["action_layout"] == evaluator.DUAL_ARM_ACTION_LAYOUT
    assert summary["action_step_interval_ns"] == 20_000_000
    assert summary["artifacts"] == {
        "npz": "offline_result.npz",
        "canonical_action_key": "action",
        "action_contract": evaluator.DUAL_ARM_TCP_ACTION_CONTRACT,
        "action_layout": evaluator.DUAL_ARM_ACTION_LAYOUT,
        "action_frame_id": evaluator.DUAL_ARM_TCP_ACTION_FRAME,
        "action_units": evaluator.DUAL_ARM_TCP_ACTION_UNITS,
        "action_step_interval_ns": 20_000_000,
    }
    assert summary["prompt"] == "stack blocks"
    assert dataset_calls == [
        {
            "inference_index": 0,
            "history_frames": 2,
            "history_stride": 3,
            "observation_interval_s": 0.04,
            "camera_devices": ["OpenCVCameraDevice_0", "OpenCVCameraDevice_2"],
            "robot_device": "RizonRobot_1",
        }
    ]
    assert policy_calls[0]["chunk_length"] == 3
    observation = policy_calls[0]["observation"]
    assert isinstance(observation, dict)
    assert all(len(values) == 2 for values in observation.values())
    assert summary["history_frames"] == 2
    assert summary["history_stride"] == 3
    assert summary["observation_interval_ns"] == 40_000_000


def test_predict_rejects_single_arm_action() -> None:
    class SingleArmPolicy:
        def predict(self, _observation: object, **kwargs: object) -> dict[str, np.ndarray]:
            action_config = kwargs["action_configs"]
            assert isinstance(action_config, list)
            return {action_config[0]["device_name"]: np.zeros((2, 8))}

    with pytest.raises(ValueError, match=r"\[T,16\]"):
        evaluator._predict_like_live(
            SingleArmPolicy(),
            {"RizonRobot_1": np.zeros(44)},
            robot_device="RizonRobot_1",
            chunk_length=2,
        )
