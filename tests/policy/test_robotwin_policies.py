"""Focused policy tests that do not need a RoboTwin or torch installation."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import policy.robotwin_act_policy as robotwin_act_module
from policy.robotwin_act_policy import RoboTwinACTPolicy
from policy.robotwin_replay_policy import RoboTwinReplayPolicy
from policy.robotwin_xpolicylab_policy import RoboTwinXPolicyLabPolicy


def _joint14_control() -> list[dict[str, object]]:
    return [{"device_name": "RoboTwinSimRobot_0", "action_dim": 14}]


def test_replay_policy_holds_measured_joint_state_without_a_file() -> None:
    policy = RoboTwinReplayPolicy(robot_device="RoboTwinSimRobot_0")
    policy.load()
    state = np.arange(14, dtype=np.float64)

    result = policy.predict(
        {"RoboTwinSimRobot_0": state}, _joint14_control(), chunk_length=3
    )

    np.testing.assert_allclose(
        result["RoboTwinSimRobot_0"], np.repeat(state[None, :], 3, axis=0)
    )


def test_replay_policy_replays_overlapping_chunks(tmp_path: Path) -> None:
    actions = np.arange(56, dtype=np.float64).reshape(4, 14)
    path = tmp_path / "actions.npy"
    np.save(path, actions)
    policy = RoboTwinReplayPolicy(actions_path=str(path), advance_per_prediction=1)
    policy.load()

    first = policy.predict({}, _joint14_control(), chunk_length=3)["RoboTwinSimRobot_0"]
    second = policy.predict({}, _joint14_control(), chunk_length=3)[
        "RoboTwinSimRobot_0"
    ]

    np.testing.assert_allclose(first, actions[:3])
    np.testing.assert_allclose(second, actions[1:])


def test_act_policy_checks_artifacts_before_importing_torch(tmp_path: Path) -> None:
    policy = RoboTwinACTPolicy(
        ckpt_dir=str(tmp_path / "missing"), act_source=str(tmp_path / "source")
    )

    with pytest.raises(FileNotFoundError, match="policy_last.ckpt"):
        policy.load()


def test_act_build_suppresses_unneeded_backbone_download(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pretrained_flags: list[bool] = []

    def original_resnet18(*_args: object, **kwargs: object) -> str:
        pretrained_flags.append(bool(kwargs["pretrained"]))
        return "backbone"

    torchvision = SimpleNamespace(models=SimpleNamespace(resnet18=original_resnet18))
    original_import = robotwin_act_module.importlib.import_module

    def import_module(name: str, *args: object, **kwargs: object) -> object:
        if name == "torchvision":
            return torchvision
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(robotwin_act_module.importlib, "import_module", import_module)

    class _Act:
        @staticmethod
        def ACT(_args: object, _config: object) -> str:
            return torchvision.models.resnet18(pretrained=True)

    result = RoboTwinACTPolicy._build_model_without_backbone_download(
        _Act, {"backbone": "resnet18"}
    )

    assert result == "backbone"
    assert pretrained_flags == [False]
    assert torchvision.models.resnet18 is original_resnet18


def test_act_build_restores_backbone_factory_after_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def original_resnet18(*_args: object, **_kwargs: object) -> str:
        return "backbone"

    torchvision = SimpleNamespace(models=SimpleNamespace(resnet18=original_resnet18))
    original_import = robotwin_act_module.importlib.import_module

    def import_module(name: str, *args: object, **kwargs: object) -> object:
        if name == "torchvision":
            return torchvision
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(robotwin_act_module.importlib, "import_module", import_module)

    class _FailingAct:
        @staticmethod
        def ACT(_args: object, _config: object) -> object:
            assert torchvision.models.resnet18 is not original_resnet18
            raise RuntimeError("model construction failed")

    with pytest.raises(RuntimeError, match="model construction failed"):
        RoboTwinACTPolicy._build_model_without_backbone_download(
            _FailingAct, {"backbone": "resnet18"}
        )

    assert torchvision.models.resnet18 is original_resnet18


def test_act_load_restores_on_cpu_then_moves_fp16_model_to_cuda(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ckpt_dir = tmp_path / "checkpoint"
    ckpt_dir.mkdir()
    (ckpt_dir / "policy_last.ckpt").touch()
    (ckpt_dir / "dataset_stats.pkl").touch()
    act_source = tmp_path / "source"
    (act_source / "ACT").mkdir(parents=True)

    class _Cuda:
        @staticmethod
        def is_available() -> bool:
            return True

    class _Torch:
        float16 = "float16"
        float32 = "float32"
        cuda = _Cuda()

        @staticmethod
        def device(value: str) -> str:
            return f"device:{value}"

        @staticmethod
        def load(*_args: object, **_kwargs: object) -> object:
            return object()

    class _Policy:
        def __init__(self) -> None:
            self.calls: list[object] = []

        def half(self) -> None:
            self.calls.append("half")

        def to(self, device: str) -> None:
            self.calls.append(("to", device))

    model = SimpleNamespace(policy=_Policy(), device=None)
    captured_args: list[dict[str, object]] = []
    load_kwargs: list[dict[str, object]] = []

    def load(*_args: object, **kwargs: object) -> object:
        load_kwargs.append(dict(kwargs))
        return object()

    _Torch.load = staticmethod(load)

    def build_without_backbone(_act_module: object, args: dict[str, object]) -> object:
        captured_args.append(dict(args))
        _Torch.load("policy_last.ckpt")
        return model

    original_import = robotwin_act_module.importlib.import_module

    def import_module(name: str, *args: object, **kwargs: object) -> object:
        if name == "torch":
            return _Torch
        if name == "ACT.act_policy":
            return SimpleNamespace()
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(robotwin_act_module.importlib, "import_module", import_module)
    monkeypatch.setattr(
        RoboTwinACTPolicy,
        "_build_model_without_backbone_download",
        staticmethod(build_without_backbone),
    )

    policy = RoboTwinACTPolicy(
        ckpt_dir=str(ckpt_dir),
        act_source=str(act_source),
        inference_dtype="fp16",
    )
    policy.load()

    assert captured_args[0]["device"] == "cpu"
    assert load_kwargs == [{"map_location": "cpu"}]
    assert model.policy.calls == ["half", ("to", "cuda:0")]
    assert model.device == "device:cuda:0"
    assert policy._model_dtype == "float16"


def test_xpolicylab_policy_formats_robotwin_observation_and_joint_chunk(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    xpl_root = tmp_path / "XPolicyLab"
    (xpl_root / "policy" / "Pi_05").mkdir(parents=True)

    class _Client:
        instances: list["_Client"] = []

        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs
            self.calls: list[tuple[str | None, object]] = []
            _Client.instances.append(self)

        def call(self, func_name: str | None = None, obs: object = None) -> object:
            self.calls.append((func_name, obs))
            if func_name == "get_action":
                return [
                    {
                        "left_arm_joint_state": np.arange(6, dtype=np.float32),
                        "left_ee_joint_state": np.asarray([0.25], dtype=np.float32),
                        "right_arm_joint_state": np.arange(10, 16, dtype=np.float32),
                        "right_ee_joint_state": np.asarray([0.75], dtype=np.float32),
                    },
                    {
                        "left_arm_joint_state": np.arange(20, 26, dtype=np.float32),
                        "left_ee_joint_state": np.asarray([0.5], dtype=np.float32),
                        "right_arm_joint_state": np.arange(30, 36, dtype=np.float32),
                        "right_ee_joint_state": np.asarray([1.0], dtype=np.float32),
                    },
                ]
            return None

        def close(self) -> None:
            self.calls.append(("close", None))

    client_server = types.ModuleType("client_server")
    client_server.__path__ = []
    ws = types.ModuleType("client_server.ws")
    ws.WsModelClient = _Client
    monkeypatch.setitem(sys.modules, "client_server", client_server)
    monkeypatch.setitem(sys.modules, "client_server.ws", ws)

    policy = RoboTwinXPolicyLabPolicy(
        policy_name="Pi_05",
        xpolicylab_root=str(xpl_root),
        host="127.0.0.1",
        port=6000,
        task_name="beat_block_hammer",
        task_config="demo_clean",
        instruction="beat block with the hammer",
        frequency=15.0,
    )
    policy.load()
    policy.set_prediction_context(123456)

    observation = {
        "RoboTwinSimRobot_0": np.arange(14, dtype=np.float64),
        "RoboTwinCamera_0": np.zeros((240, 320, 3), dtype=np.uint8),
        "RoboTwinCamera_1": np.ones((240, 320, 3), dtype=np.uint8),
        "RoboTwinCamera_2": np.full((240, 320, 3), 2, dtype=np.uint8),
    }

    result = policy.predict(observation, _joint14_control(), chunk_length=4)

    client = _Client.instances[0]
    assert client.kwargs["url"] == "ws://127.0.0.1:6000"
    assert [call[0] for call in client.calls[:4]] == [
        "prepare_case",
        "reset",
        "update_obs",
        "get_action",
    ]
    xpl_obs = client.calls[2][1]
    assert isinstance(xpl_obs, dict)
    assert xpl_obs["instruction"] == "beat block with the hammer"
    assert xpl_obs["timestamp_ns"] == 123456
    assert xpl_obs["additional_info"] == {"frequency": 15}
    assert set(xpl_obs["vision"]) == {"cam_head", "cam_left_wrist", "cam_right_wrist"}
    np.testing.assert_allclose(
        xpl_obs["state"]["left_arm_joint_state"], np.arange(6, dtype=np.float32)
    )
    np.testing.assert_allclose(
        xpl_obs["state"]["right_arm_joint_state"], np.arange(7, 13, dtype=np.float32)
    )

    chunk = result["RoboTwinSimRobot_0"]
    assert chunk.shape == (4, 14)
    np.testing.assert_allclose(
        chunk[0],
        np.asarray([0, 1, 2, 3, 4, 5, 0.25, 10, 11, 12, 13, 14, 15, 0.75]),
    )
    np.testing.assert_allclose(chunk[2], chunk[1])
    assert policy.last_prediction_timing_ns["policy_predict_ns"] >= 0


def test_xpolicylab_policy_rejects_ee_actions_for_joint_control(tmp_path: Path) -> None:
    xpl_root = tmp_path / "XPolicyLab"
    (xpl_root / "policy" / "FastWAM").mkdir(parents=True)
    policy = RoboTwinXPolicyLabPolicy(policy_name="FastWAM", xpolicylab_root=str(xpl_root))

    with pytest.raises(ValueError, match="returned eef16"):
        policy._flatten_action_chunk(
            [
                {
                    "action_type": "ee",
                    "left_ee_pose": np.zeros(7),
                    "left_ee_joint_state": np.zeros(1),
                    "right_ee_pose": np.zeros(7),
                    "right_ee_joint_state": np.zeros(1),
                }
            ],
            fallback_joint14=np.zeros(14),
            expected_schema="joint14",
            chunk_length=1,
        )


def test_xpolicylab_policy_accepts_packed_action_wrappers(tmp_path: Path) -> None:
    xpl_root = tmp_path / "XPolicyLab"
    (xpl_root / "policy" / "Pi_05").mkdir(parents=True)
    policy = RoboTwinXPolicyLabPolicy(policy_name="Pi_05", xpolicylab_root=str(xpl_root))

    actions = np.arange(28, dtype=np.float32).reshape(2, 14)
    chunk = policy._flatten_action_chunk(
        {"actions": actions},
        fallback_joint14=np.zeros(14),
        expected_schema="joint14",
        chunk_length=3,
    )

    np.testing.assert_allclose(chunk[:2], actions)
    np.testing.assert_allclose(chunk[2], actions[1])
