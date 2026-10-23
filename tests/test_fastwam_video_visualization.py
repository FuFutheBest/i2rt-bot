"""Hardware-free contracts for the optional FastWAM video protocol."""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip(
    "torch",
    reason="FastWAM video contracts require the optional PyTorch model runtime",
)

import torch

import consumer.inference.fastwam_server as fastwam_server_module
from consumer.fastwam_server import FastWAMRuntime, _encode_video_prediction_frames
from policy.fastwam_policy import FastWAMPolicy
from visualizers.fastwam_video_prediction import FastWAMVideoPredictionVisualizer


def test_video_prediction_frames_are_sampled_and_jpeg_encoded() -> None:
    video = np.zeros((7, 8, 12, 3), dtype=np.uint8)
    encoded = _encode_video_prediction_frames(video, max_frames=3, jpeg_quality=80)

    assert len(encoded) == 3
    assert all(frame[:2] == b"\xff\xd8" for frame in encoded)


def test_video_prediction_encoder_accepts_chw_time_tensor() -> None:
    video = torch.zeros((3, 5, 8, 12), dtype=torch.float32)
    video[0] = 1.0
    encoded = _encode_video_prediction_frames(video, max_frames=2)

    assert len(encoded) == 2
    assert all(frame[:2] == b"\xff\xd8" for frame in encoded)


def test_mp4_writer_uses_browser_compatible_h264(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        captured.update(command=command, input=kwargs["input"])
        Path(command[-1]).write_bytes(b"h264-mp4")
        return SimpleNamespace(returncode=0, stderr=b"")

    monkeypatch.setattr(fastwam_server_module.shutil, "which", lambda _: "/usr/bin/ffmpeg")
    monkeypatch.setattr(fastwam_server_module.subprocess, "run", fake_run)
    output = tmp_path / "prediction.mp4"
    frames = [np.zeros((4, 6, 3), dtype=np.uint8) for _ in range(2)]

    fastwam_server_module._write_rgb_mp4(output, frames, 7.5)

    command = captured["command"]
    assert isinstance(command, list)
    assert command[command.index("-c:v") + 1] == "libx264"
    assert command[command.index("-pix_fmt", command.index("-c:v")) + 1] == "yuv420p"
    assert command[command.index("-movflags") + 1] == "+faststart"
    assert len(captured["input"]) == 2 * 4 * 6 * 3
    assert output.read_bytes() == b"h264-mp4"


def test_runtime_optional_joint_path_returns_transport_metadata() -> None:
    action_condition = torch.ones((2, 2))

    class FakeModel:
        enable_video_prediction = True

        def infer_joint(self, **kwargs: object) -> dict[str, object]:
            assert kwargs["num_video_frames"] == 13
            assert kwargs["action"] is action_condition
            return {
                "video": [np.zeros((4, 8, 3), dtype=np.uint8) for _ in range(5)],
                "action": torch.zeros((2, 2)),
            }

    runtime = FastWAMRuntime.__new__(FastWAMRuntime)
    runtime.model = FakeModel()
    runtime.params = {}
    runtime.video_visualization_config = {
        "enabled": True,
        "max_prediction_frames": 3,
    }
    runtime.contract = {
        "num_frames": 49,
        "video_num_frames": 13,
        "training_enable_video_prediction": True,
        "video_action_conditioned": True,
        "concat_multi_camera": "horizontal",
        "image_meta": [{}, {}],
    }
    runtime._last_preprocessed_camera_frames = [
        np.zeros((4, 4, 3), dtype=np.uint8),
        np.zeros((4, 4, 3), dtype=np.uint8),
    ]

    result = runtime._maybe_predict_video_for_visualization(
        requested=True,
        model_prompt="prompt",
        context=None,
        context_mask=None,
        input_image=torch.zeros((3, 4, 8)),
        proprio=torch.zeros(2),
        action=action_condition,
        action_horizon=2,
        num_inference_steps=2,
        seed=1,
    )

    assert result is not None
    assert result["frame_count"] == 5
    assert len(result["frames"]) == 3
    assert result["view_count"] == 2
    assert len(result["input_frames"]) == 2
    assert all(frame[:2] == b"\xff\xd8" for frame in result["input_frames"])


def test_runtime_saves_prediction_when_configured(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeModel:
        enable_video_prediction = True

        def infer_joint(self, **_: object) -> dict[str, object]:
            return {
                "video": [
                    np.full((4, 8, 3), index * 20, dtype=np.uint8)
                    for index in range(3)
                ]
            }

    written: dict[str, object] = {}

    def fake_write(path: Path, frames: list[object], fps: float) -> None:
        written.update(path=path, frame_count=len(frames), fps=fps)
        path.write_bytes(b"test-mp4")

    monkeypatch.setattr(fastwam_server_module, "_write_rgb_mp4", fake_write)
    runtime = FastWAMRuntime.__new__(FastWAMRuntime)
    runtime.model = FakeModel()
    runtime.params = {}
    runtime.video_visualization_config = {
        "enabled": True,
        "save_predictions": True,
        "save_dir": str(tmp_path),
        "save_fps": 7.5,
        "save_input_frames": True,
        "max_prediction_frames": 13,
    }
    runtime.contract = {
        "video_num_frames": 13,
        "training_enable_video_prediction": True,
        "video_action_conditioned": False,
        "concat_multi_camera": "horizontal",
        "image_meta": [{}, {}],
    }
    runtime._last_preprocessed_camera_frames = [
        np.zeros((4, 4, 3), dtype=np.uint8),
        np.ones((4, 4, 3), dtype=np.uint8),
    ]

    result = runtime._maybe_predict_video_for_visualization(
        requested=True,
        model_prompt="prompt",
        context=None,
        context_mask=None,
        input_image=torch.zeros((3, 4, 8)),
        proprio=torch.zeros(2),
        action_horizon=48,
        num_inference_steps=10,
        seed=7,
        observation_timestamp_ns=123,
    )

    assert result is not None
    artifact_dir = Path(result["saved_artifact_dir"])
    assert written["path"] == artifact_dir / "prediction_combined.mp4"
    assert written["frame_count"] == 3
    assert written["fps"] == pytest.approx(7.5)
    assert (artifact_dir / "model_input_camera_0.png").is_file()
    assert (artifact_dir / "model_input_camera_1.png").is_file()
    metadata = json.loads((artifact_dir / "metadata.json").read_text())
    assert metadata["observation_timestamp_ns"] == 123
    assert metadata["video_action_conditioned"] is False
    assert metadata["frame_count"] == 3


def test_runtime_does_not_run_joint_path_when_visualization_is_disabled() -> None:
    class FailingModel:
        def infer_joint(self, **_: object) -> dict[str, object]:
            raise AssertionError("infer_joint must stay disabled")

    runtime = FastWAMRuntime.__new__(FastWAMRuntime)
    runtime.model = FailingModel()
    runtime.params = {}
    runtime.video_visualization_config = {"enabled": False}
    runtime.contract = {"num_frames": 5, "image_meta": [{}, {}]}

    assert runtime._maybe_predict_video_for_visualization(
        requested=True,
        model_prompt="prompt",
        context=None,
        context_mask=None,
        input_image=torch.zeros((3, 4, 8)),
        proprio=torch.zeros(2),
        action=None,
        action_horizon=2,
        num_inference_steps=2,
        seed=1,
    ) is None


def test_runtime_rejects_action_only_checkpoint_video_prediction() -> None:
    class FailingModel:
        enable_video_prediction = False

        def infer_joint(self, **_: object) -> dict[str, object]:
            raise AssertionError("infer_joint must stay disabled")

    runtime = FastWAMRuntime.__new__(FastWAMRuntime)
    runtime.model = FailingModel()
    runtime.params = {}
    runtime.video_visualization_config = {"enabled": True}
    runtime.contract = {
        "training_enable_video_prediction": False,
        "video_num_frames": 13,
        "image_meta": [{}, {}],
    }

    assert runtime._maybe_predict_video_for_visualization(
        requested=True,
        model_prompt="prompt",
        context=None,
        context_mask=None,
        input_image=torch.zeros((3, 4, 8)),
        proprio=torch.zeros(2),
        action=None,
        action_horizon=2,
        num_inference_steps=2,
        seed=1,
    ) is None


def test_runtime_disables_video_prediction_after_consecutive_failures() -> None:
    class FailingModel:
        enable_video_prediction = True

        def infer_joint(self, **_: object) -> dict[str, object]:
            raise RuntimeError("test failure")

    runtime = FastWAMRuntime.__new__(FastWAMRuntime)
    runtime.model = FailingModel()
    runtime.params = {}
    runtime.video_visualization_config = {"enabled": True, "max_failures": 2}
    runtime.contract = {
        "training_enable_video_prediction": True,
        "video_action_conditioned": True,
        "video_num_frames": 13,
        "image_meta": [{}, {}],
    }
    kwargs = {
        "requested": True,
        "model_prompt": "prompt",
        "context": None,
        "context_mask": None,
        "input_image": torch.zeros((3, 4, 8)),
        "proprio": torch.zeros(2),
        "action": None,
        "action_horizon": 2,
        "num_inference_steps": 2,
        "seed": 1,
    }

    assert runtime._maybe_predict_video_for_visualization(**kwargs) is None
    assert runtime._maybe_predict_video_for_visualization(**kwargs) is None
    assert runtime._video_prediction_disabled is True


def test_policy_does_not_request_video_without_running_dashboard() -> None:
    policy = FastWAMPolicy.__new__(FastWAMPolicy)
    policy.video_visualization_enabled = True
    policy.video_prediction_save_enabled = False
    policy._video_visualization_suppressed = False
    policy._video_visualizer = None
    policy._video_visualization_requests_disabled = False
    policy._video_visualization_request_count = 0
    policy.video_visualization_interval = 1

    assert policy._should_request_video_prediction() is False
    assert policy._video_visualization_request_count == 0


def test_policy_requests_video_for_saving_without_dashboard() -> None:
    policy = FastWAMPolicy.__new__(FastWAMPolicy)
    policy.video_visualization_enabled = True
    policy.video_prediction_save_enabled = True
    policy._video_visualization_suppressed = False
    policy._video_visualizer = None
    policy._video_visualization_requests_disabled = False
    policy._video_visualization_request_count = 0
    policy.video_visualization_interval = 1

    assert policy._should_request_video_prediction() is True
    assert policy._video_visualization_request_count == 1


def test_policy_splits_horizontal_prediction_into_two_views() -> None:
    policy = FastWAMPolicy.__new__(FastWAMPolicy)
    source = np.zeros((2, 6, 12, 3), dtype=np.uint8)
    source[:, :, :6] = [255, 0, 0]
    source[:, :, 6:] = [0, 255, 0]
    encoded = _encode_video_prediction_frames(source, max_frames=0)

    views = policy._split_video_prediction(
        {"frames": encoded, "concat_mode": "horizontal", "view_count": 2}
    )

    assert views is not None
    assert len(views) == 2
    assert len(views[0]) == len(views[1]) == 2
    assert views[0][0].shape == (6, 6, 3)
    assert float(views[0][0][..., 0].mean()) > 200
    assert float(views[1][0][..., 1].mean()) > 200


def test_policy_rejects_non_dual_prediction_metadata() -> None:
    policy = FastWAMPolicy.__new__(FastWAMPolicy)
    encoded = _encode_video_prediction_frames(
        np.zeros((1, 6, 12, 3), dtype=np.uint8), max_frames=1
    )

    assert policy._split_video_prediction(
        {"frames": encoded, "concat_mode": "horizontal", "view_count": 1}
    ) is None


def test_policy_stops_requesting_video_after_empty_responses() -> None:
    policy = FastWAMPolicy.__new__(FastWAMPolicy)
    policy._video_visualizer = object()
    policy.video_visualization_config = {"max_failures": 2}
    policy._video_visualization_failure_count = 0
    policy._video_visualization_requests_disabled = False

    policy._update_video_visualizer({}, {"data": {}}, requested=True)
    policy._update_video_visualizer({}, {"data": {}}, requested=True)

    assert policy._video_visualization_failure_count == 2
    assert policy._video_visualization_requests_disabled is True


def test_policy_labels_prediction_with_source_observation_index() -> None:
    class RecordingVisualizer:
        def __init__(self) -> None:
            self.observation_index: int | None = None

        def update(
            self,
            input_frames: object,
            prediction_frames: object,
            *,
            observation_index: int,
        ) -> None:
            del input_frames, prediction_frames
            self.observation_index = observation_index

    policy = FastWAMPolicy.__new__(FastWAMPolicy)
    visualizer = RecordingVisualizer()
    policy._video_visualizer = visualizer
    policy.video_visualization_config = {"max_failures": 3}
    policy._video_visualization_failure_count = 0
    policy._video_visualization_requests_disabled = False
    policy._video_visualization_request_count = 7
    source = np.zeros((2, 6, 12, 3), dtype=np.uint8)
    encoded_prediction = _encode_video_prediction_frames(source, max_frames=0)
    encoded_inputs = _encode_video_prediction_frames(
        np.zeros((2, 6, 6, 3), dtype=np.uint8), max_frames=0
    )

    policy._update_video_visualizer(
        {},
        {
            "data": {
                "video_prediction": {
                    "frames": encoded_prediction,
                    "input_frames": encoded_inputs,
                    "concat_mode": "horizontal",
                    "view_count": 2,
                }
            }
        },
        requested=True,
    )

    assert visualizer.observation_index == 7


def test_video_dashboard_serves_six_panel_page() -> None:
    visualizer = FastWAMVideoPredictionVisualizer(
        host="127.0.0.1",
        port=0,
        display_fps=7.5,
        max_prediction_frames=13,
    )
    try:
        url = visualizer.start()
        if url is None:
            pytest.skip(
                reason="Localhost socket binding is unavailable in this test environment"
            )
        assert url is not None
        with urllib.request.urlopen(url, timeout=2.0) as response:
            page = response.read().decode("utf-8")
        assert response.status == 200
        assert page.count("/stream/") == 4
        assert page.count("/snapshot/") >= 2
        assert "grid-template-rows:repeat(2" in page
        assert "grid-template-columns:repeat(3" in page
        assert "FastWAM Prediction Monitor" in page
        assert "Prediction chunk" in page
        assert "Display FPS" in page
        assert "Max frames" in page
        panel_titles = (
            "Left hand | camera",
            "Left hand | input observation",
            "Left hand | prediction",
            "Right hand | camera",
            "Right hand | input observation",
            "Right hand | prediction",
        )
        assert [page.index(title) for title in panel_titles] == sorted(
            page.index(title) for title in panel_titles
        )

        input_frames = [np.zeros((8, 12, 3), dtype=np.uint8) for _ in range(2)]
        prediction_frames = [
            [np.zeros((8, 12, 3), dtype=np.uint8) for _ in range(3)]
            for _ in range(2)
        ]
        visualizer.update(input_frames, prediction_frames, observation_index=42)
        with urllib.request.urlopen(f"{url}status", timeout=2.0) as response:
            status = json.loads(response.read())
        assert status["observation_index"] == 42
        assert status["prediction_chunk_index"] == 1
        assert status["prediction_frame_count"] == 3
        assert status["display_fps"] == 7.5
        assert status["max_prediction_frames"] == 13
        with urllib.request.urlopen(f"{url}snapshot/left_input", timeout=2.0) as response:
            assert response.read()[:2] == b"\xff\xd8"
        with urllib.request.urlopen(f"{url}healthz", timeout=2.0) as response:
            assert response.read() == b'{"ok":true}'
    finally:
        visualizer.close()
