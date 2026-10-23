from __future__ import annotations

import os
from pathlib import Path
import threading

from incubator.control_panel import server
from incubator.control_panel.recording_paths import (
    configured_recording_root,
    iter_recording_episode_roots,
    latest_sibling_recording_root,
)


def _episode(root: Path, name: str = "episode_0001") -> Path:
    episode = root / name
    episode.mkdir(parents=True)
    return episode


def test_configured_recording_root_preserves_the_yaml_path() -> None:
    config = {
        "policy": {
            "infer_rollout": {
                "output_dir": "data/rollout_data/task",
            }
        }
    }
    repo_root = Path("/repo")
    configured = configured_recording_root(config, repo_root=repo_root)
    assert configured == (repo_root / "data/rollout_data/task").resolve()
    assert config["policy"]["infer_rollout"]["output_dir"] == (
        "data/rollout_data/task"
    )


def test_episode_roots_keep_flat_history_and_add_date_children(tmp_path: Path) -> None:
    root = tmp_path / "rollouts"
    _episode(root)
    _episode(root / "20260715", "episode_0002")
    _episode(root / "not-a-date", "episode_0003")

    assert list(iter_recording_episode_roots(root)) == [
        root.resolve(),
        (root / "20260715").resolve(),
    ]


def test_latest_sibling_root_is_used_only_as_a_cold_replay_fallback(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "rollout_data"
    configured = parent / "configured-but-missing"
    older = parent / "older"
    latest = parent / "latest"
    _episode(older)
    _episode(latest)
    os.utime(older, ns=(1_000_000_000, 1_000_000_000))
    os.utime(latest, ns=(2_000_000_000, 2_000_000_000))

    assert latest_sibling_recording_root(configured) == latest.resolve()


def test_latest_sibling_root_prefers_newest_same_dataset_across_catalogs(
    tmp_path: Path,
) -> None:
    configured = tmp_path / "current" / "task"
    legacy = tmp_path / "legacy"
    current_copy = configured
    legacy_copy = legacy / "task"
    _episode(current_copy)
    _episode(legacy_copy)
    os.utime(current_copy, ns=(1_000_000_000, 1_000_000_000))
    os.utime(legacy_copy, ns=(2_000_000_000, 2_000_000_000))

    assert latest_sibling_recording_root(
        configured,
        sibling_parents=(legacy,),
    ) == legacy_copy.resolve()


def test_controller_uses_explicit_replay_catalog_without_selected_config(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    configured = tmp_path / "rollout_data" / "configured-but-missing"
    latest = tmp_path / "legacy_rollout_data" / "latest"
    _episode(latest)
    (config_dir / "config.yaml").write_text(
        "policy:\n"
        "  infer_rollout:\n"
        "    enabled: true\n"
        f'    output_dir: "{configured.as_posix()}"\n',
        encoding="utf-8",
    )

    controller = object.__new__(server.DeviceController)
    controller._lock = threading.RLock()
    controller.config_dir = config_dir
    controller.selected_config = None
    controller._effective_config = None
    controller._fallback_recording_config = None
    controller._fallback_recording_config_loaded = False

    monkeypatch.setenv(
        server.RECORDING_REPLAY_ROOTS_ENV,
        str(latest.parent),
    )
    assert controller.recording_source_root() == latest.resolve()
    assert controller.selected_config_name() is None


def test_selected_config_replay_root_does_not_depend_on_prior_inference_path(
    tmp_path: Path,
) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    configured = tmp_path / "current" / "selected"
    prior_inference = tmp_path / "prior" / "other"
    _episode(configured)
    _episode(prior_inference)

    controller = object.__new__(server.DeviceController)
    controller._lock = threading.RLock()
    controller.config_dir = config_dir
    controller.selected_config = "selected"
    controller._effective_config = {
        "policy": {"infer_rollout": {"output_dir": str(configured)}}
    }
    controller._rollout_enabled = False
    controller._rollout_output_root = prior_inference
    controller._rollout_output_root_user_set = False

    assert controller.recording_source_root() == configured.resolve()

    controller._rollout_output_root_user_set = True
    assert controller.recording_source_root() == prior_inference.resolve()
