"""Hardware-free tests for RoboTwin ACT source and checkpoint preparation."""

from __future__ import annotations

import importlib.util
import io
import tarfile
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
BOOTSTRAP_PATH = REPO_ROOT / "scripts" / "robotwin" / "bootstrap_act_source.py"
DOWNLOAD_PATH = REPO_ROOT / "scripts" / "robotwin" / "download_act_checkpoint.py"


def _load_module(path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bootstrap_skips_git_archive_directory_entries(tmp_path: Path) -> None:
    bootstrap = _load_module(BOOTSTRAP_PATH, "robotwin_bootstrap_act_test")
    payload = b"from __future__ import annotations\n"
    archive_buffer = io.BytesIO()
    with tarfile.open(fileobj=archive_buffer, mode="w") as archive:
        directory = tarfile.TarInfo("policy/ACT")
        directory.type = tarfile.DIRTYPE
        archive.addfile(directory)
        source = tarfile.TarInfo("policy/ACT/act_policy.py")
        source.size = len(payload)
        archive.addfile(source, io.BytesIO(payload))

    assert bootstrap._safe_extract_act(archive_buffer.getvalue(), tmp_path) == 1
    assert (tmp_path / "ACT" / "act_policy.py").read_bytes() == payload


def test_official_checkpoint_layout_matches_the_joint14_preset(tmp_path: Path) -> None:
    downloader = _load_module(DOWNLOAD_PATH, "robotwin_download_act_test")

    assert downloader.OFFICIAL_REVISION == "790104e1bbe9f70574b8c77d3676f229e6f2daeb"

    destination, remote_paths = downloader.checkpoint_layout(
        checkpoint_root=tmp_path,
        task="beat_block_hammer",
        task_config="demo_clean",
        expert_data_num=50,
    )

    assert destination == tmp_path / "beat_block_hammer" / "demo_clean-50"
    assert remote_paths == {
        "policy_last.ckpt": downloader.PurePosixPath(
            "act_ckpt/act-beat_block_hammer/demo_clean-50/policy_last.ckpt"
        ),
        "dataset_stats.pkl": downloader.PurePosixPath(
            "act_ckpt/act-beat_block_hammer/demo_clean-50/dataset_stats.pkl"
        ),
    }
