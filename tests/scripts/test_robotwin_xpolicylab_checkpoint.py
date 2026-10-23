"""Hardware-free checks for RoboTwin XPolicyLab checkpoint contracts."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DOWNLOAD_PATH = REPO_ROOT / "scripts" / "robotwin" / "download_xpolicylab_checkpoint.py"
PI05_DEPLOY_PATH = (
    REPO_ROOT
    / "third_party"
    / "RoboTwin"
    / "XPolicyLab"
    / "policy"
    / "Pi_05"
    / "deploy.yml"
)


def _load_downloader() -> Any:
    spec = importlib.util.spec_from_file_location(
        "robotwin_download_xpolicylab_test", DOWNLOAD_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_pi05_deploy(path: Path = PI05_DEPLOY_PATH) -> dict[str, Any]:
    if not path.is_file():
        pytest.skip(
            reason=(
                "RoboTwin XPolicyLab deploy metadata requires the optional "
                "submodule checkout"
            )
        )
    deploy = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(deploy, dict)
    return deploy


def test_pi05_downloader_matches_official_lerobot_v30_asset() -> None:
    downloader = _load_downloader()
    spec = downloader.SPECS["pi05"]

    assert (
        spec.remote_archive.as_posix()
        == "cotrain_ckpt/Pi_05/RoboTwin-lerobot_v30-aloha_agilex-joint-0.tar.gz"
    )
    assert spec.run_dir_name == "RoboTwin-lerobot_v30-aloha_agilex-joint-0"


def test_pi05_deploy_repo_id_matches_downloader() -> None:
    downloader = _load_downloader()
    deploy = _load_pi05_deploy()
    spec = downloader.SPECS["pi05"]

    assert deploy.get("ckpt_name") is None or deploy["ckpt_name"] == spec.ckpt_name
    assert deploy["repo_id"] == "RoboTwin_lerobot_v30"


def test_missing_pi05_deploy_is_an_explicit_optional_skip(tmp_path: Path) -> None:
    with pytest.raises(pytest.skip.Exception, match="optional submodule checkout"):
        _load_pi05_deploy(tmp_path / "deploy.yml")
