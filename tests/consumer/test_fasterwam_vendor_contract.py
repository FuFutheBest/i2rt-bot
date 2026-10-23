"""Static FasterWAM vendor contract checks without model-runtime imports."""

from __future__ import annotations

import ast
import shutil
import subprocess
from pathlib import Path

import pytest


EXPECTED_NMX_VLA_COMMIT = "b81a39f8722ecf1a370b83b0d1026e5f6e3346a5"
NMX_VLA_PATH = "third_party/nmx_vla"


def _pinned_nmx_vla_commit(repo_root: Path, git: str) -> str:
    """Read the gitlink from the superproject without using the checkout."""
    return subprocess.check_output(
        [git, "rev-parse", f"HEAD:{NMX_VLA_PATH}"],
        cwd=repo_root,
        text=True,
    ).strip()


def _runtime_method(source: str, method_name: str) -> ast.FunctionDef:
    module = ast.parse(source)
    runtime = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "FastWAMRuntime"
    )
    return next(
        node
        for node in runtime.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )


def test_fasterwam_vendor_pin_matches_one_pass_api_contract(repo_root: Path) -> None:
    """Keep the gitlink aligned without requiring a submodule checkout."""
    git = shutil.which("git")
    if git is None:
        pytest.skip(
            reason="Git executable is required to inspect the superproject gitlink"
        )
    assert _pinned_nmx_vla_commit(repo_root, git) == EXPECTED_NMX_VLA_COMMIT


def test_fasterwam_adapter_implements_one_pass_api_contract(repo_root: Path) -> None:
    """Keep the local adapter contract covered in Git-free test jobs."""

    source = (
        repo_root / "consumer" / "inference" / "fastwam_server.py"
    ).read_text(encoding="utf-8")
    method = _runtime_method(source, "_infer_model_action")
    string_literals = {
        node.value
        for node in ast.walk(method)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }

    assert {
        "one_pass_future_cache",
        "infer_action_one_pass_future_cache",
        "num_video_frames",
        "c4_video_denoise_steps",
        "c4_video_num_inference_steps",
        "c4_reverse_video_timestep",
    } <= string_literals
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "getattr"
        and len(node.args) >= 2
        and isinstance(node.args[0], ast.Attribute)
        and node.args[0].attr == "model"
        and isinstance(node.args[1], ast.Constant)
        and node.args[1].value == "infer_action_one_pass_future_cache"
        for node in ast.walk(method)
    )


def test_fasterwam_checkpoint_loading_fails_closed_on_incompatible_mot(
    repo_root: Path,
) -> None:
    """The pinned loader must not continue with a partly initialized action head."""
    source = (
        repo_root / "consumer" / "inference" / "fastwam_server.py"
    ).read_text(encoding="utf-8")
    method = _runtime_method(source, "_load_model")

    load_calls = [
        node
        for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "load_checkpoint"
    ]

    assert len(load_calls) == 1
    assert any(
        keyword.arg == "fail_on_incompatible_mot"
        and isinstance(keyword.value, ast.Constant)
        and keyword.value.value is True
        for keyword in load_calls[0].keywords
    )
