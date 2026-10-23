"""Repository ownership, compatibility, and documentation governance contracts.

Author: SuperInference Contributors
"""

from __future__ import annotations

import ast
import importlib
import re
from pathlib import Path
from typing import Iterable

import pytest


pytestmark = pytest.mark.architecture
ROOT = Path(__file__).resolve().parents[2]

ROOT_PYTHON_ALLOWLIST = {"device_starter.py", "windows_device_start.py"}
STALE_PATHS = {
    "rollout_episode/": "consumer/inference/common/episode_controller.py",
    "scripts.pinocchio_kinematics_solver": "scripts.analysis.pinocchio_kinematics_solver",
    "scripts/evaluation_camera/": "scripts/analysis/evaluation_camera/",
    "scripts/robot_latency.py": "scripts/replay/robot_record_replay.py",
    "scripts/test_policy_gripper_control": "scripts/legacy/test_policy_gripper_control",
    "scripts/test_max_contact_wrench.py": "scripts/legacy/test_max_contact_wrench1.py",
    "scripts/waypoint_trajectory_planner.py": "scripts/benchmark/test_trajectory_planner.py",
    "consumer/async_infer_data_saver_NEDF3.py": "consumer/io/async_infer_data_saver_NEDF3.py",
    "consumer/openpi_policy_infer": "consumer/inference/openpi_policy_infer",
    "scripts/policy_launch_plan.py": "tests/config/policy_launch_plan.py",
    "scripts/test_policy_launch_plan.py": "tests/config/test_policy_launch_plan.py",
    "configs/policy_infer/devices/": "configs/devices/ or configs/devices/infer/",
}
TEXT_SUFFIXES = {".md", ".py", ".sh", ".yaml", ".yml"}
LEGACY_POLICY_CONFIG_NAME_RE = re.compile(
    r"(?<!policy_infer/)\b(?:openpi|fastwam|lingbotva|robotwin)/config_infer_"
)
CRITICAL_PATHS = (
    "useful_tools_commands.md",
    "openpi_inference_commands.md",
    "fastwam_inference_commands.md",
    "fastwam_memory_inference_commands.md",
    "fasterwam_inference_commands.md",
    "robotwin_inference_commands.md",
    "useful_prompts.md",
    "configs/policy_infer/openpi/config_infer_fisheye_umi3_dual_openpi.yaml",
    "configs/policy_infer/openpi/config_infer_fisheye_umi3_dual_openpi_rtc.yaml",
    "configs/policy_infer/robotwin/config_infer_robotwin_replay_joint14.yaml",
    "consumer/inference/common/episode_controller.py",
    "consumer/inference/common/rollout_controller.py",
    "consumer/io/async_infer_data_saver_NEDF3.py",
    "consumer/inference/openpi_policy_infer.py",
    "devices/cameras/openpi.py",
    "devices/robots/rizon/robot.py",
    "tests/config/policy_launch_plan.py",
    "tests/config/test_policy_launch_plan.py",
    "utils/episode_anomaly_algorithms.py",
    "utils/yam_head_tracking.py",
)
MACHINE_PYTHON_PATH_RE = re.compile(
    r"/(?:home/[^/<\s\"'`]+|root)/[^\s\"'`]*?/bin/python(?:\d+(?:\.\d+)?)?"
)
ABSOLUTE_CONFIG_PYTHON_RE = re.compile(
    r"(?m)^\s*python_path:\s*[\"']?/(?:home|root)/"
)
NAMED_CONDA_RUNTIME_RE = re.compile(
    r"\bconda\s+(?:activate|run\s+-n)\s+([A-Za-z0-9][A-Za-z0-9._-]*)"
)


def _method_names(path: Path, class_name: str) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return {
                child.name
                for child in node.body
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
    raise AssertionError(f"{class_name} not found in {path.relative_to(ROOT)}")


def _governed_text_files() -> Iterable[Path]:
    roots = (
        *sorted(ROOT.glob("*.md")),
        ROOT / "docs",
        ROOT / "tests",
        ROOT / "configs",
        ROOT / "scripts",
        ROOT / ".claude" / "skills",
    )
    for root in roots:
        if root.is_file():
            yield root
            continue
        yield from (
            path
            for path in root.rglob("*")
            if path.is_file()
            and path.suffix in TEXT_SUFFIXES
            and path.resolve() != Path(__file__).resolve()
        )


def test_root_python_entrypoints_are_explicit() -> None:
    assert {path.name for path in ROOT.glob("*.py")} == ROOT_PYTHON_ALLOWLIST


def test_machine_specific_python_paths_stay_in_local_runtime_registry() -> None:
    governed: list[Path] = []
    for directory in (
        ROOT / "consumer",
        ROOT / "devices",
        ROOT / "manager",
        ROOT / "policy",
        ROOT / "scripts",
        ROOT / "utils",
        ROOT / "incubator" / "control_panel",
    ):
        governed.extend(
            path
            for path in directory.rglob("*")
            if path.is_file()
            and path.suffix in {".py", ".sh"}
            and "tests" not in path.parts
        )
    governed.extend((ROOT / ".claude" / "skills").glob("*/SKILL.md"))
    governed.extend(ROOT.glob("*.md"))

    violations: list[str] = []
    for path in governed:
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if MACHINE_PYTHON_PATH_RE.search(line):
                violations.append(f"{path.relative_to(ROOT)}:{line_number}: {line.strip()}")

    for path in (ROOT / "configs").rglob("*.yaml"):
        if ABSOLUTE_CONFIG_PYTHON_RE.search(path.read_text(encoding="utf-8")):
            violations.append(
                f"{path.relative_to(ROOT)} contains an absolute python_path"
            )

    assert violations == []


def test_runtime_profile_registry_contract_is_tracked_but_local_data_is_ignored() -> None:
    assert (ROOT / "utils/runtime_profiles.py").is_file()
    assert (ROOT / "scripts/diagnostics/runtime_profiles.py").is_file()
    assert (ROOT / ".claude/skills/runtime-profile-management/SKILL.md").is_file()
    assert "/.superinference/local/" in (ROOT / ".gitignore").read_text(
        encoding="utf-8"
    )


def test_configs_do_not_use_fixed_device_initialization_delays() -> None:
    violations = [
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "configs").rglob("*.yaml")
        if re.search(
            r"(?m)^\s*device_init_delay\s*:",
            path.read_text(encoding="utf-8"),
        )
    ]

    assert violations == []


def test_agents_requires_runtime_profile_governance_for_inference_work() -> None:
    instructions = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    required_contracts = (
        ".claude/skills/runtime-profile-management/SKILL.md",
        "scripts/diagnostics/runtime_profiles.py",
        "`command --json`",
        "`doctor`",
        "`run --role <role>`",
        ".superinference/local/",
    )
    for contract in required_contracts:
        assert contract in instructions


def test_split_inference_launcher_uses_runtime_roles_instead_of_named_conda_envs() -> None:
    launcher = (ROOT / "scripts/lingbotva/run_infer_split.sh").read_text(
        encoding="utf-8"
    )
    assert "runtime_profiles.py" in launcher
    assert "--role device" in launcher
    assert "--role policy" in launcher
    assert "DEVICE_ENV=" not in launcher
    assert "POLICY_ENV=" not in launcher
    assert "conda activate" not in launcher


def test_robotwin_policy_server_configs_do_not_guess_environment_names() -> None:
    for name in (
        "policy_infer/robotwin/config_infer_robotwin_pi05_joint14.yaml",
        "policy_infer/robotwin/config_infer_robotwin_fastwam_joint14.yaml",
    ):
        text = (ROOT / "configs" / name).read_text(encoding="utf-8")
        assert "runtime_env:" in text
        assert "not_configured" in text
    assert "ROBOTWIN_PI05_UV_ENV,uv" not in (
        ROOT / "configs/policy_infer/robotwin/config_infer_robotwin_pi05_joint14.yaml"
    ).read_text(encoding="utf-8")
    assert "ROBOTWIN_FASTWAM_CONDA_ENV,fastwam" not in (
        ROOT / "configs/policy_infer/robotwin/config_infer_robotwin_fastwam_joint14.yaml"
    ).read_text(encoding="utf-8")


def test_robotwin_top_level_configs_are_grouped_by_subsystem() -> None:
    expected = {
        "config_infer_robotwin_replay_joint14.yaml",
        "config_infer_robotwin_act_joint14.yaml",
        "config_infer_robotwin_eef16.yaml",
        "config_infer_robotwin_pi05_joint14.yaml",
        "config_infer_robotwin_fastwam_joint14.yaml",
    }

    assert {
        path.name
        for path in (ROOT / "configs/policy_infer/robotwin").glob("*.yaml")
    } == expected
    assert list((ROOT / "configs").glob("config_infer_robotwin_*.yaml")) == []


def test_active_runtime_instructions_do_not_name_conda_environments() -> None:
    governed: list[Path] = []
    for directory in (ROOT / "scripts", ROOT / "incubator", ROOT / "configs"):
        governed.extend(
            path
            for path in directory.rglob("*")
            if path.is_file()
            and path.suffix in {".py", ".sh", ".md", ".yaml", ".yml"}
            and "legacy" not in path.parts
            and "tests" not in path.parts
        )
    governed.extend((ROOT / ".claude" / "skills").glob("*/SKILL.md"))
    governed.extend(ROOT.glob("*.md"))

    violations = [
        f"{path.relative_to(ROOT)} names conda environment {match.group(1)!r}"
        for path in governed
        for match in NAMED_CONDA_RUNTIME_RE.finditer(path.read_text(encoding="utf-8"))
    ]
    assert violations == []


def test_moved_paths_do_not_reappear_in_governed_text() -> None:
    violations: list[str] = []
    for path in _governed_text_files():
        text = path.read_text(encoding="utf-8")
        for stale, replacement in STALE_PATHS.items():
            if stale in text:
                violations.append(
                    f"{path.relative_to(ROOT)} references {stale}; use {replacement}"
                )
        if LEGACY_POLICY_CONFIG_NAME_RE.search(text):
            violations.append(
                f"{path.relative_to(ROOT)} uses a policy config name outside policy_infer/"
            )
    assert violations == []


def test_optional_dependency_skips_have_precise_reasons() -> None:
    violations: list[str] = []
    for path in (ROOT / "tests").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            function = node.func
            call_name = (
                function.attr
                if isinstance(function, ast.Attribute)
                and isinstance(function.value, ast.Name)
                and function.value.id == "pytest"
                else None
            )
            if call_name not in {"importorskip", "skip", "skipif"}:
                continue
            reason = next(
                (keyword.value for keyword in node.keywords if keyword.arg == "reason"),
                None,
            )
            if (
                not isinstance(reason, ast.Constant)
                or not isinstance(reason.value, str)
                or len(reason.value.split()) < 5
            ):
                violations.append(
                    f"{path.relative_to(ROOT)}:{node.lineno} needs a precise skip reason"
                )
    assert violations == []


def test_shared_module_implementations_are_unique() -> None:
    forbidden_copies = (
        "incubator/control_panel/services/lktech_gripper_server.py",
        "incubator/control_panel/services/lktech_websocket.py",
        "incubator/control_panel/yam_disconnect_safety.py",
    )
    assert [path for path in forbidden_copies if (ROOT / path).exists()] == []

    canonical_yam = _method_names(
        ROOT / "utils" / "yam_head_tracking.py",
        "YamHeadTrackingStrategy",
    )
    assert canonical_yam
    for facade in (
        ROOT / "incubator" / "control_panel" / "controllers" / "yam_head_tracking.py",
        ROOT / "scripts" / "yam_head_hacking" / "yam_head_tracking.py",
    ):
        tree = ast.parse(facade.read_text(encoding="utf-8"), filename=str(facade))
        assert not any(
            isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            for node in tree.body
        ), facade.relative_to(ROOT)

    shared_anomaly_methods = _method_names(
        ROOT / "utils" / "episode_anomaly_algorithms.py",
        "EpisodeAnomalyAlgorithms",
    )
    panel_methods = _method_names(
        ROOT / "incubator" / "control_panel" / "episode_anomaly_detector.py",
        "EpisodeAnomalyDetector",
    )
    assert shared_anomaly_methods.isdisjoint(panel_methods)

    canonical_yam_module = importlib.import_module("utils.yam_head_tracking")
    for facade_name in (
        "incubator.control_panel.controllers.yam_head_tracking",
        "scripts.yam_head_hacking.yam_head_tracking",
    ):
        facade_module = importlib.import_module(facade_name)
        assert (
            facade_module.YamHeadTrackingStrategy
            is canonical_yam_module.YamHeadTrackingStrategy
        )


def test_legacy_consumer_facades_export_identical_objects() -> None:
    exports = (
        ("consumer.base", "consumer.core.base", "BaseConsumer"),
        (
            "consumer.policy_connector",
            "consumer.inference.common.policy_connector",
            "PolicyConnector",
        ),
        (
            "consumer.openpi_policy_infer",
            "consumer.inference.openpi_policy_infer",
            "_OpenPiPolicyInferMixin",
        ),
    )
    for facade_name, implementation_name, attribute in exports:
        facade = importlib.import_module(facade_name)
        implementation = importlib.import_module(implementation_name)
        assert getattr(facade, attribute) is getattr(implementation, attribute)


@pytest.mark.hardware
def test_legacy_device_facade_exports_identical_webcam_object() -> None:
    pytest.importorskip(
        "cv2",
        reason="device facade identity requires the optional OpenCV hardware profile",
    )
    facade = importlib.import_module("devices.web_camera_openpi")
    implementation = importlib.import_module("devices.cameras.openpi")
    assert facade.WebCameraOpenPi is implementation.WebCameraOpenPi


def test_legacy_robot_facade_exports_identical_base_object() -> None:
    facade = importlib.import_module("devices.robot")
    implementation = importlib.import_module("devices.robots.base")
    assert facade.BaseRobot is implementation.BaseRobot


@pytest.mark.docs
def test_documented_critical_paths_exist() -> None:
    assert [path for path in CRITICAL_PATHS if not (ROOT / path).is_file()] == []


@pytest.mark.docs
def test_local_markdown_links_resolve() -> None:
    markdown_files = (*sorted(ROOT.glob("*.md")), *sorted((ROOT / "docs").glob("*.md")))
    link_pattern = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
    broken: list[str] = []
    for markdown in markdown_files:
        for target in link_pattern.findall(markdown.read_text(encoding="utf-8")):
            if target.startswith(("http://", "https://", "#", "mailto:")):
                continue
            relative_target = target.split("#", maxsplit=1)[0]
            if relative_target and not (markdown.parent / relative_target).resolve().exists():
                broken.append(f"{markdown.relative_to(ROOT)} -> {target}")
    assert broken == []
