import ast
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

from incubator.control_panel.release import doctor as release_doctor
from incubator.control_panel.release.build_release import (
    FLEXIV_RDK_VERSIONS,
    NEDF_SDK_PACKAGES,
    PROFILE_EXCLUDES,
    ReleaseBuildError,
    _absolute_entry,
    _deduplicate_tree,
    _install_nedf_sdk_wheels,
    _require_python_version,
    _same_python_profile,
    _sanitize_profile_site,
    _write_wrappers,
)
from incubator.control_panel.release.runtime_bootstrap import (
    ReleaseRuntimeError,
    build_profile_environment,
)


def _fake_release(tmp_path: Path) -> Path:
    root = tmp_path / "release"
    (root / "app").mkdir(parents=True)
    (root / "app/third_party/i2rt").mkdir(parents=True)
    for version in ("python310/bin/python3.10", "python312/bin/python3.12"):
        path = root / "runtime" / version
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("python", encoding="utf-8")
    for profile in ("device", "policy", "recording", "converter"):
        (root / "runtime/profiles" / profile / "site-packages").mkdir(parents=True)
    for version in ("1.7.0", "1.8.0", "1.8.1", "1.9.0", "1.9.1", "1.9.2"):
        (root / f"runtime/flexivrdk/rdk-{version}/site-packages").mkdir(
            parents=True
        )
    full_decode_ffmpeg = root / "runtime/converter/full-decode-ffmpeg/bin/ffmpeg"
    full_decode_ffmpeg.parent.mkdir(parents=True)
    full_decode_ffmpeg.write_text("ffmpeg", encoding="utf-8")
    (root / "bin").mkdir()
    return root


def test_device_profile_selects_only_one_explicit_rdk_runtime(tmp_path) -> None:
    root = _fake_release(tmp_path)

    environment, python = build_profile_environment(
        root,
        "device",
        rdk_version="1.9.0",
        base_environment={"HOME": str(tmp_path)},
    )

    paths = environment["PYTHONPATH"].split(os.pathsep)
    assert python == root / "runtime/python312/bin/python3.12"
    assert str(root / "app/third_party/i2rt") in paths
    assert str(root / "runtime/flexivrdk/rdk-1.9.0/site-packages") in paths
    assert not any("rdk-1.8.1" in path for path in paths)
    assert not any("rdk-1.9.2" in path for path in paths)
    assert environment["SUPERINFERENCE_FLEXIV_RDK_VERSION"] == "1.9.0"
    assert environment["SUPERINFERENCE_FLEXIV_RDK_ROOT"] == str(
        root / "runtime/flexivrdk"
    )
    assert environment["SUPERINFERENCE_DEVICE_PROFILE_NAME"] == "device312"
    assert environment["SUPERINFERENCE_INFERENCE_PROFILE_NAME"] == "openpi312"
    assert environment["SUPERINFERENCE_CONVERTER_FULL_DECODE_FFMPEG"] == str(
        root / "runtime/converter/full-decode-ffmpeg/bin/ffmpeg"
    )


@pytest.mark.parametrize("profile", ["control-panel", "device", "policy"])
def test_python312_profiles_prefer_one_shared_main_site(
    tmp_path, profile: str
) -> None:
    root = _fake_release(tmp_path)
    shared = root / "runtime/profiles/main/site-packages"
    shared.mkdir(parents=True)

    environment, python = build_profile_environment(
        root, profile, base_environment={"HOME": str(tmp_path)}
    )

    paths = environment["PYTHONPATH"].split(os.pathsep)
    assert python == root / "runtime/python312/bin/python3.12"
    assert str(shared) in paths
    assert str(root / "runtime/profiles/device/site-packages") not in paths
    assert str(root / "runtime/profiles/policy/site-packages") not in paths


@pytest.mark.parametrize("profile", ["recording", "converter"])
def test_nedf_profiles_prefer_one_shared_main_site(
    tmp_path, profile: str
) -> None:
    root = _fake_release(tmp_path)
    shared = root / "runtime/profiles/main/site-packages"
    shared.mkdir(parents=True)

    environment, python = build_profile_environment(
        root, profile, base_environment={"HOME": str(tmp_path)}
    )

    paths = environment["PYTHONPATH"].split(os.pathsep)
    assert python == root / "runtime/python312/bin/python3.12"
    assert str(shared) in paths
    assert str(root / "runtime/profiles/recording/site-packages") not in paths
    assert str(root / "runtime/profiles/converter/site-packages") not in paths


def test_release_builder_only_unifies_identical_profile_sites(tmp_path) -> None:
    first = {"purelib": str(tmp_path / "main/site-packages")}
    same = {"purelib": str(tmp_path / "main/../main/site-packages")}
    different = {"purelib": str(tmp_path / "policy/site-packages")}

    assert _same_python_profile(first, same)
    assert not _same_python_profile(first, different)


def test_control_panel_exposes_rdk_inventory_without_loading_one(tmp_path) -> None:
    root = _fake_release(tmp_path)

    environment, _python = build_profile_environment(
        root, "control-panel", base_environment={"HOME": str(tmp_path)}
    )

    paths = environment["PYTHONPATH"].split(os.pathsep)
    assert not any("runtime/flexivrdk" in path for path in paths)
    assert environment["SUPERINFERENCE_FLEXIV_RDK_VERSIONS"].split(",") == [
        "1.9.2",
        "1.9.1",
        "1.9.0",
        "1.8.1",
        "1.8.0",
        "1.7.0",
    ]


def test_standalone_bootstrap_adds_release_app_before_rdk_import(tmp_path) -> None:
    root = _fake_release(tmp_path)
    package = root / "app/incubator/control_panel"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "flexiv_rdk.py").write_text(
        "\n".join(
            [
                "SUPPORTED_RDK_VERSIONS = ('1.9.0',)",
                "RDK_AVAILABLE_ENV = 'SUPERINFERENCE_FLEXIV_RDK_VERSIONS'",
                "RDK_ROOT_ENV = 'SUPERINFERENCE_FLEXIV_RDK_ROOT'",
                "RDK_SITE_ENV = 'SUPERINFERENCE_FLEXIV_RDK_SITE'",
                "RDK_VERSION_ENV = 'SUPERINFERENCE_FLEXIV_RDK_VERSION'",
                "def normalize_rdk_version(value): return str(value)",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    bootstrap = (
        Path(__file__).resolve().parents[1] / "release/runtime_bootstrap.py"
    )
    code = "\n".join(
        [
            "import importlib.util",
            "from pathlib import Path",
            f"spec = importlib.util.spec_from_file_location('isolated_bootstrap', {str(bootstrap)!r})",
            "module = importlib.util.module_from_spec(spec)",
            "spec.loader.exec_module(module)",
            f"environment, _ = module.build_profile_environment(Path({str(root)!r}), 'device', rdk_version='1.9.0', base_environment={{'HOME': {str(tmp_path)!r}}})",
            "assert environment['SUPERINFERENCE_FLEXIV_RDK_VERSION'] == '1.9.0'",
        ]
    )

    subprocess.run(
        [sys.executable, "-I", "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )


def test_elements_version_no_longer_selects_an_rdk(tmp_path) -> None:
    root = _fake_release(tmp_path)

    with pytest.raises(ReleaseRuntimeError, match="Elements version does not determine"):
        build_profile_environment(
            root,
            "device",
            elements_version="3.11",
            base_environment={"HOME": str(tmp_path)},
        )


def test_release_writes_one_click_server_launcher(tmp_path) -> None:
    root = _fake_release(tmp_path)
    source_launcher = root / "app/incubator/control_panel/start_server.sh"
    source_launcher.parent.mkdir(parents=True)
    source_launcher.write_text("#!/usr/bin/env bash\n", encoding="utf-8")

    _write_wrappers(root)

    launcher = root / "start_server.sh"
    content = launcher.read_text(encoding="utf-8")
    assert launcher.stat().st_mode & 0o111
    assert "app/incubator/control_panel/start_server.sh" in content
    assert "exec bash" in content
    assert "runtime/python312/bin/python3.12" in (root / "bin/python-device").read_text(
        encoding="utf-8"
    )
    assert "runtime/python312/bin/python3.12" in (root / "bin/python-policy").read_text(
        encoding="utf-8"
    )
    assert "runtime/python312/bin/python3.12" in (
        root / "bin/python-recording"
    ).read_text(encoding="utf-8")
    assert "runtime/python312/bin/python3.12" in (
        root / "bin/python-converter"
    ).read_text(encoding="utf-8")
    doctor = (root / "bin/superinference-doctor").read_text(encoding="utf-8")
    assert "PYTHONDONTWRITEBYTECODE=1" in doctor
    assert "SUPERINFERENCE_READ_ONLY_DOCTOR=1" in doctor


def test_source_tree_exposes_one_click_server_launcher() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    module_launcher = (repo_root / "incubator/control_panel/start_server.sh").read_text(
        encoding="utf-8"
    )

    assert not (repo_root / "start_server.sh").exists()
    assert "bin/superinference-control-panel" in module_launcher
    assert "python3" in module_launcher
    assert "SUPERINFERENCE_CONTROL_PANEL_HOST" in module_launcher
    assert "SUPERINFERENCE_CONTROL_PANEL_PORT" in module_launcher


def test_sha256_pinned_converter_scripts_are_forced_to_lf() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    attributes = (repo_root / ".gitattributes").read_text(encoding="utf-8")

    for script in (
        "convert_accepted_episode.py",
        "merge_lerobot_shards.py",
        "nedf3_rollout_to_lerobot.py",
    ):
        assert f"/incubator/control_panel/tools/{script} text eol=lf" in attributes


def test_yam_runtime_dependencies_are_part_of_the_release_contract() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    release_root = repo_root / "incubator/control_panel/release"
    requirements_root = release_root / "requirements"
    build_constraints = (requirements_root / "build_constraints.txt").read_text(
        encoding="utf-8"
    )
    doctor = (release_root / "doctor.py").read_text(encoding="utf-8")
    policy_requirements = (requirements_root / "policy.txt").read_text(
        encoding="utf-8"
    )
    main_requirements = (requirements_root / "main_py312.txt").read_text(
        encoding="utf-8"
    )
    source_builder = (release_root / "create_source_py312_env.sh").read_text(
        encoding="utf-8"
    )

    assert "numpy==1.26.4" in main_requirements
    assert "opencv-python==4.11.0.86" in main_requirements
    assert "rerun-sdk==0.23.1" in main_requirements
    assert "crcmod-plus==2.3.1" in main_requirements
    assert "dill==0.3.8" in main_requirements
    assert "fsspec==2025.3.0" in main_requirements
    assert 'repo_root=$(cd "$script_dir/../../.." && pwd -P)' in source_builder
    assert 'sync --python "$python_bin"' in source_builder
    assert "third_party/lktech_motor_sdk" in source_builder
    assert '-e "$repo_root/third_party/lktech_motor_sdk"' in source_builder
    assert "--frozen --no-dev" in source_builder
    assert "--no-install-package mujoco" in source_builder
    assert "--no-deps" in source_builder
    assert "actual_conflicts -ne 3" in source_builder
    assert "undocumented dependency conflicts" in source_builder
    assert "third_party/i2rt" in source_builder
    assert '--find-links "$wheelhouse"' in source_builder
    recording_installer = (
        release_root / "install_source_recording_profile.sh"
    ).read_text(encoding="utf-8")
    assert "dm3_core-0.2.18-*.whl" in recording_installer
    assert "nmx_msg-2.2.0-*.whl" in recording_installer
    assert "from consumer.io.async_infer_data_saver_NEDF3 import" in recording_installer
    assert "import openpi" in source_builder
    assert "import i2rt" in source_builder

    assert "third_party/i2rt" in source_builder
    assert "python-can==4.5.0" in main_requirements
    assert "mujoco==3.9.0" in main_requirements
    assert "tyro==1.0.15" in main_requirements
    assert "websockets==16.0" in main_requirements
    assert "incubator/control_panel/release/requirements/build_constraints.txt" in source_builder
    assert "scikit-build-core==0.9.10" in build_constraints
    assert "import i2rt" in doctor
    assert "import can" in doctor
    assert "import i2rt.motor_drivers.dm_driver" in doctor
    assert "import mujoco" in doctor
    assert "find_spec('flexivrdk') is None" in doctor
    assert "'flexivrdk_unselected': True" in doctor
    assert "from devices.flexiv_adapter import FlexivAdapter" in doctor
    assert "'adapter_version':" in doctor
    assert "MVCAM_COMMON_RUNENV" in doctor
    assert "from devices import WebCameraOpenPi" in doctor
    assert "from incubator.control_panel.gripper_control import" in doctor
    assert "from third_party.lktech_motor_sdk.example.lktech_gripper import" in doctor
    assert "'gripper_transport': 'direct_serial_worker'" in doctor
    assert "'_has_position_calibration'" in doctor
    assert "hydra-core==1.3.4" in policy_requirements
    assert "loguru==0.7.3" in policy_requirements
    assert "matplotlib==3.10.9" in policy_requirements
    assert "scikit-image==0.25.2" in policy_requirements
    assert "from policy.openpi_umi_policy import OpenPiUmiPolicy" in doctor
    assert "NEDFFactory.get_reader" in doctor
    assert "NEDFReaderBase.iter_decoded_messages" in doctor
    assert "NEDFWriterBase.write_message" in doctor
    assert "metadata.version('nmx-nedf-api') == '0.3.1'" in doctor
    assert "metadata.version('nm-storage-api') == '0.3.2'" in doctor
    assert "metadata.version('nmx-msg') == '2.2.0'" in doctor
    assert "'conversion_scripts': conversion_scripts" in doctor


def test_webcam_module_has_no_top_level_hik_sdk_import() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    camera_source = (repo_root / "devices/cameras/webcam.py").read_text(encoding="utf-8")
    tree = ast.parse(camera_source)
    top_level_imports = [
        node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))
    ]

    assert not any(
        isinstance(node, ast.ImportFrom)
        and str(node.module or "").startswith("third_party.MvImport")
        for node in top_level_imports
    )
    assert "third_party.MvImport" not in camera_source


def test_policy_profile_uses_python312_and_release_openpi_source(tmp_path) -> None:
    root = _fake_release(tmp_path)
    openpi = root / "app/third_party/openpi/src"
    client = root / "app/third_party/openpi/packages/openpi-client/src"
    openpi.mkdir(parents=True)
    client.mkdir(parents=True)

    environment, python = build_profile_environment(
        root, "policy", base_environment={"HOME": str(tmp_path)}
    )

    paths = environment["PYTHONPATH"].split(os.pathsep)
    assert python == root / "runtime/python312/bin/python3.12"
    assert str(openpi) in paths
    assert str(client) in paths
    assert "SUPERINFERENCE_FLEXIV_RDK_VERSION" not in environment
    assert environment["SUPERINFERENCE_DEVICE_PROFILE_NAME"] == "device312"
    assert environment["SUPERINFERENCE_INFERENCE_PROFILE_NAME"] == "openpi312"
    assert environment["SUPERINFERENCE_FASTWAM_SERVER_PYTHON"] == str(
        root / "bin/python-policy"
    )


def test_release_fastwam_server_python_does_not_inherit_stale_path(tmp_path) -> None:
    root = _fake_release(tmp_path)

    environment, _python = build_profile_environment(
        root,
        "control-panel",
        base_environment={
            "HOME": str(tmp_path),
            "SUPERINFERENCE_INFERENCE_PYTHON": "/old/release/client",
            "SUPERINFERENCE_FASTWAM_SERVER_PYTHON": "/old/release/python",
        },
    )

    assert environment["SUPERINFERENCE_INFERENCE_PYTHON"] == str(
        root / "bin/python-policy"
    )
    assert environment["SUPERINFERENCE_FASTWAM_SERVER_PYTHON"] == str(
        root / "bin/python-policy"
    )


def test_release_fastwam_server_python_can_use_explicit_external_environment(
    tmp_path: Path,
) -> None:
    root = _fake_release(tmp_path)
    external_python = tmp_path / "nmx-env" / "bin" / "python"
    external_python.parent.mkdir(parents=True)
    external_python.write_text("python", encoding="utf-8")

    environment, _python = build_profile_environment(
        root,
        "control-panel",
        base_environment={
            "HOME": str(tmp_path),
            "SUPERINFERENCE_FASTWAM_SERVER_PYTHON": str(external_python),
            "SUPERINFERENCE_FASTWAM_SERVER_PYTHON_EXTERNAL": "1",
        },
    )

    assert environment["SUPERINFERENCE_FASTWAM_SERVER_PYTHON"] == str(
        external_python
    )


def test_release_fastwam_external_server_python_requires_absolute_path(
    tmp_path: Path,
) -> None:
    root = _fake_release(tmp_path)

    with pytest.raises(ReleaseRuntimeError, match="absolute path"):
        build_profile_environment(
            root,
            "control-panel",
            base_environment={
                "HOME": str(tmp_path),
                "SUPERINFERENCE_FASTWAM_SERVER_PYTHON": "./nmx-python",
                "SUPERINFERENCE_FASTWAM_SERVER_PYTHON_EXTERNAL": "true",
            },
        )


@pytest.mark.parametrize("profile", ["recording", "converter"])
def test_nedf_profiles_remain_on_python312(tmp_path, profile: str) -> None:
    root = _fake_release(tmp_path)

    _environment, python = build_profile_environment(
        root, profile, base_environment={"HOME": str(tmp_path)}
    )

    assert python == root / "runtime/python312/bin/python3.12"


def test_release_builder_enforces_main_and_nedf_python_versions() -> None:
    _require_python_version({"version": "3.12.13"}, "Device", "3.12")
    _require_python_version({"version": "3.12.13"}, "Recording", "3.12")

    with pytest.raises(ReleaseBuildError, match="Policy profile must use Python 3.12"):
        _require_python_version({"version": "3.11.14"}, "Policy", "3.12")


def test_release_bundles_every_reviewed_firmware_rdk() -> None:
    assert FLEXIV_RDK_VERSIONS == (
        "1.7.0",
        "1.8.0",
        "1.8.1",
        "1.9.0",
        "1.9.1",
        "1.9.2",
    )


def _write_fixture_wheel(
    wheelhouse: Path,
    package: str,
    version: str,
) -> Path:
    normalized = package.replace("-", "_")
    wheel = wheelhouse / f"{normalized}-{version}-cp312-cp312-manylinux_2_17_x86_64.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(
            f"{normalized}-{version}.dist-info/METADATA",
            f"Metadata-Version: 2.1\nName: {package}\nVersion: {version}\n",
        )
        archive.writestr(
            f"{normalized}-{version}.dist-info/WHEEL",
            "Wheel-Version: 1.0\n"
            "Generator: SuperInference test\n"
            "Root-Is-Purelib: false\n"
            "Tag: cp312-cp312-manylinux_2_17_x86_64\n",
        )
    return wheel


def test_release_bundles_exact_nedf3_sdk_wheels_with_hashes(tmp_path) -> None:
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    expected = {
        package: _write_fixture_wheel(wheelhouse, package, version)
        for package, version in NEDF_SDK_PACKAGES
    }
    _write_fixture_wheel(wheelhouse, "unrelated-package", "9.9.9")
    release = tmp_path / "release"

    records = _install_nedf_sdk_wheels(wheelhouse, release)

    assert [(item["package"], item["version"]) for item in records] == list(
        NEDF_SDK_PACKAGES
    )
    assert all(item["wheel"].startswith("vendor/wheels/nedf3/") for item in records)
    assert {
        path.name for path in (release / "vendor/wheels/nedf3").iterdir()
    } == {path.name for path in expected.values()}
    assert all(len(item["sha256"]) == 64 for item in records)


def test_release_rejects_incomplete_nedf3_sdk_wheelhouse(tmp_path) -> None:
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    _write_fixture_wheel(wheelhouse, "nmx-nedf-api", "0.3.1")

    with pytest.raises(ReleaseBuildError, match="Expected one dm3-core"):
        _install_nedf_sdk_wheels(wheelhouse, tmp_path / "release")


def test_release_profile_excludes_any_preinstalled_global_rdk() -> None:
    assert {
        "flexivrdk/",
        "flexivrdk.libs/",
        "flexivrdk-*.dist-info/",
    }.issubset(PROFILE_EXCLUDES)


def test_release_profile_removes_absolute_editable_paths(tmp_path) -> None:
    site = tmp_path / "site-packages"
    site.mkdir()
    stale = site / "_editable_impl_openpi.pth"
    stale.write_text("/home/old/source/openpi/src\n", encoding="utf-8")
    relative = site / "rerun_sdk.pth"
    relative.write_text("rerun_sdk\n", encoding="utf-8")

    _sanitize_profile_site(site)

    assert not stale.exists()
    assert relative.read_text(encoding="utf-8") == "rerun_sdk\n"


def test_release_deduplication_uses_hardlinks_for_identical_runtime_files(
    tmp_path,
) -> None:
    first = tmp_path / "device/library.bin"
    second = tmp_path / "recording/library.bin"
    first.parent.mkdir()
    second.parent.mkdir()
    content = b"runtime-library" * 2048
    first.write_bytes(content)
    second.write_bytes(content)

    linked, saved, _hashes = _deduplicate_tree(tmp_path)

    assert linked == 1
    assert saved == len(content)
    assert os.path.samefile(first, second)


def test_release_keeps_the_venv_entry_instead_of_resolving_its_python_symlink(
    tmp_path,
) -> None:
    target = tmp_path / "runtime/python3.10"
    target.parent.mkdir(parents=True)
    target.write_text("python", encoding="utf-8")
    entry = tmp_path / "profile/bin/python"
    entry.parent.mkdir(parents=True)
    try:
        entry.symlink_to(target)
    except OSError:
        return

    absolute = _absolute_entry(entry)

    assert absolute == entry.absolute()
    assert absolute != entry.resolve()


def test_release_doctor_parses_marked_json_amid_native_stdout() -> None:
    output = "\n".join(
        [
            "native warning",
            '{"unrelated": true}',
            (
                f'{release_doctor.PROBE_JSON_MARKER}'
                '{"python":"3.12.13","ok":true}'
                "native shutdown text"
            ),
        ]
    )

    assert release_doctor._parse_probe_json(output) == {
        "python": "3.12.13",
        "ok": True,
    }


def test_release_doctor_retries_only_unparseable_success_output(monkeypatch) -> None:
    outputs = iter(
        [
            "native initialization without a payload",
            release_doctor.PROBE_JSON_MARKER + '{"ok":true}',
        ]
    )
    calls = []

    def fake_run(command, *, environment=None):
        calls.append((command, environment))
        return next(outputs)

    monkeypatch.setattr(release_doctor, "_run", fake_run)

    assert release_doctor._run_json_probe(["probe"]) == {"ok": True}
    assert len(calls) == 2


def test_release_doctor_does_not_retry_failed_probe_process(monkeypatch) -> None:
    calls = []

    def failed_run(command, *, environment=None):
        calls.append((command, environment))
        raise release_doctor.DoctorFailure("probe exited 1")

    monkeypatch.setattr(release_doctor, "_run", failed_run)

    with pytest.raises(release_doctor.DoctorFailure, match="probe exited 1"):
        release_doctor._run_json_probe(["probe"])
    assert len(calls) == 1


def test_release_doctor_processes_are_read_only(monkeypatch) -> None:
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured.update(kwargs)
        return subprocess.CompletedProcess(command, 0, "ok\n", "")

    monkeypatch.setattr(release_doctor.subprocess, "run", fake_run)

    assert release_doctor._run(
        ["probe"], environment={"CALLER_VALUE": "preserved"}
    ) == "ok"
    assert captured["env"]["CALLER_VALUE"] == "preserved"
    assert captured["env"]["PYTHONDONTWRITEBYTECODE"] == "1"
    assert captured["env"][release_doctor.READ_ONLY_DOCTOR_ENV] == "1"


def test_read_only_doctor_import_does_not_create_log_files(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    environment = dict(os.environ)
    environment[release_doctor.READ_ONLY_DOCTOR_ENV] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPATH"] = str(repo_root)

    subprocess.run(
        [sys.executable, "-c", "import utils.logger_config"],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    assert not (tmp_path / "logs").exists()
