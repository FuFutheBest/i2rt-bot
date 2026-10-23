"""Contracts for the explicit real-robot SuperOffline model catalog."""

from __future__ import annotations

from pathlib import Path

import pytest

import utils.superoffline_registry as superoffline_registry
from utils.runtime_profiles import RuntimeProfileNotFoundError, RuntimeSelection
from utils.superoffline_registry import (
    RuntimeProfileComparisonRunner,
    SUPER_OFFLINE_ADAPTERS,
    SuperOfflineAdapter,
    SuperOfflineAdapterError,
    SuperOfflineAdapterRegistry,
    SuperOfflineExecutionError,
    SuperOfflinePairError,
    resolve_model_runtime,
)


def _pair() -> list[dict[str, object]]:
    return [
        {
            "adapter_id": "fastwam",
            "variant_id": "fasterwam",
            "checkpoint": "checkpoints/fasterwam",
        },
        {
            "adapter_id": "openpi",
            "config_name": "policy_infer/openpi/config_infer_fisheye_umi3_dual_openpi",
            "checkpoint": "checkpoints/openpi",
        },
    ]


def test_default_registry_lists_explicit_real_robot_model_contracts() -> None:
    models = {
        record["adapter_id"]: record for record in SUPER_OFFLINE_ADAPTERS.public_models()
    }

    assert set(models) == {"diffusion", "fastwam", "lingbotva", "openpi"}
    assert models["fastwam"]["runtime_role"] == "fastwam-server"
    assert models["fastwam"]["real_robot"] is True
    assert [variant["id"] for variant in models["fastwam"]["variants"]] == [
        "fastwam",
        "fastwam-memory",
        "fasterwam",
    ]
    assert models["openpi"]["default_config_name"] == (
        "policy_infer/openpi/config_infer_fisheye_umi3_dual_openpi"
    )
    assert models["openpi"]["builtin_execution"] is True
    assert models["openpi"]["allowed_config_names"] == [
        "policy_infer/openpi/config_infer_fisheye_umi3_dual_openpi"
    ]
    assert models["diffusion"]["default_config_name"] == "policy_infer/diffusion/config_infer_iphone_umi_dual"
    assert models["diffusion"]["available"] is False
    assert models["diffusion"]["config_contracts"] == []
    assert "no reviewed TCP/UMI" in models["diffusion"]["availability_error"]
    assert models["lingbotva"]["runtime_role"] == "policy-server"
    assert models["lingbotva"]["allowed_config_names"] == [
        "policy_infer/lingbotva/config_infer_fisheye_umi3_dual_lingbotva",
        "policy_infer/lingbotva/config_infer_fisheye_umi3_dual_lingbotva_async",
        "policy_infer/lingbotva/config_infer_iphone_umi3_dual_lingbotva",
    ]


def test_pair_normalization_uses_the_fastwam_variant_config() -> None:
    request = SUPER_OFFLINE_ADAPTERS.parse_request({"models": _pair()})

    fastwam, openpi = request.models
    assert fastwam.variant_id == "fasterwam"
    assert fastwam.config_name == "policy_infer/fastwam/config_infer_fisheye_umi3_dual_fasterwam"
    assert fastwam.adapter.runtime_role == "fastwam-server"
    assert openpi.config_name == "policy_infer/openpi/config_infer_fisheye_umi3_dual_openpi"
    assert fastwam.source_profiles == (
        "rizon-umi3-fisheye-openpi-rgb.v2",
        "rizon-umi3-fisheye-openpi-rgb.v1",
        "rizon-umi3-fisheye-openpi-ids-2-0-rgb.v1",
    )
    assert fastwam.target_device_roles == {
        "WebCameraOpenPi_0": "left_wrist_rgb",
        "WebCameraOpenPi_1": "right_wrist_rgb",
        "RizonRobot_2": "rizon_dual_arm_state",
    }


def test_manual_fastwam_checkpoint_infers_variant_without_ui_model_selector(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    weights = checkpoint / "checkpoints" / "weights"
    weights.mkdir(parents=True)
    weight = weights / "step_001000.pt"
    weight.write_bytes(b"placeholder")
    (checkpoint / "config.yaml").write_text(
        "framework:\n  fastwam:\n    default_action_infer_mode: one_pass_future_cache\n",
        encoding="utf-8",
    )

    model = SUPER_OFFLINE_ADAPTERS.resolve_model(
        {"adapter_id": "fastwam", "checkpoint": str(weight)}
    )

    assert model.variant_id == "fasterwam"
    assert model.config_name.endswith("config_infer_fisheye_umi3_dual_fasterwam")


def test_pair_accepts_any_two_available_real_models_before_recorder_preflight() -> None:
    models = _pair()
    models[1] = {
        "adapter_id": "lingbotva",
        "checkpoint": "checkpoints/lingbotva",
        "config_name": "policy_infer/lingbotva/config_infer_fisheye_umi3_dual_lingbotva",
    }

    request = SUPER_OFFLINE_ADAPTERS.parse_request({"models": models})

    assert [model.adapter.id for model in request.models] == ["fastwam", "lingbotva"]


def test_pair_rejects_a_real_adapter_without_reviewed_calibration() -> None:
    models = _pair()
    models[1] = {
        "adapter_id": "diffusion",
        "checkpoint": "checkpoints/diffusion",
    }

    with pytest.raises(SuperOfflineAdapterError, match="no reviewed TCP/UMI"):
        SUPER_OFFLINE_ADAPTERS.parse_request({"models": models})


def test_pair_rejects_wrong_count_duplicate_and_simulation_model() -> None:
    with pytest.raises(SuperOfflinePairError, match="two to four"):
        SUPER_OFFLINE_ADAPTERS.parse_request({"models": _pair()[:1]})

    duplicate = _pair()[:1] * 2
    with pytest.raises(SuperOfflinePairError, match="must be distinct"):
        SUPER_OFFLINE_ADAPTERS.parse_request({"models": duplicate})

    simulated = _pair()
    simulated[0]["simulation"] = True
    with pytest.raises(SuperOfflineAdapterError, match="simulation models"):
        SUPER_OFFLINE_ADAPTERS.parse_request({"models": simulated})


def test_comparison_accepts_four_distinct_models_and_rejects_five() -> None:
    models = [
        {
            "adapter_id": "fastwam",
            "variant_id": variant,
            "checkpoint": f"checkpoints/{variant}",
        }
        for variant in ("fastwam", "fastwam-memory", "fasterwam")
    ]
    models.append(
        {
            "adapter_id": "openpi",
            "checkpoint": "checkpoints/openpi",
        }
    )

    request = SUPER_OFFLINE_ADAPTERS.parse_request({"models": models})

    assert len(request.models) == 4
    with pytest.raises(SuperOfflinePairError, match="two to four"):
        SUPER_OFFLINE_ADAPTERS.parse_request({"models": [*models, dict(models[0])]})


def test_openpi_adapter_rejects_a_simulation_config_namespace() -> None:
    models = _pair()
    models[1]["config_name"] = "policy_infer/robotwin/config_infer_robotwin_fastwam_joint14"

    with pytest.raises(SuperOfflineAdapterError, match="registered config names"):
        SUPER_OFFLINE_ADAPTERS.parse_request({"models": models})


def test_explicit_model_adapters_reject_nonregistered_native_configs() -> None:
    with pytest.raises(SuperOfflineAdapterError, match="no reviewed TCP/UMI"):
        SUPER_OFFLINE_ADAPTERS.resolve_model(
            {
                "adapter_id": "diffusion",
                "checkpoint": "checkpoints/diffusion",
                "config_name": "policy_infer/diffusion/config_infer_rizon_eef_dual_abs_mode_umi",
            }
        )
    with pytest.raises(SuperOfflineAdapterError, match="registered config names"):
        SUPER_OFFLINE_ADAPTERS.resolve_model(
            {
                "adapter_id": "lingbotva",
                "checkpoint": "checkpoints/lingbotva",
                "config_name": "policy_infer/lingbotva/config_infer_iphone_umi_single_lingbotva",
            }
        )


def test_simulation_adapter_is_neither_listed_nor_selectable() -> None:
    registry = SuperOfflineAdapterRegistry(
        (
            SuperOfflineAdapter(
                id="simulator",
                label="Simulator",
                runtime_role="simulation",
                is_simulation=True,
            ),
        )
    )

    assert registry.public_models() == []
    with pytest.raises(SuperOfflineAdapterError, match="simulation adapter"):
        registry.resolve_model(
            {
                "adapter_id": "simulator",
                "checkpoint": "checkpoints/sim",
            }
        )


def test_runtime_preflight_uses_the_model_config_and_verified_profile() -> None:
    request = SUPER_OFFLINE_ADAPTERS.parse_request({"models": _pair()})
    calls: list[tuple[str, dict[str, object]]] = []

    class Profile:
        name = "fastwam-runtime"

        def process_environment(self, *, config_name: str) -> dict[str, str]:
            return {"SUPERINFERENCE_CONFIG_NAME": config_name}

    def selector(role: str, **kwargs: object) -> RuntimeSelection:
        calls.append((role, kwargs))
        return RuntimeSelection(
            python=Path("/runtime/python"),
            source="profile:fastwam-runtime",
            profile=Profile(),  # type: ignore[arg-type]
        )

    context = resolve_model_runtime(request.models[0], selector=selector)

    assert context.profile_name == "fastwam-runtime"
    assert context.environment["SUPERINFERENCE_CONFIG_NAME"] == (
        "policy_infer/fastwam/config_infer_fisheye_umi3_dual_fasterwam"
    )
    assert calls == [
        (
            "fastwam-server",
            {
                "config_name": "policy_infer/fastwam/config_infer_fisheye_umi3_dual_fasterwam",
                "repo_root": None,
                "require_verified_profile": True,
                "required": True,
            },
        )
    ]


def test_runtime_preflight_translates_an_unavailable_profile() -> None:
    request = SUPER_OFFLINE_ADAPTERS.parse_request({"models": _pair()})

    def selector(_role: str, **_kwargs: object) -> RuntimeSelection:
        raise RuntimeProfileNotFoundError("no local profile")

    with pytest.raises(SuperOfflineExecutionError, match="verified runtime profile"):
        resolve_model_runtime(request.models[0], selector=selector)


def test_injected_executor_receives_profile_checked_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = SUPER_OFFLINE_ADAPTERS.parse_request({"models": _pair()})
    monkeypatch.setattr(
        superoffline_registry,
        "prepare_superoffline_input",
        lambda *_args, **_kwargs: object(),
    )

    class Profile:
        name = "profile"

        def process_environment(self, *, config_name: str) -> dict[str, str]:
            return {"SUPERINFERENCE_CONFIG_NAME": config_name}

    def selector(_role: str, **_kwargs: object) -> RuntimeSelection:
        return RuntimeSelection(
            python=Path("/runtime/python"),
            source="profile:profile",
            profile=Profile(),  # type: ignore[arg-type]
        )

    observed: dict[str, object] = {}

    class Executor:
        def validate_request(self, request_arg: object) -> None:
            observed["validated"] = request_arg

        def run(self, request_arg: object, contexts: object) -> dict[str, object]:
            observed["request"] = request_arg
            observed["contexts"] = contexts
            return {"artifact": "comparison.json"}

    runner = RuntimeProfileComparisonRunner(Executor(), selector=selector)  # type: ignore[arg-type]

    assert runner.run(request) == {"artifact": "comparison.json"}
    assert observed["validated"] is request
    assert observed["request"] is request
    contexts = observed["contexts"]
    assert isinstance(contexts, tuple)
    assert [context.model.adapter.id for context in contexts] == ["fastwam", "openpi"]


def test_object_executor_run_method_receives_profile_checked_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = SUPER_OFFLINE_ADAPTERS.parse_request({"models": _pair()})
    monkeypatch.setattr(
        superoffline_registry,
        "prepare_superoffline_input",
        lambda *_args, **_kwargs: object(),
    )

    class Profile:
        name = "profile"

        def process_environment(self, *, config_name: str) -> dict[str, str]:
            return {"SUPERINFERENCE_CONFIG_NAME": config_name}

    def selector(_role: str, **_kwargs: object) -> RuntimeSelection:
        return RuntimeSelection(
            python=Path("/runtime/python"),
            source="profile:profile",
            profile=Profile(),  # type: ignore[arg-type]
        )

    class Executor:
        def validate_request(self, request_arg: object) -> None:
            assert request_arg is request

        def run(self, request_arg: object, contexts: object) -> dict[str, object]:
            assert request_arg is request
            assert isinstance(contexts, tuple)
            return {"artifact": "comparison.json"}

    runner = RuntimeProfileComparisonRunner(Executor(), selector=selector)  # type: ignore[arg-type]

    assert runner.run(request) == {"artifact": "comparison.json"}


def test_runner_rejects_an_executor_without_trusted_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = SUPER_OFFLINE_ADAPTERS.parse_request({"models": _pair()})
    monkeypatch.setattr(
        superoffline_registry,
        "prepare_superoffline_input",
        lambda *_args, **_kwargs: object(),
    )

    def selector(_role: str, **_kwargs: object) -> RuntimeSelection:
        raise AssertionError("runtime selection must not precede executor preflight")

    def plain_executor(_request: object, _contexts: object) -> dict[str, object]:
        return {"artifact": "comparison.json"}

    runner = RuntimeProfileComparisonRunner(plain_executor, selector=selector)

    with pytest.raises(SuperOfflineAdapterError, match="validate_request"):
        runner.validate_request(request)
