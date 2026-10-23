"""Explicit contracts for real-robot SuperOffline comparisons.

The registry deliberately does not inspect ``policy.POLICY_CLASSES``.  A
comparison is only valid when both selections resolve through this module's
real-robot adapter registry and share the recorded-observation/action
contracts below.  Model imports remain lazy so the browser service can start
without a model runtime installed.
"""

from __future__ import annotations

import inspect
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

from utils.runtime_profiles import (
    RuntimeProfileError,
    RuntimeSelection,
    normalize_config_name,
    select_runtime_python,
)
from utils.superoffline_action_calibration import ActionCalibrationContract
from utils.superoffline_observation import (
    FISHEYE_LINGBOTVA_SOURCE_PROFILE,
    FISHEYE_OPENPI_RTC_SOURCE_PROFILE,
    FISHEYE_OPENPI_SOURCE_PROFILE,
    IPHONE_SOURCE_PROFILE,
)
from utils.superoffline_preflight import (
    PreparedSuperOfflineInput,
    SuperOfflinePreflightError,
    prepare_superoffline_input,
)

REAL_ROBOT_OBSERVATION_CONTRACT = "superoffline.real_robot_observation.v1"
DUAL_ARM_TCP_ACTION_CONTRACT = "superoffline.dual_arm_tcp_action_16d.v1"
DUAL_ARM_TCP_ACTION_LAYOUT = "dual_tcp_xyz_quaternion_xyzw_gripper"
DUAL_ARM_TCP_ACTION_FRAME = "rizon.base_tcp.v1"
DUAL_ARM_TCP_ACTION_UNITS = "position_m_quaternion_xyzw_gripper_width_m"

_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")


class SuperOfflineError(ValueError):
    """Base error for a request that cannot become a safe comparison job."""


class SuperOfflineAdapterError(SuperOfflineError):
    """A requested model adapter or model variant is not available."""


class SuperOfflinePairError(SuperOfflineError):
    """The request does not describe two to four compatible real models."""


class SuperOfflineExecutionError(SuperOfflineError):
    """A validated job could not be passed to an offline execution adapter."""


@dataclass(frozen=True)
class SuperOfflineModelContract:
    """Reviewed source and action semantics for one registered config."""

    config_name: str
    source_profiles: tuple[str, ...]
    target_device_roles: Mapping[str, str]
    action_calibration: ActionCalibrationContract
    action_frame_id: str = DUAL_ARM_TCP_ACTION_FRAME
    action_units: str = DUAL_ARM_TCP_ACTION_UNITS

    def public_metadata(self) -> dict[str, Any]:
        return {
            "config_name": self.config_name,
            "source_profiles": list(self.source_profiles),
            "target_device_roles": dict(self.target_device_roles),
            **self.action_calibration.public_metadata(),
            "action_frame_id": self.action_frame_id,
            "action_units": self.action_units,
        }


def _model_contract(
    config_name: str,
    *,
    source_profile: str,
    left_camera: str,
    right_camera: str,
    action_calibration: ActionCalibrationContract,
    robot_device: str = "RizonRobot_2",
    compatible_source_profiles: tuple[str, ...] = (),
) -> SuperOfflineModelContract:
    from utils.superoffline_observation import SUPER_OFFLINE_REVIEWED_SOURCE_PROFILES

    current_profile = SUPER_OFFLINE_REVIEWED_SOURCE_PROFILES.get(source_profile)
    return SuperOfflineModelContract(
        config_name=config_name,
        source_profiles=(*((current_profile,) if current_profile else ()), source_profile, *compatible_source_profiles),
        target_device_roles={
            left_camera: "left_wrist_rgb",
            right_camera: "right_wrist_rgb",
            robot_device: "rizon_dual_arm_state",
        },
        action_calibration=action_calibration,
    )


_DIRECT_TCP_ACTION_CALIBRATION = ActionCalibrationContract(
    calibration_id="rizon.base-tcp.direct-state.v1",
    absent_parameters=(
        "tcp_umi_json_path",
        "left_robot_base_to_world_transform",
        "right_robot_base_to_world_transform",
    ),
    required_policy_values={"normalize_gripper_width": False},
)
_FASTWAM_GRIPPER_POLICY_VALUES = {
    "gripper_max_width": 0.0925319933860716,
    "normalize_gripper_width": False,
    "clip_gripper_width": True,
}
_FASTWAM_DIRECT_TCP_ACTION_CALIBRATION = ActionCalibrationContract(
    calibration_id="rizon.base-tcp.fastwam-state.v1",
    absent_parameters=(
        "tcp_umi_json_path",
        "left_robot_base_to_world_transform",
        "right_robot_base_to_world_transform",
    ),
    required_policy_values=_FASTWAM_GRIPPER_POLICY_VALUES,
)
_FASTWAM_TCP_UMI_ACTION_CALIBRATION = ActionCalibrationContract(
    calibration_id="rizon.umi3.tcp-umi.v1",
    file_parameters={"tcp_umi_json_path": "tcp_to_umi"},
    required_policy_values=_FASTWAM_GRIPPER_POLICY_VALUES,
)
_OPENPI_TCP_UMI_ACTION_CALIBRATION = ActionCalibrationContract(
    calibration_id="rizon.umi3.tcp-umi.v1",
    file_parameters={"tcp_umi_json_path": "tcp_to_umi"},
    required_policy_values={
        "gripper_open_max": 0.11,
        "left_gripper_offset": 0.0,
        "right_gripper_offset": 0.0,
    },
)
SUPER_OFFLINE_MODEL_CONTRACTS: dict[str, SuperOfflineModelContract] = {
    "policy_infer/fastwam/config_infer_fisheye_umi3_dual_fastwam": _model_contract(
        "policy_infer/fastwam/config_infer_fisheye_umi3_dual_fastwam",
        source_profile=FISHEYE_OPENPI_SOURCE_PROFILE,
        compatible_source_profiles=(FISHEYE_OPENPI_RTC_SOURCE_PROFILE,),
        left_camera="WebCameraOpenPi_0",
        right_camera="WebCameraOpenPi_1",
        action_calibration=_FASTWAM_DIRECT_TCP_ACTION_CALIBRATION,
    ),
    "policy_infer/fastwam/config_infer_fisheye_umi3_dual_fastwam_memory": _model_contract(
        "policy_infer/fastwam/config_infer_fisheye_umi3_dual_fastwam_memory",
        source_profile=FISHEYE_OPENPI_SOURCE_PROFILE,
        compatible_source_profiles=(FISHEYE_OPENPI_RTC_SOURCE_PROFILE,),
        left_camera="WebCameraOpenPi_0",
        right_camera="WebCameraOpenPi_1",
        action_calibration=_FASTWAM_DIRECT_TCP_ACTION_CALIBRATION,
    ),
    "policy_infer/fastwam/config_infer_fisheye_umi3_dual_fasterwam": _model_contract(
        "policy_infer/fastwam/config_infer_fisheye_umi3_dual_fasterwam",
        source_profile=FISHEYE_OPENPI_SOURCE_PROFILE,
        compatible_source_profiles=(FISHEYE_OPENPI_RTC_SOURCE_PROFILE,),
        left_camera="WebCameraOpenPi_0",
        right_camera="WebCameraOpenPi_1",
        action_calibration=_FASTWAM_TCP_UMI_ACTION_CALIBRATION,
    ),
    "policy_infer/openpi/config_infer_fisheye_umi3_dual_openpi": _model_contract(
        "policy_infer/openpi/config_infer_fisheye_umi3_dual_openpi",
        source_profile=FISHEYE_OPENPI_SOURCE_PROFILE,
        compatible_source_profiles=(FISHEYE_OPENPI_RTC_SOURCE_PROFILE,),
        left_camera="WebCameraOpenPi_0",
        right_camera="WebCameraOpenPi_1",
        action_calibration=_OPENPI_TCP_UMI_ACTION_CALIBRATION,
    ),
    "policy_infer/lingbotva/config_infer_iphone_umi3_dual_lingbotva": _model_contract(
        "policy_infer/lingbotva/config_infer_iphone_umi3_dual_lingbotva",
        source_profile=IPHONE_SOURCE_PROFILE,
        left_camera="IPhoneCameraDevice_0",
        right_camera="IPhoneCameraDevice_1",
        action_calibration=_DIRECT_TCP_ACTION_CALIBRATION,
    ),
    "policy_infer/lingbotva/config_infer_fisheye_umi3_dual_lingbotva": _model_contract(
        "policy_infer/lingbotva/config_infer_fisheye_umi3_dual_lingbotva",
        source_profile=FISHEYE_LINGBOTVA_SOURCE_PROFILE,
        left_camera="WebCameraDevice_0",
        right_camera="WebCameraDevice_1",
        action_calibration=_DIRECT_TCP_ACTION_CALIBRATION,
    ),
    "policy_infer/lingbotva/config_infer_fisheye_umi3_dual_lingbotva_async": _model_contract(
        "policy_infer/lingbotva/config_infer_fisheye_umi3_dual_lingbotva_async",
        source_profile=FISHEYE_LINGBOTVA_SOURCE_PROFILE,
        left_camera="WebCameraDevice_0",
        right_camera="WebCameraDevice_1",
        action_calibration=_DIRECT_TCP_ACTION_CALIBRATION,
    ),
}


def get_superoffline_model_contract(config_name: str) -> SuperOfflineModelContract:
    """Return the reviewed source/action contract for a registered config."""
    normalized = normalize_config_name(config_name)
    if normalized is None:
        raise SuperOfflineAdapterError("config_name is required")
    try:
        return SUPER_OFFLINE_MODEL_CONTRACTS[normalized]
    except KeyError as exc:
        raise SuperOfflineAdapterError(
            f"config {normalized!r} has no registered SuperOffline source contract"
        ) from exc


def _identifier(value: Any, *, field: str) -> str:
    normalized = str(value or "").strip()
    if _IDENTIFIER_RE.fullmatch(normalized) is None:
        raise SuperOfflineAdapterError(
            f"{field} must contain only letters, digits, dot, underscore, or dash"
        )
    return normalized


def _required_text(value: Any, *, field: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise SuperOfflineAdapterError(f"{field} is required")
    return normalized


def _mapping(value: Any, *, field: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise SuperOfflineAdapterError(f"{field} must be an object")
    return {str(key): item for key, item in value.items()}


def _variant_value(variant: Any, field: str) -> Any:
    if isinstance(variant, Mapping):
        return variant.get(field)
    return getattr(variant, field, None)


@dataclass(frozen=True)
class SuperOfflineAdapter:
    """A model family permitted to participate in a real-robot comparison."""

    id: str
    label: str
    runtime_role: str
    default_config_name: str | None = None
    variant_family: bool = False
    builtin_execution: bool = False
    allowed_config_prefixes: tuple[str, ...] = ()
    allowed_config_names: tuple[str, ...] = ()
    observation_contract: str = REAL_ROBOT_OBSERVATION_CONTRACT
    action_contract: str = DUAL_ARM_TCP_ACTION_CONTRACT
    is_simulation: bool = False
    availability_error: str | None = None

    def public_metadata(self) -> dict[str, Any]:
        return {
            "adapter_id": self.id,
            "label": self.label,
            "runtime_role": self.runtime_role,
            "default_config_name": self.default_config_name,
            "variant_family": self.variant_family,
            "builtin_execution": self.builtin_execution,
            "allowed_config_prefixes": list(self.allowed_config_prefixes),
            "allowed_config_names": list(self.allowed_config_names),
            "observation_contract": self.observation_contract,
            "action_contract": self.action_contract,
            "real_robot": not self.is_simulation,
            "availability_error": self.availability_error,
        }


@dataclass(frozen=True)
class ResolvedSuperOfflineModel:
    """One normalized selection accepted by the adapter registry."""

    adapter: SuperOfflineAdapter
    label: str
    checkpoint: str
    config_name: str
    variant_id: str | None = None
    policy_params: Mapping[str, Any] | None = None
    source_profiles: tuple[str, ...] = ()
    target_device_roles: Mapping[str, str] | None = None
    action_calibration: ActionCalibrationContract | None = None
    action_frame_id: str = DUAL_ARM_TCP_ACTION_FRAME
    action_units: str = DUAL_ARM_TCP_ACTION_UNITS
    capabilities: tuple[str, ...] = ()

    @property
    def identity(self) -> tuple[str, str | None, str, str]:
        return (
            self.adapter.id,
            self.variant_id,
            self.checkpoint,
            self.config_name,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "adapter_id": self.adapter.id,
            "label": self.label,
            "checkpoint": self.checkpoint,
            "config_name": self.config_name,
            "variant_id": self.variant_id,
            "policy_params": dict(self.policy_params or {}),
            "capabilities": list(self.capabilities),
            "runtime_role": self.adapter.runtime_role,
            "source_profiles": list(self.source_profiles),
            "target_device_roles": dict(self.target_device_roles or {}),
            "action_calibration_id": (
                None
                if self.action_calibration is None
                else self.action_calibration.calibration_id
            ),
            "action_frame_id": self.action_frame_id,
            "action_units": self.action_units,
        }


@dataclass(frozen=True)
class SuperOfflineComparisonRequest:
    """A validated two-to-four-model request for an execution adapter."""

    models: tuple[ResolvedSuperOfflineModel, ...]
    observation_path: str | None = None
    metadata: Mapping[str, Any] | None = None
    job_id: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "models": [model.as_dict() for model in self.models],
            "observation_path": self.observation_path,
            "metadata": dict(self.metadata or {}),
            "job_id": self.job_id,
        }


@dataclass(frozen=True)
class SuperOfflineRuntimeContext:
    """A selected, verified runtime profile for one comparison model."""

    model: ResolvedSuperOfflineModel
    python: Path
    profile_name: str
    environment: Mapping[str, str]


def _fastwam_registry_api() -> tuple[Any, Any, Any]:
    """Import the shared FastWAM variant contract only when it is needed."""
    try:
        from consumer.inference.fastwam_model_registry import (
            FASTWAM_MODEL_VARIANTS,
            get_fastwam_model_variant,
            validate_fastwam_variant_config,
        )
    except ModuleNotFoundError as exc:
        if exc.name == "consumer.inference.fastwam_model_registry":
            raise SuperOfflineAdapterError(
                "FastWAM variant registry is unavailable in this checkout"
            ) from exc
        raise
    return (
        FASTWAM_MODEL_VARIANTS,
        get_fastwam_model_variant,
        validate_fastwam_variant_config,
    )


def _fastwam_variants() -> list[dict[str, Any]]:
    variants, _, _ = _fastwam_registry_api()
    records = variants.values() if isinstance(variants, Mapping) else variants
    result: list[dict[str, Any]] = []
    for variant in records:
        variant_id = _identifier(
            _variant_value(variant, "variant_id")
            if _variant_value(variant, "variant_id") is not None
            else _variant_value(variant, "id"),
            field="variant id",
        )
        config_name = normalize_config_name(_variant_value(variant, "config_name"))
        if config_name is None:
            raise SuperOfflineAdapterError(
                f"FastWAM variant {variant_id!r} has no config_name"
            )
        label = _required_text(_variant_value(variant, "label"), field="variant label")
        contract = get_superoffline_model_contract(config_name)
        result.append(
            {
                "id": variant_id,
                "label": label,
                "config_name": config_name,
                **contract.public_metadata(),
            }
        )
    return result


def _infer_fastwam_variant_from_checkpoint(checkpoint: str) -> str:
    """Infer the public variant for a manual checkpoint path without loading it."""
    path = Path(checkpoint).expanduser()
    search_root = path if path.is_dir() else path.parent
    config_path = next(
        (
            candidate / "config.yaml"
            for candidate in (search_root, *search_root.parents)
            if (candidate / "config.yaml").is_file()
        ),
        search_root / "config.yaml",
    )
    try:
        text = config_path.read_text(encoding="utf-8", errors="replace").lower()
    except OSError:
        text = ""
    checkpoint_hint = str(path).lower()
    if (
        "one_pass_future_cache" in text
        or "fasterwam" in checkpoint_hint
        or re.search(r"(?:^|[/_.-])c4(?:$|[/_.-])", checkpoint_hint)
    ):
        return "fasterwam"
    if (
        re.search(r"condition_history_frames\s*:\s*4\b", text)
        or re.search(r"obs_horizon\s*:\s*4\b", text)
        or "fastwam-memory" in checkpoint_hint
        or "fastwam_memory" in checkpoint_hint
        or "b1_1_s4" in checkpoint_hint
    ):
        return "fastwam-memory"
    return "fastwam"


class SuperOfflineAdapterRegistry:
    """Explicit model catalog and request normalizer for SuperOffline."""

    def __init__(self, adapters: Sequence[SuperOfflineAdapter] = ()) -> None:
        self._adapters: dict[str, SuperOfflineAdapter] = {}
        for adapter in adapters:
            self.register(adapter)

    def register(self, adapter: SuperOfflineAdapter) -> None:
        adapter_id = _identifier(adapter.id, field="adapter id")
        if adapter_id in self._adapters:
            raise SuperOfflineAdapterError(
                f"SuperOffline adapter {adapter_id!r} is already registered"
            )
        if not str(adapter.label).strip():
            raise SuperOfflineAdapterError("adapter label is required")
        if not str(adapter.runtime_role).strip():
            raise SuperOfflineAdapterError("adapter runtime_role is required")
        if adapter.availability_error is not None and not str(
            adapter.availability_error
        ).strip():
            raise SuperOfflineAdapterError(
                "adapter availability_error must be a non-empty string when supplied"
            )
        default_config_name = (
            None
            if adapter.default_config_name is None
            else normalize_config_name(adapter.default_config_name)
        )
        if isinstance(adapter.allowed_config_prefixes, (str, bytes)) or not isinstance(
            adapter.allowed_config_prefixes, Sequence
        ):
            raise SuperOfflineAdapterError("adapter config prefixes must be an array")
        for prefix in adapter.allowed_config_prefixes:
            if not isinstance(prefix, str):
                raise SuperOfflineAdapterError("adapter config prefixes must be strings")
            normalized_prefix = prefix.strip()
            if normalized_prefix != prefix:
                raise SuperOfflineAdapterError(
                    "adapter config prefixes cannot contain leading or trailing whitespace"
                )
            if not normalized_prefix.endswith("/"):
                raise SuperOfflineAdapterError(
                    "adapter config prefixes must end with a slash"
                )
            normalize_config_name(normalized_prefix + "config")
        if isinstance(adapter.allowed_config_names, (str, bytes)) or not isinstance(
            adapter.allowed_config_names, Sequence
        ):
            raise SuperOfflineAdapterError("adapter config names must be an array")
        allowed_config_names = tuple(
            normalize_config_name(config_name)
            for config_name in adapter.allowed_config_names
        )
        if any(config_name is None for config_name in allowed_config_names):
            raise SuperOfflineAdapterError("adapter config names must not be empty")
        if len(set(allowed_config_names)) != len(allowed_config_names):
            raise SuperOfflineAdapterError("adapter config names must be unique")
        if (
            default_config_name is not None
            and allowed_config_names
            and default_config_name not in allowed_config_names
        ):
            raise SuperOfflineAdapterError(
                "adapter default_config_name must be one of its registered config names"
            )
        if adapter.availability_error is None:
            for config_name in allowed_config_names:
                get_superoffline_model_contract(str(config_name))
        self._adapters[adapter_id] = replace(
            adapter,
            default_config_name=default_config_name,
            allowed_config_names=tuple(
                str(config_name) for config_name in allowed_config_names
            ),
        )

    @staticmethod
    def _validate_config_name(adapter: SuperOfflineAdapter, config_name: str) -> str:
        normalized = normalize_config_name(config_name)
        assert normalized is not None
        if (
            adapter.allowed_config_names
            and normalized not in adapter.allowed_config_names
        ):
            expected = ", ".join(
                repr(config_name) for config_name in adapter.allowed_config_names
            )
            raise SuperOfflineAdapterError(
                f"{adapter.id!r} only accepts registered config names {expected}"
            )
        if adapter.allowed_config_prefixes and not any(
            normalized.startswith(prefix) for prefix in adapter.allowed_config_prefixes
        ):
            expected = ", ".join(repr(prefix) for prefix in adapter.allowed_config_prefixes)
            raise SuperOfflineAdapterError(
                f"{adapter.id!r} only accepts config names below {expected}"
            )
        return normalized

    def get(self, adapter_id: str) -> SuperOfflineAdapter:
        normalized = _identifier(adapter_id, field="adapter id")
        try:
            return self._adapters[normalized]
        except KeyError as exc:
            raise SuperOfflineAdapterError(
                f"{normalized!r} is not a registered SuperOffline adapter"
            ) from exc

    def public_models(self) -> list[dict[str, Any]]:
        """Return browser-safe metadata without importing a model runtime."""
        records: list[dict[str, Any]] = []
        for adapter in self._adapters.values():
            if adapter.is_simulation:
                continue
            record = adapter.public_metadata()
            if adapter.availability_error is not None:
                record.update(
                    {
                        "available": False,
                        "availability_error": adapter.availability_error,
                        "config_contracts": [],
                        "variants": [],
                    }
                )
                records.append(record)
                continue
            if adapter.allowed_config_names:
                record["config_contracts"] = [
                    get_superoffline_model_contract(config_name).public_metadata()
                    for config_name in adapter.allowed_config_names
                ]
            if adapter.variant_family:
                try:
                    variants = _fastwam_variants()
                except SuperOfflineAdapterError as exc:
                    record.update(
                        {
                            "available": False,
                            "availability_error": str(exc),
                            "variants": [],
                        }
                    )
                else:
                    record.update({"available": bool(variants), "variants": variants})
            else:
                record["available"] = True
            records.append(record)
        return records

    def resolve_model(self, payload: Mapping[str, Any]) -> ResolvedSuperOfflineModel:
        if not isinstance(payload, Mapping):
            raise SuperOfflineAdapterError("each model must be an object")
        if payload.get("simulation") is True:
            raise SuperOfflineAdapterError(
                "simulation models are not accepted by SuperOffline"
            )

        adapter = self.get(str(payload.get("adapter_id", "")))
        if adapter.is_simulation:
            raise SuperOfflineAdapterError(
                f"{adapter.id!r} is a simulation adapter and cannot be compared here"
            )
        if adapter.availability_error is not None:
            raise SuperOfflineAdapterError(adapter.availability_error)
        checkpoint = _required_text(payload.get("checkpoint"), field="checkpoint")
        policy_params = _mapping(payload.get("policy_params"), field="policy_params")

        if adapter.variant_family:
            raw_variant_id = payload.get("variant_id")
            variant_id = _identifier(
                _infer_fastwam_variant_from_checkpoint(checkpoint)
                if raw_variant_id in (None, "", "auto")
                else raw_variant_id,
                field="variant_id",
            )
            _, get_variant, validate_variant = _fastwam_registry_api()
            try:
                variant = get_variant(variant_id)
            except Exception as exc:
                raise SuperOfflineAdapterError(
                    f"unknown FastWAM variant {variant_id!r}"
                ) from exc
            raw_config_name = _variant_value(variant, "config_name")
            if raw_config_name is None:
                raise SuperOfflineAdapterError(
                    f"FastWAM variant {variant_id!r} has no config_name"
                )
            config_name = self._validate_config_name(adapter, str(raw_config_name))
            contract = get_superoffline_model_contract(config_name)
            requested_config = normalize_config_name(payload.get("config_name"))
            if requested_config is not None and requested_config != config_name:
                raise SuperOfflineAdapterError(
                    f"FastWAM variant {variant_id!r} requires config {config_name!r}"
                )
            if "policy_params" in payload:
                try:
                    validate_variant(variant_id, policy_params)
                except Exception as exc:
                    raise SuperOfflineAdapterError(
                        f"FastWAM policy_params do not match {variant_id!r}"
                    ) from exc
            resolved_variant_id = _identifier(
                _variant_value(variant, "variant_id")
                if _variant_value(variant, "variant_id") is not None
                else variant_id,
                field="variant_id",
            )
            label = _required_text(
                _variant_value(variant, "label"), field="variant label"
            )
            return ResolvedSuperOfflineModel(
                adapter=adapter,
                label=label,
                checkpoint=checkpoint,
                config_name=config_name,
                variant_id=resolved_variant_id,
                policy_params=policy_params,
                source_profiles=contract.source_profiles,
                target_device_roles=contract.target_device_roles,
                action_calibration=contract.action_calibration,
                action_frame_id=contract.action_frame_id,
                action_units=contract.action_units,
            )

        if payload.get("variant_id") not in (None, ""):
            raise SuperOfflineAdapterError(
                f"{adapter.id!r} does not accept a FastWAM variant_id"
            )
        raw_config_name = payload.get("config_name") or adapter.default_config_name
        if raw_config_name is None:
            raise SuperOfflineAdapterError(f"config_name is required for {adapter.id!r}")
        config_name = self._validate_config_name(adapter, str(raw_config_name))
        contract = get_superoffline_model_contract(config_name)
        return ResolvedSuperOfflineModel(
            adapter=adapter,
            label=adapter.label,
            checkpoint=checkpoint,
            config_name=config_name,
            policy_params=policy_params,
            source_profiles=contract.source_profiles,
            target_device_roles=contract.target_device_roles,
            action_calibration=contract.action_calibration,
            action_frame_id=contract.action_frame_id,
            action_units=contract.action_units,
        )

    def validate_pair(
        self, models: Sequence[Mapping[str, Any]]
    ) -> tuple[ResolvedSuperOfflineModel, ...]:
        if isinstance(models, (str, bytes)) or not isinstance(models, Sequence):
            raise SuperOfflinePairError("models must be an array containing two to four models")
        if not 2 <= len(models) <= 4:
            raise SuperOfflinePairError(
                "SuperOffline comparisons require two to four registered real-robot models"
            )
        resolved = tuple(self.resolve_model(model) for model in models)
        if len({model.identity for model in resolved}) != len(resolved):
            raise SuperOfflinePairError("comparison models must be distinct")
        first = resolved[0]
        for model in resolved[1:]:
            if first.adapter.observation_contract != model.adapter.observation_contract:
                raise SuperOfflinePairError(
                    "the selected models require different observation contracts"
                )
            if first.adapter.action_contract != model.adapter.action_contract:
                raise SuperOfflinePairError(
                    "the selected models emit different action contracts"
                )
            if first.action_frame_id != model.action_frame_id:
                raise SuperOfflinePairError("the selected models emit different action frames")
            if first.action_units != model.action_units:
                raise SuperOfflinePairError("the selected models emit different action units")
        return resolved

    def parse_request(self, payload: Mapping[str, Any]) -> SuperOfflineComparisonRequest:
        if not isinstance(payload, Mapping):
            raise SuperOfflinePairError("comparison payload must be an object")
        models = self.validate_pair(payload.get("models", ()))
        observation_value = payload.get("observation_path")
        if observation_value in (None, ""):
            observation_path = None
        elif isinstance(observation_value, str):
            observation_path = observation_value.strip() or None
        else:
            raise SuperOfflinePairError("observation_path must be a string")
        return SuperOfflineComparisonRequest(
            models=models,
            observation_path=observation_path,
            metadata=_mapping(payload.get("metadata"), field="metadata"),
        )


def resolve_model_runtime(
    model: ResolvedSuperOfflineModel,
    *,
    repo_root: Path | None = None,
    selector: Callable[..., RuntimeSelection | None] = select_runtime_python,
) -> SuperOfflineRuntimeContext:
    """Resolve one model through a verified local runtime profile only."""
    try:
        selection = selector(
            model.adapter.runtime_role,
            config_name=model.config_name,
            repo_root=repo_root,
            require_verified_profile=True,
            required=True,
        )
    except RuntimeProfileError as exc:
        raise SuperOfflineExecutionError(
            f"verified runtime profile is unavailable for {model.label}: {exc}"
        ) from exc
    if selection is None or selection.profile is None:
        raise SuperOfflineExecutionError(
            "SuperOffline requires a verified runtime profile; no direct Python "
            "path or environment fallback is accepted"
        )
    profile = selection.profile
    return SuperOfflineRuntimeContext(
        model=model,
        python=selection.python,
        profile_name=profile.name,
        environment=profile.process_environment(config_name=model.config_name),
    )


class ComparisonExecutor(Protocol):
    """Object contract for a read-only SuperOffline comparison executor."""

    def validate_request(self, request: SuperOfflineComparisonRequest) -> None:
        """Apply executor-specific validation after common preflight."""

    def run(
        self,
        request: SuperOfflineComparisonRequest,
        contexts: tuple[SuperOfflineRuntimeContext, ...],
        *,
        event_callback: Callable[[Mapping[str, Any]], None] | None = None,
        cancel_event: Any | None = None,
    ) -> Mapping[str, Any]:
        """Run the validated pair and return a JSON-serializable summary."""


class RuntimeProfileComparisonRunner:
    """Preflight model runtimes before delegating to an injected executor.

    The runner intentionally owns no policy classes.  A production integration
    supplies an executor object that validates a request before loading one
    model at a time and writes its own artifacts.  Validation is mandatory so
    an alternate executor cannot bypass recorder and action-contract checks.
    """

    def __init__(
        self,
        executor: ComparisonExecutor | None = None,
        *,
        repo_root: Path | None = None,
        selector: Callable[..., RuntimeSelection | None] = select_runtime_python,
    ) -> None:
        self._executor = executor
        self._repo_root = repo_root
        self._selector = selector

    def validate_request(self, request: SuperOfflineComparisonRequest) -> None:
        """Run common input preflight before executor and runtime validation."""
        self._validate_and_resolve(request)

    def _validate_and_resolve(
        self,
        request: SuperOfflineComparisonRequest,
    ) -> tuple[tuple[SuperOfflineRuntimeContext, ...], PreparedSuperOfflineInput]:
        if self._executor is None:
            raise SuperOfflineAdapterError(
                "No SuperOffline executor is registered for the selected models"
            )
        try:
            prepared = prepare_superoffline_input(request, repo_root=self._repo_root)
        except SuperOfflinePreflightError as exc:
            raise SuperOfflineAdapterError(str(exc)) from exc
        validate = getattr(self._executor, "validate_request", None)
        if not callable(validate):
            raise SuperOfflineAdapterError(
                "SuperOffline executor must implement validate_request(request)"
            )
        executor_run = getattr(self._executor, "run", None)
        if not callable(executor_run):
            raise SuperOfflineAdapterError(
                "SuperOffline executor must implement run(request, contexts)"
            )
        validate_prepared = getattr(self._executor, "validate_prepared_request", None)
        if callable(validate_prepared):
            validate_prepared(request, prepared)
        else:
            validate(request)
        contexts = tuple(
            resolve_model_runtime(
                model,
                repo_root=self._repo_root,
                selector=self._selector,
            )
            for model in request.models
        )
        return contexts, prepared

    def public_metadata(self) -> dict[str, Any]:
        """Expose executor capability without leaking runtime profile paths."""
        metadata = getattr(self._executor, "public_metadata", None)
        if callable(metadata):
            payload = metadata()
            if isinstance(payload, Mapping):
                return dict(payload)
        return {"mode": "external-executor"}

    def request_warnings(
        self, request: SuperOfflineComparisonRequest
    ) -> list[dict[str, Any]]:
        """Return non-blocking executor warnings after common validation."""
        warnings = getattr(self._executor, "request_warnings", None)
        if not callable(warnings):
            return []
        payload = warnings(request)
        if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)):
            raise SuperOfflineAdapterError(
                "SuperOffline executor request_warnings must return an array"
            )
        return [dict(item) for item in payload if isinstance(item, Mapping)]

    def run(
        self,
        request: SuperOfflineComparisonRequest,
        *,
        event_callback: Callable[[Mapping[str, Any]], None] | None = None,
        cancel_event: Any | None = None,
    ) -> Mapping[str, Any]:
        contexts, prepared_input = self._validate_and_resolve(request)
        if self._executor is None:
            raise SuperOfflineExecutionError(
                "No SuperOffline executor is registered. Start the service with "
                "an executor that implements the adapter contract."
            )
        executor_run = getattr(self._executor, "run", None)
        if not callable(executor_run):
            raise SuperOfflineExecutionError(
                "SuperOffline executor must implement run(request, contexts)"
            )
        optional_kwargs: dict[str, Any] = {}
        try:
            parameters = inspect.signature(executor_run).parameters.values()
        except (TypeError, ValueError):
            parameters = ()
        parameter_names = {parameter.name for parameter in parameters}
        accepts_keywords = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters
        )
        if accepts_keywords or "event_callback" in parameter_names:
            optional_kwargs["event_callback"] = event_callback
        if accepts_keywords or "cancel_event" in parameter_names:
            optional_kwargs["cancel_event"] = cancel_event
        if accepts_keywords or "prepared_input" in parameter_names:
            optional_kwargs["prepared_input"] = prepared_input
        result = executor_run(request, contexts, **optional_kwargs)
        if not isinstance(result, Mapping):
            raise SuperOfflineExecutionError("SuperOffline executor must return an object")
        return dict(result)


SUPER_OFFLINE_ADAPTERS = SuperOfflineAdapterRegistry(
    (
        SuperOfflineAdapter(
            id="fastwam",
            label="FastWAM",
            runtime_role="fastwam-server",
            variant_family=True,
            builtin_execution=True,
            allowed_config_names=(
                "policy_infer/fastwam/config_infer_fisheye_umi3_dual_fastwam",
                "policy_infer/fastwam/config_infer_fisheye_umi3_dual_fastwam_memory",
                "policy_infer/fastwam/config_infer_fisheye_umi3_dual_fasterwam",
            ),
        ),
        SuperOfflineAdapter(
            id="openpi",
            label="OpenPi",
            runtime_role="policy",
            default_config_name="policy_infer/openpi/config_infer_fisheye_umi3_dual_openpi",
            builtin_execution=True,
            allowed_config_names=(
                "policy_infer/openpi/config_infer_fisheye_umi3_dual_openpi",
            ),
        ),
        SuperOfflineAdapter(
            id="diffusion",
            label="Diffusion (legacy iPhone dual)",
            runtime_role="policy",
            default_config_name="policy_infer/diffusion/config_infer_iphone_umi_dual",
            builtin_execution=False,
            allowed_config_names=(
                "policy_infer/diffusion/config_infer_iphone_umi_dual",
            ),
            availability_error=(
                "Diffusion's legacy 062931/00044 iPhone-Dahuan rig has no "
                "reviewed TCP/UMI and base-to-world calibration bundle."
            ),
        ),
        SuperOfflineAdapter(
            id="lingbotva",
            label="LingBotVA",
            runtime_role="policy-server",
            default_config_name="policy_infer/lingbotva/config_infer_fisheye_umi3_dual_lingbotva",
            builtin_execution=True,
            allowed_config_names=(
                "policy_infer/lingbotva/config_infer_fisheye_umi3_dual_lingbotva",
                "policy_infer/lingbotva/config_infer_fisheye_umi3_dual_lingbotva_async",
                "policy_infer/lingbotva/config_infer_iphone_umi3_dual_lingbotva",
            ),
        ),
    )
)


__all__ = [
    "ComparisonExecutor",
    "DUAL_ARM_TCP_ACTION_CONTRACT",
    "DUAL_ARM_TCP_ACTION_FRAME",
    "DUAL_ARM_TCP_ACTION_LAYOUT",
    "DUAL_ARM_TCP_ACTION_UNITS",
    "FISHEYE_LINGBOTVA_SOURCE_PROFILE",
    "FISHEYE_OPENPI_SOURCE_PROFILE",
    "IPHONE_SOURCE_PROFILE",
    "REAL_ROBOT_OBSERVATION_CONTRACT",
    "ResolvedSuperOfflineModel",
    "RuntimeProfileComparisonRunner",
    "SUPER_OFFLINE_ADAPTERS",
    "SuperOfflineAdapter",
    "SuperOfflineAdapterError",
    "SuperOfflineAdapterRegistry",
    "SuperOfflineComparisonRequest",
    "SuperOfflineError",
    "SuperOfflineExecutionError",
    "SuperOfflineModelContract",
    "SuperOfflinePairError",
    "SuperOfflineRuntimeContext",
    "SUPER_OFFLINE_MODEL_CONTRACTS",
    "get_superoffline_model_contract",
    "resolve_model_runtime",
]
