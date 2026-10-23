"""Tests for the single-pass device construction contract."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from devices.config import StateFeedbackConfig
from devices.contracts import DeviceKind
from devices import DEVICE_CLASSES
from utils.device_config import DeviceConfigError, parse_device_config


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _module_source_paths(module_name: str) -> list[Path]:
    """Return module source paths without resolving optional device imports."""
    module_path = REPOSITORY_ROOT.joinpath(*module_name.split("."))
    source_path = module_path.with_suffix(".py")
    if source_path.is_file():
        return [source_path]
    if module_path.is_dir():
        return sorted(module_path.rglob("*.py"))
    return []


def _find_class_definition(
    module_name: str, class_name: str
) -> tuple[Path, ast.Module, ast.ClassDef] | None:
    for source_path in _module_source_paths(module_name):
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                return source_path, tree, node
    return None


def _source_module_name(source_path: Path) -> str:
    relative_path = source_path.relative_to(REPOSITORY_ROOT)
    parts = (
        relative_path.parent.parts
        if relative_path.name == "__init__.py"
        else relative_path.with_suffix("").parts
    )
    return ".".join(parts)


def _relative_import_module(
    source_path: Path, source_module: str, imported: ast.ImportFrom
) -> str | None:
    package_parts = (
        source_module.split(".")
        if source_path.name == "__init__.py"
        else source_module.split(".")[:-1]
    )
    if imported.level:
        if imported.level > len(package_parts) + 1:
            return None
        package_parts = package_parts[: len(package_parts) - imported.level + 1]
    imported_parts = imported.module.split(".") if imported.module else []
    module_parts = [*package_parts, *imported_parts]
    return ".".join(module_parts) or None


def _inherited_constructor_targets(
    source_path: Path, tree: ast.Module, class_node: ast.ClassDef
) -> list[tuple[str, str]]:
    source_module = _source_module_name(source_path)
    locally_declared = {
        node.name for node in tree.body if isinstance(node, ast.ClassDef)
    }
    targets: list[tuple[str, str]] = []

    for base in class_node.bases:
        if isinstance(base, ast.Name):
            if base.id in locally_declared:
                targets.append((source_module, base.id))
                continue
            for imported in tree.body:
                if not isinstance(imported, ast.ImportFrom):
                    continue
                imported_module = _relative_import_module(
                    source_path, source_module, imported
                )
                if imported_module is None:
                    continue
                for alias in imported.names:
                    if (alias.asname or alias.name) == base.id:
                        targets.append((imported_module, alias.name))
                        break
        elif isinstance(base, ast.Attribute):
            parts: list[str] = []
            attribute: ast.expr = base
            while isinstance(attribute, ast.Attribute):
                parts.append(attribute.attr)
                attribute = attribute.value
            if isinstance(attribute, ast.Name):
                parts.append(attribute.id)
                qualified_parts = list(reversed(parts))
                for imported in tree.body:
                    if not isinstance(imported, ast.Import):
                        continue
                    for alias in imported.names:
                        if alias.asname == qualified_parts[0]:
                            qualified_parts = [
                                *alias.name.split("."),
                                *qualified_parts[1:],
                            ]
                            break
                targets.append((".".join(qualified_parts[:-1]), qualified_parts[-1]))

    return targets


def _constructor_parameter_names(
    module_name: str,
    class_name: str,
    seen: set[tuple[str, str]] | None = None,
) -> set[str]:
    """Read an adapter constructor signature from source without importing SDKs."""
    target = (module_name, class_name)
    seen = set() if seen is None else seen
    if target in seen:
        raise AssertionError(f"Cyclic constructor inheritance: {module_name}.{class_name}")

    definition = _find_class_definition(module_name, class_name)
    if definition is None:
        raise AssertionError(f"Registry class not found: {module_name}.{class_name}")
    source_path, tree, class_node = definition
    initializer = next(
        (
            child
            for child in class_node.body
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            and child.name == "__init__"
        ),
        None,
    )
    if initializer is not None:
        return {
            argument.arg
            for argument in (
                *initializer.args.posonlyargs,
                *initializer.args.args,
                *initializer.args.kwonlyargs,
            )
        }

    for inherited_module, inherited_class in _inherited_constructor_targets(
        source_path, tree, class_node
    ):
        try:
            return _constructor_parameter_names(
                inherited_module,
                inherited_class,
                seen | {target},
            )
        except AssertionError:
            continue
    raise AssertionError(
        f"Registry class has no resolvable constructor: {module_name}.{class_name}"
    )


def _state(shape, dtype="float64", types=None):
    return {
        "fps": 30.0,
        "buffer_size": 10,
        "hardware_latency_ms": 0.0,
        "format": {
            "shape": shape,
            "dtype": dtype,
            "type": types or [f"linear:{shape[0] if len(shape) == 1 else 1}"],
        },
    }


def test_constructor_contract_source_check_follows_inherited_init(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package_path = tmp_path / "adapters"
    package_path.mkdir()
    (package_path / "base.py").write_text(
        "class AdapterBase:\n"
        "    def __init__(self, state_feedback, *, control=None):\n"
        "        pass\n",
        encoding="utf-8",
    )
    (package_path / "device.py").write_text(
        "from .base import AdapterBase\n\n"
        "class InheritedAdapter(AdapterBase):\n"
        "    pass\n",
        encoding="utf-8",
    )
    (package_path / "alias_device.py").write_text(
        "import adapters.base as base_module\n\n"
        "class AliasInheritedAdapter(base_module.AdapterBase):\n"
        "    pass\n",
        encoding="utf-8",
    )
    monkeypatch.setitem(globals(), "REPOSITORY_ROOT", tmp_path)

    expected_parameters = {
        "self",
        "state_feedback",
        "control",
    }
    assert _constructor_parameter_names("adapters.device", "InheritedAdapter") == expected_parameters
    assert (
        _constructor_parameter_names("adapters.alias_device", "AliasInheritedAdapter")
        == expected_parameters
    )


def test_parse_device_config_returns_typed_contracts_once() -> None:
    parsed = parse_device_config(
        {
            "class": "BaseDevice",
            "device_class_id": 3,
            "state_feedback": _state([7], types=["linear:7"]),
        },
        device_kind=DeviceKind.LOW_DIM_R,
    )

    assert isinstance(parsed.state_feedback, StateFeedbackConfig)
    assert parsed.control is None
    assert parsed.constructor_kwargs()["state_feedback"] is parsed.state_feedback


def test_camera_requires_rgb_state_format() -> None:
    with pytest.raises(DeviceConfigError, match="type must be rgb"):
        parse_device_config(
            {
                "class": "WebCameraDevice",
                "device_class_id": 0,
                "state_feedback": _state([480, 640, 3], types=["linear:921600"]),
            },
            device_kind=DeviceKind.CAMERA,
        )


def test_writable_device_requires_control() -> None:
    with pytest.raises(DeviceConfigError, match="requires a control mapping"):
        parse_device_config(
            {
                "class": "BaseRobot",
                "device_class_id": 0,
                "state_feedback": _state([7], types=["j1-7"]),
            },
            device_kind=DeviceKind.LOW_DIM_RW,
        )


def test_state_feedback_requires_all_common_fields() -> None:
    state = _state([1], types=["scalar"])
    del state["hardware_latency_ms"]
    with pytest.raises(DeviceConfigError, match="hardware_latency_ms"):
        parse_device_config(
            {
                "class": "BaseDevice",
                "device_class_id": 0,
                "state_feedback": state,
            },
            device_kind=DeviceKind.LOW_DIM_R,
        )


def test_registered_device_constructors_declare_common_contracts() -> None:
    for name, (module_name, class_name) in DEVICE_CLASSES._specs.items():
        parameters = _constructor_parameter_names(module_name, class_name)
        assert "state_feedback" in parameters, name
        if DEVICE_CLASSES.kind(name).writable:
            assert "control" in parameters, name
