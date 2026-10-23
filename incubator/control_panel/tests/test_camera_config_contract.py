from pathlib import Path
from typing import Any, Iterator

import yaml


def _mapping_nodes(value: Any) -> Iterator[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for nested in value.values():
            yield from _mapping_nodes(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _mapping_nodes(nested)


def test_all_web_camera_openpi_configs_disable_hardware_latency() -> None:
    config_root = Path(__file__).resolve().parents[3] / "configs"
    cameras: list[tuple[Path, dict[str, Any]]] = []

    for path in sorted(config_root.rglob("*.yaml")):
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        cameras.extend(
            (path, node)
            for node in _mapping_nodes(document)
            if node.get("class") == "WebCameraOpenPi"
        )

    assert cameras, "no WebCameraOpenPi config entries were found"
    violations = [
        f"{path.relative_to(config_root)}:{camera.get('device_class_id')}="
        f"{camera.get('state_feedback', {}).get('hardware_latency_ms')!r}"
        for path, camera in cameras
        if camera.get("state_feedback", {}).get("hardware_latency_ms") != 0
    ]
    assert not violations, "WebCameraOpenPi latency must be zero: " + ", ".join(
        violations
    )
