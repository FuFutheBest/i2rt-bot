"""Regression tests for Control Panel config-picker metadata caching."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from incubator.control_panel import server


def _write_launch_config(config_dir: Path, *, marker: str = "initial") -> Path:
    config_dir.mkdir(parents=True, exist_ok=True)
    launch = config_dir / "launch.yaml"
    launch.write_text(
        f"""# {marker}
devices:
  - class: BaseDevice
    device_class_id: 1
""",
        encoding="utf-8",
    )
    return launch


def _fake_runtime_config() -> dict[str, Any]:
    return {
        "devices": [{"class": "BaseDevice", "device_class_id": 1}],
        "metadata": {"label": "Cached launch"},
    }


def test_config_picker_reuses_metadata_and_invalidates_launch_changes(
    tmp_path: Path, monkeypatch
) -> None:
    config_dir = tmp_path / "configs"
    _write_launch_config(config_dir)
    calls: list[tuple[str, Path]] = []

    def compose(name: str, *, config_dir: Path) -> dict[str, Any]:
        calls.append((name, config_dir))
        return _fake_runtime_config()

    monkeypatch.setattr(server, "compose_runtime_config", compose)

    first = server.control_panel_config_items(config_dir, selected="launch")
    assert [name for name, _ in calls] == ["launch"]
    assert first[0]["selected"] is True
    assert calls[0][1] == config_dir.resolve()
    first[0]["label"] = "mutated response"

    second = server.control_panel_config_items(config_dir)
    assert [name for name, _ in calls] == ["launch"]
    assert second[0]["label"] == "Cached launch"
    assert second[0]["selected"] is False

    _write_launch_config(config_dir, marker="changed-and-longer")

    third = server.control_panel_config_items(config_dir)
    assert [name for name, _ in calls] == ["launch", "launch"]
    assert third[0]["label"] == "Cached launch"


def test_config_picker_invalidates_for_nested_hydra_input(
    tmp_path: Path, monkeypatch
) -> None:
    config_dir = tmp_path / "configs"
    _write_launch_config(config_dir)
    nested = config_dir / "devices" / "shared.yaml"
    nested.parent.mkdir()
    nested.write_text("devices: []\n", encoding="utf-8")
    calls: list[str] = []

    def compose(name: str, *, config_dir: Path) -> dict[str, Any]:
        calls.append(name)
        return _fake_runtime_config()

    monkeypatch.setattr(server, "compose_runtime_config", compose)

    first = server.control_panel_config_items(config_dir)
    second = server.control_panel_config_items(config_dir)
    assert calls == ["launch"]
    assert first == second

    nested.write_text("# changed\ndevices: []\n", encoding="utf-8")

    refreshed = server.control_panel_config_items(config_dir)
    assert calls == ["launch", "launch"]
    assert refreshed[0]["name"] == "launch"


def test_config_picker_cache_is_scoped_to_config_directory(
    tmp_path: Path, monkeypatch
) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    _write_launch_config(left)
    _write_launch_config(right)
    calls: list[Path] = []

    def compose(name: str, *, config_dir: Path) -> dict[str, Any]:
        calls.append(config_dir)
        return _fake_runtime_config()

    monkeypatch.setattr(server, "compose_runtime_config", compose)

    server.control_panel_config_items(left)
    server.control_panel_config_items(right)
    server.control_panel_config_items(left)
    server.control_panel_config_items(right)

    assert calls == [left.resolve(), right.resolve()]


def test_config_picker_does_not_cache_payload_built_during_config_change(
    tmp_path: Path, monkeypatch
) -> None:
    config_dir = tmp_path / "configs"
    _write_launch_config(config_dir)
    calls: list[str] = []

    def compose(name: str, *, config_dir: Path) -> dict[str, Any]:
        calls.append(name)
        if len(calls) == 1:
            _write_launch_config(config_dir, marker="changed-during-compose")
        return _fake_runtime_config()

    monkeypatch.setattr(server, "compose_runtime_config", compose)

    first = server.control_panel_config_items(config_dir)
    second = server.control_panel_config_items(config_dir)
    third = server.control_panel_config_items(config_dir)

    assert first[0]["name"] == "launch"
    assert second[0]["name"] == "launch"
    assert third[0]["name"] == "launch"
    assert calls == ["launch", "launch"]
