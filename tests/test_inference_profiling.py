"""Configuration parsing tests for the optional inference profiler."""

from __future__ import annotations

from utils.inference_profiling import InferenceProfiler


def test_resource_monitor_enabled_parses_string_false(monkeypatch) -> None:
    monkeypatch.delenv("NMX_INFERENCE_RESOURCES", raising=False)

    profiler = InferenceProfiler(
        {
            "mode": "off",
            "resources": {"enabled": "false"},
        },
        component="server",
    )

    assert profiler._resources.enabled is False
