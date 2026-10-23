"""Real dm3-core round trip driven by simulated I2RT device streams."""

from pathlib import Path

import pytest

pytest.importorskip(
    "dm3_core.nedf",
    reason="I2RT NEDF3 round-trip tests require the optional recording SDK",
)
pytest.importorskip(
    "nmx_msg",
    reason="I2RT NEDF3 round-trip tests require the optional protobuf package",
)

from scripts.diagnostics.smoke_i2rt_nedf3_unreal import run_unreal_smoke


def test_unreal_i2rt_nedf3_pipeline(tmp_path: Path) -> None:
    episode_dir = run_unreal_smoke(tmp_path, frames=3)

    assert episode_dir == tmp_path / "episode_0000"
    assert (episode_dir / "metadata.json").is_file()
    assert (episode_dir / "episode_stats.json").is_file()
