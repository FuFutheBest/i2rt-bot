from __future__ import annotations

import numpy as np

from manager.utils.rtc_blend_manager import RtcBlendChunkManager


def test_rtc_blend_manager_publishes_switch_diagnostics() -> None:
    manager = RtcBlendChunkManager(drop_first_actions=0, blend_steps=2)
    manager.put({"robot": np.zeros((4, 16), dtype=np.float64)}, observation_timestamp=1)
    manager.put({"robot": np.ones((4, 16), dtype=np.float64)}, observation_timestamp=2)

    diagnostics = manager.get_last_switch_diagnostics()
    assert diagnostics["manager_type"] == "RtcBlendChunkManager"
    assert diagnostics["chunk_replaced"] is True
    assert diagnostics["rtc_blend_applied"] is True
    assert diagnostics["rtc_blend_actual_steps"] == 2
