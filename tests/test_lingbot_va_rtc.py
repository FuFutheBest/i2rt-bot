from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip(
    "torch",
    reason="LingBotVA RTC tensor contracts require the optional PyTorch model stack",
)
pytest.importorskip(
    "diffusers",
    reason="LingBotVA RTC utilities import the optional Diffusers model stack",
)

pytestmark = pytest.mark.model

ROOT = Path(__file__).resolve().parents[1]
NMX_VLA_ROOT = ROOT / "third_party" / "nmx_vla"
if str(NMX_VLA_ROOT) not in sys.path:
    sys.path.append(str(NMX_VLA_ROOT))

from manager.utils.action_manager import BaseChunkManager  # noqa: E402
from policy.lingbot_va_policy import LingBotVAPolicy  # noqa: E402
from nmx_vla.model.modules.lingbot_va.utils.rtc import (  # noqa: E402
    apply_prefix_inpainting,
    build_prefix_weights,
)


class _SchedulerStub:
    sigmas = torch.tensor([1.0, 0.5, 0.0], dtype=torch.float32)


def test_lingbot_rtc_prefix_weights_exp() -> None:
    weights = build_prefix_weights(
        total_steps=8,
        inference_delay=2,
        prefix_attention_horizon=3,
        schedule="exp",
    )

    assert torch.allclose(weights[:2], torch.ones(2))
    assert weights[2] > weights[3] > weights[4] > 0
    assert torch.allclose(weights[5:], torch.zeros(3))


def test_lingbot_rtc_inpainting_uses_noised_prefix() -> None:
    action = torch.zeros((1, 2, 1, 3, 1), dtype=torch.float32)
    prefix = torch.ones_like(action)
    initial_noise = torch.zeros_like(action)
    weights = torch.ones((1, 1, 1, 3, 1), dtype=torch.float32)

    blended = apply_prefix_inpainting(
        action=action,
        prefix_action=prefix,
        initial_noise=initial_noise,
        scheduler=_SchedulerStub(),
        next_step_index=1,
        weights=weights,
    )

    assert torch.allclose(blended, torch.full_like(action, 0.5))


def test_base_chunk_manager_reports_original_index_after_drop() -> None:
    manager = BaseChunkManager(drop_first_actions=2)
    manager.put({"robot": np.arange(5, dtype=np.float64).reshape(5, 1)}, 100)

    first = manager.get()
    assert first is not None
    assert first["robot"].item() == 0
    assert manager.get_original_current_step() == 0

    manager.put({"robot": np.arange(10, 15, dtype=np.float64).reshape(5, 1)}, 200)
    second = manager.get()
    assert second is not None
    assert second["robot"].item() == 12
    assert manager.get_original_current_step() == 2


def test_lingbot_policy_rtc_delay_uses_mean_latency() -> None:
    policy = LingBotVAPolicy(
        enable_rtc=True,
        execution_fps=10.0,
        rtc_default_delay_steps=3,
        rtc_latency_stat="mean",
    )

    assert policy._compute_rtc_inference_delay_steps() == 3
    policy._latency_tracker.add(0.21)
    policy._latency_tracker.add(0.29)
    assert policy._compute_rtc_inference_delay_steps() == 3
