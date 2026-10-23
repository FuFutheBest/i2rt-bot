from __future__ import annotations

from incubator.r1.rizon_state_layout import RizonDualArmStateLayout


def test_r1_state_layout_preserves_command_and_wrench_tails() -> None:
    assert RizonDualArmStateLayout.LEFT_COMMAND_TCP == slice(48, 55)
    assert RizonDualArmStateLayout.RIGHT_EXTERNAL_WRENCH == slice(68, 74)
    assert RizonDualArmStateLayout.COMMAND_GRIPPER_SIZE == 88
