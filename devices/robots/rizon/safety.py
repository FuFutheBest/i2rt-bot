#!/usr/bin/env python3
"""Rizon robot helper mixin.

Author: OpenAI Assistant
"""

from __future__ import annotations

import time


from utils.logger_config import logger


class RizonSafetyMixin:
    """Focused RizonRobot helper methods from safety.py."""

    def _record_safety_clip_event(self, arm_name: str) -> None:
        """Hook for downstream metrics on safety clip events (rate-limited)."""
        now = time.monotonic()
        last = self._safety_clip_last_log_time.get(arm_name, 0.0)
        if now - last < 1.0:
            return
        self._safety_clip_last_log_time[arm_name] = now
        logger.warning(
            f"safety clip event recorded for arm={arm_name or 'single'} "
            f"(bounds enabled, see prior 'safety clip applied' message for details)"
        )

    def _check_operational_status_change(self) -> None:
        """Edge-detect transitions in operational_status and log them.

        Used primarily to surface wrench-limit safety events (transitions
        in/out of IN_REDUCED_STATE caused by SetMaxContactWrench). Failures
        are silently swallowed so robot polling never breaks because of
        diagnostic logging.
        """
        try:
            if self.is_dual_arm:
                handle = self.robot_left or self.robot_right
            else:
                handle = self.robot
            if handle is None:
                return
            current = handle.operational_status()
            if current == self._last_operational_status:
                return
            previous = self._last_operational_status
            self._last_operational_status = current
            current_name = getattr(current, "name", str(current))
            previous_name = (
                getattr(previous, "name", str(previous))
                if previous is not None
                else "UNKNOWN"
            )
            if str(current_name) == "IN_REDUCED_STATE":
                logger.warning(
                    f"contact wrench limit triggered: operational_status "
                    f"{previous_name} -> {current_name} "
                    f"(max_contact_wrench={self.max_contact_wrench})"
                )
            elif str(previous_name) == "IN_REDUCED_STATE":
                logger.info(
                    f"contact wrench cleared: operational_status "
                    f"{previous_name} -> {current_name}"
                )
            else:
                logger.debug(
                    f"operational_status changed: {previous_name} -> {current_name}"
                )
        except Exception as exc:
            logger.debug(f"operational_status polling skipped: {exc}")

