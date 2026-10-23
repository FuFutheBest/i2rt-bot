#!/usr/bin/env python3
"""
Policy history manager facade.

Author: OpenAI Assistant
"""

from __future__ import annotations

from typing import Any

from consumer.policy_history.frames import PolicyHistoryFramesMixin
from consumer.policy_history.picker import PolicyHistoryPickerMixin
from consumer.policy_history.timing import PolicyHistoryTimingMixin
from consumer.policy_history.update import PolicyHistoryUpdateMixin
from consumer.policy_history.video import PolicyHistoryVideoMixin


class PolicyHistoryManager:
    """Compose policy-history timing, frame, picker, video, and update helpers."""

    def __init__(self, owner: Any) -> None:
        self.owner = owner
        component_types = (
            PolicyHistoryTimingMixin,
            PolicyHistoryVideoMixin,
            PolicyHistoryFramesMixin,
            PolicyHistoryPickerMixin,
            PolicyHistoryUpdateMixin,
        )
        self._components = [component_type() for component_type in component_types]
        for component in self._components:
            component.owner = owner
        # Components call one another through their public helper methods. Bind
        # missing peer operations explicitly while retaining each implementation
        # on its single-responsibility component.
        for component in self._components:
            for peer in self._components:
                for name in dir(peer):
                    if name.startswith("_") or hasattr(component, name):
                        continue
                    value = getattr(peer, name)
                    if callable(value):
                        setattr(component, name, value)

    def __getattr__(self, name: str) -> Any:
        for component in self._components:
            if hasattr(component, name):
                return getattr(component, name)
        raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        object.__setattr__(self, name, value)
        if name.startswith("_"):
            return
        for component in self.__dict__.get("_components", []):
            if callable(value):
                setattr(component, name, value)
