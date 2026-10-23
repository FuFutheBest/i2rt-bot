#!/usr/bin/env python3
"""
Policy runtime wrapper for loading and invoking policy adapters.

Author: OpenAI Assistant
"""

from __future__ import annotations

import inspect
from typing import Any, Dict, List, Optional

import numpy as np

from policy import POLICY_CLASSES

class PolicyRuntime:
    """Own the policy object lifecycle while preserving the existing policy API."""

    def __init__(self, policy_class: str, policy_params: Optional[Dict[str, Any]] = None) -> None:
        self.policy_class = policy_class
        self.policy_params: Dict[str, Any] = dict(policy_params or {})
        self.policy = self._load_policy()

    def _load_policy(self) -> Any:
        """Instantiate and load the configured policy class."""
        if self.policy_class not in POLICY_CLASSES:
            raise ValueError(f"Unknown policy_class: {self.policy_class}")

        policy_cls = POLICY_CLASSES[self.policy_class]
        if policy_cls is None:
            raise ValueError(
                f"Policy class '{self.policy_class}' is registered but not available (import failed)"
            )

        policy = policy_cls(**self.policy_params)
        policy.load()
        return policy

    def predict(
        self,
        observation: Any,
        action_configs: List[Dict[str, Any]],
        chunk_length: int,
        **kwargs: Any,
    ) -> Dict[str, np.ndarray]:
        """Delegate prediction to the loaded policy."""
        # The refactored loop supplies optional timing/RTC arguments, while a
        # few legacy policies still implement the original three-argument
        # ``predict`` contract.  Filter only unsupported keyword arguments;
        # policies with ``**kwargs`` retain the complete extension surface.
        forwarded_kwargs = kwargs
        try:
            signature = inspect.signature(self.policy.predict)
        except (TypeError, ValueError):
            pass
        else:
            accepts_extra = any(
                parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in signature.parameters.values()
            )
            if not accepts_extra:
                forwarded_kwargs = {
                    key: value
                    for key, value in kwargs.items()
                    if key in signature.parameters
                }
        return self.policy.predict(
            observation,
            action_configs,
            chunk_length=chunk_length,
            **forwarded_kwargs,
        )

    def predict_value(self, observation: Any, predicted_action: Any) -> Any:
        """Delegate value prediction when the policy supports it."""
        if not hasattr(self.policy, "predict_value"):
            raise ValueError(f"Policy class '{self.policy_class}' does not support predict_value")
        return self.policy.predict_value(observation, predicted_action)

    def warm_up(
        self,
        action_configs: List[Dict[str, Any]],
        *,
        num_observations: int,
        sample_observation: Any,
    ) -> None:
        """Warm up RTC-capable policies without exposing lifecycle details to callers."""
        if hasattr(self.policy, "warm_up"):
            self.policy.warm_up(
                action_configs,
                num_observations=num_observations,
                sample_observation=sample_observation,
            )
