#!/usr/bin/env python3
"""Compatibility facade for the policy connector.

The implementation lives in :mod:`consumer.inference.common.policy_connector`.

Author: Jun Lv, Han Xue, Zheng Wang
"""

from consumer.inference.common.policy_connector import (
    ActionType,
    PolicyConnector,
    _build_policy_constructor_kwargs,
    _make_run_timestamp,
    _resolve_action_chunk_observation_timestamp,
)

__all__ = [
    "ActionType",
    "PolicyConnector",
    "_build_policy_constructor_kwargs",
    "_make_run_timestamp",
    "_resolve_action_chunk_observation_timestamp",
]
