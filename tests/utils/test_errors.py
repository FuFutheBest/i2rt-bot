"""Tests for explicit SuperInference error types.

Author: OpenAI Assistant
"""

from __future__ import annotations

import pytest

from utils.errors import (
    ConfigError,
    DataStaleError,
    DeviceRuntimeError,
    PolicyRuntimeError,
    SHMContractError,
    SuperInferenceError,
)


@pytest.mark.unit
def test_errors_share_common_base_class() -> None:
    """Explicit errors should be catchable through SuperInferenceError."""
    for error_class in (
        ConfigError,
        SHMContractError,
        DataStaleError,
        PolicyRuntimeError,
        DeviceRuntimeError,
    ):
        assert issubclass(error_class, SuperInferenceError)


@pytest.mark.unit
def test_error_context_is_included_in_message() -> None:
    """Context should make contract failures actionable in logs."""
    error = ConfigError("missing class", context="devices[0]")

    assert str(error) == "devices[0]: missing class"
    assert error.message == "missing class"
    assert error.context == "devices[0]"
