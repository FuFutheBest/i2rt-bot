#!/usr/bin/env python3
"""
Explicit error hierarchy for SuperInference contracts and runtime boundaries.

Author: OpenAI Assistant
"""

from __future__ import annotations

from typing import Optional


class SuperInferenceError(Exception):
    """Base class for all explicit SuperInference errors."""

    def __init__(self, message: str, *, context: Optional[str] = None) -> None:
        self.message = message
        self.context = context
        full_message = f"{context}: {message}" if context else message
        super().__init__(full_message)


class ConfigError(SuperInferenceError):
    """Raised when configuration is missing, invalid, or inconsistent."""


class SHMContractError(SuperInferenceError):
    """Raised when a shared-memory layout or metadata contract is violated."""


class DataStaleError(SuperInferenceError):
    """Raised when data is too old for the required runtime contract."""


class PolicyRuntimeError(SuperInferenceError):
    """Raised when policy loading, prediction, or output validation fails."""


class DeviceRuntimeError(SuperInferenceError):
    """Raised when device connection, reading, writing, or control fails."""
