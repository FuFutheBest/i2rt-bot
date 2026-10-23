"""Compatibility alias for :mod:`utils.recording_control`."""

from __future__ import annotations

import sys

from utils import recording_control as _implementation

sys.modules[__name__] = _implementation
