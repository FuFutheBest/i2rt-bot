"""Compatibility alias for :mod:`utils.inference_control`."""

from __future__ import annotations

import sys

from utils import inference_control as _implementation

sys.modules[__name__] = _implementation
