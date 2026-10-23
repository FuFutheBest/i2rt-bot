"""Compatibility alias for :mod:`utils.home_transition`."""

from __future__ import annotations

import sys

from utils import home_transition as _implementation

sys.modules[__name__] = _implementation
