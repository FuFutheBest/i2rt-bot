#!/usr/bin/env python3
"""Compatibility facade for :mod:`consumer.inference.fastwam_server`."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if __name__ == "__main__":
    runpy.run_module("consumer.inference.fastwam_server", run_name="__main__")
else:
    from consumer.inference import fastwam_server as _implementation

    # Keep private helper imports used by legacy diagnostics and tests working
    # while the implementation lives in the refactored package.
    for _name, _value in vars(_implementation).items():
        if not _name.startswith("__"):
            globals()[_name] = _value
