#!/usr/bin/env python3
"""Compatibility facade for :mod:`consumer.inference.fastwam_policy_infer`."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if __name__ == "__main__":
    runpy.run_module("consumer.inference.fastwam_policy_infer", run_name="__main__")
else:
    from consumer.inference import fastwam_policy_infer as _implementation

    for _name, _value in vars(_implementation).items():
        if not _name.startswith("__"):
            globals()[_name] = _value
