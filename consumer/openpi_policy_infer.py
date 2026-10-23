#!/usr/bin/env python3
"""Compatibility facade for consumer.inference.openpi_policy_infer."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if __name__ == "__main__":
    import runpy
    runpy.run_module("consumer.inference.openpi_policy_infer", run_name="__main__")
else:
    from consumer.inference.openpi_policy_infer import *  # noqa: F403
    from consumer.inference.openpi_policy_infer import _OpenPiPolicyInferMixin  # noqa: F401
