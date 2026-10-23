#!/usr/bin/env python3
"""Compatibility facade for consumer.inference.lingbot_va_infer."""
if __name__ == "__main__":
    import sys
    import runpy
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    runpy.run_module("consumer.inference.lingbot_va_infer", run_name="__main__")
else:
    from consumer.inference.lingbot_va_infer import *  # noqa: F403
