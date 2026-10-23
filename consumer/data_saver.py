#!/usr/bin/env python3
"""Compatibility facade for consumer.io.data_saver."""

if __name__ == "__main__":
    import sys
    import runpy
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    runpy.run_module("consumer.io.data_saver", run_name="__main__")
else:
    from consumer.io.data_saver import *  # noqa: F403
