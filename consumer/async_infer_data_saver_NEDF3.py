#!/usr/bin/env python3
"""Compatibility facade for the canonical NEDF3 inference saver."""
if __name__ == "__main__":
    import sys
    import runpy
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    runpy.run_module("consumer.io.async_infer_data_saver_NEDF3", run_name="__main__")
else:
    from consumer.io.async_infer_data_saver_NEDF3 import *  # noqa: F403
