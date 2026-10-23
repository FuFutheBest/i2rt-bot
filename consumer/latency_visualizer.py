#!/usr/bin/env python3
"""Compatibility facade for consumer.visualization.latency_visualizer."""
if __name__ == "__main__":
    import sys
    import runpy
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    runpy.run_module("consumer.visualization.latency_visualizer", run_name="__main__")
else:
    from consumer.visualization.latency_visualizer import *  # noqa: F403
