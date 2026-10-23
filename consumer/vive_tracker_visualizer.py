#!/usr/bin/env python3
"""Compatibility facade for consumer.visualization.vive_tracker_visualizer."""
if __name__ == "__main__":
    import sys
    import runpy
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    runpy.run_module("consumer.visualization.vive_tracker_visualizer", run_name="__main__")
else:
    from consumer.visualization.vive_tracker_visualizer import *  # noqa: F403
