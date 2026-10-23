#!/usr/bin/env python3
"""Launch the I2RT NEDF3 Rerun visualizer from a source checkout."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from consumer.io.i2rt_nedf3_rerun import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
