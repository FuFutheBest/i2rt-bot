#!/usr/bin/env python3
"""Discover, bind, and inspect machine-local Wan assets for FastWAM."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.fastwam_model_cache import (
    FastWAMModelCacheError,
    build_model_cache_plan,
    prepare_model_cache,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("discover", "prepare", "doctor"))
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--home", default=str(Path.home()))
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def _print_human(payload: dict[str, object]) -> None:
    print(f"FastWAM model cache ready: {payload['ready']}")
    for item in payload["assets"]:  # type: ignore[index]
        print(
            f"- {item['model_id']}: {item['status']} "  # type: ignore[index]
            f"source={item.get('source')} target={item.get('target')}"  # type: ignore[union-attr]
        )
    for warning in payload.get("warnings", []):
        print(f"warning: {warning}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.max_depth < 1 or args.max_depth > 12:
        print("error: --max-depth must be between 1 and 12", file=sys.stderr)
        return 2
    try:
        repo_root = Path(args.repo_root).expanduser().resolve()
        plan = build_model_cache_plan(args.config_name, repo_root=repo_root)
        payload = prepare_model_cache(
            plan,
            repo_root=repo_root,
            home=Path(args.home).expanduser().resolve(),
            environment=os.environ,
            max_depth=args.max_depth,
            apply=args.command == "prepare",
            discover_unbound=args.command != "doctor",
        )
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
        else:
            _print_human(payload)
        if args.command == "doctor" and not payload["ready"]:
            return 2
        return 0
    except (FastWAMModelCacheError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
