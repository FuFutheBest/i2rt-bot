#!/usr/bin/env python3
"""
Visualize camera frames from a ReplayBuffer zarr.zip using OpenCV windows.

This script opens a zarr ZipStore produced by the HDF5→zarr converter and
displays the two camera streams (`camera0_rgb`, `camera1_rgb`) for quick
inspection.

Author: SuperInference Team
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional, Tuple

import cv2
import numpy as np
import zarr

from utils.logger_config import logger

def _try_register_codecs() -> Optional[object]:
    """Try to import and run register_codecs from diffusion_policy.

    Tries both import directly and via adding third_party/data-scaling-laws to sys.path.
    Returns the callable if available, else None.
    """
    # Attempt direct import first
    try:
        from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs  # type: ignore
        return register_codecs
    except Exception:
        pass

    # Attempt to add third_party/data-scaling-laws to sys.path, then import
    try:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        third_party_root = os.path.join(repo_root, "third_party", "data-scaling-laws")
        if os.path.isdir(third_party_root) and third_party_root not in sys.path:
            sys.path.append(third_party_root)
        from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs  # type: ignore
        return register_codecs
    except Exception:
        return None


def _find_camera_group(root: zarr.hierarchy.Group) -> zarr.hierarchy.Group:
    """Return the group where camera arrays are stored.

    ReplayBuffer usually stores arrays under a `data` group. If not present,
    use the root itself.
    """
    try:
        if "data" in root.group_keys():
            return root["data"]
    except Exception:
        pass
    return root


def _list_all_arrays(group: zarr.hierarchy.Group, prefix: str = "") -> List[Tuple[str, zarr.Array]]:
    """Recursively list all arrays under the group, returning (path, array)."""
    results: List[Tuple[str, zarr.Array]] = []
    for key in group.array_keys():
        try:
            results.append((os.path.join(prefix, key) if prefix else key, group[key]))
        except Exception:
            continue
    for key in group.group_keys():
        try:
            sub = group[key]
        except Exception:
            continue
        results.extend(_list_all_arrays(sub, os.path.join(prefix, key) if prefix else key))
    return results


def _find_camera_keys(group: zarr.hierarchy.Group) -> List[str]:
    """Find camera dataset keys under the provided group.

    Prefer canonical `camera0_rgb`, `camera1_rgb`. Otherwise, use heuristics:
    - arrays whose name contains both "camera" and "rgb"
    - arrays of shape (N, H, W, 3) and dtype uint8
    Returns up to two keys.
    """
    # Direct canonical keys
    direct_keys = list(group.array_keys())
    expected = ["camera0_rgb", "camera1_rgb"]
    found = [k for k in expected if k in direct_keys]
    if len(found) >= 2:
        return found[:2]

    # Heuristic by name at current level
    name_candidates = [k for k in direct_keys if ("camera" in k and "rgb" in k)]
    name_candidates.sort()
    if len(name_candidates) >= 2:
        return name_candidates[:2]

    # Recursive search by name and shape
    arrays = _list_all_arrays(group)
    name_based = [path for path, arr in arrays if ("camera" in path and "rgb" in path)]
    name_based.sort()
    if len(name_based) >= 2:
        return name_based[:2]

    # Shape/dtype based
    shape_based = []
    for path, arr in arrays:
        try:
            if arr.ndim == 4 and arr.shape[-1] == 3 and np.issubdtype(arr.dtype, np.uint8):
                shape_based.append(path)
        except Exception:
            continue
    shape_based.sort()
    return shape_based[:2]


def visualize_zarr_cameras(
    zarr_zip_path: str,
    start: int = 0,
    end: Optional[int] = None,
    step: int = 1,
    wait_ms: int = 1,
    show_cam0: bool = True,
    show_cam1: bool = True,
    cam0_key_override: Optional[str] = None,
    cam1_key_override: Optional[str] = None,
) -> None:
    """Visualize camera frames from zarr.zip using OpenCV.

    Args:
        zarr_zip_path: Absolute/relative path to the .zarr.zip file.
        start: Start frame index (inclusive).
        end: End frame index (exclusive). If None, use min length of cameras.
        step: Frame stride (e.g., 2 to skip every other frame).
        wait_ms: Milliseconds to wait in cv2.waitKey per frame.
        show_cam0: Whether to show `camera0_rgb`.
        show_cam1: Whether to show `camera1_rgb`.
    """
    if not os.path.exists(zarr_zip_path):
        logger.error(f"zarr file does not exist: {zarr_zip_path}")
        return

    # Ensure codecs are registered for compressed datasets
    register = _try_register_codecs()
    if register is not None:
        try:
            register()
        except Exception as exc:  # pragma: no cover
            logger.warning(f"Failed to register image codecs, attempting to proceed: {exc}")
    else:
        logger.warning("register_codecs not available; loading may fail if datasets are compressed.")

    try:
        with zarr.ZipStore(zarr_zip_path, mode="r") as store:
            root = zarr.open(store, mode="r")

            data_group = _find_camera_group(root)
            all_arrays = _list_all_arrays(data_group)
            try:
                available = ", ".join([p for p, _ in all_arrays]) or "<none>"
            except Exception:
                available = "<unknown>"
            logger.info(f"Available arrays: {available}")

            # Determine keys: priority 1) user override; 2) canonical; 3) heuristic
            cam0_key = cam0_key_override
            cam1_key: Optional[str] = cam1_key_override

            if cam0_key is None and cam1_key is None:
                camera_keys = _find_camera_keys(data_group)
                if not camera_keys:
                    logger.error("No camera datasets found in zarr store.")
                    return
                cam0_key = "camera0_rgb"
                cam1_key = "camera1_rgb"
                # Use detected keys if canonical names are missing
                if cam0_key not in camera_keys and (cam1_key not in camera_keys if cam1_key else True):
                    if len(camera_keys) >= 2:
                        cam0_key, cam1_key = camera_keys[0], camera_keys[1]
                    elif len(camera_keys) == 1:
                        cam0_key, cam1_key = camera_keys[0], None

            # Resolve arrays possibly in nested groups
            def _get_array_by_path(g: zarr.hierarchy.Group, path: Optional[str]) -> Optional[zarr.Array]:
                if path is None:
                    return None
                try:
                    return g[path]
                except Exception:
                    return None

            cam0 = _get_array_by_path(data_group, cam0_key) if show_cam0 else None
            cam1 = _get_array_by_path(data_group, cam1_key) if show_cam1 else None

            if cam0 is None and cam1 is None:
                logger.error("Requested cameras are not available in the zarr store.")
                return

            lengths: List[int] = []
            if cam0 is not None:
                lengths.append(int(cam0.shape[0]))
            if cam1 is not None:
                lengths.append(int(cam1.shape[0]))
            if not lengths:
                logger.error("No valid camera arrays found to visualize.")
                return

            num_frames = min(lengths)
            start_idx = max(0, start)
            end_idx = num_frames if end is None else max(0, min(end, num_frames))

            logger.info(
                f"Visualizing frames [{start_idx}, {end_idx}) step={step}, wait={wait_ms}ms | "
                f"cam0={cam0_key if cam0 is not None else 'disabled'} | "
                f"cam1={cam1_key if cam1 is not None else 'disabled'}"
            )

            for frame_index in range(start_idx, end_idx, max(1, step)):
                if cam0 is not None:
                    frame0 = cam0[frame_index]
                    if not isinstance(frame0, np.ndarray):
                        frame0 = np.asarray(frame0)
                    # Convert RGB -> BGR for OpenCV display if shape matches HxWx3
                    if frame0.ndim == 3 and frame0.shape[-1] == 3:
                        frame0 = cv2.cvtColor(frame0, cv2.COLOR_RGB2BGR)
                    cv2.imshow("camera0_rgb", frame0)

                if cam1 is not None:
                    frame1 = cam1[frame_index]
                    if not isinstance(frame1, np.ndarray):
                        frame1 = np.asarray(frame1)
                    # Convert RGB -> BGR for OpenCV display if shape matches HxWx3
                    if frame1.ndim == 3 and frame1.shape[-1] == 3:
                        frame1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2BGR)
                    cv2.imshow("camera1_rgb", frame1)

                key = cv2.waitKey(wait_ms) & 0xFF
                if key in (27, ord('q')):  # ESC or 'q'
                    break

    except Exception as exc:  # pragma: no cover - runtime errors vary by data
        logger.exception(f"Failed to visualize zarr cameras: {exc}")
    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize camera streams from ReplayBuffer zarr.zip using OpenCV."
    )
    parser.add_argument(
        "zarr_zip",
        type=str,
        help="Path to episodes .zarr.zip",
    )
    parser.add_argument("--start", type=int, default=0, help="Start frame index (inclusive)")
    parser.add_argument("--end", type=int, default=None, help="End frame index (exclusive)")
    parser.add_argument("--step", type=int, default=1, help="Visualize every N frames")
    parser.add_argument("--wait", type=int, default=1, help="cv2.waitKey wait in milliseconds per frame")
    parser.add_argument(
        "--cams",
        type=str,
        default="both",
        choices=["both", "0", "1"],
        help="Which camera(s) to show",
    )
    parser.add_argument(
        "--cam0-key",
        type=str,
        default=None,
        help="Override dataset path for camera0 (e.g., data/cam_left_rgb)",
    )
    parser.add_argument(
        "--cam1-key",
        type=str,
        default=None,
        help="Override dataset path for camera1 (e.g., data/cam_right_rgb)",
    )
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    show_cam0 = args.cams in ("both", "0")
    show_cam1 = args.cams in ("both", "1")

    visualize_zarr_cameras(
        zarr_zip_path=args.zarr_zip,
        start=args.start,
        end=args.end,
        step=args.step,
        wait_ms=args.wait,
        show_cam0=show_cam0,
        show_cam1=show_cam1,
        cam0_key_override=args.cam0_key,
        cam1_key_override=args.cam1_key,
    )


if __name__ == "__main__":
    main()


