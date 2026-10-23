#!/usr/bin/env python3
"""Inspect an R1 NEDF3 episode saved by incubator/r1/async_data_saver_NEDF3_r1.py.

Reads the R1 HAL schema (nmx/hal/arm/{left,right}/tcp/state, nmx/hal/head/state,
nmx/hal/hand/{left,right}/state, nmx/hal/camera/camera_{0,1,2}/rgbd) back through
``nmx_nedf_api`` and either:

  * plays the 3 cameras side-by-side, time-synchronized, with the left / right /
    ego EEF pose + gripper width overlaid (default, needs a display), or
  * prints a headless integrity summary (``--summary``): per-topic message
    counts, duration / rate, first&last pose, gripper range, monotonic-timestamp
    check -- useful to confirm "is my recorded data correct" over SSH.

Episodes are named by UUID (e.g. saved_data/<uuid>/ or saved_data/<task>/<uuid>/).

Usage:
    # GUI replay of an episode
    python3 scripts/diagnostics/runtime_profiles.py run \
        --config-name quest3_collect_r1 --role recording -- \
        -m incubator.r1.inspect_nedf3_episode saved_data/<uuid>

    # Headless integrity check (no window)
    python3 scripts/diagnostics/runtime_profiles.py run \
        --config-name quest3_collect_r1 --role recording -- \
        -m incubator.r1.inspect_nedf3_episode saved_data/<uuid> --summary

GUI keys:  SPACE pause/resume | , / . step back/fwd (paused) | r restart |
           +/- speed | q or ESC quit
"""

import argparse
import bisect
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    from nmx_nedf_api import NEDFFactory, NEDFReaderConfig
except Exception as exc:  # pragma: no cover
    print(
        f"ERROR: cannot import nmx_nedf_api ({exc}). "
        "Run through the verified recording runtime profile."
    )
    sys.exit(1)


# R1 (NEDF3) HAL schema written by incubator/r1/async_data_saver_NEDF3_r1.py. Rather than
# hard-coding the exact topic set (which breaks the moment the camera count or
# arm set changes), we DISCOVER the topics actually present in the episode and
# match them by these patterns. So any number of cameras / arms in the file works.
#   poses use the MEASURED tcp/state (ArmTcpState.pose); "ego" = YAM head arm.
#   gripper width = HandState.joints[gripper].pos (head arm has no gripper).
CAMERA_RE = re.compile(r"^nmx/hal/camera/([^/]+)/rgbd$")
ARM_POSE_RE = re.compile(r"^nmx/hal/arm/([^/]+)/tcp/state$")
HEAD_POSE_RE = re.compile(r"^nmx/hal/head/state$")
HAND_RE = re.compile(r"^nmx/hal/hand/([^/]+)/state$")

# Best-effort human labels for the known R1 camera slots; unknown ids fall back
# to the raw id. Purely cosmetic -- presence is decided by what's in the file.
CAMERA_ROLE = {
    "camera_0": "right_wrist",
    "camera_1": "left_wrist",
    "camera_2": "head/ego",
}
# Preferred ordering for arm-pose readouts; any extra arms are appended.
POSE_ORDER = ["left", "right", "ego"]


def _camera_label(cam_id: str) -> str:
    role = CAMERA_ROLE.get(cam_id)
    return f"{cam_id} ({role})" if role else cam_id


def _discover_topics(reader) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
    """Read the episode's actual topic list and split it into
    (cameras{id->topic}, poses{side->topic}, hands{side->topic}) by pattern."""
    cameras: Dict[str, str] = {}
    poses: Dict[str, str] = {}
    hands: Dict[str, str] = {}
    for topic in reader.get_raw_topics_timestamps().keys():
        m = CAMERA_RE.match(topic)
        if m:
            cameras[m.group(1)] = topic
            continue
        m = ARM_POSE_RE.match(topic)
        if m:
            poses[m.group(1)] = topic
            continue
        if HEAD_POSE_RE.match(topic):
            poses["ego"] = topic
            continue
        m = HAND_RE.match(topic)
        if m:
            hands[m.group(1)] = topic
    return cameras, poses, hands


def _camera_display_order(cameras: Dict[str, str]) -> List[str]:
    """Wrist cams first (sorted by id), head/ego last, to read left|right|head."""
    def _key(cam_id: str):
        role = CAMERA_ROLE.get(cam_id, "")
        is_head = "head" in role or "ego" in role
        return (is_head, cam_id)
    return sorted(cameras, key=_key)


def _pose_order(poses: Dict[str, str]) -> List[str]:
    ordered = [s for s in POSE_ORDER if s in poses]
    ordered += [s for s in poses if s not in POSE_ORDER]
    return ordered


def _resolve_metadata(path: Path) -> Path:
    """Accept an episode dir or a metadata.json path; return the metadata file."""
    if path.is_dir():
        meta = path / "metadata.json"
        if not meta.exists():
            raise FileNotFoundError(f"No metadata.json in {path}")
        return meta
    return path


def _open(episode: Path):
    meta = _resolve_metadata(episode)
    reader = NEDFFactory.get_reader(NEDFReaderConfig(metadata_file_path=meta))
    return reader


def _pose7_from_msg(decoded) -> Tuple[float, ...]:
    p = decoded.pose.pos
    q = decoded.pose.quat  # w, x, y, z
    return (p.x, p.y, p.z, q.w, q.x, q.y, q.z)


def _gripper_from_state(decoded) -> Optional[float]:
    for j in decoded.joints:
        return float(j.pos)  # single 'gripper' joint; None when empty
    return None


def _load_camera(reader, topic: str) -> List[Tuple[int, np.ndarray]]:
    """Return [(ts_ns, jpg_uint8_array)], decoding lazily at display time."""
    out = []
    for msg in reader.iter_decoded_messages(topic):
        jpg = np.frombuffer(msg.decoded_message.rgb.data, dtype=np.uint8)
        out.append((int(msg.message.log_time), jpg))
    out.sort(key=lambda x: x[0])
    return out


def _load_series(reader, topic: str, extractor) -> List[Tuple[int, object]]:
    out = []
    try:
        for msg in reader.iter_decoded_messages(topic):
            val = extractor(msg.decoded_message)
            out.append((int(msg.message.log_time), val))
    except Exception:
        return []
    out.sort(key=lambda x: x[0])
    return out


def _nearest(series: List[Tuple[int, object]], ts: int):
    if not series:
        return None
    keys = [s[0] for s in series]
    i = bisect.bisect_left(keys, ts)
    if i == 0:
        return series[0][1]
    if i >= len(series):
        return series[-1][1]
    before = series[i - 1]
    after = series[i]
    return before[1] if (ts - before[0]) <= (after[0] - ts) else after[1]


# ---------------------------------------------------------------------------
# Summary mode
# ---------------------------------------------------------------------------

def run_summary(episode: Path) -> int:
    reader = _open(episode)
    print(f"Episode: {episode}")
    try:
        print(f"  data_format: {reader.metadata.data_format}")
    except Exception:
        pass
    print(f"  check_valid(): {reader.check_valid()}")

    topics = reader.get_raw_topics_timestamps()
    if not topics:
        print("  NO TOPICS FOUND -- episode is empty/corrupt.")
        return 1

    cameras, pose_topics, hand_topics = _discover_topics(reader)

    t0 = min(ts[0] for ts in topics.values() if ts)
    t1 = max(ts[-1] for ts in topics.values() if ts)
    dur = (t1 - t0) / 1e9
    print(f"  duration: {dur:.2f} s   ({len(topics)} topics)")
    print("  per-topic:")
    for topic in sorted(topics):
        ts = topics[topic]
        n = len(ts)
        rate = (n - 1) / ((ts[-1] - ts[0]) / 1e9) if n > 1 else 0.0
        mono = all(ts[i] <= ts[i + 1] for i in range(n - 1))
        flag = "" if mono else "  <-- NON-MONOTONIC!"
        print(f"    {topic:40s} n={n:5d}  ~{rate:6.1f} Hz{flag}")

    print("  poses (first/last):")
    for side in _pose_order(pose_topics):
        series = _load_series(reader, pose_topics[side], _pose7_from_msg)
        if not series:
            print(f"    {side:5s}: (none)")
            continue
        f = series[0][1]
        l = series[-1][1]
        print(f"    {side:5s}: first xyz=({f[0]:.3f},{f[1]:.3f},{f[2]:.3f}) "
              f"last xyz=({l[0]:.3f},{l[1]:.3f},{l[2]:.3f})  n={len(series)}")

    print("  gripper width (min/max):")
    for side, topic in sorted(hand_topics.items()):
        series = _load_series(reader, topic, _gripper_from_state)
        vals = [v for _, v in series if v is not None]
        if not vals:
            print(f"    {side:5s}: (empty / no gripper)")
        else:
            print(f"    {side:5s}: min={min(vals):.4f} max={max(vals):.4f} n={len(vals)}")

    print("  cameras:")
    for cam in _camera_display_order(cameras):
        n = len(topics.get(cameras[cam], []))
        print(f"    {_camera_label(cam):28s} frames={n}")
    return 0


# ---------------------------------------------------------------------------
# GUI replay
# ---------------------------------------------------------------------------

def _overlay_text(img, lines, org=(8, 22), color=(0, 255, 0)):
    y = org[1]
    for ln in lines:
        cv2.putText(img, ln, (org[0], y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, ln, (org[0], y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    color, 1, cv2.LINE_AA)
        y += 20


def run_gui(episode: Path, height: int, speed: float) -> int:
    reader = _open(episode)
    print(f"Loading {episode} ...")

    camera_topics, pose_topics, hand_topics = _discover_topics(reader)
    if not camera_topics:
        print("No camera topics found in this episode; use --summary for a headless check.")
        return 1
    display_order = _camera_display_order(camera_topics)

    cams = {cam: _load_camera(reader, topic) for cam, topic in camera_topics.items()}
    poses = {s: _load_series(reader, t, _pose7_from_msg) for s, t in pose_topics.items()}
    grips = {s: _load_series(reader, t, _gripper_from_state) for s, t in hand_topics.items()}

    # Master clock = whichever camera has the most frames.
    master_cam = max(cams, key=lambda c: len(cams[c]))
    master = cams[master_cam]
    if not master:
        print("No camera frames found; use --summary for a headless check.")
        return 1
    t0 = master[0][0]
    n = len(master)
    print(f"{n} master frames ({_camera_label(master_cam)}), "
          f"{(master[-1][0]-t0)/1e9:.1f}s. SPACE=pause q=quit ,/.=step +/-=speed")

    win = f"NEDF3 inspection - {Path(episode).name}"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    i = 0
    paused = False
    last_wall = time.time()
    while True:
        ts = master[i][0]
        tiles = []
        for cam in display_order:
            frame = _nearest(cams[cam], ts)
            if frame is None:
                tile = np.zeros((height, int(height * 4 / 3), 3), np.uint8)
            else:
                img = cv2.imdecode(frame, cv2.IMREAD_COLOR)
                h, w = img.shape[:2]
                tile = cv2.resize(img, (int(w * height / h), height))
            _overlay_text(tile, [_camera_label(cam)], color=(0, 200, 255))
            tiles.append(tile)
        # pad tiles to same height (already equal) and concat
        composite = cv2.hconcat(tiles)

        # pose + gripper overlay
        lines = [f"t={(ts - t0)/1e9:6.2f}s  frame {i+1}/{n}  speed x{speed:.2f}"
                 f"{'  [PAUSED]' if paused else ''}"]
        for s in _pose_order(pose_topics):
            p = _nearest(poses[s], ts)
            g = _nearest(grips.get(s, []), ts)
            if p is None:
                lines.append(f"{s:5s}: (no pose)")
            else:
                gtxt = f" grip={g:.3f}" if g is not None else ""
                lines.append(f"{s:5s} xyz=({p[0]:+.3f},{p[1]:+.3f},{p[2]:+.3f}) "
                             f"q=({p[3]:+.2f},{p[4]:+.2f},{p[5]:+.2f},{p[6]:+.2f}){gtxt}")
        _overlay_text(composite, lines, org=(8, composite.shape[0] - 96),
                      color=(0, 255, 0))

        cv2.imshow(win, composite)

        # pacing
        if not paused and i + 1 < n:
            dt = (master[i + 1][0] - ts) / 1e9 / max(speed, 1e-3)
        else:
            dt = 0.03
        wait_ms = max(1, int(dt * 1000))
        key = cv2.waitKey(wait_ms) & 0xFF

        if key in (ord('q'), 27):
            break
        elif key == ord(' '):
            paused = not paused
        elif key == ord('r'):
            i = 0
        elif key == ord('.'):
            paused = True
            i = min(n - 1, i + 1)
        elif key == ord(','):
            paused = True
            i = max(0, i - 1)
        elif key in (ord('+'), ord('=')):
            speed = min(16.0, speed * 1.5)
        elif key == ord('-'):
            speed = max(0.0625, speed / 1.5)
        elif not paused:
            i += 1
            if i >= n:
                paused = True
                i = n - 1

        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Inspect/verify a NEDF3 (R1) episode")
    ap.add_argument("episode", help="episode dir (or metadata.json), e.g. saved_data/<uuid>")
    ap.add_argument("--summary", action="store_true", help="headless integrity check, no window")
    ap.add_argument("--height", type=int, default=480, help="per-camera tile height (px)")
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed multiplier")
    args = ap.parse_args()

    episode = Path(args.episode).expanduser()
    if not episode.exists():
        print(f"ERROR: {episode} does not exist")
        return 1

    if args.summary:
        return run_summary(episode)
    return run_gui(episode, height=args.height, speed=args.speed)


if __name__ == "__main__":
    sys.exit(main())
