#!/usr/bin/env python3
"""
Policy history helper mixins.

Author: OpenAI Assistant
"""

from __future__ import annotations

import html
import time
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, unquote, urlparse

import imageio.v2 as imageio
import numpy as np

from utils.logger_config import logger


class PolicyHistoryPickerMixin:
    """Mixin containing a cohesive subset of policy-history behavior."""

    def history_manual_pose_summary(self, frame_data: Dict[str, Any]) -> str:
        """Summarize robot pose values for the manual history picker."""
        owner = self.owner
        parts: List[str] = []
        for device_name, (timestamp, data) in frame_data.get("data", {}).items():
            if not owner._is_robot_state_device(device_name):
                continue
            flat = np.asarray(data, dtype=np.float64).reshape(-1)
            arm_parts: List[str] = []
            if flat.size >= 44:
                arm_parts.append(f"L=({flat[14]:.4f}, {flat[15]:.4f}, {flat[16]:.4f})")
                arm_parts.append(f"R=({flat[36]:.4f}, {flat[37]:.4f}, {flat[38]:.4f})")
            elif flat.size >= 22:
                arm_parts.append(f"({flat[14]:.4f}, {flat[15]:.4f}, {flat[16]:.4f})")
            elif flat.size >= 3:
                arm_parts.append(f"({flat[0]:.4f}, {flat[1]:.4f}, {flat[2]:.4f})")
            if arm_parts:
                parts.append(f"{device_name}@{int(timestamp)}: " + " ".join(arm_parts))
        return " | ".join(parts) if parts else "no robot state"

    def history_manual_image_preview(
        self,
        frame_data: Dict[str, Any],
        output_path: Path,
    ) -> bool:
        """Write a combined camera preview image for manual history selection."""
        owner = self.owner
        camera_names = list(owner.camera_devices)
        if not camera_names:
            camera_names = [
                key for key in frame_data.get("data", {}).keys()
                if "camera" in key.lower()
            ]
        frames: List[np.ndarray] = []
        for camera_name in camera_names:
            if camera_name not in frame_data.get("data", {}):
                continue
            frame = self.convert_history_video_frame(
                frame_data["data"][camera_name][1]
            )
            if frame is not None:
                frames.append(frame)
        if not frames:
            return False
        max_h = max(int(frame.shape[0]) for frame in frames)
        padded = []
        for frame in frames:
            pad_h = max_h - int(frame.shape[0])
            if pad_h > 0:
                frame = np.pad(frame, ((0, pad_h), (0, 0), (0, 0)), constant_values=0)
            padded.append(frame)
        try:
            imageio.imwrite(str(output_path), np.hstack(padded))
            return True
        except Exception as exc:
            logger.warning(f"PolicyConnector: Failed to write manual history preview: {exc}")
            return False

    def serve_history_manual_picker(
        self,
        html_path: Path,
        root_dir: Path,
        expected_image_count: int,
        expected_pose_count: int,
    ) -> Optional[Dict[str, List[int]]]:
        """Serve manual picker HTML and wait for selected frame indices."""
        owner = self.owner
        result: Dict[str, List[int]] = {}
        submitted = threading.Event()

        class PickerHandler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                return

            def do_GET(self) -> None:
                parsed = urlparse(self.path)
                if parsed.path == "/submit":
                    params = parse_qs(parsed.query)
                    try:
                        image_indices = [
                            int(params[f"image_{idx}"][0])
                            for idx in range(expected_image_count)
                        ]
                        pose_indices = [
                            int(params[f"pose_{idx}"][0])
                            for idx in range(expected_pose_count)
                        ]
                    except Exception as exc:
                        self.send_response(400)
                        self.end_headers()
                        self.wfile.write(f"Invalid selection: {exc}".encode("utf-8"))
                        return
                    result["image"] = image_indices
                    result["pose"] = pose_indices
                    submitted.set()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.end_headers()
                    self.wfile.write(
                        b"<html><body><h2>Selection submitted.</h2>"
                        b"<p>You can close this tab and return to SuperInference.</p>"
                        b"</body></html>"
                    )
                    return

                if parsed.path.startswith("/files/"):
                    rel = unquote(parsed.path[len("/files/") :])
                    file_path = (root_dir / rel).resolve()
                    try:
                        file_path.relative_to(root_dir.resolve())
                    except ValueError:
                        self.send_response(403)
                        self.end_headers()
                        return
                    if not file_path.exists():
                        self.send_response(404)
                        self.end_headers()
                        return
                    self.send_response(200)
                    self.send_header("Content-Type", "image/png")
                    self.end_headers()
                    self.wfile.write(file_path.read_bytes())
                    return

                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(html_path.read_bytes())

        server = ThreadingHTTPServer(("127.0.0.1", owner.history_manual_selection_port), PickerHandler)
        port = int(server.server_address[1])
        url = f"http://127.0.0.1:{port}/"
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        logger.info(f"PolicyConnector: Manual history picker waiting at {url}")
        if owner.history_manual_selection_auto_open:
            try:
                webbrowser.open(url)
            except Exception as exc:
                logger.warning(f"PolicyConnector: Failed to open browser for picker: {exc}")

        timeout = owner.history_manual_selection_timeout_s
        ok = submitted.wait(None if timeout <= 0 else timeout)
        server.shutdown()
        server.server_close()
        if not ok:
            logger.warning("PolicyConnector: Manual history picker timed out")
            return None
        return result

    @staticmethod
    def format_history_picker_time(timestamp_ns: int) -> str:
        """Format timestamp for manual picker labels."""
        return time.strftime(
            "%H:%M:%S",
            time.localtime(int(timestamp_ns) / 1e9),
        ) + f".{int(timestamp_ns / 1e6) % 1000:03d}"

    def write_history_manual_picker_html(
        self,
        *,
        html_path: Path,
        image_groups: List[Dict[str, Any]],
        pose_groups: List[Dict[str, Any]],
    ) -> None:
        """Write the manual history picker HTML page."""
        def candidate_rows(group: Dict[str, Any], prefix: str) -> str:
            rows = []
            candidates = group["candidates"]
            if not candidates:
                return '<p class="empty">No candidates in range.</p>'
            default_index = min(
                range(len(candidates)),
                key=lambda idx: abs(float(candidates[idx]["diff_ms"])),
            )
            for idx, candidate in enumerate(candidates):
                checked = " checked" if idx == default_index else ""
                preview = ""
                if candidate.get("preview"):
                    preview = (
                        f'<img src="/files/{html.escape(candidate["preview"])}" '
                        f'alt="preview {idx}">'
                    )
                pose_text = html.escape(str(candidate.get("pose_summary", "")))
                rows.append(
                    "<label class='candidate'>"
                    f"<input type='radio' name='{prefix}_{group['index']}' "
                    f"value='{int(candidate['frame_index'])}'{checked}>"
                    "<span>"
                    f"<b>{self.format_history_picker_time(int(candidate['frame_timestamp_ns']))}</b>"
                    f" | diff={float(candidate['diff_ms']):+.1f} ms"
                    f" | frame={int(candidate['frame_index'])}"
                    f"<small>{pose_text}</small>"
                    f"{preview}"
                    "</span>"
                    "</label>"
                )
            return "\n".join(rows)

        sections = []
        for group in image_groups:
            sections.append(
                "<section>"
                f"<h2>Image step {group['step']} "
                f"<small>target {self.format_history_picker_time(group['target_timestamp_ns'])}</small></h2>"
                f"{candidate_rows(group, 'image')}"
                "</section>"
            )
        for group in pose_groups:
            sections.append(
                "<section>"
                f"<h2>Pose step {group['step']} "
                f"<small>target {self.format_history_picker_time(group['target_timestamp_ns'])}</small></h2>"
                f"{candidate_rows(group, 'pose')}"
                "</section>"
            )

        html_text = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Manual History Picker</title>
  <style>
    body {{ margin: 0; font-family: sans-serif; background: #f6f7f9; color: #17202a; }}
    header {{ position: sticky; top: 0; z-index: 2; background: #fff; border-bottom: 1px solid #d7dce2; padding: 12px 16px; }}
    h1 {{ margin: 0; font-size: 20px; }}
    form {{ display: grid; gap: 12px; padding: 12px; }}
    section {{ background: #fff; border: 1px solid #d7dce2; border-radius: 8px; padding: 10px; }}
    h2 {{ margin: 0 0 8px; font-size: 15px; }}
    small {{ color: #657182; font-weight: 400; }}
    .candidate {{ display: grid; grid-template-columns: 22px 1fr; gap: 8px; padding: 8px; border-top: 1px solid #eef1f4; cursor: pointer; }}
    .candidate:first-of-type {{ border-top: 0; }}
    .candidate input {{ margin-top: 3px; }}
    .candidate span {{ display: grid; gap: 5px; }}
    .candidate img {{ max-width: min(720px, 100%); border: 1px solid #d7dce2; border-radius: 6px; background: #000; }}
    .empty {{ color: #9aa3af; }}
    .submit {{ position: sticky; bottom: 0; background: #fff; border: 1px solid #d7dce2; border-radius: 8px; padding: 10px; }}
    button {{ min-height: 40px; padding: 0 18px; border: 1px solid #17202a; border-radius: 6px; background: #17202a; color: white; font-weight: 700; }}
  </style>
</head>
<body>
  <header>
    <h1>Manual History Picker</h1>
    <small>Select one summary frame for every image and pose history step, then submit.</small>
  </header>
  <form action="/submit" method="get">
    {''.join(sections)}
    <div class="submit"><button type="submit">Use selected history</button></div>
  </form>
</body>
</html>
"""
        html_path.write_text(html_text, encoding="utf-8")

    def manual_select_policy_history(
        self,
        history_spec: Dict[str, Any],
    ) -> Optional[Tuple[List[Dict[str, np.ndarray]], Optional[List[Dict[str, np.ndarray]]]]]:
        """Run manual history selection and return selected observations."""
        owner = self.owner
        image_targets, pose_targets = self.manual_target_specs(history_spec)
        if not image_targets:
            return None

        selection_index = owner._history_manual_selection_index
        owner._history_manual_selection_index += 1
        picker_dir = owner.history_manual_selection_dir / f"selection_{selection_index:06d}"
        picker_dir.mkdir(parents=True, exist_ok=True)

        def build_group(target: Dict[str, int], kind: str) -> Dict[str, Any]:
            candidates = self.summary_frame_candidates(
                int(target["target_timestamp_ns"]),
                window_ms=owner.history_manual_selection_window_ms,
                max_candidates=owner.history_manual_selection_max_candidates,
            )
            enriched = []
            for cand_idx, candidate in enumerate(candidates):
                frame_data = self.read_summary_frame_by_physical_index(
                    int(candidate["frame_index"])
                )
                if frame_data is None:
                    continue
                item = dict(candidate)
                if kind == "image":
                    preview_name = (
                        f"{kind}_{target['frame_index']:03d}_{cand_idx:03d}.png"
                    )
                    if self.history_manual_image_preview(frame_data, picker_dir / preview_name):
                        item["preview"] = preview_name
                else:
                    item["pose_summary"] = self.history_manual_pose_summary(frame_data)
                enriched.append(item)
            return {
                "index": int(target["frame_index"]),
                "step": int(target["step"]),
                "target_timestamp_ns": int(target["target_timestamp_ns"]),
                "candidates": enriched,
            }

        image_groups = [build_group(target, "image") for target in image_targets]
        pose_groups = [build_group(target, "pose") for target in pose_targets]
        if any(not group["candidates"] for group in image_groups):
            logger.warning("PolicyConnector: Manual history picker missing image candidates")
            return None
        if any(not group["candidates"] for group in pose_groups):
            logger.warning("PolicyConnector: Manual history picker missing pose candidates")
            return None

        html_path = picker_dir / "index.html"
        self.write_history_manual_picker_html(
            html_path=html_path,
            image_groups=image_groups,
            pose_groups=pose_groups,
        )
        selection = self.serve_history_manual_picker(
            html_path,
            picker_dir,
            expected_image_count=len(image_groups),
            expected_pose_count=len(pose_groups),
        )
        if selection is None:
            return None

        keyframe_observations = self.collect_history_frames_by_indices(
            selection["image"],
            history_kind="history_image",
            history_spec=history_spec,
            target_specs=image_targets,
        )
        if keyframe_observations is None:
            return None

        pose_observations = None
        if pose_targets:
            pose_observations = self.collect_history_frames_by_indices(
                selection["pose"],
                history_kind="history_pose",
                history_spec=history_spec,
                target_specs=pose_targets,
            )
            if pose_observations is None:
                return None
        return keyframe_observations, pose_observations

