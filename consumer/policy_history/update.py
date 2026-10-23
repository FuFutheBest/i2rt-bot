#!/usr/bin/env python3
"""
Policy history helper mixins.

Author: OpenAI Assistant
"""

from __future__ import annotations

import time
from typing import Dict, Optional

import numpy as np

from utils.logger_config import logger


class PolicyHistoryUpdateMixin:
    """Mixin containing a cohesive subset of policy-history behavior."""

    def maybe_update_policy_history(
        self,
        latest_obs_time: int,
        block: bool = True,
        current_obs: Optional[Dict[str, np.ndarray]] = None,
    ) -> bool:
        """Update the policy history cache when a pending history spec is ready."""
        owner = self.owner
        history_spec = owner._pending_policy_history_spec
        if history_spec is None:
            return True
        if not hasattr(owner.policy, "update_history_with_observations"):
            self.clear_state()
            return True

        keyframe_after_steps = [
            int(step) for step in history_spec.get("keyframe_after_steps", [])
        ]
        if not keyframe_after_steps:
            self.clear_state()
            return True

        # In "arrival" anchor mode, anchor both image and pose timelines from
        # the moment the arm reaches the last dispatched action.
        if (
            str(history_spec.get("anchor_mode", "now")) == "arrival"
            and "base_timestamp_ns" not in history_spec
        ):
            now_ns = time.time_ns()
            last_action_dispatched_timestamp_ns = (
                self.resolve_last_dispatch_timestamp(history_spec)
            )
            if (
                last_action_dispatched_timestamp_ns is None
                or int(latest_obs_time) <= int(last_action_dispatched_timestamp_ns)
            ):
                return False

            if owner._policy_history_arrival_wait_start_ns is None:
                owner._policy_history_arrival_wait_start_ns = now_ns

            reached = True
            if hasattr(owner.policy, "is_history_anchor_reached"):
                reached = bool(owner.policy.is_history_anchor_reached(current_obs or {}))

            timeout_ns = int(history_spec.get("arrival_timeout_ns", 0))
            timed_out = (
                timeout_ns > 0
                and now_ns - owner._policy_history_arrival_wait_start_ns >= timeout_ns
            )
            if not reached and timed_out:
                if not history_spec.get("_arrival_timeout_warned"):
                    logger.warning(
                        "PolicyConnector: Arm did not reach the last action pose within "
                        f"{timeout_ns / 1e9:.2f}s; continuing to wait for arrival."
                    )
                    history_spec["_arrival_timeout_warned"] = True

            if not reached:
                return False

            keyframe_interval_ns = self.resolve_keyframe_interval_ns(history_spec)
            history_spec["base_timestamp_ns"] = int(latest_obs_time) - (
                int(max(keyframe_after_steps)) * keyframe_interval_ns
            )
            if (
                history_spec.get("pose_after_steps")
                and history_spec.get("pose_interval_ns")
            ):
                pose_interval_ns = int(history_spec["pose_interval_ns"])
                pose_steps = [int(s) for s in history_spec["pose_after_steps"]]
                history_spec["pose_base_timestamp_ns"] = int(latest_obs_time) - (
                    int(max(pose_steps)) * pose_interval_ns
                )
            owner._policy_history_arrival_wait_start_ns = None
        else:
            if (
                history_spec.get("pose_after_steps")
                and history_spec.get("pose_interval_ns")
                and "pose_base_timestamp_ns" not in history_spec
            ):
                history_spec["pose_base_timestamp_ns"] = int(
                    history_spec["base_timestamp_ns"]
                )

        tolerance_ns = int(float(history_spec.get("tolerance_ms", 80.0)) * 1_000_000)
        keyframe_interval_ns = self.resolve_keyframe_interval_ns(history_spec)
        last_target_timestamp = max(
            self.resolve_keyframe_target_timestamp_ns(
                base_timestamp_ns=int(history_spec["base_timestamp_ns"]),
                step=step,
                interval_ns=keyframe_interval_ns,
                frame_index=idx,
                frame_count=len(keyframe_after_steps),
                history_kind="history_image",
                history_spec=history_spec,
            )
            for idx, step in enumerate(keyframe_after_steps)
        )
        if latest_obs_time + tolerance_ns < last_target_timestamp:
            return False

        if owner.history_manual_selection_enabled:
            manual_history = self.manual_select_policy_history(history_spec)
            if manual_history is None:
                return False
            keyframe_observations, pose_observations = manual_history
        else:
            keyframe_observations = self.collect_policy_history_observations(
                history_spec,
                block=block,
            )
            if keyframe_observations is None:
                return False

            pose_observations = self.collect_policy_history_poses(history_spec, block=block)

        self.save_policy_history_keyframe_images(history_spec, keyframe_observations)

        kv_cache_update_logged = False
        try:
            kv_cache_call_start_ns = time.time_ns()
            owner.policy.update_history_with_observations(
                keyframe_observations,
                pose_observations=pose_observations,
            )
            owner._maybe_log_policy_history_relative_xyz(history_spec)
            kv_cache_call_end_ns = time.time_ns()
            kv_cache_summary = getattr(
                owner.policy,
                "_last_kv_cache_update_summary",
                {},
            )
            kv_cache_update_start_ns = int(
                kv_cache_summary.get("trigger_timestamp_ns", kv_cache_call_start_ns)
            )
            kv_cache_update_end_ns = int(
                kv_cache_summary.get("end_timestamp_ns", kv_cache_call_end_ns)
            )
            kv_cache_update_duration_ns = int(
                kv_cache_summary.get(
                    "duration_ns",
                    kv_cache_update_end_ns - kv_cache_update_start_ns,
                )
            )
            owner.action_state_xyz_logger.log_meta(
                "kv_cache_update_start",
                timestamp_ns=kv_cache_update_start_ns,
                device_name="policy_kv_cache",
                history_frames=int(
                    kv_cache_summary.get("history_frames", len(keyframe_observations))
                ),
                pose_frames=int(
                    kv_cache_summary.get(
                        "pose_frames",
                        len(pose_observations) if pose_observations else 0,
                    )
                ),
                state_source=kv_cache_summary.get("state_source"),
                state_shape=kv_cache_summary.get("state_shape"),
                policy_connector_call_start=kv_cache_call_start_ns,
                history_base_timestamp_ns=history_spec.get("base_timestamp_ns"),
                history_pose_base_timestamp_ns=history_spec.get("pose_base_timestamp_ns"),
            )
            owner.action_state_xyz_logger.log_meta(
                "kv_cache_update_end",
                timestamp_ns=kv_cache_update_end_ns,
                device_name="policy_kv_cache",
                history_frames=int(
                    kv_cache_summary.get("history_frames", len(keyframe_observations))
                ),
                pose_frames=int(
                    kv_cache_summary.get(
                        "pose_frames",
                        len(pose_observations) if pose_observations else 0,
                    )
                ),
                kv_cache_update_start=kv_cache_update_start_ns,
                kv_cache_update_end=kv_cache_update_end_ns,
                kv_cache_update_duration_ns=kv_cache_update_duration_ns,
                policy_connector_call_start=kv_cache_call_start_ns,
                policy_connector_call_end=kv_cache_call_end_ns,
                state_source=kv_cache_summary.get("state_source"),
                state_shape=kv_cache_summary.get("state_shape"),
                success=bool(kv_cache_summary.get("success", True)),
                history_base_timestamp_ns=history_spec.get("base_timestamp_ns"),
                history_pose_base_timestamp_ns=history_spec.get("pose_base_timestamp_ns"),
            )
            kv_cache_update_logged = True
            owner._maybe_log_policy_history_relative_xyz(history_spec)
        except Exception as exc:
            if not kv_cache_update_logged:
                kv_cache_call_end_ns = time.time_ns()
                kv_cache_summary = getattr(
                    owner.policy,
                    "_last_kv_cache_update_summary",
                    {},
                )
                kv_cache_update_start_ns = int(
                    kv_cache_summary.get(
                        "trigger_timestamp_ns",
                        locals().get("kv_cache_call_start_ns", kv_cache_call_end_ns),
                    )
                )
                kv_cache_update_end_ns = int(
                    kv_cache_summary.get("end_timestamp_ns", kv_cache_call_end_ns)
                )
                kv_cache_update_duration_ns = int(
                    kv_cache_summary.get(
                        "duration_ns",
                        kv_cache_update_end_ns - kv_cache_update_start_ns,
                    )
                )
                owner.action_state_xyz_logger.log_meta(
                    "kv_cache_update_start",
                    timestamp_ns=kv_cache_update_start_ns,
                    device_name="policy_kv_cache",
                    history_frames=int(
                        kv_cache_summary.get("history_frames", len(keyframe_observations))
                    ),
                    pose_frames=int(
                        kv_cache_summary.get(
                            "pose_frames",
                            len(pose_observations) if pose_observations else 0,
                        )
                    ),
                    state_source=kv_cache_summary.get("state_source"),
                    state_shape=kv_cache_summary.get("state_shape"),
                    success=False,
                    exception_type=type(exc).__name__,
                    exception_message=str(exc),
                    policy_connector_call_start=locals().get("kv_cache_call_start_ns"),
                    history_base_timestamp_ns=history_spec.get("base_timestamp_ns"),
                    history_pose_base_timestamp_ns=history_spec.get("pose_base_timestamp_ns"),
                )
                owner.action_state_xyz_logger.log_meta(
                    "kv_cache_update_end",
                    timestamp_ns=kv_cache_update_end_ns,
                    device_name="policy_kv_cache",
                    history_frames=int(
                        kv_cache_summary.get("history_frames", len(keyframe_observations))
                    ),
                    pose_frames=int(
                        kv_cache_summary.get(
                            "pose_frames",
                            len(pose_observations) if pose_observations else 0,
                        )
                    ),
                    kv_cache_update_start=kv_cache_update_start_ns,
                    kv_cache_update_end=kv_cache_update_end_ns,
                    kv_cache_update_duration_ns=kv_cache_update_duration_ns,
                    success=False,
                    exception_type=type(exc).__name__,
                    exception_message=str(exc),
                    policy_connector_call_start=locals().get("kv_cache_call_start_ns"),
                    policy_connector_call_end=kv_cache_call_end_ns,
                    state_source=kv_cache_summary.get("state_source"),
                    state_shape=kv_cache_summary.get("state_shape"),
                    history_base_timestamp_ns=history_spec.get("base_timestamp_ns"),
                    history_pose_base_timestamp_ns=history_spec.get("pose_base_timestamp_ns"),
                )
            logger.warning(
                f"PolicyConnector: Stateful history update failed ({exc}); falling back to stateless recovery."
            )
            if hasattr(owner.policy, "recover_stateless"):
                owner.policy.recover_stateless()
        finally:
            self.clear_state()
        return True

