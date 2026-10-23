"""Static and Node contracts for the standalone recording replay page."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "templates" / "recordings.html"
CONTROL_TEMPLATE = ROOT / "templates" / "device_control.html"
SCRIPT = ROOT / "static" / "recording_replay.js"
ARM_VIEWER = ROOT / "static" / "arm_viewer.js"
STYLES = ROOT / "static" / "recording_replay.css"
NODE_TEST = ROOT / "tests" / "recording_replay.test.mjs"


def test_control_and_recording_pages_have_bidirectional_navigation() -> None:
    control_template = CONTROL_TEMPLATE.read_text(encoding="utf-8")
    replay_template = TEMPLATE.read_text(encoding="utf-8")

    assert 'id="recordings-navigation"' in control_template
    assert 'href="/recordings"' in control_template
    assert '>录制回放</span>' in control_template
    assert 'id="control-panel-navigation"' in replay_template
    assert 'href="/"' in replay_template
    assert '>控制台</span>' in replay_template


def test_recording_replay_is_an_independent_page_module() -> None:
    template = TEMPLATE.read_text(encoding="utf-8")
    script = SCRIPT.read_text(encoding="utf-8")

    assert "device_control.js" not in template
    assert "device_control.css" not in template
    assert "recording_replay.css" in template
    assert "recording_replay.js" in template
    assert "v='20260806-select-accepted-v1'" in template
    assert "<title>NEDF 录制检查</title>" in template
    assert '<p class="eyebrow" id="recording-topic-format">NEDF</p>' in template
    assert 'id="recording-episode-list"' in template
    assert 'id="recording-arm-viewer-root"' in template
    assert 'id="recording-dense-track"' in template
    assert 'id="recording-topic-list"' in template
    assert 'id="recording-camera-role-health"' in template
    assert 'id="recording-scrubber"' in template
    assert 'id="recording-speed"' in template
    assert 'class="timeline-transport"' in template
    assert 'id="recordings-page-previous"' in template
    assert 'id="recordings-page-next"' in template
    assert 'id="recordings-select-accepted"' in template
    assert '>全部接受</button>' in template
    assert 'id="recording-export-dialog"' in template
    assert 'id="recording-export-submit"' in template
    assert 'id="recording-delete-open"' in template
    assert 'id="recording-delete-dialog"' in template
    assert 'id="recording-delete-confirm"' in template
    assert 'href="/"' in template
    assert 'const RECORDINGS_URL = "/api/recordings";' in script
    assert 'const RECORDING_EXPORTS_URL = "/api/recording-exports";' in script
    assert "selectedExportIds: new Set()" in script
    assert "selectedItems: new Map()" in script
    assert 'method: "DELETE"' in script
    assert 'JSON.stringify({ episode_ids: episodeIds, confirm: true })' in script
    assert "recordingItemExportEligible" in script
    assert "collectSuccessfulRecordingItems" in script
    assert "loadExportState" in script
    assert "normalizeCameraRoles(payload.camera_roles)" in script
    assert 'safeString(payload.data_format, "NEDF3", 16).toUpperCase()' in script
    assert 'factRow("格式", summary.dataFormat)' in script
    assert "elements.topicFormat.textContent = summary.dataFormat" in script
    assert "`${summary.dataFormat} 视频缓存不可用`" in script
    assert "cameraRoleIssues(cameraRoles)" in script
    assert '["Device", camera.deviceKey]' in script
    assert '["Topic", camera.topic]' in script
    assert '["SHM", camera.shmName]' in script
    assert '"未配置映射"' in script
    assert "const warning = cameraRoleWarning(camera.reason);" in script
    assert "CAMERA_ROLE_STATUS_LABELS[camera.status]" in script


def test_recording_replay_uses_the_shared_dark_theme() -> None:
    template = TEMPLATE.read_text(encoding="utf-8")
    styles = STYLES.read_text(encoding="utf-8")

    assert '<meta name="color-scheme" content="dark">' in template
    assert "color-scheme: dark" in styles
    assert "--bg: #000000" in styles
    assert "--panel: #090909" in styles
    assert "background: #fff" not in styles


def test_replay_uses_recorded_tcp_directly_and_never_runs_ik() -> None:
    script = SCRIPT.read_text(encoding="utf-8")
    template = TEMPLATE.read_text(encoding="utf-8")

    assert "robotStateUrl: null" in script
    assert "homePoseCandidatesUrl: null" in script
    assert "/api/robots/state" not in script
    assert "/api/home-pose-candidates" not in script
    assert "inverseKinematics" not in script
    assert "不会使用 TCP 做 IK" in script
    assert "/playback-trajectory?" in script
    assert "viewer.setRecordedTcpTrajectories?.(" in script
    assert "viewer.updateRecordedTcpTrails?.(" in script
    assert "viewer.updateTcpPoses(" in script
    assert "viewer.updateGripperPositions?.({" in script
    assert "sample.grippers?.leftAvailable" in script
    assert "summary?.visualization?.configuredTools" in script
    assert "viewer.setArmModelsVisible?.(jointsAvailable)" in script
    assert "sample.tcp?.leftAvailable || sample.tcp?.rightAvailable" in script
    assert "dimension === JOINT_DIMENSION" in script
    for role in ("head", "left", "right"):
        assert f'id="recording-video-{role}"' in template
        assert f'data-video-role="{role}"' in template


def test_replay_performance_and_cleanup_contracts_are_explicit() -> None:
    script = SCRIPT.read_text(encoding="utf-8")

    assert "const TRAJECTORY_CACHE_ENTRIES = 3;" in script
    assert "const MAX_PLAYBACK_FPS = 30;" in script
    assert "const SCRUB_DEBOUNCE_MS = 100;" in script
    assert "const DENSE_DOM_NODE_LIMIT = 1000;" in script
    assert "const CACHE_POLL_DELAYS_MS = Object.freeze([250, 500, 1000, 2000]);" in script
    assert 'import { DualArmViewer } from "./arm_viewer.js"' not in script
    assert 'import("./arm_viewer.js?v=20260726-recording-gripper-v4")' in script
    assert "async function ensureViewer" in script
    assert 'statusMode: "external"' in script
    assert "await viewer.ready;" in script
    assert "planTrajectoryRequest" in script
    assert "interpolateTrajectoryFrame" in script
    assert "normalizeDenseCheckpoint" in script
    assert "planDenseTimeline" in script
    assert "scheduleScrubFrame" in script
    assert "abortTrajectorySlot" in script
    assert "trajectoryPrefetch" in script
    assert "preserveTrajectory" in script
    assert "pollCacheOperation" in script
    assert "waitForCacheRetry" in script
    assert "const summaryController = new AbortController();" in script
    assert "app.summaryController = summaryController;" in script
    assert "signal: app.summaryController.signal" not in script
    assert 'stateError.cacheState !== "building"' in script
    assert 'error.cacheState === "failed"' in script
    assert '"正在准备快速回放，请稍候…"' in script
    assert "可刷新或重新选择 episode 重试" in script
    assert 'scheduleScrubFrame(Number(elements.scrubber.value))' in script
    assert 'showFrame(Number(elements.scrubber.value))' not in script
    assert "preserveTrajectory: preserveSelection && previousSelectedId === app.selectedId" in script
    assert 'document.visibilityState === "hidden"' in script
    assert 'window.addEventListener("pagehide"' in script
    assert "viewer?.dispose()" in script
    assert "abortTrajectoryRequests()" in script
    assert "this.chunks.delete(this.chunks.keys().next().value)" in script
    assert "this.generation === generation" in script


def test_replay_distinguishes_unavailable_root_from_an_empty_directory() -> None:
    script = SCRIPT.read_text(encoding="utf-8")

    assert "payload.root_available === true" in script
    assert "payload.root_available == null && rootConfigured" in script
    assert 'rootState === "missing" && rootConfigured' in script
    assert '"unreadable"' in script
    assert '"录制目录尚未创建"' in script
    assert '"录制目录暂不可用"' in script
    assert '"录制目录暂不可用。"' in script


def test_replay_layout_has_stable_desktop_and_mobile_tracks() -> None:
    styles = STYLES.read_text(encoding="utf-8")

    assert "grid-template-columns: minmax(260px, 300px) minmax(480px, 1fr) minmax(220px, 270px);" in styles
    assert "grid-column: 2 / 4;" in styles
    assert "--timeline-control-column: 142px" in styles
    assert styles.count(
        "grid-template-columns: var(--timeline-control-column) minmax(0, 1fr)"
    ) == 2
    assert "height: calc(100dvh - var(--topbar-height));" in styles
    assert "@media (max-width: 980px)" in styles
    assert "grid-template-columns: minmax(0, 1fr);" in styles
    assert ".viewer-unavailable[hidden] { display: none; }" in styles
    assert ".dense-checkpoint {" in styles


def test_tcp_only_replay_frames_the_recorded_path_instead_of_hidden_arms() -> None:
    source = ARM_VIEWER.read_text(encoding="utf-8")

    assert "const allPoints = [];" in source
    assert "allPoints.push(...points);" in source
    assert "this.recordedTcpScenePoints = allPoints;" in source
    assert "!this.armModelsVisible && this.recordedTcpScenePoints.length" in source
    assert "this._frameRecordedTcpTrajectories(this.recordedTcpScenePoints);" in source
    assert "if (this.armModelsVisible) this._frameModel();" in source
    assert "else this._frameRecordedTcpTrajectories(allPoints);" in source
    assert "_frameRecordedTcpTrajectories(points)" in source
    assert 'this._invalidateRender("frame-recorded-tcp")' in source


def test_recording_replay_node_contracts() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is not installed")
    completed = subprocess.run(
        [node, "--test", str(NODE_TEST)],
        cwd=ROOT.parent,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
