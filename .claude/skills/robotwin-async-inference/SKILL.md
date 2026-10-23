---
name: robotwin-async-inference
description: Run or modify RoboTwin's SuperInference asynchronous simulation bridge, including direct SAPIEN drives, joint14/EEF16 contracts, trace diagnostics, historical ACT, and XPolicyLab Pi_05/FastWAM checkpoints.
---

# RoboTwin Async Inference

Use this skill for the RoboTwin simulation presets under `configs/policy_infer/robotwin/config_infer_robotwin_*.yaml`. The bridge is a debugging environment for asynchronous policy inference and actuator dynamics. SuperInference owns policy inference, chunk scheduling, wall-clock latency measurement, and sim-time action dispatch; one `RoboTwinSimRobot` process owns SAPIEN, task state, cameras, and GUI.
The short operator-facing launch sequence is maintained in
[`robotwin_inference_commands.md`](../../../robotwin_inference_commands.md).

## Core Contract

- Do not call `Base_Task.take_action()` or RoboTwin planner/TOPP APIs. `RoboTwinSimRobot` does not use `BaseRobot.start_server()` or the inherited `execute_control_command()` path; its own physics loop reads control SHM, interpolates each high-level `joint14` target over exactly `1 / robotwin.control_fps` of simulation time, updates SAPIEN drive position and velocity targets at `robotwin.physics_fps`, and calls `scene.step()` without waiting for arrival.
- RoboTwin action dispatch must use `action_executor.execution_clock: sim_time`. The bridge-owned sim clock is derived from the first policy command endpoint as `<device>_<device_class_id>_sim_clock`; it is never configured as a raw SHM name. The wall clock measures real inference and transport latency; it remains the time base for `chunk_manager.drop_first_actions: auto`. The sim clock decides when SAPIEN receives the next high-level target, so a slow simulator does not compress a 15Hz action chunk in simulation time. If one executor poll crosses several sim-time deadlines, the executor consumes the corresponding old-chunk rows and dispatches only the newest row, keeping the action index aligned with the simulator.
- `joint14` is the default and checkpoint-compatible action layout: six left arm joints, left gripper, six right arm joints, right gripper. It is fixed as `float64` in control SHM.
- `robotwin.renderer_shader_dir` defaults to SAPIEN's raster `default` shader. RoboTwin's upstream setup selects `rt`, but the bridge must remain usable on non-ray-tracing GPUs. Select `rt` explicitly only on supported GPUs.
- The bridge renders only `head_camera`, `left_camera`, and `right_camera`; do not call RoboTwin's bulk `Camera.update_picture()`, which also allocates a target for the unpublished `front_camera`. Camera capture/write runs in the RoboTwin bridge's camera publisher thread, not inside the physics loop. One SAPIEN camera capture returns all three ACT camera frames, so publish the synchronized stream set from that capture and schedule the next camera tick from the capture start time. Do not let tiny per-camera due-time skew trigger repeated full-scene captures.
- Treat `robotwin.state_fps`, `robotwin.camera_fps`, `robotwin.control_fps`, and `robotwin.physics_fps` as separate clocks. `policy.fps` is only the policy query cadence; it must not silently throttle camera capture or control interpolation.
- Keep the RoboTwin `BaseChunkManager` at `drop_first_actions: auto`. `auto` uses wall-clock observation age because that is the real policy/transport delay. It is intentionally separate from sim-time dispatch; the bridge now emits a `realtime_factor` warning when sim/wall leaves the configured near-`1.0` range (the default threshold yields approximately `[0.95, 1.05]`).
- When `physics_fps / control_fps` is non-integral, sim-time dispatch lands on quantized physics ticks. Keep interpolation duration exact, but finish an almost-due active segment with a small slack capped by both physics step size and control period. This prevents false `sim_control_superseded` records without changing the average sim-time cadence.
- Policy-visible robot state in `RoboTwinSimRobot_0_data` is measured joint feedback, not the current drive target. Trace records still include both `sim_measured_state.measured_qpos` and `sim_measured_state.target_qpos` so control timing and actuator tracking can be diagnosed separately.
- Do not send zero velocity targets while interpolating a moving joint target. SAPIEN drive damping uses the velocity target, and zero velocity makes measured qpos lag badly behind the moving drive target. The bridge should pass the segment slope as the left/right arm and gripper drive velocity, matching RoboTwin `take_action()`'s `set_arm_joints(position, velocity)` contract.
- The current ACT baseline uses `policy.control_fps: 15.0`, `robotwin.physics_fps: 50.0`, `robotwin.camera_fps: 6.0`, `robotwin.gui_render_fps: 6.0`, `joint_stiffness: 2500.0`, `joint_damping: 100.0`, `gripper_stiffness: 3000.0`, and `gripper_damping: 120.0`. This trades physics resolution and camera cadence for wall-clock/sim-time alignment. Re-tune from trace metrics rather than visual impressions: a healthy run should have `sim_control_superseded=0`, low sustained loop overruns, `realtime_factor` near `1.0`, low `deadline_miss_count`, effective control frequency near the configured rate, and low p95 `measured_qpos - target_qpos` error.
- Pi_05 and FastWAM RoboTwin rollouts must use `RoboTwinXPolicyLabPolicy` and the RoboTwin `XPolicyLab` websocket contract. Do not reuse SuperInference's existing `OpenPiPolicy` or `FastWAMPolicy` for these presets: those adapters target different robot/camera coordinate conventions. The XPolicyLab adapter sends RoboTwin camera keys as `cam_head`, `cam_left_wrist`, and `cam_right_wrist`, sends measured `joint14` feedback split into XPolicyLab state fields, then flattens returned `action_type: joint` chunks back to `RoboTwinSimRobot_0` `joint14` control chunks. The official cotrain checkpoints are downloaded by `scripts/robotwin/download_xpolicylab_checkpoint.py` from the `TianxingChen/RoboTwin2.0` dataset and installed under XPolicyLab's standard `RoboTwin-<ckpt>-aloha_agilex-joint-0` run directories.
- Register the RoboTwin simulation interpreter and run launch/preflight commands
  through [`runtime-profile-management`](../runtime-profile-management/SKILL.md).
  The launcher hides user/system site packages after startup, but it cannot import
  Hydra/OmegaConf when no verified launcher profile exists.
- On GPUs with about 8 GB of VRAM, validate the weight-free replay path before learned policies. Pi_05 and especially FastWAM may exceed the practical memory floor because model restore can overlap simulator allocation.
- The RoboTwin preset uses a disjoint custom layout: DeviceStarter `[0,1]`, DeviceManager `[2,3]`, RoboTwinSimRobot `[4..15]`, and PolicyRunner `[16..31]`. Its `min_cpus: 32` gate disables affinity on smaller hosts instead of partially applying invalid CPU IDs. Re-tune or disable this layout when the host topology differs.
- Keep `robotwin.trace_timing_interval_s` enabled during performance work. It emits `sim_loop_timing` records with per-section nanosecond timings for control polling, EEF polling, target application, `scene.step()`, optional GUI render, state read/publish, runtime-lock wait, requested sleep, actual sleep, spin wait, sleep overshoot, overrun, and bounded catch-up state. Camera work emits separate `sim_camera_timing` records with total publish, SAPIEN capture, SHM write, and post-capture wait timings because it no longer runs on the physics-loop path.
- The timing trace also exposes `realtime_factor = sim_time_delta / wall_monotonic_delta`, its warning threshold, `effective_control_hz_wall`, `effective_control_hz_sim`, and the number of accepted high-level control targets. The control frequency is measured from `sim_control_accepted` timestamps, not from the physics-loop frequency: it answers how often a new target entered the bridge. The ActionExecutor trace adds `effective_action_dispatch_hz_*`, `deadline_due_count`, `deadline_miss_count`, `sim_dispatch_lateness_ns`, the number of chunk rows consumed during catch-up, and the final original action index/offset used for that dispatch. `action_dispatch_metrics.dispatch_emitted=false` records a due cycle with no SHM write and gives `dispatch_skip_reason` (`no_action_chunk`, `chunk_exhausted`, `device_delay_not_ready`, or `no_robot_write`).
- `deadline_miss_count` counts extra action deadlines crossed by one executor poll; ordinary small polling lateness is reported separately as `sim_dispatch_lateness_ns`. This keeps scheduler misses distinguishable from the expected quantization between the physics tick and the control period.
- `robotwin.viewer_shader_dir` defaults to `minimal` independently of the policy camera shader. It reduces GUI VRAM use; use `default` only when its visual fidelity is required. `viewer_resolution` defaults to `640x480` instead of SAPIEN Viewer's `1920x1080` default; raise it deliberately for inspection.
- The bridge passes SAPIEN's own conservative `renderer_max_num_materials=128` and `renderer_max_num_textures=512`, instead of RoboTwin's upstream `50000/50000` resource pool. This preserves VRAM for the policy process; increase capacities deliberately only for a scene that exhausts them.
- The shipped ACT GUI preset uses `inference_dtype: float16`: restore the historical checkpoint on CPU, convert it before the CUDA transfer, then use matching FP16 input tensors inside CUDA autocast. This keeps the `joint14` ABI unchanged and avoids peak VRAM duplication. Override to `float32` only where GPU headroom is sufficient and that numeric path is required.
- `eef16` is a separate launch-time preset: left/right `xyz + xyzw quaternion + gripper`. It requires a pure-NumPy `eef_resolver`; its worker coalesces stale targets so slow resolution never blocks physics or queues old commands.
- Virtual camera descriptors use `managed_by: RoboTwinSimRobot_0`. Do not start them as independent devices; the bridge creates all three SHMs.
- `DeviceStarter` waits for every configured stream to publish changing live frames before it starts DeviceManager. This includes each `managed_by` virtual stream. Keep the RoboTwin preset's `managed_device_ready_timeout` high enough for a cold simulator import; an owner process that exits first fails startup immediately.
- A managed-stream readiness failure or an unrecoverable bridge, PolicyRunner, or DeviceManager exit makes `device_starter.py` exit nonzero. Preserve that behavior so `scripts/robotwin/launch.py` reports a real startup/runtime failure rather than a misleading clean exit.
- On shutdown, `DeviceStarter` sends SIGINT to a managed-stream owner before joining it. `RoboTwinSimRobot` handles that signal locally by leaving its physics loop, then closes SAPIEN and generates the optional trace page in `finally`. Keep `managed_device_owner_shutdown_timeout` larger than the trace visualizer timeout, and keep the launcher wait budget larger still.
- `SynchronizedDeviceManager` consumes `device_manager.master_device_class_id` at the top level. This selects the device's logical `device_class_id`; IDs equal positions in the complete configured device list, including disabled entries. Resolve the master by ID after filtering enabled devices. The `device_manager.params.master_device_class_id` form is normalized before construction; never set both to different values.
- The direct robot shim deliberately avoids RoboTwin's MPLib/CuRobo imports. RoboTwin's own task modules still import their baseline simulation dependencies.

## Integration Guide

### 1. Prepare The Checkout

The parent repository pins RoboTwin as `third_party/RoboTwin`. Initialize it, then confirm that the intended checkout is on the `superinference` branch:

```bash
git submodule update --init third_party/RoboTwin
git -C third_party/RoboTwin status --branch
```

If the private GitLab submodule is not accessible, the public `RoboTwin-Platform/RoboTwin` checkout can be used as a fallback only after confirming the target task and preflight still pass. The public repository may not contain the parent repo's pinned gitlink commit, so treat it as a compatibility risk rather than a drop-in equivalent.

The bridge uses `beat_block_hammer` with RoboTwin's `demo_clean` task profile by default. It does not require the RoboTwin XPolicyLab package or its policy server.

### 2. Prepare Runtime Assets And Dependencies

Use a dedicated RoboTwin simulation environment. Install the SuperInference core and the upstream RoboTwin
requirements into that environment, then use preflight as the source of truth for the direct bridge dependencies:

```bash
python -m pip install -r requirements/requirements-robotwin.txt
python scripts/robotwin/preflight.py
```

`requirements/requirements-all.txt` is not a RoboTwin substitute: it does not supply RoboTwin's SAPIEN or Hugging Face Hub
dependencies. `requirements/requirements-robotwin.txt` combines the core and upstream requirements, pins `setuptools<82` because
`sapien==3.0.0b1` imports `pkg_resources`, pins `rerun-sdk==0.22.1` because newer releases require NumPy 2.x, and retains the NumPy 1.x ABI. Do not install a planner just for this
bridge: the direct-drive shim avoids MPLib and CuRobo.

On machines with a small root filesystem, avoid letting pip unpack large CUDA wheels under `/tmp`; put `TMPDIR` on the larger `/home` volume and disable inherited user site packages during install and validation:

```bash
mkdir -p .tmp-pip
TMPDIR="$PWD/.tmp-pip" PYTHONNOUSERSITE=1 python -m pip install --no-cache-dir \
  -r requirements/requirements-robotwin.txt
PYTHONNOUSERSITE=1 python scripts/robotwin/preflight.py
```

`scripts/robotwin/launch.py` relaunches itself with `PYTHONNOUSERSITE=1` when needed, then passes that isolation to DeviceStarter and the policy connector. This keeps Conda prefix environments from importing stale user or system packages during the multi-process launch.

After the direct preflight passes, register the actual interpreter once and
bind all RoboTwin configs' launcher-side roles. Set `<ROBOTWIN_PYTHON>` from the
environment you just validated; do not infer it from a conventional venv name.
The generated registry remains local to this worktree:

```bash
python3 scripts/diagnostics/runtime_profiles.py register robotwin \
  --python <ROBOTWIN_PYTHON> \
  --cwd "$(git rev-parse --show-toplevel)" \
  --env PYTHONNOUSERSITE=1 \
  --import hydra --import omegaconf --import sapien
for config in \
  policy_infer/robotwin/config_infer_robotwin_replay_joint14 \
  policy_infer/robotwin/config_infer_robotwin_act_joint14 \
  policy_infer/robotwin/config_infer_robotwin_eef16 \
  policy_infer/robotwin/config_infer_robotwin_pi05_joint14 \
  policy_infer/robotwin/config_infer_robotwin_fastwam_joint14; do
  for role in launcher planner device policy; do
    python3 scripts/diagnostics/runtime_profiles.py bind \
      --config-name "$config" --role "$role" --profile robotwin
  done
done
python3 scripts/diagnostics/runtime_profiles.py doctor \
  --config-name policy_infer/robotwin/config_infer_robotwin_replay_joint14
```

RoboTwin does not commit task assets. Download and extract the upstream asset archives before starting the bridge:

```bash
cd third_party/RoboTwin
HF_ENDPOINT=https://huggingface.co bash scripts/_download_assets.sh
cd ../..
python scripts/robotwin/preflight.py
```

Preflight must find `third_party/RoboTwin/assets/embodiments/aloha-agilex/config.yml`. It also checks the actual Python imports used by the bridge, such as SAPIEN, Gymnasium, transforms3d, OpenCV, Open3D, trimesh, imageio, and h5py.

### 3. Run The Weight-Free Transport Check

Start here before adding a learned checkpoint. `RoboTwinReplayPolicy` repeats measured `joint14` qpos when `actions_path` is null, exercising the exact asynchronous observation -> policy -> action executor -> SAPIEN transport without causing deliberate motion:

```bash
python3 scripts/diagnostics/runtime_profiles.py run \
  --config-name policy_infer/robotwin/config_infer_robotwin_replay_joint14 --role launcher -- \
  scripts/robotwin/launch.py --gui
```

To inject a known trajectory, point `policy.params.actions_path` in `configs/policy_infer/robotwin/config_infer_robotwin_replay_joint14.yaml` at a `.npy` or `.npz` file with an `actions` array shaped `[T, 14]`. The launcher starts DeviceStarter first, waits for the aggregate SHM, then starts the policy connector. Stop it with Ctrl+C so the bridge can close SAPIEN cleanly and write the trace page.

### 4. Run The Historical ACT Checkpoint

The ACT adapter is intentionally standalone. First export the historical ACT source from the local RoboTwin history:

```bash
python3 scripts/diagnostics/runtime_profiles.py run \
  --config-name policy_infer/robotwin/config_infer_robotwin_act_joint14 --role launcher -- \
  scripts/robotwin/bootstrap_act_source.py
HF_ENDPOINT=https://huggingface.co \
python3 scripts/diagnostics/runtime_profiles.py run \
  --config-name policy_infer/robotwin/config_infer_robotwin_act_joint14 --role launcher -- \
  scripts/robotwin/download_act_checkpoint.py
```

The downloader retrieves the official
[`TianxingChen/RoboTwin2.0 act_ckpt`](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/790104e1bbe9f70574b8c77d3676f229e6f2daeb/act_ckpt/act-beat_block_hammer/demo_clean-50)
pair for `beat_block_hammer/demo_clean-50` and places it under the configured directory:

```text
checkpoints/robotwin/act/beat_block_hammer/demo_clean-50/
  policy_last.ckpt
  dataset_stats.pkl
```

ACT needs CUDA, torch, torchvision, and IPython. Validate the exact preset and launch it:

```bash
python3 scripts/diagnostics/runtime_profiles.py run \
  --config-name policy_infer/robotwin/config_infer_robotwin_act_joint14 --role launcher -- \
  scripts/robotwin/preflight.py \
  --act-source third_party/robotwin_act/policy \
  --act-ckpt checkpoints/robotwin/act/beat_block_hammer/demo_clean-50
python3 scripts/diagnostics/runtime_profiles.py run \
  --config-name policy_infer/robotwin/config_infer_robotwin_act_joint14 --role launcher -- \
  scripts/robotwin/launch.py --config-name policy_infer/robotwin/config_infer_robotwin_act_joint14 --gui
```

The adapter receives state from `RoboTwinSimRobot_0` and images from `RoboTwinCamera_0`, `RoboTwinCamera_1`, and `RoboTwinCamera_2`. Raw camera SHM frames are `240x320x3`; the adapter resizes them internally to the historical ACT `480x640` input profile.

### 5. Run Pi_05 Or FastWAM Through XPolicyLab

Initialize the RoboTwin XPolicyLab submodule and validate the selected policy directory:

```bash
git -C third_party/RoboTwin submodule update --init --recursive XPolicyLab
python3 scripts/diagnostics/runtime_profiles.py run \
  --config-name policy_infer/robotwin/config_infer_robotwin_pi05_joint14 --role launcher -- \
  scripts/robotwin/preflight.py \
  --xpolicylab-root third_party/RoboTwin/XPolicyLab \
  --xpolicylab-policy Pi_05
python3 scripts/diagnostics/runtime_profiles.py run \
  --config-name policy_infer/robotwin/config_infer_robotwin_fastwam_joint14 --role launcher -- \
  scripts/robotwin/preflight.py \
  --xpolicylab-root third_party/RoboTwin/XPolicyLab \
  --xpolicylab-policy FastWAM
```

Install the policy runtime environments and the official cotrain checkpoints. Pi_05 uses its bundled OpenPI uv environment; FastWAM uses its own policy environment. The SuperInference FastWAM async eval path composes `FastWAM/FastWAM/configs/sim_robotwin.yaml`, which sets `model.skip_dit_load_from_pretrain: true` and `model.action_dit_pretrained_path: null`; it does not need the ActionDiT preprocessing step unless training or switching to an upstream config that loads pretrained DiT weights. Keep `PYTHONNOUSERSITE=1` and put `TMPDIR` in a workspace location with enough capacity; otherwise large CUDA wheels can fill the system temporary filesystem.

```bash
mkdir -p .tmp-pip
ROBOTWIN_PYTHON="$(python3 scripts/diagnostics/runtime_profiles.py resolve \
  --config-name policy_infer/robotwin/config_infer_robotwin_pi05_joint14 \
  --role launcher --require-verified)"
TMPDIR="$PWD/.tmp-pip" "$ROBOTWIN_PYTHON" -m pip install uv
TMPDIR="$PWD/.tmp-pip" PATH="$(dirname "$ROBOTWIN_PYTHON"):$PATH" PYTHONNOUSERSITE=1 \
  bash third_party/RoboTwin/XPolicyLab/policy/Pi_05/install.sh

PI05_PYTHON="${PI05_PYTHON:?set the interpreter created by the reviewed Pi_05 install}"
FASTWAM_PYTHON="${FASTWAM_PYTHON:?set the interpreter created by the reviewed FastWAM install}"
PYTHONNOUSERSITE=1 "$FASTWAM_PYTHON" -m pip install --force-reinstall \
  numpy==1.26.4 opencv-python==4.11.0.86 opencv-python-headless==4.11.0.86

FASTWAM_ROOT=third_party/RoboTwin/XPolicyLab/policy/FastWAM/FastWAM
DIFFSYNTH_MODEL_BASE_PATH="$PWD/${FASTWAM_ROOT}/checkpoints" \
DIFFSYNTH_DOWNLOAD_SOURCE=modelscope \
PYTHONPATH="$PWD/${FASTWAM_ROOT}:$PWD/${FASTWAM_ROOT}/src:${PYTHONPATH:-}" \
PYTHONNOUSERSITE=1 \
"$FASTWAM_PYTHON" - <<'PY'
from fastwam.models.wan22.helpers.loader import _resolve_configs

_, text_config, vae_config, tokenizer_config = _resolve_configs(
    model_id="Wan-AI/Wan2.2-TI2V-5B",
    tokenizer_model_id="Wan-AI/Wan2.1-T2V-1.3B",
    redirect_common_files=True,
)
for config in (vae_config, text_config, tokenizer_config):
    config.download_if_necessary()
    print(config.path)
PY

python3 scripts/diagnostics/runtime_profiles.py register xpolicylab-pi05 \
  --python "$PI05_PYTHON" --cwd third_party/RoboTwin/XPolicyLab/policy/Pi_05/openpi \
  --env PYTHONNOUSERSITE=1 --import jax --import websockets
python3 scripts/diagnostics/runtime_profiles.py bind \
  --config-name policy_infer/robotwin/config_infer_robotwin_pi05_joint14 \
  --role policy-server --profile xpolicylab-pi05

python3 scripts/diagnostics/runtime_profiles.py register xpolicylab-fastwam \
  --python "$FASTWAM_PYTHON" --cwd "$FASTWAM_ROOT" \
  --env PYTHONNOUSERSITE=1 --import torch --import websockets
python3 scripts/diagnostics/runtime_profiles.py bind \
  --config-name policy_infer/robotwin/config_infer_robotwin_fastwam_joint14 \
  --role policy-server --profile xpolicylab-fastwam

python3 scripts/diagnostics/runtime_profiles.py run \
  --config-name policy_infer/robotwin/config_infer_robotwin_pi05_joint14 --role launcher -- \
  scripts/robotwin/download_xpolicylab_checkpoint.py --policy all --dry-run
HF_ENDPOINT=https://huggingface.co HF_HUB_ENABLE_HF_TRANSFER=0 \
  python3 scripts/diagnostics/runtime_profiles.py run \
  --config-name policy_infer/robotwin/config_infer_robotwin_pi05_joint14 --role launcher -- \
  scripts/robotwin/download_xpolicylab_checkpoint.py --policy pi05
HF_ENDPOINT=https://huggingface.co HF_HUB_ENABLE_HF_TRANSFER=0 \
  python3 scripts/diagnostics/runtime_profiles.py run \
  --config-name policy_infer/robotwin/config_infer_robotwin_fastwam_joint14 --role launcher -- \
  scripts/robotwin/download_xpolicylab_checkpoint.py --policy fastwam
```

The downloader installs:

```text
third_party/RoboTwin/XPolicyLab/policy/Pi_05/checkpoints/RoboTwin-lerobot_v30-aloha_agilex-joint-0/
third_party/RoboTwin/XPolicyLab/policy/FastWAM/checkpoints/RoboTwin-cotrain-aloha_agilex-joint-0/
```

Use `policy_infer/robotwin/config_infer_robotwin_pi05_joint14` or `policy_infer/robotwin/config_infer_robotwin_fastwam_joint14`. By default these presets expect the matching XPolicyLab websocket server to already be running on the configured host/port:

```bash
python3 scripts/diagnostics/runtime_profiles.py run \
  --config-name policy_infer/robotwin/config_infer_robotwin_pi05_joint14 --role launcher -- \
  scripts/robotwin/launch.py --config-name policy_infer/robotwin/config_infer_robotwin_pi05_joint14 --gui
python3 scripts/diagnostics/runtime_profiles.py run \
  --config-name policy_infer/robotwin/config_infer_robotwin_fastwam_joint14 --role launcher -- \
  scripts/robotwin/launch.py --config-name policy_infer/robotwin/config_infer_robotwin_fastwam_joint14 --gui
```

To let the launcher start the XPolicyLab server as a child process, pass `--start-xpolicylab-server`. The default checkpoint names already match the official downloads, so normal launches do not need `--xpolicylab-ckpt-name`:

```bash
python3 scripts/diagnostics/runtime_profiles.py run \
  --config-name policy_infer/robotwin/config_infer_robotwin_pi05_joint14 --role launcher -- \
  scripts/robotwin/launch.py \
  --config-name policy_infer/robotwin/config_infer_robotwin_pi05_joint14 \
  --start-xpolicylab-server \
  --gui
python3 scripts/diagnostics/runtime_profiles.py run \
  --config-name policy_infer/robotwin/config_infer_robotwin_fastwam_joint14 --role launcher -- \
  scripts/robotwin/launch.py \
  --config-name policy_infer/robotwin/config_infer_robotwin_fastwam_joint14 \
  --start-xpolicylab-server \
  --gui
```

`ROBOTWIN_XPL_HOST`, `ROBOTWIN_PI05_PORT`, `ROBOTWIN_FASTWAM_PORT`, `ROBOTWIN_PI05_CKPT`, and `ROBOTWIN_FASTWAM_CKPT` override the preset defaults. Use `--xpolicylab-ckpt-name` only to select another installed run, and pass the real checkpoint name without shell angle brackets. The official defaults are `lerobot_v30` for Pi_05 and `cotrain` for FastWAM. These checkpoints use `env_cfg_type: aloha_agilex` and `action_type: joint`; keep that contract unless the replacement checkpoint was trained for a different RoboTwin embodiment/action contract.

### 6. Plug In A Custom Joint Policy

Create a normal SuperInference `BasePolicy` adapter and configure it as `policy.class`. Its `predict()` result must target `RoboTwinSimRobot_0` and return a float64 chunk shaped `[chunk_length, 14]`:

```python
return {"RoboTwinSimRobot_0": action_chunk}
```

The `joint14` order is:

```text
left_j0..left_j5, left_gripper, right_j0..right_j5, right_gripper
```

Keep all three declarations aligned: `policy.controls[*].shape`, `action_executor.robot_control_shms`, and `devices.devices[*].control.format.shape`. The shape is static shared-memory ABI, so action-space changes require a separate preset and a restart.

### 7. Use The EEF16 Adapter Template

`configs/policy_infer/robotwin/config_infer_robotwin_eef16.yaml` is an explicit template, not a usable IK implementation. It accepts per-arm `xyz + xyzw quaternion + gripper`, then invokes a pure-NumPy resolver that returns `joint14`. The resolver executes off the SAPIEN thread and intermediate targets are coalesced:

```bash
export ROBOTWIN_EEF_RESOLVER=your_package.your_module:resolve_eef16
export ROBOTWIN_EEF_REPLAY_ACTIONS=/absolute/path/actions_eef16.npz
python3 scripts/diagnostics/runtime_profiles.py run \
  --config-name policy_infer/robotwin/config_infer_robotwin_eef16 --role launcher -- \
  scripts/robotwin/launch.py --config-name policy_infer/robotwin/config_infer_robotwin_eef16 --gui
```

The action file must contain an `actions` array shaped `[T, 16]`; the resolver signature is `(target_eef16, current_joint14) -> joint14`. It must not touch SAPIEN or wait for robot arrival.

### 8. Inspect GUI And Traces

`--gui` enables the SAPIEN Viewer. On a clean bridge shutdown, the default trace configuration writes JSONL under `logs/action_state_xyz/` and automatically generates a sibling `.robotwin.html` page. Use `--open-trace` to open it after creation.

For cross-process latency analysis, join policy/action dispatch and bridge records using `control_write_timestamp_ns`. `sim_step` and `sim_time_ns` are simulator diagnostics only; do not use them as wall-clock join keys.

### 9. Diagnose Startup Failures

| Symptom | Check |
| --- | --- |
| Preflight cannot find embodiment assets | Run `(cd third_party/RoboTwin && bash scripts/_download_assets.sh)`; confirm the Aloha config path above exists. |
| Asset download fails with Hugging Face metadata errors | Override any broken mirror with `HF_ENDPOINT=https://huggingface.co` and rerun RoboTwin's asset downloader. |
| pip fails with no space left under `/tmp` | Create `.tmp-pip` on a filesystem with enough space and run with `TMPDIR=$PWD/.tmp-pip`; CUDA/Torch wheels need several GB while unpacking. |
| pip reports `rerun-sdk` requires NumPy 2.x | Install through `requirements/requirements-robotwin.txt`, which pins `rerun-sdk==0.22.1` to match RoboTwin's `numpy==1.26.4`. |
| DeviceStarter fails with `cannot import name 'AttrsInstance' from 'attr'` or SciPy warns from `/usr/lib/python3/dist-packages` | The runtime is importing system packages ahead of the RoboTwin environment. Use `scripts/robotwin/launch.py`, which enforces `PYTHONNOUSERSITE=1`; for direct helper/debug commands, set `PYTHONNOUSERSITE=1` manually. |
| `sapien` fails with `No module named 'pkg_resources'` | Run `python -m pip install -r requirements/requirements-robotwin.txt`; the profile pins `setuptools<82`, which is required by the current SAPIEN build. |
| SAPIEN says the selected GPU does not support ray tracing | Keep `robotwin.renderer_shader_dir: default`. Use `rt` only on a GPU with ray-tracing support. |
| ACT policy fails with CUDA OOM while the scene is alive | Keep the bridge resource capacities at `128` materials and `512` textures; RoboTwin's upstream `50000` material/texture preallocation is not appropriate for a single-task async test. |
| ACT downloader returns a 404 under `main` | Use its default pinned revision. The mutable official dataset branch no longer keeps every historical ACT artifact pair. |
| Pi_05 install fails with `uv not found` | Install `uv` into the active RoboTwin venv and prepend that venv to `PATH` before running `policy/Pi_05/install.sh`. |
| Pi_05 server exits with missing `assets/arx_x5_sim/norm_stats.json` | The official `lerobot_v30` checkpoint ships its OpenPI stats as `assets/RoboTwin_lerobot_v30/norm_stats.json`; keep `policy/Pi_05/deploy.yml` `repo_id: RoboTwin_lerobot_v30` when using the official cotrain archive. |
| Pi_05 server logs XLA/BFC `ran out of memory` before port 6000 opens | Do not keep XPolicyLab's old 30% JAX cap. `policy/Pi_05/setup_eval_policy_server.sh` should default `XLA_PYTHON_CLIENT_PREALLOCATE=false` and `XLA_PYTHON_CLIENT_MEM_FRACTION=0.85`. When the launcher starts the server, wait for port 6000 before starting SAPIEN/GUI so checkpoint restore peak VRAM does not overlap simulator allocation. |
| Pi_05 still OOMs when launched standalone with `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95` | Treat an 8 GB-class GPU as below the practical memory floor for the official JAX `lerobot_v30` checkpoint. Use a larger-VRAM GPU, a smaller converted policy, or CPU-only JAX for slow debugging. |
| FastWAM imports user/system packages or `pip` from `/usr/lib/python3/dist-packages` | Run install, validation, and server launch with `PYTHONNOUSERSITE=1`. After XPolicyLab editable install, keep `numpy==1.26.4`, `opencv-python==4.11.0.86`, and `opencv-python-headless==4.11.0.86`; `pip check` should report no broken requirements. |
| `scripts/robotwin/launch.py` fails with `ModuleNotFoundError: No module named 'omegaconf'` | The launcher role is missing, stale, or bypassed. Run `runtime_profiles.py doctor --config-name <robotwin-config>` and launch through `runtime_profiles.py run --role launcher`. |
| FastWAM starts loading then reports missing `ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt` | Confirm the async preset is composing `FastWAM/FastWAM/configs/sim_robotwin.yaml`; that path should set `skip_dit_load_from_pretrain: true` and not request ActionDiT. Only upstream training or non-sim configs should run `scripts/preprocess_action_dit_backbone.py`. |
| FastWAM exits with CUDA OOM on an 8 GB-class GPU | Treat it as a capacity limit unless memory/offload/runtime settings are redesigned; installed assets alone do not prove the full server fits. |
| Pi_05/FastWAM preset hangs before first inference | Confirm the XPolicyLab websocket server is reachable at the configured host/port, or launch with `--start-xpolicylab-server` after `scripts/robotwin/download_xpolicylab_checkpoint.py --policy all`. The client intentionally retries while the heavy policy server starts. |
| DeviceStarter exits before policy starts | Read its nonzero exit log; managed virtual cameras publish only after the bridge has started SAPIEN and produced a fresh frame. |
| ACT preset fails before inference | Run both ACT helpers, confirm their two artifact files, CUDA availability, and the ACT-only preflight checks. |
| No HTML trace page | Stop through the launcher with Ctrl+C rather than killing the bridge; the page is generated during graceful shutdown. |
| Control shape/dtype error | Use exactly `[14]` or `[16]` per preset and `float64` control SHM. |

## Run And Diagnose

Start with the weight-free transport check:

```bash
python3 scripts/diagnostics/runtime_profiles.py run \
  --config-name policy_infer/robotwin/config_infer_robotwin_replay_joint14 --role launcher -- \
  scripts/robotwin/launch.py --gui
```

Use the historical ACT preset only after its source and artifacts exist:

```bash
python3 scripts/diagnostics/runtime_profiles.py run \
  --config-name policy_infer/robotwin/config_infer_robotwin_act_joint14 --role launcher -- \
  scripts/robotwin/bootstrap_act_source.py
python3 scripts/diagnostics/runtime_profiles.py run \
  --config-name policy_infer/robotwin/config_infer_robotwin_act_joint14 --role launcher -- \
  scripts/robotwin/download_act_checkpoint.py
python3 scripts/diagnostics/runtime_profiles.py run \
  --config-name policy_infer/robotwin/config_infer_robotwin_act_joint14 --role launcher -- \
  scripts/robotwin/launch.py --config-name policy_infer/robotwin/config_infer_robotwin_act_joint14 --gui
```

Run `runtime_profiles.py run --role launcher -- scripts/robotwin/preflight.py`
first. The RoboTwin submodule does not include embodiment assets, so
`assets/embodiments/aloha-agilex/config.yml` must be provisioned separately. The
ACT preset additionally needs `policy_last.ckpt` and `dataset_stats.pkl` at
`checkpoints/robotwin/act/beat_block_hammer/demo_clean-50/`; download them
explicitly through the helper rather than treating them as an implicit local
prerequisite.

`scripts/robotwin/launch.py` composes the selected preset before preflight, so a custom task or ACT artifact path is checked rather than the default task by mistake.

The P0 task is `beat_block_hammer` with `demo_clean`; its raw camera SHM profile is `240x320x3`. The ACT adapter resizes frames to `480x640` internally, matching the historical deployment path.

## Trace Semantics

Join SuperInference dispatch records and bridge records using `control_write_timestamp_ns`. The bridge writes `sim_control_accepted`, `sim_action_applied`, and `sim_measured_state` JSONL records with full target/measured joint vectors. `sim_action_applied.target_qpos` is the high-level target accepted from SHM; `sim_measured_state.target_qpos` is the interpolated drive target currently being applied to SAPIEN. `sim_measured_state.measured_qpos` is the physical articulation state and is also what the policy-visible robot state SHM publishes. `sim_step` and `sim_time_ns` diagnose simulator timing only; they are not replacements for the wall-clock join key.

In `execution_clock: sim_time`, `all_actions` and `aligned_actions` are chunk previews on an estimated wall-clock dispatch timeline. Their `timestamp_ns` is computed from the chunk switch wall time plus a future wall offset estimated from recent `action_xyz` dispatch intervals; nominal sim-time offsets remain in `sim_preview_offset_ns` / `sim_preview_offset_s`, and estimated wall offsets are in `wall_preview_offset_ns` / `wall_preview_offset_s`. Treat `timestamp_semantics: estimated_wall_dispatch_time` as a forecast, not a hardware-confirmed dispatch record. `action_xyz` remains the actual control-SHM write record.

With the default `trace_visualizer.enabled: true`, clean bridge shutdown invokes `scripts/visualization/robotwin_trace_visualizer.py` and writes a sibling `.robotwin.html` page for physical target-versus-measured joints. For the generic `action_state_xyz_visualizer.html`, load the complete run directory under `logs/action_state_xyz/<run-id>/`, not only the bridge JSONL: `action_executor_xyz.jsonl` contains dispatched, original full-chunk, aligned-chunk, and chunk-band records, while `policy_connector_xyz.jsonl` contains observations, measured joint state, and inference markers.

Use `sim_loop_timing` records before changing scheduling assumptions. `scene_step_ns` isolates physics stepping, while `overrun_ns` shows how far a loop missed `physics_fps`. Camera capture/write is reported separately as `sim_camera_timing.publish_camera_ns`; if that value is large and loop sections also stretch, the camera publisher is still competing for render/GPU resources even though it is outside the physics-loop call path.

New RoboTwin `joint14` records carry `vector_layout: joint14` plus all `joint_positions`; the generic page exposes `left_j0` through `right_gripper` in its Axis selector and automatically selects `left_j0` for a joint-only run. Joint values must never be interpreted as Cartesian `xyzw`. Virtual camera RGB tensors are excluded from robot-state tracing. Historical JSONL written before this contract cannot recover the omitted joint values, so rerun the simulation when joint chunk plotting is needed.

For every policy adapter, a successful `inference_end` marker carries the wall-clock inference duration, cumulative `actual_policy_inference_hz`, configured policy FPS, and the action-chunk horizon `chunk_length / control_fps`. If duration exceeds that horizon, the connector emits a warning because one prediction can be exhausted before the next usable prediction arrives. For reference, Pi05 is `50 / 15 = 3.333s` in the current preset and FastWAM is `24 / 15 = 1.6s`. The historical ACT adapter additionally carries `policy_input_preprocess_ns`, `policy_h2d_ns`, `policy_cuda_forward_ns`, `policy_output_sync_d2h_ns`, `policy_postprocess_ns`, and `policy_policy_predict_ns`. Treat `policy.fps` as the requested query cadence; the actual frequency is measured from consecutive inference start timestamps.

`--gui` disables only the SAPIEN Viewer. ACT still needs its three policy camera frames, and the bridge renders them in the camera publisher thread. On a single GPU this can still compete with CUDA inference and SAPIEN rendering resources; inspect both `sim_loop_timing` and `sim_camera_timing` before judging actuator dynamics. Reduce `robotwin.camera_fps` deliberately for slow-query debugging or place SAPIEN and the policy on separate GPUs. Do not silently change this rate in a latency experiment because it changes the observation cadence.

## Change Safely

Keep the static SHM shape consistent across `policy.controls`, `action_executor.robot_control_shms`, and `RoboTwinSimRobot`'s `control.format.shape`. Set robot `control.type` directly to `joints` or `eef_abs`; the adapter does not accept control-mode aliases. Changing action dimensions requires a separate config/restart, not a runtime switch.

When touching the runtime, preserve the fresh-process import order in `integrations/robotwin/runtime.py`: install the `envs.robot` direct-drive shim before importing the task module. Keep all SAPIEN calls on the bridge process/thread. Run the hardware-free integration tests when pytest is available, then run preflight; a real rollout requires the assets and simulation dependencies above.
