---
name: fastwam-inference-runtime
description: Maintain and troubleshoot FastWAM, FastWAM-memory, and FasterWAM websocket Server/client contracts and Control Panel lifecycle.
---

# FastWAM Inference Runtime

FastWAM is deliberately split into two processes. The Server owns the heavy
checkpoint/model and exposes websocket inference plus `GET /healthz`; the
client reads SuperInference summary SHM and writes policy action SHM. Keep the
model out of the device process and never make the client load a checkpoint.

## Entrypoints

Canonical implementations live under `consumer/inference/`:

```bash
<fastwam-server-prefix> consumer/fastwam_server.py \
  --config-name policy_infer/fastwam/config_infer_fisheye_umi3_dual_fastwam
<policy-prefix> -m consumer.inference.fastwam_policy_infer \
  --config-name policy_infer/fastwam/config_infer_fisheye_umi3_dual_fastwam
```

The flat scripts are compatibility facades. Start the device runtime in its
own approved device environment first; then wait for `/healthz` before starting
the client. Direct launch and Control Panel both compose the same FastWAM YAML
and therefore use the same two-wrist device set. The starter also locks its
SHM/device/robot resources to prevent a second stack.
`tests/config/policy_launch_plan.py --config-name ... --json` checks the command
contract without opening hardware.

The Server imports `third_party/nmx_vla` and may need a separate CUDA model
environment. In a packaged release, set
`SUPERINFERENCE_FASTWAM_SERVER_PYTHON` together with
`SUPERINFERENCE_FASTWAM_SERVER_PYTHON_EXTERNAL=1` to select that interpreter;
the bundled OpenPI client environment is otherwise used and inherited server
paths are deliberately ignored. In a source checkout, the client uses
its config-specific `policy` profile (or the panel interpreter), while the
Server uses the `fastwam-server` profile. The legacy
`SUPERINFERENCE_INFERENCE_PYTHON` and
`SUPERINFERENCE_FASTWAM_SERVER_PYTHON` variables remain higher-priority one-off
overrides. Manage source profiles through
[`runtime-profile-management`](../runtime-profile-management/SKILL.md).

Preserve the virtual-environment interpreter path when spawning the Server.
Resolving a path such as `.venv/bin/python` through symlinks can collapse it to
`/usr/bin/python3.x`, which bypasses the virtual environment and causes an
immediate `ModuleNotFoundError` for packages such as Torch. Make interpreter
paths absolute without resolving their symlinks.

The self-contained release builder uses one Python 3.12 runtime for its NEDF3
recording/conversion profiles as well; do not point those wrappers at a missing
Python 3.10 sidecar. The older NEDF2 workflow remains a separate contract.

## Source checkout setup

After merging FastWAM into a source checkout, initialize the locked NMX model
submodule before importing the server:

```bash
git submodule update --init third_party/nmx_vla
```

Keep the model environment separate from a pure OpenPI environment. The profile
validated on September 3, 2026 provided Torch 2.7.1+cu128, Transformers 4.57.6,
websockets 16.0, Flexiv RDK 1.8.0, and the NMX dependencies. Register each
actual workstation interpreter, then bind every role that will be started to the
exact native config. A binding does not inherit through Hydra defaults: baseline,
RTC, memory, and FasterWAM each need their own config-specific bindings. The
server, device, policy, and launcher profiles may be different. Its editable
`i2rt` and LKTech packages must point at the current checkout after a branch
switch:

```bash
FASTWAM_CONFIG=policy_infer/fastwam/config_infer_fisheye_umi3_dual_fastwam
FASTWAM_MODEL_PYTHON=/absolute/path/to/fastwam/.venv/bin/python
uv pip install --python "$FASTWAM_MODEL_PYTHON" "modelscope==1.33.0"
python3 scripts/diagnostics/runtime_profiles.py register fastwam-server \
  --python "$FASTWAM_MODEL_PYTHON" \
  --cwd "$(git rev-parse --show-toplevel)" \
  --env PYTHONNOUSERSITE=1 \
  --import torch --import transformers --import websockets --import modelscope
FASTWAM_RECORDING_PYTHON=/absolute/path/to/nedf3/.venv/bin/python
python3 scripts/diagnostics/runtime_profiles.py register fastwam-recording \
  --python "$FASTWAM_RECORDING_PYTHON" \
  --cwd "$(git rev-parse --show-toplevel)" \
  --env PYTHONNOUSERSITE=1 \
  --import dm3_core.nedf --import dm3_core.mcap --import nmx_msg \
  --import cv2 --import yaml --import numpy \
  --import consumer.io.async_infer_data_saver_NEDF3
# Repeat these exact-config bindings with the verified profile for each role.
# Do not assume the six placeholders name the same environment.
python3 scripts/diagnostics/runtime_profiles.py bind \
  --config-name "$FASTWAM_CONFIG" \
  --role fastwam-server --profile <fastwam-server-profile>
python3 scripts/diagnostics/runtime_profiles.py bind \
  --config-name "$FASTWAM_CONFIG" \
  --role policy --profile <policy-profile>
python3 scripts/diagnostics/runtime_profiles.py bind \
  --config-name "$FASTWAM_CONFIG" \
  --role device --profile <device-profile>
python3 scripts/diagnostics/runtime_profiles.py bind \
  --config-name "$FASTWAM_CONFIG" \
  --role launcher --profile <launcher-profile>
python3 scripts/diagnostics/runtime_profiles.py bind \
  --config-name "$FASTWAM_CONFIG" \
  --role planner --profile <planner-profile>
python3 scripts/diagnostics/runtime_profiles.py bind \
  --config-name "$FASTWAM_CONFIG" \
  --role recording --profile fastwam-recording
FASTWAM_PYTHON="$(python3 scripts/diagnostics/runtime_profiles.py resolve \
  --config-name "$FASTWAM_CONFIG" --role fastwam-server --require-verified)"
uv pip install --python "$FASTWAM_PYTHON" --no-deps \
  -e ./third_party/i2rt -e ./third_party/lktech_motor_sdk
python3 scripts/diagnostics/runtime_profiles.py verify --profile fastwam-server
python3 scripts/diagnostics/runtime_profiles.py run \
  --config-name "$FASTWAM_CONFIG" --role fastwam-server -- \
  scripts/diagnostics/fastwam_model_cache.py prepare \
  --config-name "$FASTWAM_CONFIG" --json
python3 scripts/diagnostics/runtime_profiles.py doctor \
  --config-name "$FASTWAM_CONFIG"
```

Run the model-cache `prepare` step whenever an AI configures a direct FastWAM,
RTC, memory, or FasterWAM environment. It performs bounded, read-only discovery
across standard ModelScope/Hugging Face caches, conventional model roots, and a
limited-depth scan of the current user's home directory. It rejects Trash,
temporary, environment, dataset, and repository-local cache trees. When a
complete source is found, it creates a machine-local symlink below
`.superinference/local/model-cache/<model-id>` and records the result in the
Git-ignored `.superinference/local/fastwam_model_sources.json`. It never moves or
copies the discovered source. An incomplete previous download is atomically
quarantined below `.superinference/local/model-cache/.incomplete` before the
link is installed. If no complete source exists, setup remains valid and reports
that the normal startup download is still required. Do not ask the user to type
a model path during normal environment setup.

The reviewed NMX loader defaults `DIFFSYNTH_DOWNLOAD_SOURCE` to `modelscope`
when a Wan component is absent from the local cache. Keep
`modelscope==1.33.0` in each FastWAM model environment and `modelscope` in the
profile import probe so `doctor` catches an incomplete download runtime before
model loading.

NMX resolves a remote-style model ID below `DIFFSYNTH_MODEL_BASE_PATH`, falling
back to `./checkpoints` when that variable is unset. All direct FastWAM variants
(`fastwam`, RTC, memory, and FasterWAM) set `policy.params.model_base_path` to the Git-ignored
`.superinference/local/model-cache` root; the Server exports it before calling
NMX. Populate the model-ID layout there with local files or machine-local
symlinks. Never commit the symlink targets or replace this relative root with a
workstation-specific absolute path.

The server and client then use the same source checkout while the heavy model
remains in the Server process. Do not install `third_party/nmx_vla/requirements.txt`
into the OpenPI policy environment; its Transformers/CUDA contract is owned by
the FastWAM Server profile. The selected FastWAM YAML owns both
`policy.params.ckpt_path` and `policy.params.task`, including the fixed memory
and FasterWAM deployments. Runtime profiles own interpreters only and must
never store checkpoints or task prompts.

For a split launch, bind and run `device`, `fastwam-server`, and `policy` for
the same selected config. A Control Panel launch also requires `launcher` and
`planner`. Every FastWAM config also requires a verified `recording` profile
that imports `dm3_core.nedf`, `dm3_core.mcap`, `nmx_msg`, `cv2`, `yaml`,
`numpy`, and the stable recorder facade. Register and bind it even while
`policy.infer_rollout.enabled` is false. Before an actual launch, run
`doctor --config-name <exact-config>` without a role so the complete required
role contract is checked. Use `command --json` when inspecting commands and
`run` only after those checks pass.
The same config name is passed to the Control Panel:

```bash
python3 scripts/diagnostics/runtime_profiles.py run \
  --config-name policy_infer/fastwam/config_infer_fisheye_umi3_dual_fastwam \
  --role launcher -- \
  -m incubator.control_panel \
  --config-name policy_infer/fastwam/config_infer_fisheye_umi3_dual_fastwam
```

The selected FastWAM YAML stores the checkpoint root in
`policy.params.ckpt_path`. A valid root contains
`config.yaml`, `dataset_statistics.json`, and `checkpoints/weights/*.pt`.

The native RTC config declares its checkpoint and task directly under
`policy.params`, like the other FastWAM variants; it does not load a hidden
`configs/local` overlay. Set `policy.params.ckpt_path` to an existing root or
exact weight file, and validate the composed config with the bound server profile
and `consumer/fastwam_server.py --validate-only`.
This checks checkpoint metadata and normalization without loading weights,
opening a server port, or touching hardware; it does not prove model inference.

## Control Panel lifecycle

The panel exposes `runtime -> server -> inference` for `workflow_kind: fastwam`.
The Server stage may start before the physical runtime and has an independent
operation lock, but inference still requires runtime readiness and Server
readiness. The Server owns `policy.params.ckpt_path`; checkpoint overrides are
validated at Server start and are not passed to the client.

`FastWAMWorkflowLifecycle` allocates a free port from `policy.params.port_range`
(default `18090-18100`) and creates a private run directory containing
identity-checked `ready.json` and bounded startup `progress.json`. Readiness
requires both files to match run id, PID, config name and port, and a matching
`/healthz` response. Do not trust an arbitrary listener on the same port.

The server process receives these ownership variables:

```text
SUPERINFERENCE_FASTWAM_READY_FILE
SUPERINFERENCE_FASTWAM_PROGRESS_FILE
SUPERINFERENCE_FASTWAM_RUN_ID
SUPERINFERENCE_FASTWAM_CONFIG_NAME
SUPERINFERENCE_FASTWAM_SERVER_PORT
```

On failure, stop the client before reporting the model stage failed. On normal
shutdown use the lifecycle stop path so the process group and private run
directory are cleaned up. `force_stop_all` is reserved for an emergency and
must preserve the control-panel web process.

Control Panel preserves the same action execution cadence as direct launch
through `BasePolicyRunner.run_connected()`. Direct `start()` connects and then
calls this shared loop; the panel connects first so it can publish readiness,
then calls the same loop. The loop applies `precise_loop_timing()` at
`execution_fps`. The panel applies the configured `device`, `device_manager`,
and `policy_runner` CPU-affinity roles to the corresponding spawned workers as
direct mode does. Do not replace this with a raw `_update_execution()` loop:
that drains a 48-step chunk to `action_index=47` almost immediately, sends sparse
end-of-chunk commands to the robot, and makes FastWAM RTC fall back because no
useful chunk overlap remains.

The remaining launch differences are lifecycle boundaries, not alternate
policy configuration. Both paths wait for observed live device/SHM frames; the
panel additionally uses multiprocessing `spawn`, owns fail-closed health
monitoring, and may pick an available FastWAM Server port while overriding only
the client endpoint. The panel may also override `policy.infer_rollout` from its
recording control and apply unsaved hardware bindings; account for those two
explicit UI settings when comparing runs.

## Loading, variants, and memory

The Server resolves either a checkpoint directory or a concrete
`checkpoints/weights/*.pt` file (the panel rejects other file suffixes), and
requires sibling `config.yaml` and `dataset_statistics.json`. Keep the
returned checkpoint payload only long enough to read scalar metadata, then
release it before attaching the text encoder. Text-encoder loading uses the
low-CPU-memory path when local paths are configured. Freeze model parameters
after loading; RTC needs gradients only with respect to the noisy action latent.

`consumer.inference.fastwam_model_registry` is the code-owned registration
contract for supported variants. Consumers that render or validate model choices
must call `iter_fastwam_model_variants()` or
`get_fastwam_model_variant(variant_id)` rather than duplicate an enum. The
native configuration names are:

| Variant | Config name | History / action path |
| --- | --- | --- |
| `fastwam` | `policy_infer/fastwam/config_infer_fisheye_umi3_dual_fastwam` | one observed frame / `observed_frame_cache` |
| `fastwam-memory` | `policy_infer/fastwam/config_infer_fisheye_umi3_dual_fastwam_memory` | four observed frames / `observed_frame_cache` |
| `fasterwam` | `policy_infer/fastwam/config_infer_fisheye_umi3_dual_fasterwam` | one observed frame / `one_pass_future_cache` |

Memory and FasterWAM are standalone top-level rollout configs. Each must directly
declare its device preset, camera binding, runtime groups, complete policy
contract, checkpoint, and task; neither may inherit the baseline FastWAM launch
config through Hydra defaults. This keeps model variants independently
preflightable and prevents baseline policy changes from silently changing another
deployment. Shared component groups such as the FastWAM device preset,
`camera_sync`, and `base_auto_drop` remain reusable.

Every native config declares `model_variant`, `condition_history_frames`,
`obs_horizon`, `action_infer_mode`, and `action_video_attention_mode`.
`validate_fastwam_variant_config()` checks these YAML-owned fields. Checkpoint
metadata owns the memory stride (`condition_history_stride`) and FasterWAM
sampled-video length (`sampled_video_num_frames`), because those may differ by
checkpoint; `validate_fastwam_variant_runtime_contract()` checks them after the
Server loads the checkpoint. Do not make either a required YAML field merely to
silence validation.

Memory clients send oldest-to-newest `image_history` plus matching source-frame
timestamps, while normal FastWAM sends `images`. After an episode reset, the
history builder pads with the anchor frame and records the anchor's actual
timestamp; pad flags indicate requested temporal positions that are not yet
available. The client validates Server metadata at initial connect and after a
reconnect, so a restarted Server cannot silently switch model contracts.

FasterWAM calls `infer_action_one_pass_future_cache` and receives C4 scheduler
arguments only on that path. Its video scheduler is independent from action
denoising: the native FasterWAM preset keeps `num_inference_steps: 10` and
`c4_video_num_inference_steps: 2`; do not couple those fields. Ordinary `infer_action` never receives C4-only
kwargs unless its model signature explicitly accepts them. A FasterWAM
checkpoint must advertise full-video attention and cache-compatible model
settings; reject an observed-frame checkpoint instead of attempting a fallback.

FasterWAM checkpoint state and relative poses use the UMI tool frame, while
Rizon commands remain in the TCP frame. Its portable config therefore sets
`tcp_umi_json_path: calibration/tcp_to_umi_transform.json`. The policy converts
TCP state to UMI before model inference, applies local or world relative motion
in UMI, then converts the target back to TCP. The offline evaluator follows the
same transform and records both TCP and UMI state arrays; do not bypass this
with a hand-written coordinate conversion or a host-specific calibration path.
The reviewed FasterWAM vendor pin is `b81a39f8722ecf1a370b83b0d1026e5f6e3346a5`:
its one-pass method requires `num_video_frames` and accepts the C4 scheduler
fields. Update this pin and `tests/consumer/test_fasterwam_vendor_contract.py`
together when reviewing a vendor API upgrade.

The FastWAM configs use the dedicated
`infer_rizon_fisheye_umi3_dual_fastwam` dual-wrist device group and are shared
by direct launch and Control Panel. This group is intentionally separate from
OpenPI's `infer_rizon_fisheye_umi3_dual_openpi` group so their camera settings
can diverge. During the current camera-port investigation, both groups set
`usb_port_requirement: any`; restore a specific requirement in the relevant
group once the binding is confirmed. Their `metadata.device_profile` values
make that ownership explicit. Keep machine-specific camera and gripper values
in the selected group rather than creating a second panel profile. The standard
and RTC FastWAM names are
`policy_infer/fastwam/config_infer_fisheye_umi3_dual_fastwam` and
`policy_infer/fastwam/config_infer_fisheye_umi3_dual_fastwam_rtc`; use the memory and
FasterWAM names in the variant table above for their dedicated checkpoints.
简明操作文档按配置分开维护：

- [`fastwam_inference_commands.md`](../../../fastwam_inference_commands.md)：普通 FastWAM 与 RTC；
- [`fastwam_memory_inference_commands.md`](../../../fastwam_memory_inference_commands.md)：FastWAM memory；
- [`fasterwam_inference_commands.md`](../../../fasterwam_inference_commands.md)：FasterWAM。

`usb_port_requirement` checks the negotiated USB link speed, not the physical
receptacle. Configure a stable `capture_id` path as well when camera identity
must remain fixed across USB enumeration changes.

`WebCameraOpenPi` has two different identities. `device_class_id` is the logical
SuperInference/SHM identity and produces a name such as
`WebCameraOpenPi_1_data`; the underscore is inserted between the class name and
the logical ID by the device adapter. `capture_id` selects the physical UVC
node. The FastWAM and OpenPi top-level configs expose
`camera.left_capture_id` and `camera.right_capture_id`, sourced from
`SUPERINFERENCE_CAMERA_LEFT_CAPTURE_ID` and
`SUPERINFERENCE_CAMERA_RIGHT_CAPTURE_ID`. Set those variables to stable
`/dev/v4l/by-id/...` or `/dev/v4l/by-path/...-video-index0` links when cameras
are unplugged/replugged. Numeric `left_capture_id`/`right_capture_id` values
select physical indices only. WebCamera requires an explicit `capture_id` and
never falls back to the logical ID. Environment numeric strings are normalized
to integers before calling OpenCV.

Every configured device, including disabled entries, has `device_class_id`
equal to its zero-based position in the complete list. IDs are unique across
classes, not per class. Device constructors, endpoint mappings, SHM headers,
signal patterns and consumers all use this field. Old `device_id` config
fields are rejected. Preflight and Control Panel update physical capture,
serial and gripper bindings without changing logical IDs or references.
Changing list order is a schema edit requiring IDs and references to migrate
together. Stop old processes before launching a migrated config. Manager master
selection resolves an enabled device by ID, not by an index into a filtered list.

Device startup waits for actual SHM frame readiness. Do not add
`global_settings.device_init_delay`: it is obsolete and is not consumed by the
launcher. The canonical FastWAM and RTC configs currently both enable RTC at
1.5 Hz with VJP disabled; treat their YAML parameters as authoritative instead
of inferring synchronous behavior from a filename. Static Control Panel
imports must keep device-contract validation lazy so importing its server does
not load device registries or create repository log files.

The canonical FastWAM and OpenPi presets compose the `camera_sync`
device-manager behavior group. Each root launch config sets the first wrist
camera (device-list index `0`) as `master_device_class_id` and sets
`params.buffer_size: 500` through the historical nested
`device_manager.device_manager` Hydra envelope. `camera_sync` keeps the other
wrist image on closest-frame matching and interpolates the typed Rizon state to
the reference image timestamp. Its Rizon
`state_feedback.format` owns shape, dtype, and ordered primitives; it declares
linear joint/velocity/translation/gripper segments and WXYZ quaternion segments
for SLERP. Robot commands live under `control`, while joint and Cartesian limits
with explicit units live under `motion_limits.velocity` and
`motion_limits.acceleration`. Do not change the shared `sync` manager behavior
to camera mode: Lingbot and legacy EEF workflows still rely on its default
robot-master binary matching. In any new camera-interpolated device
group, every low-dimensional device must declare a valid
`state_feedback.format`; targets older than retained history are skipped.
Robot states may be projected forward within the bounded horizon below.
Runtime config validation does not materialize flat
robot fields; DeviceStarter passes grouped sections directly to every registered
robot entrypoint. Their public constructors do not accept duplicate shape,
dtype, rate, or buffer arguments. Device-level
`enabled` is only the process launch switch. Per-arm robot output belongs to
`control.source_selection` declares the logical command source (or explicit
policy/teleoperation arbitration); Rizon gripper state/control is enabled by
`gripper` primitives in the corresponding format and configured by the
`gripper` mapping. Arm command availability is derived from the structured
control format rather than a per-arm control switch.
`control.type` is exactly one of `joints`, `eef_abs`, or `eef_rel`; aliases
are not converted. It must agree with
`control.format.type`: joint control uses `j1-7`, while TCP control uses `xyz`
plus one explicitly ordered quaternion primitive.
The device registry owns the `camera`, `low_dim_r`, `low_dim_rw`, and
`low_dim_w` capability. DeviceManager rejects write-only entries and validates
low-dimensional format shape/dtype against the live SHM header before creating
the summary buffer.

`torch_compile`, prompt caching, and warmup are optional policy parameters. The
canonical FastWAM configs enable the low-latency combination:
`cache_prompt_embeddings: true`, `prompt_cache_max_entries: 8`,
`torch_compile.enabled: true` with `mode: reduce-overhead`, and
`warm_up_on_load: true` with three requests. RTC configs should also set
`rtc_use_vjp: false` unless exact Jacobian guidance is explicitly required;
exact VJP runs the denoiser eagerly under autograd and is substantially slower.
Treat startup phase/progress and `model_loaded` metadata as evidence; a process
that merely binds a port is not a ready model.

With `websockets==16.0`, HTTP responses from `process_request` must be created
with `connection.respond(...)` (or the full four-field `Response` contract).
Passing the legacy three positional values constructs a malformed response and
makes `/healthz` fail with `TypeError: 'bytes' object does not support item
assignment`, leaving the Control Panel Server stage stuck at starting even
though model loading and warmup completed.

The FastWAM model environment must contain the top-level `functorch` package
that ships with the matching PyTorch wheel. Torch 2.7.1 can import
`torch._functorch` while still failing `torch.compile` if this compatibility
package was omitted during an incomplete or `--no-deps` install. Verify it
before a latency comparison:

```bash
"$FASTWAM_PYTHON" -c 'import torch, functorch; print(torch.__version__, functorch.__file__)'
```

If that import fails, reinstall the exact CUDA-matched PyTorch wheel into the
FastWAM environment (do not install the unrelated standalone `functorch`
2.0 package):

```bash
uv pip install --python "$FASTWAM_PYTHON" --reinstall \
  torch==2.7.1+cu128 --index-url https://download.pytorch.org/whl/cu128
```

After restarting the server, require `torch_compile_active: true` in its
metadata. `torch_compile_requested: true` with `torch_compile_active: false`
means inference is still eager; the server log will include the import or
compiler exception that caused the fallback.

When comparing latency, use `server_timing.infer_ms` or the FastWAM client's
`rpc_ms` for model/RPC time. `policy.fps` is the replan cadence used by
`PolicyConnector` (`1.5 Hz` means a roughly `667 ms` loop period), so it must
not be interpreted as model inference latency. After startup, verify the
server metadata reports `torch_compile_active: true`,
`prompt_cache_enabled: true`, and the intended `rtc_use_vjp` value. If the
observed delay is the action-update period rather than `infer_ms`, report it as
the configured replan cadence and keep it separate from model acceleration.

FastWAM RTC keeps the same small contract as the native Junlin implementation.
The policy computes delay and prefix horizon from tracked p95 RPC latency on
every request, then applies the optional `rtc_delay_margin_steps` or the
explicit `rtc_guidance_full_steps_override`. `rtc_max_guidance_weight` and
`rtc_use_vjp` are part of the shared policy/server contract. Do not add
`rtc_inference_fps`, temporal schedules, or reverse-guidance parameters. The
server validates non-negative delay, positive horizon and finite guidance
weight, and forwards the per-request `rtc_use_vjp` value only when the model
API accepts it.

Before loading a checkpoint, the Server performs a no-allocation CUDA
preflight and latches classified CUDA/driver failures for the process lifetime.
`GET /healthz` reports this latch and returns `503` until both the model and GPU
are healthy. A checkpoint directory may opt into a `.complete.json` marker and
SHA-256 validation with `require_checkpoint_completion_marker` and
`verify_checkpoint_sha256`; the selected weight, training config, dataset
statistics, and variant-derived action/video contract are then reported in
Server metadata. Action outputs are required to have the exact checkpoint
horizon/dimension and finite normalized and denormalized values; silent
padding/truncation is an error.

For a clean model-latency comparison, temporarily set
`infer_rollout.enabled: false`. An enabled inference saver is a separate
process that polls and locks the same summary SHM used by the policy reader and
also performs asynchronous disk writes. It is useful for data collection, but
can add scheduling/lock jitter; it cannot explain a persistent eager-model
slowdown when `torch_compile_active` is false.

## Camera synchronization retries

The canonical mode name is `camera_align`, covering both interpolation and
bounded forward extrapolation. The former `camera_interpolate` config name and
`CAMERA_INTERPOLATE` constant have been removed; use `camera_align` and
`CAMERA_ALIGN` respectively.

In `camera_align` mode, mark a master camera timestamp consumed only after
all synchronized frames have been assembled successfully. Prefer bracketed
interpolation. If the target is newer than the latest robot state (`low_dim_rw`),
use the latest two strictly increasing samples for constant-velocity forward
projection, bounded by **1.5 / state_feedback.fps seconds** beyond the latest
sample (120 Hz means 12.5 ms). Use the configured rate, not an observed gap that
could grow with stale data. Positions and other linear primitives extend linearly;
quaternions extend shortest-path angular velocity and remain normalized; explicit
`dq` and `twist` velocity primitives retain their latest value. The projected frame
has the adjusted camera target timestamp and is a state estimate, not a new measurement.
Missing/invalid rates, insufficient samples, invalid data, or an exceeded horizon
produce no synchronized frame. Return immediately and retry on the next poll.
Always read the latest camera frame so a newer frame supersedes a failed one.
Do not wait or create a pending-frame queue. Other cameras still use nearest frames.
Successful frames are still published only once; preserve
legacy `binary_match` consumption behavior.

An early consumed timestamp can starve summary updates even while inference RPCs
remain fast. Diagnose observation age separately from RPC duration: repeated
observation timestamps cause the executor to reject replacement chunks and RTC
to exhaust the previous action horizon.

## Verification and safety

- If `rollout_device_preflight.py` fails at `cv2.imshow` with
  `The function is not implemented`, the FastWAM environment imported a
  headless OpenCV build. Restore the GUI package used by the hardware profile:

  ```bash
  FASTWAM_PYTHON="$(python3 scripts/diagnostics/runtime_profiles.py resolve \
    --config-name policy_infer/fastwam/config_infer_fisheye_umi3_dual_fastwam \
    --role fastwam-server --require-verified)"
  uv pip install --python "$FASTWAM_PYTHON" \
    --reinstall opencv-python==4.13.0.92
  ```

- Before a hardware rollout, bind the USB grippers and wrist cameras against
  the same config used by both launch paths:

  ```bash
  FASTWAM_PYTHON="$(python3 scripts/diagnostics/runtime_profiles.py resolve \
    --config-name policy_infer/fastwam/config_infer_fisheye_umi3_dual_fastwam \
    --role fastwam-server --require-verified)"
  uv pip install --python "$FASTWAM_PYTHON" \
    'ruamel.yaml==0.19.1'
  python3 scripts/diagnostics/runtime_profiles.py run \
    --config-name policy_infer/fastwam/config_infer_fisheye_umi3_dual_fastwam \
    --role device -- scripts/diagnostics/rollout_device_preflight.py \
    configs/policy_infer/fastwam/config_infer_fisheye_umi3_dual_fastwam.yaml
  ```

  Add `--gripper_calibrate` when a new gripper calibration is required. This
  command writes the selected camera IDs to the FastWAM source config and the
  robot binding to its referenced device group.

- Run config, architecture, and lifecycle tests in the lightweight environment.
- Run model/runtime tests only in the policy environment with the matching Torch
  and `third_party/nmx_vla` checkout.
- A real-robot test still requires camera/robot/gripper preflight, one device
  runtime, one Server, and one client. Never launch a second model copy just to
  probe a config; use `--validate-only` or the planner when available.
