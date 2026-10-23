---
name: workstation-inference-bootstrap
description: Use when a fresh SuperInference checkout must be prepared for OpenPI inference. It selects a firmware-compatible Flexiv RDK, initializes the required submodules, builds the reviewed source environment, and turns unknown machine inputs into explicit blockers before hardware startup.
---

# Workstation Inference Bootstrap

Use this Skill when the request is equivalent to: "I cloned the repository onto a
new machine; prepare the OpenPI inference environment." It owns environment
bootstrap only. Route robot startup, calibration, rollout recording, and episode control to
[`superinference-operator`](../superinference-operator/SKILL.md) after this Skill
reports readiness.

## Deployment Mode Boundary

The canonical fresh-workstation path is a source checkout plus one reviewed Python
3.12 environment. It does not use packaged release mode. Packaged release mode is
entered only from an extracted archive's wrappers (for example `start_server.sh`)
or when `SUPERINFERENCE_RELEASE_ROOT` is set; it exists for producing relocatable
Control Panel archives and is not part of normal repository startup.

Some locked requirement files and the low-level source environment builder live
under `incubator/control_panel/release/` because source and archive builds share the
same validated dependency profile. That directory location alone does not activate
release mode. Do not set release environment variables during source bootstrap.

An environment is not ready merely because installation finished. Report `ready`
only after the post-install preflight validates Python 3.12, the NVIDIA driver,
real JAX and Torch CUDA operations, the selected Flexiv RDK, NEDF3 imports, Hydra
composition, the selected OpenPI registry, and the real checkpoint contract.

The setup is designed for Linux x86_64 workstations. It deliberately does not
guess a robot firmware version, a vendor credential, or a CUDA driver. A full
rollout-ready environment requires these external inputs:

- Git credentials that can initialize `openpi`, `i2rt`, and `lktech_motor_sdk`;
- an approved cp312 NEDF3 wheelhouse containing `dm3-core==0.2.18` and
  `nmx-msg==2.2.0`;
- the real checkpoint directory configured in `policy.params.ckpt_dir`, including
  real `params/` and normalization assets;
- the target robots' software version, which selects the reviewed Flexiv RDK;
- a working NVIDIA driver on the host.

When an input is unavailable, stop with its exact blocker and expected artifact;
do not create placeholder packages, shadow checkpoints, or guessed registrations.

Install `uv` before the preflight if it is absent. This is the only bootstrap tool
required before the repository's own installers can run:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv --version
```

The checkout itself still requires Git access, and the curl-based uv installer
requires `curl`; install those two host tools with the machine's package manager
when the base OS image omits them. NVIDIA driver installation remains host-owned
and is never attempted by the repository bootstrap.

## First Action: Read-only Preflight

From the repository root, run the bundled preflight before installing or launching:

```bash
uv run --python 3.12 --no-project python \
  .claude/skills/workstation-inference-bootstrap/scripts/inference_preflight.py \
  --repo-root "$(git rev-parse --show-toplevel)" \
  --target flexiv \
  --robot-software-version <robot-software-version>
```

Without `--device-python`, the tool only inspects files, executables, and
`nvidia-smi`; it never imports a vendor SDK or connects to hardware. With an
installed environment supplied, it additionally runs the hardware-free profile and
checkpoint probes. It exits `0` only if no `BLOCKER` remains. `ACTION` entries are
required before a real policy launch but do not make the machine install unsafe.
The OpenPI checkpoint comes from the selected config's `policy.params.ckpt_dir`;
the installed-profile probe composes that config and validates the real checkpoint
contract. Use `SUPERINFERENCE_RECORDING_PYTHON` only as a one-off compatibility
override for recording, and create a generated config when deployment values differ.

## Flexiv RDK Gate

Robot software, not Elements UI version, selects the RDK. The authoritative matrix
is in [`references/flexiv_rdk_compatibility.json`](references/flexiv_rdk_compatibility.json)
and comes from the supplied Flexiv compatibility screenshots.

| Robot software | Compatible RDK | Repository status |
| --- | --- | --- |
| `v3.6` | `v1.4` | not reviewed |
| `v3.7` | `v1.5` | not reviewed |
| `v3.7.1` | `v1.5.1` | not reviewed |
| `v3.8` | `v1.6` | not reviewed |
| `v3.9` | `v1.7` | reviewed as `1.7.0` |
| `v3.10` | `v1.8` | reviewed as `1.8.0` |
| `v3.11` | `v1.9` | reviewed as `1.9.0` |
| `v3.11.1` | `v1.9.1` or `v1.9.2` | reviewed; preflight recommends `1.9.2` |
| `v3.11.2` | `v1.9.3` | firmware-compatible, adapter/release not reviewed |
| `v3E.0` | `v2.0` | firmware-compatible, adapter/release not reviewed |
| `v3E.1` | `v2.1` | firmware-compatible, adapter/release not reviewed |

The source/release adapter is reviewed only for `1.7.0`, `1.8.0`, `1.8.1`,
`1.9.0`, `1.9.1`, and `1.9.2`. A robot running `v3.11.2`, `v3E.0`, or `v3E.1`
therefore stops at the preflight gate. Do not install RDK `1.9.3` or `2.x` as an
unreviewed replacement and do not use an Elements version as evidence of firmware
compatibility. The short discussion needed at that gate is: obtain the approved
vendor wheel, audit the 2.x adapter API, and complete the designated read-only and
hardware validation before admitting it to the reviewed list.

## Flexiv Elements Discovery

Elements Studio may be unpacked manually rather than registered with the system
package manager. Locate the active installation without launching another GUI:

```bash
pid=$(pgrep -n -f '/FlexivElementsStudio( |$)')
readlink -f "/proc/$pid/exe"
readlink -f "/proc/$pid/cwd"
rg -l 'FlexivElementsStudio' \
  /usr/share/applications ~/.local/share/applications ~/Desktop ~/\u684c\u9762 2>/dev/null
```

The executable does not reliably expose a CLI version. Use the bundled detector
to recursively scan all candidate archives and compare their embedded executable
hashes with the installed binary:

```bash
python .claude/skills/workstation-inference-bootstrap/scripts/detect_flexiv_elements.py
python .claude/skills/workstation-inference-bootstrap/scripts/detect_flexiv_elements.py \
  --executable /absolute/install/path/FlexivElementsStudio \
  --archive-root /path/to/downloads --archive-root /path/to/vendor/archive.zip
```

By default it finds the active process and recursively scans `~/Downloads` and
`~/\u4e0b\u8f7d`. It reports every exact archive match, tolerates unrelated archives
and duplicate copies of the same version, and fails on no match or conflicting
version labels. It streams archive members without extracting them. The detector
also reads `user_data_ui/License/featureID*.json` and reports cached robot serial
numbers with their `RobotControlApp` versions. Check this cache before asking the
user for a version that Elements has already recorded.

Treat cached robot data as a historical observation: match the requested serials,
require all target robots to report the same compatible software line, and report
the source files. A patch release such as `RobotControlApp v3.10.3` belongs to the
compatibility table's `v3.10` line. If target robots disagree, the cache is absent,
or the line is not in the matrix, stop instead of choosing an RDK. Do not use the
separate `FlexivElements` field for this decision.

This is only the UI application version. It is useful for finding the installed
launcher and inspecting robot information, but it is not the robot software
version used by the RDK compatibility gate. Read each connected robot's software
version from Elements/vendor tooling before selecting an RDK.

## OpenPI: Supported Source Profile

OpenPI is the supported real-robot setup path. It uses one Python 3.12 main
environment for the Control Panel, device runtime, and OpenPI policy. The existing
`create_source_py312_env.sh` installs OpenPI from its frozen `uv.lock`, applies the
repository overlay, and preserves OpenPI's NumPy ABI while adding `i2rt` without
dependency replacement. Do not substitute a bare editable OpenPI install for it.
After successful bootstrap, the installer registers this interpreter as the
machine-local `openpi-main312` profile and binds the selected config's launcher,
planner, device, policy, recording, and converter roles. Use
[`runtime-profile-management`](../runtime-profile-management/SKILL.md) for later
inspection or repair. `SUPERINFERENCE_INFERENCE_PYTHON` remains a one-off override
when a separate policy or FastWAM model environment is required.

Source runtime resolution must never use an interpreter or rollout catalog in
another checkout. It uses explicit arguments and environment overrides first,
then the repository-local verified profile binding, then the subsystem's legacy
fallback. If the selected executable is unavailable or its profile is stale,
startup fails with a configuration error. Additional replay catalogs must be
supplied explicitly.

### Canonical automated bootstrap

After collecting the external inputs, have the AI print and review the deterministic
plan first. This default mode does not write anything:

```bash
uv run --python 3.12 --no-project python \
  .claude/skills/workstation-inference-bootstrap/scripts/bootstrap_openpi_source.py \
  --repo-root "$(git rev-parse --show-toplevel)" \
  --env-dir "$HOME/.local/share/superinference/openpi-main312" \
  --nedf-wheelhouse /absolute/path/to/approved/nedf3-wheelhouse \
  --robot-software-version <robot-software-version>
```

After reviewing the resolved RDK and every absolute path, add `--execute`. The
installer downloads a uv-managed Python 3.12, initializes only the three required
submodules, builds the frozen unified environment, installs the firmware-compatible
RDK, runs the post-install preflight, and writes the ignored local runtime profile.
It never starts the Control Panel or opens hardware. On success it prints the
profile-backed Control Panel launch command.

If post-install validation says the selected `pi_config_name` is absent, inspect the
real checkpoint's asset layout and add the narrow local OpenPI registry alias before
rerunning the final preflight. For the 0813 gift deployment this registration is
intentionally workstation-local and must not change the submodule commit or parent
gitlink. The validation, rather than a copied checkpoint directory, is the gate.

The commands below are the manual repair path when one automated step must be
diagnosed independently.

Initialize exactly its required third-party sources:

```bash
git submodule update --init third_party/openpi third_party/i2rt third_party/lktech_motor_sdk
```

Create an unused, absolute environment directory outside the repository. The
script refuses to overwrite an existing path:

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
MAIN_ENV="$HOME/.local/share/superinference/openpi-main312"
NEDF_WHEELHOUSE="/absolute/path/to/approved/nedf3-wheelhouse"
PYTHON_BIN=python3.12 "$REPO_ROOT/incubator/control_panel/release/create_source_py312_env.sh" \
  "$MAIN_ENV" "$NEDF_WHEELHOUSE"
```

Once the preflight selects a reviewed RDK, install that exact profile. For example,
for robot software `v3.11.1` the recommended selection is `1.9.2`:

```bash
uv pip install --python "$MAIN_ENV/bin/python" \
  --build-constraints "$REPO_ROOT/incubator/control_panel/release/requirements/build_constraints.txt" \
  -r "$REPO_ROOT/incubator/control_panel/release/requirements/flexiv_rdk_1_9_2.txt"
```

Confirm that the environment received the intended RDK distribution, without
opening a robot connection:

```bash
python "$REPO_ROOT/.claude/skills/workstation-inference-bootstrap/scripts/inference_preflight.py" \
  --repo-root "$REPO_ROOT" --target flexiv \
  --robot-software-version v3.11.1 --rdk-version 1.9.2 \
  --device-python "$MAIN_ENV/bin/python"
```

For final readiness, include the canonical config so the same probe
also verifies GPU execution, recording dependencies, Hydra composition, registry,
and normalization dimensions:

```bash
"$MAIN_ENV/bin/python" \
  .claude/skills/workstation-inference-bootstrap/scripts/inference_preflight.py \
  --repo-root "$REPO_ROOT" --target flexiv \
  --robot-software-version v3.11.1 --rdk-version 1.9.2 \
  --config policy_infer/openpi/config_infer_fisheye_umi3_dual_openpi \
  --device-python "$MAIN_ENV/bin/python"
```

Verify the installed stack without robot connectivity:

```bash
"$MAIN_ENV/bin/python" - <<'PY'
from importlib import metadata
import jax
import jax.numpy as jnp
import openpi
import torch

print({
    "openpi": metadata.version("openpi"),
    "torch": torch.__version__,
    "torch_cuda": torch.cuda.is_available(),
    "jax_backend": jax.default_backend(),
})
print("jax_cuda_sum", float(jnp.arange(8, dtype=jnp.float32).sum()))
print("torch_cuda_sum", torch.arange(8, device="cuda").sum().item())
PY
```

The reviewed OpenPI profile pins `torch==2.7.1`, `torchvision==0.22.1`,
`jax[cuda12]==0.5.3`, and `transformers==4.53.2`; the frozen OpenPI lock remains
the source of truth. CUDA user-space libraries come from the environment, but the
NVIDIA driver remains host-owned. A failed `torch.cuda.is_available()` or JAX GPU
backend is an environment blocker, not a reason to install another Torch version.
The preflight reports installed GPU memory but cannot infer a checkpoint's memory
requirement; compare that result with the selected checkpoint's deployment record
before approving a real launch.

NEDF3 recording is part of the reviewed unified Python 3.12 source profile. The
private wheels are not fetched from a public index: pass the approved cp312/py3
wheelhouse to the source builder. To add or repair recording support in an existing
reviewed Python 3.12 environment, run:

```bash
incubator/control_panel/release/install_source_recording_profile.sh \
  "$MAIN_ENV/bin/python" "$NEDF_WHEELHOUSE"
```

Do not use the old cp310 wheels in the Python 3.12 environment.

## Common Issues

- If `rollout_device_preflight.py` reports that `ruamel.yaml` is required, the
  unified source environment predates the operator-script dependency. Install
  `ruamel.yaml==0.19.1` into that environment, or rebuild it from the current
  `main_py312.txt`. Do not fall back to PyYAML for this workflow because the
  preflight intentionally preserves comments and quoting when it writes the
  selected device config.
- If LK gripper startup reports `No module named 'serial'`, install the reviewed
  `pyserial==3.5` package into the unified source environment. The import name and
  distribution name differ; do not install the unrelated `serial` distribution.
- If the final source profile reports `No module named 'can'`, the environment was
  built before the unified profile included `python-can==4.5.0` or was assembled
  manually. Rebuild it at a new path with the canonical bootstrap; do not patch an
  otherwise unverified environment and call it rollout-ready.
- If the source builder reports a missing path under `incubator/third_party`, its
  repository-root calculation has regressed. The release script lives three levels
  below the repository root (`incubator/control_panel/release`) and must resolve
  `repo_root` with `../../..`. Keep the corresponding release-runtime assertion in
  place so bootstrap fails in CI rather than on a fresh workstation.
- On an RTX 50-series GPU, `torch.cuda.is_available()` can return true even when
  the installed wheel lacks `sm_120` kernels. Require the real JAX and Torch CUDA
  operations in the import probe above. A Torch `no kernel image is available`
  error means the frozen OpenPI profile needs a reviewed Blackwell-compatible
  wheel update; do not silently replace its pinned Torch build on one workstation.

## Handoff To Operation

After the preflight and import probe pass, report the absolute interpreter paths,
selected RDK, GPU/driver, selected config overlay, checkpoint path, and whether the
NEDF sidecar is needed. Then hand real device/policy startup to
`superinference-operator`; starting devices, opening cameras, policy control,
calibration, and rollout artifact creation are outside this Skill and remain
hardware-facing operations.

## Validation

When editing this Skill or its bundled preflight, run:

```bash
python -m py_compile .claude/skills/workstation-inference-bootstrap/scripts/inference_preflight.py
python -m py_compile .claude/skills/workstation-inference-bootstrap/scripts/detect_flexiv_elements.py
python -m py_compile .claude/skills/workstation-inference-bootstrap/scripts/bootstrap_openpi_source.py
python .claude/skills/workstation-inference-bootstrap/scripts/detect_flexiv_elements.py
python .claude/skills/workstation-inference-bootstrap/scripts/inference_preflight.py \
  --repo-root . --target flexiv \
  --robot-software-version 3.11.1 --rdk-version 1.9.2 --json
python "$HOME/.codex/skills/.system/skill-creator/scripts/quick_validate.py" \
  .claude/skills/workstation-inference-bootstrap
python -m pytest tests/scripts/test_workstation_inference_bootstrap.py -q
```
