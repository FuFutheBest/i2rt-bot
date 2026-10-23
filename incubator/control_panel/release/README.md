# Self-contained Linux release

The release builder packages the current OpenPi control-panel application with
a relocatable Python 3.12 runtime and one shared site-packages tree for the
control panel, devices, OpenPi policy, NEDF3 recording/conversion, and the
relevant `third_party` SDKs. All profile inputs must use the same prepared
Python 3.12 environment; the builder stores one unified runtime tree. The archive
also includes the private NEDF3 SDK wheels (`dm3-core 0.2.18` and
`nmx-msg 2.2.0`),
the validated AV1 FFmpeg build, and six FlexivRDK variants. The target machine
does not need conda, uv, pip, or a Python installation.

The exported application omits development-only IDE/agent metadata, test
suites, and Windows launchers. Runtime code, configs, calibration data,
visualization assets, and hardware SDK snapshots remain included.

NEDF3 is installed from the builder's pinned cp312 wheelhouse, so recording and
conversion use the same Python 3.12 runtime as the control panel and inference
client. A release build must provide the exact supported NEDF3 wheels; do not
substitute an older Python 3.10 sidecar.

The unified main profile deliberately keeps OpenPi's NumPy 1.26/OpenCV 4.11
and rerun-sdk 0.23 ABI, then installs i2rt without dependency replacement.
Import and unit probes pass; real Yam hardware validation is required before
production promotion. `gym-aloha` remains installed only because it is part of
OpenPi's upstream inference lock; its `mujoco<3` metadata is intentionally not
used by the real-robot control panel, which requires i2rt-compatible Mujoco 3.
The source environment builder fails if `uv pip check` reports anything beyond
these three documented upstream metadata conflicts.

## Flexiv RDK selection

Elements UI version does not determine robot firmware compatibility. The
release therefore bundles RDK `1.7.0`, `1.8.0`, `1.8.1`, `1.9.0`, `1.9.1`, and
`1.9.2`. RDK 1.6 lacks adapter APIs used by this project, while the 2.x API has
not yet been reviewed for real hardware.

When a Rizon start is requested, each candidate runs in a separate short-lived
process. The probe only constructs `flexivrdk.Robot` and reads operational
status; it never enables the robot, changes mode, or sends motion. The first
version accepted by every configured arm is injected only into that Rizon
worker. A successful version is tried first for later starts in the same
control-panel session.

Set `SUPERINFERENCE_FLEXIV_RDK_VERSION=1.9.0` to restrict probing to one known
firmware-compatible version. Per-candidate timeout defaults to 15 seconds per
robot and can be changed with `SUPERINFERENCE_FLEXIV_RDK_PROBE_TIMEOUT`.

The complete release targets Ubuntu 22.04+ x86_64 because RDK 1.9.x wheels
require glibc 2.35. Elements discovery remains visible as diagnostics only.

## FastWAM model environment

FastWAM keeps the heavyweight NMX model in its own process. The release's
bundled `bin/python-policy` is the OpenPI client/runtime profile; it is not
automatically a valid NMX model environment because `third_party/nmx_vla`
uses a different Transformers and CUDA dependency set. Provision a CUDA-
matched environment from `third_party/nmx_vla/requirements.txt` and opt into it
explicitly when launching the panel:

```bash
SUPERINFERENCE_FASTWAM_SERVER_PYTHON=/absolute/path/to/nmx/bin/python \
SUPERINFERENCE_FASTWAM_SERVER_PYTHON_EXTERNAL=1 \
./start_server.sh
```

Without that opt-in, the packaged launcher intentionally resets the server
interpreter to `bin/python-policy`, so an inherited path from an older release
cannot change the model process. The FastWAM Server reports missing NMX
dependencies in its startup progress/log; resolve that environment issue
before starting the client. The lightweight client remains on the bundled
profile.

## Host prerequisites

The archive includes application dependencies, but hardware and kernel-level
requirements remain host-owned:

- Ubuntu 22.04+ x86_64
- NVIDIA driver compatible with the bundled CUDA 12 OpenPi environment
- Flexiv robot network configuration and RDK license
- camera, `/dev/ttyUSB*`, and CAN permissions required by the selected config
- enough disk for the release and checkpoint; checkpoints are not bundled

Both the AV1 encoding runtime and a static software AV1 decoder are bundled;
the target host does not need a system FFmpeg installation.

Create the preferred shared source profile from an initialized Linux checkout:

```bash
PYTHON_BIN=python3.12 incubator/control_panel/release/create_source_py312_env.sh \
  /absolute/path/to/main312 /absolute/path/to/approved/nedf3-wheelhouse
```

The script installs OpenPi from its frozen `uv.lock` (replacing its obsolete,
non-cp312 `mujoco 2.3.7` with the overlay's `mujoco 3.9.0`), applies
`incubator/control_panel/release/requirements/main_py312.txt`, installs LKTech normally,
and adds i2rt without replacing OpenPi's NumPy ABI. The legacy split inputs
remain available as root `requirements_device.txt` and
`incubator/control_panel/release/requirements/policy.txt`. Recording dependencies are
merged into the shared Python 3.12 environment from
`incubator/control_panel/release/requirements/recording.txt` and the private wheelhouse.
An existing reviewed Python 3.12 environment can add or repair the same profile with
`incubator/control_panel/release/install_source_recording_profile.sh`.
The builder inspects wheel metadata, requires exactly the two supported NEDF3
SDK distributions, copies only those wheels, and records their SHA256 values in
`manifest/release.json`.

## Run

For an operator-facing walkthrough, use
[`../docs/01-deployment/README.md`](../docs/01-deployment/README.md). Release
pages should use [`RELEASE_BODY_TEMPLATE.md`](RELEASE_BODY_TEMPLATE.md) and link
to the target commit's documentation instead of duplicating version-specific
deployment commands.

```bash
tar --zstd -xf superinference-openpi-*-linux-x86_64.tar.zst
cd superinference-openpi-*-linux-x86_64
./bin/superinference-doctor
./start_server.sh
```

`start_server.sh` listens on `0.0.0.0:18088` by default. Override with
`SUPERINFERENCE_CONTROL_PANEL_HOST` / `SUPERINFERENCE_CONTROL_PANEL_PORT`, or
pass normal control-panel arguments after the script name.

Force a known robot-firmware RDK when needed:

```bash
SUPERINFERENCE_FLEXIV_RDK_VERSION=1.9.0 ./start_server.sh
```

The release writes conversion output and activation receipts under
`~/.local/share/superinference-control-panel` by default. Set
`SUPERINFERENCE_DATA_ROOT` to relocate mutable data outside the extracted
release.

## Build

Run the builder on the Linux host that owns the already-validated profiles. It
exports the main repository from `HEAD`, snapshots the specified submodule work
trees, records dirty submodule patches, downloads the six reviewed RDK wheels
from PyPI with SHA256 verification, runs the read-only doctor, and produces a
Zstandard-compressed archive plus checksum.

```bash
python -m incubator.control_panel.release.build_release \
  --version v1.0.0 \
  --repo-root /path/to/SuperInference \
  --output-dir /path/to/releases \
  --device-python /path/to/main312/bin/python \
  --recording-python /path/to/main312/bin/python \
  --converter-python /path/to/main312/bin/python \
  --policy-python /path/to/main312/bin/python \
  --ffmpeg-root /path/to/ffmpeg-av1 \
  --full-decode-ffmpeg /path/to/static-software-av1-ffmpeg \
  --vendor-wheelhouse /path/to/nedf3-sdk-wheels \
  --submodule-source openpi=/path/to/openpi \
  --submodule-source lktech_motor_sdk=/path/to/lktech_motor_sdk \
  --submodule-source i2rt=/path/to/i2rt \
  --submodule-source MvImport=/path/to/MvImport \
  --submodule-source nmx_vla=/path/to/nmx_vla
```

`manifest/release.json` records the main commit, every source profile, installed
package inventory, RDK wheel hashes, and submodule snapshot state. A dirty
submodule is preserved as a patch in `manifest/` instead of being silently
treated as a clean dependency.
