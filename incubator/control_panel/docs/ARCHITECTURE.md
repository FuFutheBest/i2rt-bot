# Architecture

The Flask application owns one lifecycle-manager generation at a time. Config
composition happens without hardware. The controller lock serializes lifecycle
replacement plus operation reservation and commit, while slow device work runs
outside it so authoritative status reads remain available. This prevents a
request holding an obsolete manager reference from starting an unmanaged worker
during a config switch.

The browser opts lifecycle requests into `Prefer: respond-async`. The server
registers the transition token, accepts the request with HTTP 202, and runs the
bounded operation outside the request thread. Different device scopes may run
in parallel, while all workflow scopes share one ordered lane. A stop cancels
older queued starts for the same lane and signals a running workflow start to
leave its readiness wait. Legacy clients that omit `Prefer` retain the original
synchronous response contract.

HTTP acceptance never means that hardware reached the target state. One shared
`StateSyncBroker` samples process, SHM, workflow, recorder, and gripper state;
SSE publishes those observations to every tab. The UI displays a local pending
transition immediately, but only an authoritative observation completes it.

Each manual start creates a dedicated `multiprocessing` spawn worker and a
one-way ownership event pipe. The child temporarily replaces
`utils.shm_utils.create_shared_memory` in its own interpreter so every primary
and secondary allocation uses exclusive `SharedMemory(create=True)`. Claims and
conflicts are reported to the parent; no existing SHM is unlinked during start.

Creator channels retain unlink authority when made read-only, so normal device
shutdown can clean both state and control SHMs. Registry finalization only
closes handles and detaches `resource_tracker`; it never performs a name-based
fallback unlink. Residual SHM is exposed as orphaned/cleanup-required and blocks
restart until an operator resolves it.

The browser never receives arbitrary resolved config. It polls allowlisted
device lifecycle state, JPEG camera snapshots/streams, and normalized seven-axis
Flexiv joint arrays. The dual-arm renderer loads only assets vendored under this
module.

`/api/robots/state` also exposes a narrow `configured_tools.left/right`
contract derived only from the selected effective Hydra config. A static tool
frame is available only when config contains an explicit finite
`flange_T_tool` translation and normalized quaternion. The viewer parents that
frame to the URDF `flange`, so home/live joint FK moves it. Fresh 44-value SHM
TCP poses are already absolute controller TCP poses: they hide and replace the
config frame and are never composed with its offset. Empty `tool_name` means
the controller's current active tool is intentionally retained; without RDK
the panel reports `controller-active-unknown` and renders no invented offset.

Preview readers use a local protocol implementation and an `O_RDONLY` POSIX
file mapping. The parent web process never imports the repository logger or
hardware registries. Device registries and producer SHM helpers are imported
only after entering the spawned worker.

## Source ownership

New control-panel implementation is kept under `incubator/control_panel/`:

- `gripper_control.py` owns direct, short-lived LKTech serial workers. There is
  no persistent gripper server or WebSocket transport in the runtime.
- `release/` owns the Linux release builder, doctor, runtime bootstrap, and
  release-specific dependency profiles.
- `static/assets/` owns every URDF and mesh used by the browser.
- `tools/` owns NEDF3 inspection, conversion, merge, and export commands.
- `tests/` owns control-panel unit, frontend, release, and integration tests.

The small integration surface that must remain in existing SuperInference
locations is deliberate: Hydra configs stay in `configs/`; recorder, policy,
device, and action-manager hooks stay beside the subsystem they extend. Those
files import shared protocols and services from `control_panel` rather than
duplicating them. The pre-existing root `requirements_device.txt` remains the
source-device environment contract, while release-only profiles live under
`incubator/control_panel/release/requirements/`.

Remote deployment snapshots, one-off patch directories, extracted release
trees, generated fixtures, and source CAD archives are not repository source.
They must be stored outside the checkout instead of under a `remote/` or patch
directory in the project root.
