# Security and operations

The control panel has no authentication or TLS. Run it on loopback and reach it
through an SSH tunnel. Treat any non-loopback bind as privileged remote process
control.

The API does not capture global application logs. Only messages explicitly
created by the controller (config name, canonical device key, action, and server
lifecycle) are returned by `/api/logs`. Config inspection emits only validated
class/id/capability/shape/dtype/FPS fields.

Stopping a motion-capable device is deliberately fail-safe: after the configured
grace period the process remains owned and visible in `stop_timeout`; the panel
does not SIGKILL it. An operator must inspect hardware state and logs before any
further action.

If a control-panel process is killed externally, use the UI status probe or
inspect `/dev/shm` before restarting. Atomic ownership checks will refuse to
replace a still-live or orphaned producer segment.

The registry deliberately does not auto-unlink a residual name after device
cleanup fails. POSIX offers no atomic "unlink only this inode" operation, so
leaving an orphan is safer than racing a replacement producer.

Shared-control implementations with a legacy "attach, unlink, recreate" path
are intercepted in the isolated worker. Their attached handle has no unlink
authority; an external flag causes an ownership conflict instead of deletion.

## Manual LKTech gripper control

Manual calibration and position checks run in a spawned isolated device worker that
imports only the bundled LKTech wrapper and constructs `Gripper(robot=None)`.
It never imports, constructs, connects, enables, or commands an RDK robot.

The browser can submit only `side`, a `0..100` opening percentage, the current
controller revision, and an idempotency key. Serial ports, motor IDs, baudrate,
calibration paths, velocity, and SDK units come from the selected effective
config. The YAML side-to-port order is authoritative; the historical `port`
field inside a calibration JSON file is not used for binding.

Every physical command uses a short motor session: enable the selected
gripper, perform one calibration or position command, wait for the motion to
settle, read the actual width, disable and disconnect, then exit the worker.
There is no browser heartbeat that can keep a motor enabled. Position requests
are coalesced latest-wins and dispatched serially at no more than 5 Hz.

Commands fail closed while any configured device, runtime, inference, or
recording process is active, when a Rizon shared-memory object exists, or when
another process holds the configured serial device. Config switches, binding
changes, device/runtime starts, and server shutdown stop the gripper worker
before continuing.

Calibration is physical motion across both limits. A dual calibration runs the
two sides serially. New JSON files are prepared and fsynced before commit;
existing files are backed up, a transaction manifest is written, and a partial
multi-file commit is rolled back.
