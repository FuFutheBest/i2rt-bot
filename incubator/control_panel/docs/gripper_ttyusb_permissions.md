# ttyUSB and Yam CAN permission recovery

When either configured gripper ttyUSB port is not readable and writable, the
control panel opens a password dialog. The submitted sudo password is fed once
to sudo through stdin, and the backend directly executes `chmod 0666` against
the already-opened descriptors for the exact left/right gripper ports disclosed
by the dialog. When the selected config contains a real `YamHeadRobot`, the
same dialog also reports its strict `canN` target. If that interface exists but
is down or is not configured for 1 Mbps, the backend executes only these fixed
commands:

```text
sudo ip link set canN down
sudo ip link set canN up type can bitrate 1000000
```

The CAN interface identity is revalidated between both commands and the final
state must be `UP` with bitrate `1000000`. A missing interface is reported as
unavailable and is not misrepresented as a password problem.

There is no helper installation, passwordless repair endpoint, or sudoers
setup in this flow. The fixed direct chmod and `ip link` argument vectors are
the only privileged recovery paths exposed by the control panel.

## Device runtime selection

In a source checkout, the gripper worker uses the same validated executable
that is already running the control panel when
`SUPERINFERENCE_DEVICE_PYTHON` is not set. This keeps source deployments
portable and avoids falling back to another machine's historical absolute
path. A packaged release remains fail-closed: it accepts only
`runtime/python312/bin/python3.12` under `SUPERINFERENCE_RELEASE_ROOT`.

An external process that already has a configured ttyUSB node open remains a
real ownership conflict. The API blocker includes the affected port plus the
visible PID and bounded command line so the operator can distinguish that
conflict from a missing runtime. The control panel does not ignore or stop an
external process automatically.

The password is not persisted or placed in argv, logs, environment variables,
URLs, browser storage, or response bodies. The request is still accepted over
the production plain-HTTP URL by explicit operator choice.

Mode `0666` makes the selected gripper serial ports accessible to all local
users. The controller pins each target inode with `O_PATH|O_NOFOLLOW`, checks
the binding revision before and after sudo, and rejects unplug/replacement or
binding changes rather than redirecting chmod to another device. CAN targets
are restricted to the current config's `canN` names and are never passed
through a shell.

## Runtime authorization state

A successful repair is not treated as a blind, permanent path grant. Before a
connection, the backend revalidates that every target is the canonical direct
`/dev/ttyUSBN` path, is a non-symlink character device, still has the same
`st_dev`/`st_ino`/`st_rdev` identity, has read/write access for the owning
process, and
has both world read and world write mode bits (`mode & 0006 == 0006`). No sudo,
chmod, or serial open is performed by this read-only validation.

For the same live device node, an existing `0666` grant is automatically
re-adopted after camera operations, unrelated controller revision changes,
lease TTL expiry, lease revocation, and control-panel server restart. These
events therefore do not ask the operator for the sudo password again while the
direct character-device identity, mode, and effective access remain valid.

Within one server process, a trusted ttyUSB path is identity-bound. If the node
disappears, is replaced or re-created with a different identity, becomes a
symlink/non-character device, loses either world read/write bit, or is no
longer effectively readable and writable, permission becomes required again.
A replacement identity is not auto-adopted even when it is already `0666`; it
must complete a new explicit two-phase authorization commit. A chmod result
whose binding/revision commit fails or is discarded also remains untrusted and
cannot be adopted by a later status poll.

Permission status includes an opaque SHA-256 `context_id` derived only from the
sorted target path and character-device identities. It remains stable across
revision, TTL, permission-state, and server-restart changes for the same nodes,
and changes when a target identity changes. The browser uses this identity
context rather than the global controller revision to decide whether a new
password prompt is necessary.
