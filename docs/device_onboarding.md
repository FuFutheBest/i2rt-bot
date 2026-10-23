# Device Onboarding

This page defines the stable integration checklist. Keep hardware-specific operating procedures in an applicable
Skill under `.claude/skills/`.

## Add the implementation

1. Start from `devices/base.py`.
2. Place cameras in `devices/cameras/`, robots in `devices/robots/`, and other focused adapters in an appropriate
   `devices/` module.
3. Keep vendor SDK imports inside the implementation module. Importing `devices` must remain hardware-free.
4. Add the class's module and attribute to the lazy specs in `devices/__init__.py`.
5. Add a group config under `configs/devices/`; put policy-inference-only variants under
   `configs/devices/infer/`, then reference the appropriate group from a top-level config only when needed.
6. Use `utils.logger_config.logger`, explicit type annotations, and
   `utils.time_control.precise_loop_timing` for fixed-rate loops.

Do not add new implementation to the compatibility facades `devices/camera.py`, `devices/robot.py`, or
`devices/web_camera_openpi.py` unless compatibility itself is the change.

## Define the contract

Document and test:

- class registry name and `device_class_id` (the zero-based position in the full
  device list, shared across classes and including disabled devices);
- physical selection separately: top-level `camera.<role>_capture_id`, passed as
  `capture_id`; robot serials and serial ports never determine logical IDs;
- data shape, dtype, units, axis order, and quaternion convention;
- SHM names and ownership;
- timestamp source and hardware-latency interpretation;
- control SHM shape and command semantics for robots;
- calibration and optional SDK requirements;
- cleanup behavior after normal stop and process failure.

Use SciPy/Rerun quaternion order `(x, y, z, w)` unless an SDK boundary explicitly requires conversion.

## Hardware-free acceptance

At minimum, add tests for lazy registry resolution and config references. Where possible, test parsing or sample
generation without opening the SDK or device:

```bash
python -m pytest tests/devices tests/config/test_registry_references.py -q
python -m pytest tests/architecture/test_incubator_boundaries.py -q
```

The current device registry test is `tests/devices/test_lazy_registry.py`.

## Hardware acceptance

Unit tests do not authorize a physical run. On the target workstation, a qualified operator must separately verify
SDK and firmware versions, serial/IP selection, calibration, frame rate, timestamps, disconnect handling, SHM
cleanup, command limits, homing, emergency stop, and safe restart behavior. Begin with motion disabled or reduced
limits where the device permits it.

Install pip-distributed adapters with
`uv pip install -r requirements/requirements-hardware.txt`. Install vendor SDKs such as
Flexiv RDK, Hik MVS, and LKTech separately for the selected workstation; they are intentionally not base
dependencies.

For LKTech gripper work, follow
[`lktech-gripper-control`](../.claude/skills/lktech-gripper-control/SKILL.md). Do not copy its calibration procedure
into this stable overview.
