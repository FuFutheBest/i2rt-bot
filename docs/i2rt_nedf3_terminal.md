# I2RT NEDF3 Terminal Collection

DeviceStarter owns the complete teleoperation rig and both physical cameras.
The default config binds the left USB2 and right USB3 openMV cameras through
their stable `/dev/v4l/by-id` paths. Override either path when needed, then
start the hardware stack:

```bash
# Optional overrides:
# export SI_I2RT_LEFT_CAMERA=/dev/v4l/by-id/<left-camera>
# export SI_I2RT_RIGHT_CAMERA=/dev/v4l/by-id/<right-camera>
.venv/bin/python device_starter.py --config-name i2rt_data_collect_async
```

Press Ctrl+C once to stop the stack. DeviceStarter forwards the shutdown to
the teleoperation owner and cameras, waits for their hardware cleanup, and
then returns to the prompt. This normally takes a few seconds; the configured
fallback timeout is 45 seconds for the composite teleoperation owner.

Start the recorder in a second terminal:

```bash
.venv/bin/python -m consumer.io.async_data_saver_i2rt_NEDF3 \
  --config configs/i2rt_nedf3_saver.yaml
```

The right leader's second button starts or finishes an episode. The left
leader's second button discards the active episode. Keyboard controls remain
available: press Enter to start or stop, enter `d` to discard, and enter `q`
to stop the recorder.

The two physical streams are stored as right `camera_0` and left `camera_1`.
The recorder generates black 360x640 frames for the missing head `camera_2`.
Arm-state topics are recorded at 90 Hz; gripper and camera topics are recorded
at 30 Hz. The device processes continue to publish robot state at 100 Hz and
capture both physical cameras at 30 Hz.
Finished episodes are placed below `data/i2rt_nedf3`; incomplete or invalid
episodes are moved to `quarantine`.

For hardware diagnostics without DeviceStarter, the same teleoperation runtime
can still be launched directly with `consumer/i2rt_bimanual_teleop.py`. Never
run that diagnostic command at the same time as the collection DeviceStarter.

For an unattended recording, use `--auto-start --duration SECONDS`.

Run the complete hardware-free writer/reader check with:

```bash
.venv/bin/python scripts/diagnostics/smoke_i2rt_nedf3_unreal.py
```

The I2RT state layout is fixed as:

```text
[left joint0..5, left gripper, right joint0..5, right gripper]
```

Every `/joint` NEDF3 topic contains an `ArmState` with six `JointState`
entries. Every `/hand_state` topic contains one gripper `JointState` inside a
`HandState` message.
