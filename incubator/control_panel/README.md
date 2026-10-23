# SuperInference Control Panel

The Control Panel is an optional web UI over the same Hydra configs used by
`device_starter.py`. It does not maintain a second config namespace or device
stack.

Start with the versioned [operator manual](docs/README.md). Source-checkout
launches must use verified runtime-profile bindings:

```bash
CONFIG=policy_infer/openpi/config_infer_fisheye_umi3_dual_openpi
python3 scripts/diagnostics/runtime_profiles.py doctor \
  --config-name "$CONFIG" --role launcher
python3 scripts/diagnostics/runtime_profiles.py run \
  --config-name "$CONFIG" --role launcher -- \
  -m incubator.control_panel --config-name "$CONFIG"
```

The default listener is `127.0.0.1:18088`. For remote access, keep it on
loopback and use an SSH tunnel with the actual host and account:

```bash
ssh -L 18088:127.0.0.1:18088 <user>@<control-panel-host>
```

Server startup and config selection do not themselves start hardware. Device,
runtime, inference, recording, calibration, replay, and Home operations retain
their own readiness and safety gates. Do not run a second producer stack for
the same physical devices or shared-memory names.

Release packaging is documented in [release/README.md](release/README.md).
Third-party asset provenance is recorded in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
