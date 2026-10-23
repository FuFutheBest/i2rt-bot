# Testing

The root suite emphasizes hardware-free configuration, import, shared-memory, and action-flow contracts. Install
the development profile before running it:

```bash
uv pip install -r requirements/requirements-dev.txt
```

Hydra and OmegaConf are base dependencies, so missing either is an installation error rather than a reason to skip
pure config logic. Tests may skip only when they exercise a clearly optional model, hardware SDK, OS facility, or
external tool, and every skip must state that missing capability precisely.

## Root commands

```bash
python -m pytest tests -q -m "not hardware and not manual"
python -m pytest tests/config -q
python -m pytest tests/consumer tests/manager tests/utils -q
python -m pytest tests/integration/test_fake_inference_smoke.py -q
python -m pytest tests/architecture/test_incubator_boundaries.py -q
python -m pytest tests/architecture -q
```

Use the complete root command before integration. Narrow commands are useful while changing one contract; the test
tree itself is the current inventory.

## Control-panel commands

`incubator/control_panel/` is isolated from the stable core and owns a separate, larger test suite. Give it a
temporary XDG data root so tests do not read or modify an operator's real application state. Install its optional
PyPI dependency profiles before the first run:

```bash
uv pip install -r requirements/requirements-hardware.txt \
  -r requirements/requirements-scripts.txt
XDG_DATA_HOME=/tmp/superinference-panel-test \
  python -m pytest incubator/control_panel/tests -q
```

For repeated or parallel runs, use a unique temporary path per run. Panel-specific release and operator validation
belongs to `incubator/control_panel/docs/README.md` and `incubator/control_panel/release/README.md`.

## What automated tests establish

- Hydra composition and registry references.
- Lazy device imports and compatibility facades.
- Shared-memory parsing, layout, cleanup, and action contracts.
- Chunk/action-manager behavior.
- Deterministic fake-policy integration.
- The one-way core-to-incubator architecture boundary.

## Hardware validation boundary

The automated suite must not be interpreted as hardware certification. It does not establish camera frame timing,
robot or gripper motion safety, vendor SDK/firmware compatibility, realtime CPU behavior, network-policy latency,
checkpoint correctness, calibration quality, or emergency-stop operation.

Keep real hardware and long-running visualization out of normal unit tests. Label and document any manual probe,
run it only on the intended workstation under operator supervision, and record the tested hardware/firmware/config
outside the generic automated result.

Markers are declared in `pytest.ini`: `unit`, `config`, `integration`, `architecture`, `docs`, `dependency`,
`model`, `hardware`, and `manual`. Unknown markers fail collection. `model`, `hardware`, and `manual` identify
optional environments; they must never be used to hide a failure in hardware-free logic.
