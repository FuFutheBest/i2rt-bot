# SuperInference Review Checklist

Use this checklist for every MR. It turns the maintenance decisions from the
refactor meeting into review evidence rather than relying on reviewer memory.

## 1. Scope and ownership

- The change has one primary purpose. Feature work, structural refactors,
  formatting-only edits, and mass deletion are split when they would obscure
  review.
- The target belongs in stable `main`. Unstable vertical workflows stay under
  `incubator/` until their owner, acceptance criteria, dependencies, and
  promotion path are agreed.
- A short design note is linked for a new module boundary, broad refactor,
  public API change, or schema migration.
- The affected core owner (`devices/`, `manager/`, `consumer/`, `policy/`,
  `configs/`, SHM/data contracts, or `third_party/`) is requested.

## 2. Compatibility and contracts

- Existing import paths, facade exports, config keys, registry names, and CLI
  behavior remain compatible, or a migration and rollback are documented.
- For SHM, trace, or data changes, the review records layout, dtype, shape,
  timestamp domain/meaning, units, serialization, and version compatibility.
- Action chunk alignment, drop/smoothing, execution rates, and original action
  indices are checked against the relevant contract tests.
- No floating-point action matching or implicit timestamp reinterpretation is
  introduced in diagnostic/recording paths.

## 3. Runtime safety and state

- No hidden global state, stale cache, signal-file leak, or cross-episode state
  is introduced.
- Reset, cleanup, reconnect, halt, home, and failure paths are reviewed for
  idempotence and safe ordering.
- Hardware-facing changes identify the operator validation needed for robot,
  gripper, camera, calibration, latency, and emergency-stop behavior.
- Secrets, credentials, IPs, device names, and host-specific paths are not
  hard-coded.

## 4. Dependencies and boundaries

- New imports belong to the correct requirements profile. The default runtime
  and root CI remain free of vendor SDKs, CUDA wheels, checkpoints, and model
  stacks.
- Optional adapters and vendor modules remain lazy; importing core packages
  does not require hardware or model dependencies.
- Core code does not import `incubator`; submodule URL/pin changes are isolated
  and reviewed as vendor changes.

## 5. Verification

- The blocking root command passes:

  ```bash
  python -m pytest tests -q -m "not hardware and not manual" --ignore=tests/test_lingbot_va_rtc.py
  ```

- Relevant focused tests pass for config schema/registry, Hydra composition,
  SHM/data contracts, action chunks, and non-hardware smoke behavior.
- Ruff contract checks and `pre-commit` pass. Full-tree Ruff findings are
  treated as advisory until historical formatting debt is removed.
- Changed entrypoints have a safe `--help` or `--dry-run` check; this must not
  start a robot, camera, policy server, or other hardware process.
- Model, panel, third-party, and hardware results are attached from their
  separate optional jobs and are not confused with root CI coverage.

## 6. Documentation and decision status

- The relevant skill and contract/reference documentation are updated.
- Any deferred work has an owner and due date.
- Do not turn unresolved meeting questions into silent policy: branch naming,
  incubator retention/promotion, exact owner identities, and the final priority
  of the camera-latency issue require an explicit follow-up decision.

## Approval bar

Approve only when required CI jobs pass, the appropriate core owner has
reviewed the change, safety/contract evidence matches the risk, and no review
discussion remains unresolved. Optional jobs may be manual or allowed to fail
only when their environment boundary is stated in the MR.
