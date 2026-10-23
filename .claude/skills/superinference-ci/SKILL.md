---
name: superinference-ci
description: Review SuperInference repository boundaries, choose the correct hardware-free or optional test suites, and prepare changes for GitLab merge without mixing vendor, model, incubator, or physical-hardware validation.
---

# SuperInference CI

Use this skill for merge readiness, repository structure reviews, CI changes, test selection, and module-boundary
work. It describes the maintained mainline contract.

## Default Merge Gate

The blocking merge path is hardware-free and uses the root Python 3.10 development profile:

```bash
python -m ruff check --select E9,F63,F7,F82 --exclude third_party .
python -m pytest tests -q -m "not hardware and not manual"
export PATH="$PWD/.venv/bin:$PATH"
{
  git diff --name-only --diff-filter=ACMR -z
  git ls-files --others --exclude-standard -z
} | xargs -0 -r python -m pre_commit run --files
git diff --check
```

Run focused tests while editing, then run the complete root suite. Do not hide a hardware-free failure behind an
optional marker or dependency skip. Full-tree Ruff and format reports remain non-blocking while historical debt is
reduced; do not bulk-format unrelated code or vendor files to make that report green.

## Test Boundaries

- `tests/`: blocking root contracts for configs, registries, imports, SHM, action flow, scripts, and utilities.
- `incubator/control_panel/tests/`: separate application suite with an isolated `XDG_DATA_HOME`.
- `model`: tests that require a model framework or checkpoint stack.
- `hardware`: tests that connect to physical devices or vendor services.
- `manual`: operator-supervised tests that must never enter the default automated selector.
- `third_party/**`: vendor-owned suites, run only in the matching checked-out dependency environment.

The root pipeline does not initialize submodules. Jobs that need Git (`update_submodules_mr`, `pre_commit`, and
`root_tests`) use the full `python:3.10-bookworm` image, which bundles Git (and curl for maintenance) instead of
installing packages from a Debian mirror during the job. Optional vendor metadata checks must skip with a precise
reason when a submodule is absent; code that is meant to be hardware-free must keep vendor imports lazy.
Keep repository-owned assertions separate from cross-submodule compatibility assertions so the former still run in
the empty-submodule checkout. The Git-capable root test image supplies the Git CLI because manifest checks inspect
`.gitmodules` and the gitlinks in `HEAD`; this does not authorize or perform submodule initialization.

The Control Panel suite additionally installs `requirements/requirements-hardware.txt` and
`requirements/requirements-scripts.txt`; these
profiles stay outside the blocking root image. Run it with an isolated application-data directory:

```bash
python -m pip install -r requirements/requirements-hardware.txt \
  -r requirements/requirements-scripts.txt
XDG_DATA_HOME=/tmp/superinference-panel-test \
  python -m pytest incubator/control_panel/tests -q
```

## Repository Contracts

- Core packages must not import `incubator`.
- `devices` imports remain hardware-free through the lazy registry.
- Hardware-free constructor-contract tests inspect lazy registry specifications and source instead of resolving
  optional adapters.
- Compatibility facades may preserve old imports, but new implementation belongs in the focused package.
- Config names are paths relative to `configs/` without `.yaml`; moving a config requires updating defaults,
  entrypoints, tests, skills, and runtime-profile binding guidance together.
- Shared-memory or action changes must state shape, dtype, units, timestamp domain, ownership, and compatibility.
- Every registered device constructor follows one typed contract: `state_feedback` is required and contains
  `fps`, `buffer_size`, `hardware_latency_ms`, and `format` (`shape`, `dtype`, `type`). Camera formats use
  `type: rgb`; low-dimensional formats declare primitive components whose widths sum to `shape`. Writable
  devices additionally require the structured `control` contract. `utils.device_config.parse_device_config`
  validates these sections once and `DeviceStarter` passes the parsed contracts directly; do not add flat stream
  aliases or reconstruct legacy fields in constructors.
- Root scripts stay categorized under `scripts/`; a static README inventory is not required because path contracts
  are enforced by tests.
- Machine-specific interpreters and runtime fingerprints stay under the ignored `.superinference/local/` registry.
- Submodule gitlinks change only in a manually reviewed commit. The scheduled job may update README status fields,
  but it must not move gitlinks.

## GitLab Jobs

`.gitlab-ci.yml` keeps four classes of work separate:

- maintenance: submodule status refresh and related repository automation;
- lint: blocking syntax/undefined-name checks, changed-range pre-commit, and non-blocking full Ruff reporting;
- test: root tests plus path-focused config/device/SHM checks;
- optional: model, Control Panel, physical hardware, and third-party validation.

The scheduled maintenance job validates its policy-compliant bot branch, resolves the GitLab API root from
`CI_API_V4_URL` with a `CI_SERVER_URL/api/v4` fallback, and retries MR creation when an earlier run left a
README-only bot branch after a successful push. It clears stale remote-tracking refs when that branch no longer exists
and fails closed on remote inspection errors.

`GIT_SUBMODULE_STRATEGY` stays disabled for the root pipeline. Hardware jobs must remain operator-supervised and
must not be made interruptible in a way that bypasses device cleanup.

## Review Workflow

1. Inspect `git status`, the complete diff, and any dirty submodule separately.
2. Identify the primary runtime/config/data contract affected by the change.
3. Review risks before summaries: behavioral regressions, hardware safety, compatibility, stale paths, missing
   tests, and undocumented migrations.
4. Run the narrow relevant tests, then the default merge gate.
5. Run optional suites only when their exact dependencies and external resources are available.
6. Update the owning skill when a workflow, invariant, troubleshooting rule, or architecture boundary changes.

Use [the review checklist](references/review-checklist.md) for a full MR review. Keep task reports, static test
inventories, branch histories, and copied CI tutorials out of the maintained knowledge base.

## Validation

```bash
python "$HOME/.codex/skills/.system/skill-creator/scripts/quick_validate.py" \
  .claude/skills/superinference-ci
python -m pytest tests/architecture tests/scripts/test_update_submodules.py -q
```
