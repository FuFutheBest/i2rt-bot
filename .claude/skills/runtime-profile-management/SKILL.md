---
name: runtime-profile-management
description: Configure, verify, resolve, and repair machine-local Python runtime profiles for SuperInference multi-process workflows. Use when planner, device, policy, policy-server, recording, converter, RoboTwin, or FastWAM commands need different environments.
---

# Runtime Profile Management

SuperInference workflows may use several Python environments at once. Treat an
environment as a reusable process profile and bind workflow roles to it; do not
assume that one inference config has one Python environment.

The tracked resolver and CLI are:

```text
utils/runtime_profiles.py
scripts/diagnostics/runtime_profiles.py
```

The generated workstation registry is:

```text
.superinference/local/runtime_profiles.json
```

That directory is Git-ignored. It survives branch switches in the same worktree
but is not a portable project artifact. Never commit it, copy another machine's
registry, or put credentials in it. The loader records and validates the lexical
repository root, so pointing one checkout at another checkout's registry fails
instead of rebasing relative paths silently.

## Roles

Use the narrowest process role:

- `launcher`: a workflow launcher or Control Panel entrypoint;
- `planner`: hardware-free Hydra/config inspection;
- `device`: DeviceStarter and vendor device workers;
- `policy`: the lightweight or in-process policy consumer;
- `policy-server`: an external model server owned by a workflow such as RoboTwin
  XPolicyLab;
- `fastwam-server`: the standalone NMX/CUDA FastWAM Server;
- `superoffline`: the local SuperOffline browser service and its NumPy-backed
  request/preflight process;
- `recording`: the NEDF3 rollout writer and inspector;
- `converter`: finalized-episode conversion.
- `data-processing`: offline dataset processors;
- `upload`: data publication tools that own separate network dependencies.

Profiles are reusable. Bind the same profile to several roles only after its
verification imports cover each role. Use config-specific bindings by default;
global role defaults are appropriate only when the workstation deliberately uses
one environment for every config.

For the source Control Panel, `launcher` and `device` normally need the same
profile because its device producers use Python's spawn context inside the panel
runtime. `policy`, `recording`, `converter`, and external server roles may use
different interpreters. Split terminal launchers such as
`scripts/lingbotva/run_infer_split.sh` resolve `device` and `policy` independently.

## Register And Bind

Registering probes the interpreter immediately and writes a fingerprint only on
success:

```bash
python3 scripts/diagnostics/runtime_profiles.py register <profile-name> \
  --python <absolute-or-repo-relative-python> \
  --cwd <working-directory> \
  --env PYTHONNOUSERSITE=1 \
  --import hydra \
  --import omegaconf
```

Bind roles to the native Hydra config name:

```bash
python3 scripts/diagnostics/runtime_profiles.py bind \
  --config-name <config-name> --role device --profile <profile-name>
python3 scripts/diagnostics/runtime_profiles.py bind \
  --config-name <config-name> --role policy --profile <profile-name>
```

The binding key is the complete native name relative to `configs/`, without the
`.yaml` suffix. Moving `configs/foo.yaml` to `configs/subsystem/foo.yaml` changes
the binding from `foo` to `subsystem/foo`; local registries do not follow file
moves automatically. Bind the new name, run `doctor`, then remove the obsolete
binding so command resolution cannot silently use stale config identity.

Use `--skip-verify` only while repairing an incomplete environment. An unverified
or stale profile is rejected by normal `run` and runtime resolution.

Some native configs declare a code-owned required-role contract. Running
`doctor --config-name <config-name>` without explicit `--role` checks that the
complete role set is bound before probing its profiles. All native FastWAM
configs require `launcher`, `planner`, `device`, `policy`, `fastwam-server`, and
`recording`; the recording binding remains mandatory when
`policy.infer_rollout.enabled` is false so enabling rollout recording never
reveals an uncalibrated environment during operation.

## Resolve And Run

Before a model or hardware launch, inspect the bindings and run their import
probes without opening hardware:

```bash
python3 scripts/diagnostics/runtime_profiles.py list
python3 scripts/diagnostics/runtime_profiles.py doctor \
  --config-name <config-name>
```

Resolve one exact interpreter when another tool needs a path:

```bash
python3 scripts/diagnostics/runtime_profiles.py resolve \
  --config-name <config-name> --role policy --require-verified
```

Prefer `run` for an actual command. It sets the registry path, active profile,
runtime role, config identity, configured local environment, and profile `bin` directory before
replacing itself with the selected interpreter:

```bash
python3 scripts/diagnostics/runtime_profiles.py run \
  --config-name <config-name> --role launcher -- \
  -m incubator.control_panel --config-name <config-name>
```

Use `command --json` when a Skill needs to report the resolved argv and managed
environment without starting it. Do not parse the JSON registry directly or
invent a Conda/uv activation command in a Skill.

## Precedence And Compatibility

Runtime code resolves in this order:

1. explicit Python argument;
2. the existing role-specific `SUPERINFERENCE_*_PYTHON` variable;
3. an exact config binding in the local registry;
4. a local default role binding;
5. the historical config/current-interpreter fallback owned by that subsystem.

Packaged releases keep their bundled-runtime restrictions. A source registry does
not override `SUPERINFERENCE_RELEASE_ROOT` policy. Existing environment variables
remain valid for one-off overrides and automation migration, but new workstation
setup should register and bind profiles instead of exporting a permanent collection
of Python paths.

## Verification And Repair

`verify` executes only the configured import probe; it does not open devices or
load a checkpoint unless an import itself has that side effect. `doctor` compares
the observed fingerprint with the saved one without modifying the registry.

```bash
python3 scripts/diagnostics/runtime_profiles.py verify --profile <profile-name>
python3 scripts/diagnostics/runtime_profiles.py doctor --role <role> \
  --config-name <config-name> --json
```

Re-run `verify` after recreating an environment, changing its interpreter,
installing core packages, or switching to a branch with a different dependency
contract. A missing executable, failed import, malformed registry, unknown profile,
stale spec, or changed package fingerprint is a blocker; do not silently fall back
to a similarly named Conda environment. Profile environment variables are applied
before workflow-owned run IDs, ports, control paths, and config identity, so the
orchestrator remains authoritative for those fields.

For Pi05 XPolicyLab, register the interpreter under the OpenPI uv project's
`.venv/bin/python` and set the profile `cwd` to that OpenPI project root. The
RoboTwin launcher passes that project root to XPolicyLab's existing uv activation
contract. Do not store XLA memory fractions, GPU selection, ports, checkpoints, or
task parameters in the runtime registry; those remain workflow configuration.
RoboTwin's Pi05 and FastWAM configs deliberately use `not_configured` when no
`policy-server` profile or explicit compatibility override exists; they never guess
the environment names `uv` or `fastwam`.

## Validation

```bash
python -m py_compile utils/runtime_profiles.py \
  scripts/diagnostics/runtime_profiles.py
python -m pytest -q tests/utils/test_runtime_profiles.py \
  incubator/control_panel/tests/test_runtime_paths.py \
  tests/scripts/test_robotwin_launch.py
python "$HOME/.codex/skills/.system/skill-creator/scripts/quick_validate.py" \
  .claude/skills/runtime-profile-management
```
