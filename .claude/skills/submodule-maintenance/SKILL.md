---
name: submodule-maintenance
description: Monitor SuperInference third-party submodule branches in the root README without changing pinned gitlinks, or maintain the scheduled GitLab status-refresh workflow.
---

# Submodule Maintenance

[submodules.json](submodules.json) is the machine-readable configuration source for every managed submodule path,
expected remote, target branch, and automation eligibility. Keep it exactly aligned with `.gitmodules` when adding a
submodule or changing its tracked branch. Do not infer branches from a repository default.

The root [README](../../../README.md) holds the human-readable status table. Its `Bound commit` and `Bound date`
describe the parent repository's reviewed gitlink. Its `Latest commit` and `Latest date` describe the most recently
observed target-branch tip. The table markers and headers are managed by the updater and must remain intact.

## Local Status Refresh

First validate the manifest, `.gitmodules`, and README table without contacting remotes:

```bash
python scripts/maintenance/update_submodules.py --validate-config
```

To fetch each enabled tracking branch and refresh only the latest-status cells:

```bash
python scripts/maintenance/update_submodules.py --report .tmp/submodule-update-report.md
git diff -- README.md
```

The updater initializes only the direct submodule checkout required for inspection, restores a missing `origin` from
the validated `.gitmodules` URL without changing an existing remote, verifies that the README binding matches the
gitlink in `HEAD`, then fetches the configured branch. It never checks out the fetched tip, stages a submodule, or
changes a parent-repository gitlink. A rewritten remote branch is recorded as its currently observed tip; review that
history before choosing a new bound pin.

## Manual Pin Promotion

Promote a submodule only in a normal, manually reviewed MR. Change the gitlink to the selected commit and update that
row's `Bound commit` and `Bound date` together. The scheduled updater refuses a binding mismatch instead of repairing
it automatically. Its next successful run will continue to refresh only the latest-status columns.

## Checkout Scope

SuperInference manages only its direct submodules. On a new machine, initialize those pins without recursion:

```bash
git submodule update --init
```

Do not use `--recursive` as the normal bootstrap command. It also initializes OpenPI's own optional `third_party`
dependencies, such as `aloha` and `libero`, which are not required for the standard SuperInference runtime. When OpenPI
development specifically needs them, initialize only the required nested path:

```bash
git -C third_party/openpi submodule update --init third_party/aloha
```

Use the analogous command for another OpenPI nested dependency when needed.

## Scheduled GitLab Workflow

`update_submodules_mr` runs only for a GitLab Pipeline Schedule. It reuses `fix/automation-update-submodules` only
when its diff from the default branch is limited to `README.md`; a legacy or otherwise changed bot branch is rebuilt
from the default branch. The job fetches configured branches, runs root hardware-free checks, then pushes a README-only
status change and creates or updates an MR into the default branch. The schedule does not merge the MR or advance any
submodule pin. The API lookup uses `CI_API_V4_URL` and falls back to `CI_SERVER_URL/api/v4` for installations that do
not expose the API root variable. If a push succeeds before the API call fails, a later schedule reuses the
README-only bot branch and retries MR creation; the job configures the bot Git identity before any recovery merge.
If a runner's cached checkout retains a deleted branch ref, the job removes it; transport or permission errors fail
instead of being treated as a missing branch. A legacy bot branch is reset to the default branch when a no-op run
finds no status changes. A README-only branch whose
merged worktree is already identical to the default branch is reset as an already-merged branch. A no-op schedule does
not push a still-pending reused README-only bot branch.

Read [GitLab scheduled updates](references/gitlab-scheduled-updates.md) before creating a token, enabling a disabled
submodule, configuring a schedule, or troubleshooting a failed update.

The required masked CI/CD variables are:

- `SUBMODULE_UPDATE_TOKEN`: a Group Access Token with `api` and `write_repository`, created at the common parent group
  of all managed private submodules.
- `SUBMODULE_UPDATE_USERNAME`: the generated Group Bot username.

The CI job rewrites SSH GitLab and GitHub URLs to HTTPS. Managed GitLab sibling submodules and the parent `origin`
use a namespace-scoped GitLab SSH rewrite with the Group Bot credential; the parent is deliberately set to SSH form
so this mapping takes precedence over GitLab Runner's `CI_JOB_TOKEN` helper. External private GitHub submodules need
their own deploy token or CI credential mechanism before they can be enabled.

Keep inaccessible or unverified repositories disabled with a concrete reason. Enable them only after confirming both
the branch and CI read access.
