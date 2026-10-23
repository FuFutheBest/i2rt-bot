# GitLab Scheduled Submodule Status Refresh

This reference explains the `update_submodules_mr` automation. It records configured remote-branch tips in the
parent repository README. It never pushes to, changes branches in, or changes the gitlink of any submodule repository.

## Mechanism

[submodules.json](../submodules.json) is the configuration source for a managed submodule path, expected
`.gitmodules` remote, tracked branch, and `enabled` flag. The table in the root [README](../../../../README.md)
records two separate states:

- `Bound commit` and `Bound date`: the reviewed gitlink stored in the parent repository.
- `Latest commit` and `Latest date`: the most recently observed tip of the configured remote branch.

The updater checks that the manifest, `.gitmodules`, and table path/branch entries agree. Before it writes a
latest-status change, it also checks that the table's bound columns still match the gitlink in `HEAD`.

```text
Pipeline Schedule on the default branch
  -> GitLab Runner reuses a README-only bot branch or resets a legacy branch from the default branch
  -> scripts/maintenance/update_submodules.py fetches each enabled target branch
  -> only README Latest commit and Latest date cells are refreshed
  -> hardware-free Ruff and pytest checks run
  -> no README change: pipeline succeeds without creating an MR
  -> README change: push fix/automation-update-submodules and create/update its MR
  -> maintainer reviews and merges the MR into the default branch
```

The job reuses the bot branch only when its full diff from the default branch is `README.md`; this prevents duplicate
MR updates for an already recorded tip. A bot branch with a legacy gitlink update or another changed file is reset
from the default branch. The bot stages only `README.md`, and CI guards fail the job if a submodule gitlink changes
or the resulting branch differs from the default branch in another file. A remote branch rewrite is recorded as its
observed current tip because no source pin is changed; review the remote history before manually promoting such a
commit. The API calls use `CI_API_V4_URL`, falling back to `CI_SERVER_URL/api/v4` when the API root variable is absent.
If a push succeeds but MR creation fails, a later run reuses the README-only branch and retries the API lookup. A
no-op run does not push a reused README-only bot branch merely because the default branch advanced. The job sets the
bot Git identity before merging a reused branch, so a default-branch advance can be recorded without relying on the
container's global Git configuration. A no-op run force-resets a legacy bot branch to the default branch, while a
reused README-only branch is left untouched. If a squash merge has already made the branch's merged worktree identical
to the default branch, the job resets that already-merged branch and finishes without opening an empty MR. When a runner
reuses a checkout, a deleted bot branch's stale remote-tracking ref is removed; network and permission errors while
checking the branch fail the job instead of being mistaken for branch deletion.

The MR contains status-record changes only. After merge, a developer still obtains the exact reviewed pins with:

```bash
git pull
git submodule update --init
```

This initializes only SuperInference's direct submodules. Do not add `--recursive` during ordinary environment
setup: it would also clone OpenPI's optional nested dependencies, including `third_party/aloha` and
`third_party/libero`. If an OpenPI development task requires one, initialize that path explicitly from inside
`third_party/openpi`, for example:

```bash
git -C third_party/openpi submodule update --init third_party/aloha
```

## Manual Pin Promotion

Only a normal, manually reviewed MR may change a bound pin. Update the parent gitlink and the corresponding
README `Bound commit` and `Bound date` cells together. The scheduled updater refuses a binding mismatch rather
than repairing it automatically. The following scheduled run can then refresh the latest-status cells normally.

## Configuration Prerequisites

Before creating a schedule, confirm all of the following:

- The committed `.gitlab-ci.yml` contains `update_submodules_mr`.
- The schedule will target the repository's default branch.
- A GitLab Runner can reach the parent GitLab instance and every enabled remote host. Public GitHub SSH URLs are
  rewritten to HTTPS by the CI job.
- A Group Access Token can push the `fix/automation-update-submodules` branch and call the Merge Request API.
- The token is created at the common parent group of the parent repository and managed private submodules, so its
  Bot inherits access to those child projects.
- Every direct managed module, including `third_party/RoboTwin`, has a complete README table row.

## 1. Create the Bot Token

Create a **Group Access Token** at the common parent group of SuperInference and all managed private submodules.
Its Bot inherits membership in child groups and projects, which lets one identity update the parent repository and
fetch the private submodule remotes.

1. Open the common parent group in GitLab.
2. Go to **Settings > Access Tokens**.
3. Create a token named `submodule-update-bot` with a planned expiration and a role that may create the source
   branch and MR. Use Maintainer unless the project permissions have been customized and a lower role is proven to
   work.
4. Select `api` and `write_repository` scopes.
5. Create the token and immediately store both its value and generated username in the team's secret manager.
   GitLab displays the token value only once.

`api` authorizes the API calls that look up and create an MR; `write_repository` authorizes the Git push.
Do not use a deploy token: it is not sufficient for this job's MR API calls.

Do not use a Project Access Token when managed private submodules are in other projects: its Bot cannot be added
directly to another project or group.

## 2. Store CI/CD Variables

In the parent project, go to **Settings > CI/CD**, expand **Variables**, then add the following *project* variables:

| Key | Value | Required settings |
| --- | --- | --- |
| `SUBMODULE_UPDATE_TOKEN` | Token value created above | Type `Variable`; visibility `Masked` or `Masked and hidden`; do not enable variable expansion. |
| `SUBMODULE_UPDATE_USERNAME` | Group-token Bot username | Type `Variable`; no secret content, but keeping it masked is acceptable. |

Use project variables, not schedule variables, for the token. Schedule variables are configuration inputs; the token
is a reusable secret.

The scheduled job rejects a `SUBMODULE_UPDATE_TOKEN` that equals `CI_JOB_TOKEN`. GitLab Runner can install a
repository-local job-token credential helper, so the job uses a namespace-scoped SSH-to-HTTPS rewrite with the Group
Bot credential for managed sibling submodules and the parent origin. The parent origin deliberately uses SSH form so
the same scoped mapping applies before Git selects a credential helper. Do not replace that rewrite with a generic
same-host rewrite: the Runner credential can take precedence and GitLab will reject cross-project access unless the
target explicitly allowlists the source job token.

Mark the variables as **Protected** only when the schedule targets a protected default branch. A protected variable
is unavailable to a schedule running on an unprotected ref, which would make the job stop at its explicit token
check.

Never put the token in `.gitlab-ci.yml`, `submodules.json`, an MR description, or a job command that prints it.
If it appears in a log, revoke it in **Settings > Access Tokens**, create a replacement, and update the variable
before the next run.

## 3. Create and Test the Schedule

1. Open **Build > Pipeline schedules** in the parent project.
2. Select **New schedule**.
3. Use a description such as `Weekly third-party submodule status refresh`.
4. Set **Target branch or tag** to the default branch, normally `main`.
5. Choose a low-contention cron. For example, `17 3 * * 1` is every Monday at 03:17 in the selected schedule
   timezone. Avoid top-of-hour schedules shared by many projects.
6. Save the schedule and select **Run** once to execute it immediately.

Do not add `SUBMODULE_UPDATE_TOKEN` as a schedule variable. The job needs no schedule-specific variables. The CI
rule itself limits the job to pipeline source `schedule`; a manually clicked **Run** for the schedule still has that
source.

Check the first pipeline's `update_submodules_mr` job:

- `No enabled submodule status records changed` is a successful no-op.
- A new observed tip creates or updates branch `fix/automation-update-submodules` and an MR titled
  `chore: refresh submodule status`.
- The job artifact `.tmp/submodule-update-report.md` lists the previous and current observed tip for each changed
  row.
- The MR diff must contain only the README latest-status cells. Do not merge until the MR pipeline and review are
  satisfactory.

## 4. Enable, Add, or Change a Submodule

Make configuration changes in a normal MR before relying on the schedule:

1. Verify the remote and intended branch without changing pins:

   ```bash
   git ls-remote --exit-code --heads <remote-url> refs/heads/<branch>
   ```

2. Confirm the CI Bot has read access when the remote is private.
3. Update `remote`, `branch`, and `enabled` in `submodules.json`; remove `disabled_reason` when enabling.
4. If the remote or tracked branch changed, update `.gitmodules` in the same MR.
5. Add or update the README table row. Its bound columns must describe the committed parent gitlink; preserve those
   columns for an automation-only branch refresh.
6. Validate the configuration:

   ```bash
   python scripts/maintenance/update_submodules.py --validate-config
   ```

7. Review every manifest, `.gitmodules`, gitlink, and bound-table change in the normal MR. After it merges, a bot
   status MR may contain only the README latest-status cells.

To intentionally track a non-default branch, record that branch explicitly. For example, `third_party/i2rt` may
track `develop`; branch names must never be inferred solely from the remote's `HEAD`.

## Troubleshooting

| Symptom | Cause and action |
| --- | --- |
| Job fails before a submodule fetch with a token message | Add `SUBMODULE_UPDATE_TOKEN` as a project CI/CD variable; check variable protection and target ref. |
| Git fetch returns access denied | Confirm the Group Token was created at a common ancestor of the target project; use an approved separate credential mechanism for a private external host. |
| Git push returns 403 after a successful fetch | Confirm the parent origin keeps its SSH form and the namespace-scoped Bot rewrite is present. If it persists, recreate/check the Bot token with `api` and `write_repository`; verify its role and branch protection permit `fix/automation-update-submodules`. |
| MR API returns 401/403 | Recreate/check the Bot token with `api`; verify its role can create Merge Requests. |
| `curl: no URL specified` during MR lookup | Check that `CI_API_V4_URL` was not overridden or emptied; the job falls back to `CI_SERVER_URL/api/v4` and fails early if neither URL is available. |
| Branch push succeeds but no MR is created | The API request may have failed after the push; rerun the schedule. The job reuses an existing README-only bot branch and retries MR creation. |
| A legacy bot branch remains after a no-op run | The next no-op schedule resets a branch whose remote diff contains files other than `README.md`; inspect branch protection if the force push is rejected. |
| A squash-merged bot branch causes a false failure | The job compares the merged worktree with the default branch, resets an already-merged branch, and exits without creating an empty MR. |
| A deleted bot branch is still selected | The runner may retain `refs/remotes/origin/fix/automation-update-submodules`; the job removes that ref when an exact `git ls-remote` query returns no ref. A failed query indicates a fetch, network, or permission problem and fails for inspection. |
| `fatal: 'origin' does not appear to be a git repository` after submodule initialization | A cached submodule Git directory can be missing its remote. The updater restores only a missing `origin` from the validated `.gitmodules` URL before fetching; if it still fails, inspect that repository's URL and CI access. |
| No MR appears | First check whether the job reports no status changes. Otherwise inspect the job artifact and the API/push log. |
| Updater reports a README binding mismatch | A manual gitlink change did not update the matching bound columns, or the table is stale. Correct both values in a normal pin-change MR. |
| A remote branch was force-pushed | The job records its current tip but never advances the gitlink. Review the rewrite, then decide whether to change the bound pin manually. |
| Merge of the bot branch with the default branch conflicts | Someone changed the same README table row or manifest. Resolve the status/binding values in a normal MR, then rerun the schedule. |
| A schedule stops running | It runs as its owner. A Maintainer/Owner can use **Take ownership** under **Build > Pipeline schedules**. |

## Operational Boundaries

- The schedule creates or refreshes an MR. It does not merge it.
- The job serializes with `resource_group: submodule-update`, so two scheduled executions do not update the same
  bot branch concurrently.
- The pipeline's fixed script executes Git, Ruff, and pytest only. It does not execute arbitrary commands from a
  submodule or its documentation.
- Keep a schedule owner with a durable project role. Scheduled pipelines use the owner's permissions; transfer
  ownership before that account is disabled.
