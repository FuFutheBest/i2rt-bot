from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPOSITORY_ROOT / "scripts/maintenance/update_submodules.py"
MANIFEST = REPOSITORY_ROOT / ".claude/skills/submodule-maintenance/submodules.json"
README = REPOSITORY_ROOT / "README.md"
GITLAB_CI = REPOSITORY_ROOT / ".gitlab-ci.yml"

MODULE_SPEC = importlib.util.spec_from_file_location("update_submodules", SCRIPT)
assert MODULE_SPEC and MODULE_SPEC.loader
UPDATER = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = UPDATER
MODULE_SPEC.loader.exec_module(UPDATER)


def test_submodule_manifest_validates() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--validate-config"],
        cwd=REPOSITORY_ROOT,
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Validated 5 submodule entries" in result.stdout


def test_git_jobs_use_a_git_capable_image_without_runtime_apt() -> None:
    pipeline = yaml.safe_load(GITLAB_CI.read_text(encoding="utf-8"))
    for job_name in ("update_submodules_mr", "pre_commit", "root_tests"):
        job = pipeline[job_name]
        assert job["image"] == "python:3.10-bookworm"
        assert any("git --version" in command for command in job["before_script"])
        assert not any("apt-get" in command for command in job["before_script"])

    assert any(
        "curl --version" in command
        for command in pipeline["update_submodules_mr"]["before_script"]
    )
    assert pipeline["variables"]["GIT_SUBMODULE_STRATEGY"] == "none"


def test_submodule_manifest_entries_match_gitmodules() -> None:
    manifest = json.loads(MANIFEST.read_text())["submodules"]
    gitmodules_entries: dict[str, dict[str, str]] = {}
    for line in subprocess.check_output(
        [
            "git",
            "config",
            "-f",
            ".gitmodules",
            "--get-regexp",
            r"^submodule\..*\.path$",
        ],
        cwd=REPOSITORY_ROOT,
        text=True,
    ).splitlines():
        key, path = line.split(maxsplit=1)
        name = key.removeprefix("submodule.").removesuffix(".path")
        gitmodules_entries[path] = {
            "remote": subprocess.check_output(
                [
                    "git",
                    "config",
                    "-f",
                    ".gitmodules",
                    "--get",
                    f"submodule.{name}.url",
                ],
                cwd=REPOSITORY_ROOT,
                text=True,
            ).strip(),
            "branch": subprocess.check_output(
                [
                    "git",
                    "config",
                    "-f",
                    ".gitmodules",
                    "--get",
                    f"submodule.{name}.branch",
                ],
                cwd=REPOSITORY_ROOT,
                text=True,
            ).strip(),
        }

    assert {entry["path"] for entry in manifest} == set(gitmodules_entries)
    assert {
        entry["path"]: {"remote": entry["remote"], "branch": entry["branch"]}
        for entry in manifest
    } == {path: gitmodules_entries[path] for path in gitmodules_entries}


def test_nmx_vla_tracks_junlin_fasterwam_inference() -> None:
    manifest = json.loads(MANIFEST.read_text())["submodules"]
    nmx_vla = next(
        entry for entry in manifest if entry["path"] == "third_party/nmx_vla"
    )

    assert nmx_vla["branch"] == "junlin/fasterwam-inference"


def test_scheduled_job_reuses_only_a_readme_only_bot_branch() -> None:
    gitlab_ci = GITLAB_CI.read_text()
    pipeline = yaml.safe_load(gitlab_ci)
    before_script = "\n".join(pipeline["update_submodules_mr"]["before_script"])
    branch = pipeline["update_submodules_mr"]["variables"]["SUBMODULE_UPDATE_BRANCH"]
    script = "\n".join(pipeline["update_submodules_mr"]["script"])
    merge_command = 'git merge --no-edit "origin/$CI_DEFAULT_BRANCH"'
    assert re.fullmatch(r"(?:feat|fix|lab)/.+", branch)
    assert branch == "fix/automation-update-submodules"
    assert "SUBMODULE_UPDATE_BRANCH must match ^(feat|fix|lab)/.+$." in gitlab_ci

    assert (
        'git diff --name-only "origin/$CI_DEFAULT_BRANCH...origin/$SUBMODULE_UPDATE_BRANCH"'
        in gitlab_ci
    )
    assert (
        'git merge-base "origin/$CI_DEFAULT_BRANCH" "origin/$SUBMODULE_UPDATE_BRANCH"'
        in gitlab_ci
    )
    assert (
        'remote_bot_refs="$(git ls-remote --heads origin '
        '"refs/heads/$SUBMODULE_UPDATE_BRANCH")"'
        in gitlab_ci
    )
    assert (
        '+refs/heads/$SUBMODULE_UPDATE_BRANCH:refs/remotes/origin/$SUBMODULE_UPDATE_BRANCH'
        in gitlab_ci
    )
    assert 'ls_remote_status=$?' in gitlab_ci
    assert (
        'git update-ref -d "refs/remotes/origin/$SUBMODULE_UPDATE_BRANCH"'
        in gitlab_ci
    )
    assert (
        'git switch -C "$SUBMODULE_UPDATE_BRANCH" "origin/$SUBMODULE_UPDATE_BRANCH"'
        in gitlab_ci
    )
    assert (
        'git switch -C "$SUBMODULE_UPDATE_BRANCH" "origin/$CI_DEFAULT_BRANCH"'
        in gitlab_ci
    )
    assert merge_command in gitlab_ci
    merge_index = before_script.index(merge_command)
    assert before_script.index(
        'git config user.name "SuperInference Submodule Bot"'
    ) < merge_index
    assert before_script.index(
        'git config user.email "submodule-bot@$CI_SERVER_HOST"'
    ) < merge_index
    assert 'if [ -z "$bot_branch_diff" ]; then' in gitlab_ci
    assert 'bot_branch_diff="__unrelated_history__"' in script
    assert (
        'bot_branch_diff="$(git diff --name-only '
        '"origin/$CI_DEFAULT_BRANCH...origin/$SUBMODULE_UPDATE_BRANCH")"'
        in script
    )
    assert 'echo "Resetting legacy bot branch to the default branch."' in script
    assert (
        'git push --force-with-lease origin "HEAD:$SUBMODULE_UPDATE_BRANCH"'
        in script
    )
    assert 'if [ -z "$local_bot_branch_diff" ]; then' in script
    assert 'echo "Resetting already-merged bot branch to the default branch."' in script
    assert 'git push --force-with-lease origin \\' in script
    assert (
        'refs/remotes/origin/$CI_DEFAULT_BRANCH:refs/heads/$SUBMODULE_UPDATE_BRANCH'
        in script
    )
    assert (
        "Reusing existing README-only bot branch; ensuring its Merge Request exists."
        in gitlab_ci
    )
    assert (
        "Bot branch may differ from the default branch only in README.md." in gitlab_ci
    )
    assert (
        '"$SUBMODULE_UPDATE_API_URL/projects/$CI_PROJECT_ID/merge_requests"'
        in script
    )
    assert '"$CI_API_V4_URL/projects/$CI_PROJECT_ID/merge_requests"' not in script
    assert (
        script.count(
            '--url "$SUBMODULE_UPDATE_API_URL/projects/$CI_PROJECT_ID/merge_requests"'
        )
        == 2
    )


def test_scheduled_job_falls_back_to_the_server_api_url() -> None:
    pipeline = yaml.safe_load(GITLAB_CI.read_text())
    before_script = "\n".join(pipeline["update_submodules_mr"]["before_script"])

    assert 'if [ -n "${CI_API_V4_URL:-}" ]; then' in before_script
    assert 'api_v4_url="${CI_SERVER_URL%/}/api/v4"' in before_script
    api_url_fallback = (
        'api_v4_url="${CI_SERVER_PROTOCOL:-https}://${CI_SERVER_FQDN:-$CI_SERVER_HOST}/api/v4"'
    )
    assert api_url_fallback in before_script
    assert 'export SUBMODULE_UPDATE_API_URL="${api_v4_url%/}"' in before_script
    assert "CI_API_V4_URL/projects/$CI_PROJECT_ID/merge_requests" not in before_script


def test_scheduled_job_uses_the_configured_bot_for_origin_and_sibling_submodules() -> None:
    gitlab_ci = GITLAB_CI.read_text()
    sibling_rewrite = (
        'url."https://${SUBMODULE_UPDATE_USERNAME}:'
        '${SUBMODULE_UPDATE_TOKEN}@${CI_SERVER_HOST}/'
        '${CI_PROJECT_NAMESPACE}/".insteadOf'
    )

    assert 'test "${SUBMODULE_UPDATE_TOKEN}" != "${CI_JOB_TOKEN:-}"' in gitlab_ci
    assert sibling_rewrite in gitlab_ci
    assert '"git@${CI_SERVER_HOST}:${CI_PROJECT_NAMESPACE}/"' in gitlab_ci
    assert 'git remote set-url origin "git@$CI_SERVER_HOST:$CI_PROJECT_PATH.git"' in gitlab_ci
    assert 'git remote set-url origin "https://$CI_SERVER_HOST/$CI_PROJECT_PATH.git"' not in gitlab_ci
    assert "credential approve" not in gitlab_ci
    assert "credential.helper store" not in gitlab_ci


def test_readme_status_table_matches_manifest() -> None:
    entries = UPDATER.load_manifest(MANIFEST)
    records = UPDATER.load_status_table(README)

    UPDATER.validate_status_table(entries, records)
    records_by_path = {record.path: record for record in records}
    for entry in entries:
        commit = subprocess.check_output(
            ["git", "ls-tree", "HEAD", entry["path"]],
            cwd=REPOSITORY_ROOT,
            text=True,
        ).split()[2]
        assert records_by_path[entry["path"]].bound_commit == commit


def test_write_status_table_changes_only_the_latest_cells(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text(README.read_text())
    records = UPDATER.load_status_table(readme)
    previous = records[0]
    refreshed = replace(previous, latest_commit="a" * 40, latest_date="2026-09-07")

    before_lines = readme.read_text().splitlines()
    UPDATER.write_status_table(readme, [refreshed, *records[1:]])
    after_lines = readme.read_text().splitlines()
    updated_records = UPDATER.load_status_table(readme)

    assert [
        after
        for before, after in zip(before_lines, after_lines, strict=True)
        if before != after
    ] == [UPDATER.table_row(refreshed)]
    assert updated_records[0].bound_commit == previous.bound_commit
    assert updated_records[0].bound_date == previous.bound_date
    assert updated_records[0].latest_commit == refreshed.latest_commit
    assert updated_records[0].latest_date == refreshed.latest_date


def test_initialize_submodule_restores_a_missing_origin(monkeypatch) -> None:
    path = "third_party/openpi"
    module_path = REPOSITORY_ROOT / path
    remote_url = "https://example.test/openpi.git"
    calls: list[tuple[str, ...]] = []

    def fake_run(
        *args: str, cwd: Path | None = None, capture_output: bool = False
    ) -> str:
        del cwd, capture_output
        calls.append(args)
        return ""

    monkeypatch.setattr(UPDATER, "run", fake_run)
    monkeypatch.setattr(
        UPDATER, "gitmodules_value", lambda configured_path, key: remote_url
    )

    assert UPDATER.initialize_submodule(path) == module_path
    assert calls == [
        ("git", "submodule", "sync", "--", path),
        ("git", "submodule", "update", "--init", "--", path),
        ("git", "-C", str(module_path), "remote"),
        (
            "git",
            "-C",
            str(module_path),
            "remote",
            "add",
            "origin",
            remote_url,
        ),
        ("git", "status", "--porcelain"),
    ]


def test_refresh_never_checks_out_or_stages_a_gitlink(monkeypatch) -> None:
    bound_commit = "b" * 40
    latest_commit = "c" * 40
    entry = {
        "name": "test-module",
        "path": "third_party/openpi",
        "remote": "https://example.test/openpi.git",
        "branch": "main",
        "enabled": True,
    }
    previous = UPDATER.StatusRecord(
        path=entry["path"],
        branch=entry["branch"],
        bound_commit=bound_commit,
        bound_date="2026-01-01",
        latest_commit="d" * 40,
        latest_date="2025-12-31",
    )
    calls: list[tuple[str, ...]] = []

    def fake_run(
        *args: str, cwd: Path | None = None, capture_output: bool = False
    ) -> str:
        del cwd, capture_output
        calls.append(args)
        if args[:3] == ("git", "ls-tree", "HEAD"):
            return f"160000 commit {bound_commit}\t{entry['path']}"
        if args == ("git", "-C", str(REPOSITORY_ROOT / entry["path"]), "remote"):
            return "origin"
        if args[:2] == ("git", "status"):
            return ""
        if "show" in args:
            commit = args[-1].split("^{", maxsplit=1)[0]
            return {bound_commit: "2026-01-01", latest_commit: "2026-02-02"}[commit]
        if "rev-parse" in args:
            return latest_commit
        return ""

    monkeypatch.setattr(UPDATER, "run", fake_run)

    refreshed = UPDATER.refresh_submodule_status(entry, previous)

    assert refreshed is not None
    assert refreshed.bound_commit == previous.bound_commit
    assert refreshed.bound_date == previous.bound_date
    assert refreshed.latest_commit == latest_commit
    assert refreshed.latest_date == "2026-02-02"
    assert all("checkout" not in command for command in calls)
    assert all("add" not in command for command in calls)


def test_refresh_rejects_a_stale_readme_binding(monkeypatch) -> None:
    entry = {
        "name": "test-module",
        "path": "third_party/openpi",
        "remote": "https://example.test/openpi.git",
        "branch": "main",
        "enabled": True,
    }
    previous = UPDATER.StatusRecord(
        path=entry["path"],
        branch=entry["branch"],
        bound_commit="b" * 40,
        bound_date="2026-01-01",
        latest_commit="b" * 40,
        latest_date="2026-01-01",
    )
    monkeypatch.setattr(UPDATER, "pinned_commit", lambda path: "c" * 40)
    monkeypatch.setattr(
        UPDATER, "initialize_submodule", lambda path: REPOSITORY_ROOT / path
    )
    monkeypatch.setattr(UPDATER, "commit_date", lambda path, commit: "2026-01-01")

    with pytest.raises(UPDATER.UpdateError, match="README binding mismatch"):
        UPDATER.refresh_submodule_status(entry, previous)
