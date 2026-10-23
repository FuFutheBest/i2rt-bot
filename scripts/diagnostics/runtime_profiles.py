"""Configure and use machine-local Python runtime profiles."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.runtime_profiles import (
    RuntimeProfile,
    RuntimeProfileError,
    RuntimeRegistry,
    normalize_config_name,
)
from utils.runtime_role_contracts import required_runtime_roles


def _environment_assignments(values: Sequence[str]) -> dict[str, str]:
    environment: dict[str, str] = {}
    for value in values:
        name, separator, raw_value = value.partition("=")
        if not separator or not name:
            raise RuntimeProfileError(f"--env must use NAME=VALUE syntax: {value!r}")
        environment[name] = raw_value
    return environment


def _registry(args: argparse.Namespace) -> RuntimeRegistry:
    return RuntimeRegistry.load(
        repo_root=Path(args.repo_root),
        path=Path(args.registry) if args.registry else None,
    )


def _profile_payload(profile: RuntimeProfile) -> dict[str, object]:
    return {
        "name": profile.name,
        "python": str(profile.python),
        "cwd": str(profile.cwd),
        "environment": profile.environment,
        "imports": list(profile.imports),
        "verified": profile.verified,
        "verification": profile.verification,
        "registry": str(profile.registry_path),
    }


def _command_arguments(args: argparse.Namespace) -> list[str]:
    arguments = list(args.arguments)
    if arguments[:1] == ["--"]:
        arguments = arguments[1:]
    if not arguments:
        raise RuntimeProfileError("a command is required after --")
    return arguments


def _require_verified(profile: RuntimeProfile, *, allow_unverified: bool) -> None:
    if profile.verified or allow_unverified:
        return
    raise RuntimeProfileError(
        f"runtime profile {profile.name!r} is unverified or stale; run verify first"
    )


def _command_payload(
    profile: RuntimeProfile,
    arguments: Sequence[str],
    *,
    config_name: str | None,
    role: str,
) -> dict[str, object]:
    environment = profile.process_environment(config_name=config_name)
    environment["SUPERINFERENCE_RUNTIME_ROLE"] = role
    managed_environment = {
        name: environment[name]
        for name in sorted(
            {
                *profile.environment,
                "PATH",
                "SUPERINFERENCE_CONFIG_NAME",
                "SUPERINFERENCE_RUNTIME_PROFILE",
                "SUPERINFERENCE_RUNTIME_REGISTRY",
                "SUPERINFERENCE_RUNTIME_ROLE",
            }
        )
        if name in environment
    }
    argv = profile.command(arguments)
    return {
        "profile": profile.name,
        "python": str(profile.python),
        "cwd": str(profile.cwd),
        "argv": argv,
        "shell": shlex.join(argv),
        "environment": managed_environment,
        "verified": profile.verified,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--registry", default=None)
    subparsers = parser.add_subparsers(dest="command", required=True)

    register = subparsers.add_parser(
        "register", help="register and verify an interpreter"
    )
    register.add_argument("name")
    register.add_argument("--python", required=True)
    register.add_argument("--cwd", default=None)
    register.add_argument("--import", dest="imports", action="append", default=[])
    register.add_argument("--env", action="append", default=[])
    register.add_argument("--skip-verify", action="store_true")
    register.add_argument("--timeout", type=float, default=30.0)
    register.add_argument("--json", action="store_true")

    bind = subparsers.add_parser("bind", help="bind a process role to a profile")
    bind.add_argument("--role", required=True)
    bind.add_argument("--profile", required=True)
    bind.add_argument("--config-name", default=None)

    unbind = subparsers.add_parser("unbind", help="remove a role binding")
    unbind.add_argument("--role", required=True)
    unbind.add_argument("--config-name", default=None)

    resolve = subparsers.add_parser("resolve", help="resolve a role to its interpreter")
    resolve.add_argument("--role", required=True)
    resolve.add_argument("--config-name", default=None)
    resolve.add_argument("--require-verified", action="store_true")
    resolve.add_argument("--json", action="store_true")

    verify = subparsers.add_parser(
        "verify", help="probe and update profile fingerprints"
    )
    verify_selection = verify.add_mutually_exclusive_group()
    verify_selection.add_argument("--profile", default=None)
    verify_selection.add_argument("--role", default=None)
    verify.add_argument("--config-name", default=None)
    verify.add_argument("--timeout", type=float, default=30.0)
    verify.add_argument("--json", action="store_true")

    doctor = subparsers.add_parser(
        "doctor", help="check bindings and run import probes"
    )
    doctor.add_argument("--config-name", default=None)
    doctor.add_argument("--role", action="append", default=[])
    doctor.add_argument("--timeout", type=float, default=30.0)
    doctor.add_argument("--json", action="store_true")

    list_parser = subparsers.add_parser("list", help="show profiles and bindings")
    list_parser.add_argument("--json", action="store_true")

    command_help = {
        "command": "print a Python command for a selected profile",
        "run": "run Python arguments with a selected profile",
    }
    for name in ("command", "run"):
        command = subparsers.add_parser(name, help=command_help[name])
        command.add_argument("--role", required=True)
        command.add_argument("--config-name", default=None)
        command.add_argument("--allow-unverified", action="store_true")
        command.add_argument("--json", action="store_true")
        command.add_argument("arguments", nargs=argparse.REMAINDER)

    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        registry = _registry(args)
        if args.command == "register":
            profile = registry.register(
                args.name,
                python=Path(args.python),
                cwd=Path(args.cwd) if args.cwd else None,
                environment=_environment_assignments(args.env),
                imports=args.imports,
            )
            if not args.skip_verify:
                profile = registry.verify(profile.name, timeout_s=args.timeout)
            registry.save()
            payload = _profile_payload(profile)
            if args.json:
                print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
            else:
                state = "verified" if profile.verified else "unverified"
                print(f"Registered {profile.name}: {profile.python} ({state})")
            return 0

        if args.command == "bind":
            registry.bind(args.role, args.profile, config_name=args.config_name)
            registry.save()
            scope = args.config_name or "default"
            print(f"Bound {scope}:{args.role} -> {args.profile}")
            return 0

        if args.command == "unbind":
            removed = registry.unbind(args.role, config_name=args.config_name)
            if removed:
                registry.save()
            scope = args.config_name or "default"
            print(f"{'Removed' if removed else 'No'} binding for {scope}:{args.role}")
            return 0

        if args.command == "resolve":
            profile = registry.resolve(
                args.role,
                config_name=args.config_name,
                required=True,
            )
            assert profile is not None
            if args.require_verified:
                _require_verified(profile, allow_unverified=False)
            if args.json:
                print(
                    json.dumps(
                        _profile_payload(profile),
                        ensure_ascii=False,
                        indent=2,
                        sort_keys=True,
                    )
                )
            else:
                print(profile.python)
            return 0

        if args.command == "verify":
            if args.profile:
                names = [args.profile]
            elif args.role:
                profile = registry.resolve(
                    args.role, config_name=args.config_name, required=True
                )
                assert profile is not None
                names = [profile.name]
            else:
                names = sorted(registry.profiles)
            results = []
            for name in names:
                profile = registry.verify(name, timeout_s=args.timeout)
                results.append(_profile_payload(profile))
            registry.save()
            if args.json:
                print(json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True))
            else:
                for result in results:
                    print(f"Verified {result['name']}: {result['python']}")
            return 0

        if args.command == "doctor":
            normalized_config = normalize_config_name(args.config_name)
            required_roles: tuple[str, ...] = ()
            checked_roles: list[str] = []
            if args.role:
                profiles: set[str] = set()
                for role in args.role:
                    profile = registry.resolve(
                        role,
                        config_name=normalized_config,
                        required=True,
                    )
                    assert profile is not None
                    profiles.add(profile.name)
                    checked_roles.append(role)
            elif normalized_config:
                bound_roles = registry.bindings.get(normalized_config, {})
                required_roles = required_runtime_roles(normalized_config)
                checked_roles = sorted({*bound_roles, *required_roles})
                if not checked_roles:
                    raise RuntimeProfileError(
                        f"config {normalized_config!r} has no runtime profile bindings"
                    )
                missing_roles = [
                    role
                    for role in required_roles
                    if registry.bound_profile_name(
                        role,
                        config_name=normalized_config,
                    )
                    is None
                ]
                if missing_roles:
                    raise RuntimeProfileError(
                        f"config {normalized_config!r} is missing required runtime "
                        f"role bindings: {', '.join(missing_roles)}"
                    )
                profiles = set()
                for role in checked_roles:
                    profile = registry.resolve(
                        role,
                        config_name=normalized_config,
                        required=True,
                    )
                    assert profile is not None
                    profiles.add(profile.name)
            else:
                profiles = set(registry.profiles)
            results = []
            failed = False
            for name in sorted(profiles):
                status = registry.status(name)
                try:
                    probe = registry.probe(name, timeout_s=args.timeout)
                except RuntimeProfileError as exc:
                    failed = True
                    status["probe_ok"] = False
                    status["error"] = str(exc)
                else:
                    status["probe_ok"] = True
                    status["observed_fingerprint"] = probe["fingerprint"]
                    status["fingerprint_matches"] = (
                        status.get("fingerprint") == probe["fingerprint"]
                    )
                    if not status["verified"]:
                        failed = True
                        status["error"] = (
                            "profile is unverified or stale; run verify before launch"
                        )
                    elif not status["fingerprint_matches"]:
                        failed = True
                        status["error"] = (
                            "environment fingerprint changed; run verify after reviewing "
                            "the interpreter and package versions"
                        )
                results.append(status)
            payload = {
                "registry": str(registry.path),
                "config_name": normalized_config,
                "ok": not failed,
                "checked_roles": checked_roles,
                "required_roles": list(required_roles),
                "profiles": results,
            }
            if args.json:
                print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
            else:
                for result in results:
                    state = (
                        "PASS"
                        if result.get("probe_ok")
                        and result.get("verified")
                        and result.get("fingerprint_matches")
                        else "FAIL"
                    )
                    print(f"{state} {result['name']}: {result['python']}")
                    if result.get("error"):
                        print(f"  {result['error']}")
            return 0 if not failed else 1

        if args.command == "list":
            payload = {
                "registry": str(registry.path),
                "profiles": [
                    registry.status(name) for name in sorted(registry.profiles)
                ],
                "defaults": registry.defaults,
                "bindings": registry.bindings,
            }
            if args.json:
                print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
            else:
                print(f"Registry: {registry.path}")
                for profile in payload["profiles"]:
                    state = "verified" if profile["verified"] else "unverified"
                    print(f"  {profile['name']}: {profile['python']} ({state})")
                for role, profile in sorted(registry.defaults.items()):
                    print(f"  default:{role} -> {profile}")
                for config_name, roles in sorted(registry.bindings.items()):
                    for role, profile in sorted(roles.items()):
                        print(f"  {config_name}:{role} -> {profile}")
            return 0

        if args.command in {"command", "run"}:
            profile = registry.resolve(
                args.role, config_name=args.config_name, required=True
            )
            assert profile is not None
            _require_verified(profile, allow_unverified=args.allow_unverified)
            arguments = _command_arguments(args)
            payload = _command_payload(
                profile,
                arguments,
                config_name=args.config_name,
                role=args.role,
            )
            if args.command == "command":
                if args.json:
                    print(
                        json.dumps(
                            payload, ensure_ascii=False, indent=2, sort_keys=True
                        )
                    )
                else:
                    print(payload["shell"])
                return 0
            environment = profile.process_environment(config_name=args.config_name)
            environment["SUPERINFERENCE_RUNTIME_ROLE"] = args.role
            os.chdir(profile.cwd)
            os.execvpe(str(profile.python), profile.command(arguments), environment)
            raise AssertionError("os.execvpe returned unexpectedly")

        raise RuntimeProfileError(f"unsupported command: {args.command}")
    except RuntimeProfileError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"error: cannot execute selected runtime: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
