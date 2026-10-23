#!/usr/bin/env python3
"""Headless DM3 cloud-upload CLI (Linux port of windows_DM3/gui.py).

Same flow as the Windows GUI, without tkinter:

    login (phone + password)
      -> [optional] list my DM3 task executions
      -> fetch OSS STS token (resource_type, default "teleoperation_r1")
      -> upload finalized episode_* dirs to Aliyun OSS
      -> update local upload.json status (pending/uploading/success/failed)

Episode layout produced by incubator/r1/async_data_saver_NEDF3_r1.py:

    <data_dir>/
      <uuid>/                    <- finalized (has metadata.json) -> uploaded
        metadata.json
        data.mcap
      <uuid>_temp/               <- still writing (SKIPPED)

OSS object keys become: {sts.dir}/<uuid>/<relative_file_path>

Credentials (in priority order):
    --phone / --password  >  env DM3_PHONE / DM3_PASSWORD  >  interactive prompt

Examples
--------
    # Upload every finalized episode under saved_data to DM3 OSS
    python -m r1_teleop_dm3_upload.cli upload saved_data --phone 13800000000

    # Upload a single episode directory
    python -m r1_teleop_dm3_upload.cli upload saved_data/<uuid>

    # Force re-upload everything, even files already on OSS (no skipping)
    python -m r1_teleop_dm3_upload.cli upload saved_data --force

    # List my DM3 task executions
    python -m r1_teleop_dm3_upload.cli tasks --phone 13800000000
"""

from __future__ import annotations

import argparse
import getpass
import os
import sys
from pathlib import Path
from typing import Optional

from utils.logger_config import logger

from .dm3_api import DM3ApiClient, DM3ApiError
from .oss_uploader import upload_episode_data_dir
from .upload_state import update_upload_state


def _resolve_credentials(args: argparse.Namespace) -> tuple[str, str]:
    """Resolve (phone, password) from args, env, then interactive prompt."""
    phone = args.phone or os.environ.get("DM3_PHONE")
    if not phone:
        phone = input("DM3 phone number: ").strip()
    if not phone:
        raise SystemExit("A DM3 phone number is required (--phone / DM3_PHONE).")

    password = args.password or os.environ.get("DM3_PASSWORD")
    if not password:
        password = getpass.getpass("DM3 password: ")
    if not password:
        raise SystemExit("A DM3 password is required (--password / DM3_PASSWORD).")
    return phone, password


def _make_client(args: argparse.Namespace) -> DM3ApiClient:
    return DM3ApiClient(base_url=args.base_url, timeout_s=args.timeout)


def _cmd_tasks(args: argparse.Namespace) -> int:
    phone, password = _resolve_credentials(args)
    client = _make_client(args)
    client.login(phone, password)
    try:
        response = client.get_my_task_executions(
            page=args.page, page_size=args.page_size, execute_day=args.execute_day
        )
        tasks = client.normalize_task_executions(response)
        if not tasks:
            logger.info("No task executions found for this user.")
            return 0
        logger.info(f"Found {len(tasks)} task execution(s):")
        for t in tasks:
            logger.info(f"  id={t.id}  status={t.status!r}  name={t.name!r}")
        return 0
    finally:
        _safe_logout(client)


def _cmd_upload(args: argparse.Namespace) -> int:
    data_dir = Path(args.data_dir).expanduser().resolve()
    if not data_dir.is_dir():
        raise SystemExit(f"Not a directory: {data_dir}")

    phone, password = _resolve_credentials(args)
    client = _make_client(args)
    client.login(phone, password)

    resource_type = args.resource_type
    try:
        update_upload_state(
            data_dir,
            status="uploading",
            note=f"Uploading to DM3 OSS resource_type={resource_type}",
            resource_type=resource_type,
        )
        logger.info(f"Fetching OSS STS token (resource_type={resource_type})...")
        sts_token = client.fetch_oss_sts_token(resource_type=resource_type)

        if args.force:
            logger.info("Force mode: re-uploading all files (not skipping existing OSS objects)")
        result = upload_episode_data_dir(
            data_dir,
            sts_token,
            progress=logger.info,
            token_provider=lambda: client.fetch_oss_sts_token(resource_type=resource_type),
            skip_existing=not args.force,
        )

        update_upload_state(
            data_dir,
            status="success",
            note=f"Uploaded {len(result.files)} file(s) to oss://{result.bucket}/{result.prefix}",
            resource_type=resource_type,
        )
        logger.info(
            f"DONE: uploaded {len(result.files)} file(s) to "
            f"oss://{result.bucket}/{result.prefix}"
        )
        return 0
    except Exception as exc:
        try:
            update_upload_state(
                data_dir,
                status="failed",
                note=f"Upload failed: {exc}",
                resource_type=resource_type,
            )
        except Exception:
            pass
        logger.error(f"Upload failed: {exc}")
        return 1
    finally:
        _safe_logout(client)


def _safe_logout(client: DM3ApiClient) -> None:
    try:
        client.logout()
    except DM3ApiError as exc:
        logger.warning(f"DM3 logout failed (ignored): {exc}")


def _add_common_auth_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--phone", default=None, help="DM3 phone number (or env DM3_PHONE)")
    p.add_argument("--password", default=None, help="DM3 password (or env DM3_PASSWORD)")
    p.add_argument(
        "--base-url",
        default="https://dm3.noematrix.cn",
        help="DM3 API base URL",
    )
    p.add_argument("--timeout", type=float, default=15.0, help="HTTP timeout seconds")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DM3 cloud-upload CLI (Linux). Login + OSS upload of UUID episode dirs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_up = sub.add_parser("upload", help="Upload finalized episode (UUID) dirs to DM3 OSS")
    p_up.add_argument("data_dir", help="Task dir containing <uuid> episode dirs (or a single episode dir)")
    p_up.add_argument(
        "--resource-type",
        default="teleoperation_r1",
        help="OSS resource_type passed to the DM3 STS endpoint",
    )
    p_up.add_argument(
        "--force",
        action="store_true",
        help="Re-upload every file even if it already exists on OSS with the same "
             "size (default: skip already-uploaded files)",
    )
    _add_common_auth_args(p_up)
    p_up.set_defaults(func=_cmd_upload)

    p_tasks = sub.add_parser("tasks", help="List my DM3 task executions")
    p_tasks.add_argument("--page", type=int, default=1)
    p_tasks.add_argument("--page-size", type=int, default=10)
    p_tasks.add_argument("--execute-day", default=None, help="Filter by day (YYYY-MM-DD)")
    _add_common_auth_args(p_tasks)
    p_tasks.set_defaults(func=_cmd_tasks)

    return parser


def main(argv: Optional[list] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except DM3ApiError as exc:
        logger.error(f"DM3 API error: {exc}")
        return 1
    except KeyboardInterrupt:
        logger.info("Interrupted.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
