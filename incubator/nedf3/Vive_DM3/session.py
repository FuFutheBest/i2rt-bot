#!/usr/bin/env python3
"""
DM3 session state for the Vive_DM3 TUI.

Keeps login and task selection in memory and mirrors them to
``dm3_collection_session.json`` for compatibility with existing savers.

Author: Yichen Xie
"""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Dict, List, Optional

from incubator.nedf3.windows_DM3.collection_session import (
    clear_collection_session,
    extract_task_description,
    extract_user_from_login,
    normalize_task_description,
    resolve_session_user_fields,
    update_login_session,
    update_selected_task_session,
)
from incubator.nedf3.windows_DM3.dm3_api import DM3ApiClient, DM3TaskExecution


@dataclass
class CollectionSession:
    """Active DM3 session for collection and upload."""

    client: Optional[DM3ApiClient] = None
    login_response: Dict[str, Any] = field(default_factory=dict)
    login_phone: str = ""
    user_id: str = ""
    user_name: str = ""
    selected_task: Optional[DM3TaskExecution] = None
    task_description: str = ""
    raw_tasks: List[DM3TaskExecution] = field(default_factory=list)

    @property
    def is_logged_in(self) -> bool:
        return self.client is not None and self.client.is_authenticated

    @property
    def has_selected_task(self) -> bool:
        return self.selected_task is not None

    def to_public_dict(self) -> Dict[str, Any]:
        task = self.selected_task
        return {
            "logged_in": self.is_logged_in,
            "login_phone": self.login_phone,
            "user_id": self.user_id,
            "user_name": self.user_name,
            "task_selected": self.has_selected_task,
            "task_execution_id": task.id if task else None,
            "task_name": task.name if task else "",
            "task_status": task.status if task else "",
            "task_description": self.task_description,
        }

    def metadata_for_saver(self) -> Dict[str, str]:
        return {
            "user_id": self.user_id,
            "user_name": self.user_name,
            "task_description": self.task_description,
        }


class SessionStore:
    """Thread-safe DM3 session holder for the TUI."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._session = CollectionSession()

    def public_state(self) -> Dict[str, Any]:
        with self._lock:
            return self._session.to_public_dict()

    def get_client(self) -> DM3ApiClient:
        with self._lock:
            if self._session.client is None or not self._session.client.is_authenticated:
                raise RuntimeError("请先登录 DM3。")
            return self._session.client

    def login(self, phone: str, password: str, base_url: str) -> Dict[str, Any]:
        client = DM3ApiClient(base_url=base_url)
        login_response = client.login(phone, password)
        user_id, user_name = extract_user_from_login(login_response)
        resolved_id, resolved_name = resolve_session_user_fields(
            {
                "user_id": user_id,
                "user_name": user_name,
                "login_phone": phone,
                "login_response": login_response,
            }
        )
        update_login_session(
            resolved_id,
            resolved_name,
            login_phone=phone,
            login_response=login_response,
        )
        with self._lock:
            self._session = CollectionSession(
                client=client,
                login_response=login_response,
                login_phone=phone.strip(),
                user_id=resolved_id,
                user_name=resolved_name,
            )
            return self._session.to_public_dict()

    def logout(self) -> None:
        with self._lock:
            if self._session.client is not None:
                try:
                    self._session.client.logout()
                except Exception:
                    pass
            self._session = CollectionSession()
        clear_collection_session()

    def fetch_tasks(
        self,
        *,
        page: int = 1,
        page_size: int = 10,
        execute_day: Optional[str] = None,
    ) -> List[DM3TaskExecution]:
        with self._lock:
            if self._session.client is None:
                raise RuntimeError("请先登录 DM3。")
            client = self._session.client
        response = client.get_my_task_executions(
            page=page,
            page_size=page_size,
            execute_day=execute_day or None,
        )
        tasks = client.normalize_task_executions(response)
        with self._lock:
            self._session.raw_tasks = tasks
        return tasks

    def select_task_by_index(self, index: int) -> Dict[str, Any]:
        with self._lock:
            if self._session.client is None:
                raise RuntimeError("请先登录 DM3。")
            try:
                task = self._session.raw_tasks[index]
            except IndexError as exc:
                raise ValueError("无效的任务索引。") from exc
            task_description = extract_task_description(task.raw)
            self._session.selected_task = task
            self._session.task_description = normalize_task_description(task_description)
            update_selected_task_session(task.raw, task_description=task_description)
            return self._session.to_public_dict()

    def require_ready_for_collection(self) -> CollectionSession:
        with self._lock:
            session = self._session
        if not session.is_logged_in:
            raise RuntimeError("请先登录 DM3。")
        if not session.has_selected_task:
            raise RuntimeError("请先选择 DM3 任务。")
        return session

