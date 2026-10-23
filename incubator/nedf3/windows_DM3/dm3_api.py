#!/usr/bin/env python3
"""
Minimal DM3 API client for Windows local data collection tools.

Author: yichen Xie, zhiyang zhang
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests

from utils.logger_config import logger


DEFAULT_BASE_URL = "https://dm3.noematrix.cn"


class DM3ApiError(RuntimeError):
    """Raised when a DM3 API request fails."""


@dataclass
class DM3TaskExecution:
    """Task execution summary returned by the DM3 "my tasks" endpoint."""

    id: Optional[int]
    name: str
    status: str
    raw: Dict[str, Any]


class DM3ApiClient:
    """Small wrapper around the DM3 endpoints documented in RobopocketApi."""

    def __init__(self, base_url: str = DEFAULT_BASE_URL, timeout_s: float = 15.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.token: Optional[str] = None
        self.last_login_response: Dict[str, Any] = {}
        self.session = requests.Session()

    @property
    def is_authenticated(self) -> bool:
        """Return whether the client currently has a bearer token."""
        return bool(self.token)

    def login(self, phone_number: str, password: str) -> Dict[str, Any]:
        """Login with phone number and password, storing the returned token in memory."""
        payload = {"phone_number": phone_number, "password": password}
        data = self._request(
            "POST",
            "/api/v1/auth/login-by-phone",
            json=payload,
            auth_required=False,
        )
        token = self._extract_token(data)
        if not token:
            raise DM3ApiError("Login response did not contain a token.")
        self.token = token
        self.last_login_response = data
        logger.info("DM3 login succeeded")
        return data

    def logout(self) -> None:
        """Logout the current DM3 session if authenticated."""
        if not self.token:
            return
        self._request("POST", "/api/v1/auth/logout", expect_json=False)
        self.token = None
        logger.info("DM3 logout succeeded")

    def get_my_task_executions(
        self,
        page: int = 1,
        page_size: int = 10,
        execute_day: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Fetch task executions assigned to the current user."""
        params: Dict[str, Any] = {"page": page, "page_size": page_size}
        if execute_day:
            params["execute_day"] = execute_day
        return self._request("GET", "/api/v1/daq/task-execution/my", params=params)

    def fetch_oss_sts_token(self, resource_type: str = "raw") -> Dict[str, Any]:
        """Fetch a temporary OSS STS token from DM3."""
        return self._request(
            "GET",
            "/api/v1/oss/sts-token",
            params={"resource_type": resource_type},
        )

    def normalize_task_executions(self, response: Dict[str, Any]) -> List[DM3TaskExecution]:
        """Normalize common paginated response shapes into task execution summaries."""
        items = self._extract_items(response)
        tasks: List[DM3TaskExecution] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            task_info = item.get("task") if isinstance(item.get("task"), dict) else {}
            name = (
                str(item.get("name") or "")
                or str(task_info.get("name") or "")
                or f"task_execution_{item.get('id', 'unknown')}"
            )
            status = str(item.get("status") or item.get("execution_status") or "")
            task_id = item.get("id")
            tasks.append(
                DM3TaskExecution(
                    id=int(task_id) if isinstance(task_id, int) else None,
                    name=name,
                    status=status,
                    raw=item,
                )
            )
        return tasks

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        auth_required: bool = True,
        expect_json: bool = True,
    ) -> Dict[str, Any]:
        """Send an HTTP request and return the JSON object response."""
        url = f"{self.base_url}{path}"
        headers = {"accept": "application/json"}
        if json is not None:
            headers["Content-Type"] = "application/json"
        if auth_required:
            if not self.token:
                raise DM3ApiError("Not logged in.")
            headers["Authorization"] = f"Bearer {self.token}"

        try:
            response = self.session.request(
                method,
                url,
                params=params,
                json=json,
                headers=headers,
                timeout=self.timeout_s,
            )
        except requests.RequestException as exc:
            raise DM3ApiError(f"Network error: {exc}") from exc

        if response.status_code == 401:
            raise DM3ApiError("登录状态已失效")
        if response.status_code < 200 or response.status_code >= 300:
            raise DM3ApiError(f"HTTP {response.status_code}: {response.text[:500]}")
        if not expect_json or not response.content:
            return {}
        try:
            parsed = response.json()
        except ValueError as exc:
            raise DM3ApiError(f"Response is not JSON: {response.text[:500]}") from exc
        if not isinstance(parsed, dict):
            return {"data": parsed}
        return parsed

    @staticmethod
    def _extract_token(data: Dict[str, Any]) -> Optional[str]:
        """Extract the bearer token from the confirmed DM3 login response shape."""
        token = data.get("token")
        if token:
            return str(token)
        print(f"DM3 login token field not found. Response keys: {list(data.keys())}")
        return None

    @staticmethod
    def _extract_items(response: Dict[str, Any]) -> List[Any]:
        """Extract list items from common DM3 pagination response shapes."""
        candidates: List[Any] = [
            response.get("items"),
            response.get("records"),
            response.get("list"),
            response.get("results"),
        ]
        data = response.get("data")
        if isinstance(data, list):
            candidates.append(data)
        elif isinstance(data, dict):
            candidates.extend(
                [
                    data.get("items"),
                    data.get("records"),
                    data.get("list"),
                    data.get("results"),
                ]
            )
        for candidate in candidates:
            if isinstance(candidate, list):
                return candidate
        return []

