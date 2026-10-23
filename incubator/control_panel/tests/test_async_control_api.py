from __future__ import annotations

import threading
import time
from typing import Any, Dict, Mapping, Optional

from incubator.control_panel.server import create_app


class StubStateSync:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_scope: Dict[str, Dict[str, Any]] = {}
        self._by_token: Dict[str, Dict[str, Any]] = {}
        self.failed = threading.Event()

    def begin(
        self,
        scope: str,
        action: str,
        desired: str,
        *,
        timeout_ms: int,
        token: Optional[str] = None,
    ) -> Dict[str, Any]:
        requested = str(token or f'token-{len(self._by_token) + 1:08d}')
        with self._lock:
            existing = self._by_token.get(requested)
            if existing is not None:
                return {**existing, 'duplicate': True}
            transition = {
                'token': requested,
                'scope': scope,
                'action': action,
                'desired': desired,
                'status': 'pending',
                'acknowledged': False,
                'timeout_ms': timeout_ms,
                'duplicate': False,
            }
            self._by_scope[scope] = transition
            self._by_token[requested] = transition
            return dict(transition)

    def acknowledge(self, token: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            transition = self._by_token.get(token)
            if transition is None:
                return None
            if transition['status'] == 'pending':
                transition['status'] = 'acknowledged'
                transition['acknowledged'] = True
            return dict(transition)

    def fail(self, token: str, reason: Any) -> Optional[Dict[str, Any]]:
        with self._lock:
            transition = self._by_token.get(token)
            if transition is None:
                return None
            transition['status'] = 'failed'
            transition['reason'] = str(reason)
            result = dict(transition)
        self.failed.set()
        return result

    def sample_if_due(self) -> Dict[str, Any]:
        with self._lock:
            transitions = {
                scope: dict(value) for scope, value in self._by_scope.items()
            }
        return {
            'revision': 0,
            'devices': [],
            'workflow': {},
            'recording': {},
            'grippers': {},
            'state_sync': {
                'authoritative': True,
                'stale': False,
                'transitions': transitions,
            },
        }

    def current(self) -> Mapping[str, Any]:
        return self.sample_if_due()

    def start(self) -> None:
        return None

    def close(self) -> None:
        return None


def _app(tmp_path):
    broker = StubStateSync()
    app = create_app(
        config_dir=tmp_path / 'configs',
        state_sync_factory=lambda source: broker,
    )
    app.testing = True
    return app, broker


def _cleanup(app) -> None:
    app.config['CONTROL_OPERATION_DISPATCHER'].close()
    app.config['DEVICE_CONTROLLER'].shutdown()


def test_browser_async_preference_returns_before_slow_device_operation(
    tmp_path,
) -> None:
    app, broker = _app(tmp_path)
    entered = threading.Event()
    release = threading.Event()
    second_finished = threading.Event()

    def slow_control(action: str, device_class: str, device_class_id: Any) -> Dict[str, Any]:
        if str(device_class_id) == '1':
            second_finished.set()
            return {'ok': True, 'action': action}
        entered.set()
        assert release.wait(timeout=2.0)
        return {'ok': True, 'action': action}

    app.config['DEVICE_CONTROLLER'].control_device = slow_control
    client = app.test_client()
    started = time.perf_counter()
    response = client.post(
        '/api/devices/WebCameraOpenPi/0/start',
        headers={
            'Prefer': 'respond-async',
            'X-Control-Transition': 'test-token-async-001',
        },
    )
    elapsed = time.perf_counter() - started

    try:
        assert response.status_code == 202
        assert response.headers['Preference-Applied'] == 'respond-async'
        assert response.get_json()['accepted'] is True
        assert response.get_json()['transition']['status'] == 'acknowledged'
        assert elapsed < 0.25
        assert entered.wait(timeout=1.0)
        state_started = time.perf_counter()
        state = client.get('/api/state').get_json()
        state_elapsed = time.perf_counter() - state_started
        assert state['state_sync']['transitions'][
            'device:WebCameraOpenPi:0'
        ]['status'] == 'acknowledged'
        assert state_elapsed < 0.25

        second_started = time.perf_counter()
        second = client.post(
            '/api/devices/WebCameraOpenPi/1/start',
            headers={
                'Prefer': 'respond-async',
                'X-Control-Transition': 'test-token-async-004',
            },
        )
        second_elapsed = time.perf_counter() - second_started
        assert second.status_code == 202
        assert second_elapsed < 0.25
        assert second_finished.wait(timeout=1.0)
    finally:
        release.set()
        _cleanup(app)

    assert broker.failed.is_set() is False


def test_legacy_client_without_preference_keeps_synchronous_result(tmp_path) -> None:
    app, _broker = _app(tmp_path)

    def control(action: str, device_class: str, device_class_id: Any) -> Dict[str, Any]:
        return {
            'ok': True,
            'action': action,
            'device': f'{device_class}:{device_class_id}',
        }

    app.config['DEVICE_CONTROLLER'].control_device = control
    response = app.test_client().post('/api/devices/WebCameraOpenPi/0/start')
    try:
        assert response.status_code == 202
        assert 'Preference-Applied' not in response.headers
        assert response.get_json()['device'] == 'WebCameraOpenPi:0'
        assert response.get_json().get('asynchronous') is None
    finally:
        _cleanup(app)


def test_async_operation_failure_is_reported_after_acceptance(tmp_path) -> None:
    app, broker = _app(tmp_path)

    def fail(action: str, device_class: str, device_class_id: Any) -> Dict[str, Any]:
        raise RuntimeError(f'{device_class}:{device_class_id} failed')

    app.config['DEVICE_CONTROLLER'].control_device = fail
    response = app.test_client().post(
        '/api/devices/WebCameraOpenPi/0/start',
        headers={
            'Prefer': 'respond-async',
            'X-Control-Transition': 'test-token-async-002',
        },
    )
    try:
        assert response.status_code == 202
        assert response.get_json()['accepted'] is True
        assert broker.failed.wait(timeout=2.0)
        transition = broker.sample_if_due()['state_sync']['transitions'][
            'device:WebCameraOpenPi:0'
        ]
        assert transition['status'] == 'failed'
        assert 'RuntimeError: WebCameraOpenPi:0 failed' in transition['reason']
    finally:
        _cleanup(app)


def test_duplicate_async_token_does_not_enqueue_operation_twice(tmp_path) -> None:
    app, _broker = _app(tmp_path)
    entered = threading.Event()
    release = threading.Event()
    calls = 0

    def slow_control(action: str, device_class: str, device_class_id: Any) -> Dict[str, Any]:
        nonlocal calls
        calls += 1
        entered.set()
        assert release.wait(timeout=2.0)
        return {'ok': True}

    app.config['DEVICE_CONTROLLER'].control_device = slow_control
    client = app.test_client()
    headers = {
        'Prefer': 'respond-async',
        'X-Control-Transition': 'test-token-async-003',
    }
    first = client.post('/api/devices/WebCameraOpenPi/0/start', headers=headers)
    assert entered.wait(timeout=1.0)
    duplicate = client.post('/api/devices/WebCameraOpenPi/0/start', headers=headers)
    try:
        assert first.status_code == 202
        assert duplicate.status_code == 202
        assert duplicate.get_json()['duplicate'] is True
        assert calls == 1
    finally:
        release.set()
        _cleanup(app)
