"""Hardware-free FastWAM model-loading contracts."""

from __future__ import annotations

import gc
import http
import json
import sys
import types
import weakref
from pathlib import Path

import pytest

pytest.importorskip(
    "torch",
    reason="FastWAM model-loading contracts require the optional PyTorch runtime",
)

import torch
from websockets.asyncio import server as websocket_server
from websockets.datastructures import Headers
from websockets.server import ServerProtocol
from omegaconf import OmegaConf

from consumer.fastwam_server import FastWAMRuntime, FastWAMWebsocketServer, _resolve_checkpoint

pytest.importorskip(
    "nmx_vla",
    reason="FastWAM loading contracts require the checked-out NMX VLA model package",
)


class _CheckpointModel:
    def __init__(self) -> None:
        self.checkpoint_tensor_ref = None
        self.load_checkpoint_kwargs: dict[str, object] = {}
        self.eval_called = False
        self.requires_grad_value = None

    def load_checkpoint(self, _path: str, **kwargs: object):
        self.load_checkpoint_kwargs = kwargs
        checkpoint_tensor = torch.ones(1)
        self.checkpoint_tensor_ref = weakref.ref(checkpoint_tensor)
        return {"mot": {"weight": checkpoint_tensor}, "step": 15000}

    def eval(self):
        self.eval_called = True
        return self

    def requires_grad_(self, value: bool):
        self.requires_grad_value = value
        return self


class _WebsocketConnection:
    def respond(self, status: http.HTTPStatus, text: str) -> websocket_server.Response:
        return ServerProtocol().reject(status, text)


def test_health_response_matches_websockets_16_contract() -> None:
    server = FastWAMWebsocketServer.__new__(FastWAMWebsocketServer)
    server._health_identity = {"run_id": "run-1", "model_loaded": True}
    server._api_key = None
    request = websocket_server.Request(path="/healthz", headers=Headers())

    response = server._process_request(_WebsocketConnection(), request)

    assert response is not None
    assert response.status_code == http.HTTPStatus.OK
    assert response.headers["Content-Type"] == "application/json"
    assert json.loads(response.body) == {
        "ok": True,
        "run_id": "run-1",
        "model_loaded": True,
    }
    response.headers["Server"] = "test"
    assert response.serialize().startswith(b"HTTP/1.1 200 OK")


def test_load_model_releases_checkpoint_payload_before_text_encoder(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from nmx_vla.model.modules import wam

    runtime = FastWAMRuntime.__new__(FastWAMRuntime)
    runtime.params = {"device": "cpu", "param_dtype": "bf16"}
    runtime.weight_path = tmp_path / "checkpoint.pt"
    runtime._build_model_config = types.MethodType(
        lambda _self: {
            "model_id": "unused",
            "tokenizer_model_id": "unused",
            "video_dit_config": {},
            "load_text_encoder": False,
        },
        runtime,
    )
    model = _CheckpointModel()
    monkeypatch.setattr(wam, "create_fastwam", lambda **_kwargs: model)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    def check_payload_was_released(_self, attached_model, *_args):
        gc.collect()
        assert attached_model.checkpoint_tensor_ref() is None

    runtime._attach_text_encoder_if_configured = types.MethodType(
        check_payload_was_released,
        runtime,
    )

    assert runtime._load_model() is model
    assert model.load_checkpoint_kwargs == {"fail_on_incompatible_mot": True}
    assert model.eval_called is True
    assert model.requires_grad_value is False


def test_load_model_forwards_reverse_temporal_chunks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from nmx_vla.model.modules import wam

    runtime = FastWAMRuntime.__new__(FastWAMRuntime)
    runtime.params = {"device": "cpu", "param_dtype": "bf16"}
    runtime.weight_path = tmp_path / "checkpoint.pt"
    runtime._build_model_config = types.MethodType(
        lambda _self: {
            "model_id": "unused",
            "tokenizer_model_id": "unused",
            "video_dit_config": {},
            "load_text_encoder": False,
            "reverse_temporal_chunks": True,
        },
        runtime,
    )
    model = _CheckpointModel()
    received = {}

    def create_model(**kwargs):
        received.update(kwargs)
        return model

    monkeypatch.setattr(wam, "create_fastwam", create_model)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    runtime._attach_text_encoder_if_configured = types.MethodType(
        lambda *_args: None,
        runtime,
    )

    assert runtime._load_model() is model
    assert received["reverse_temporal_chunks"] is True


def test_runtime_can_disable_checkpoint_reverse_temporal_chunks() -> None:
    runtime = FastWAMRuntime.__new__(FastWAMRuntime)
    runtime.params = {"reverse_temporal_chunks": False}
    runtime.training_cfg = OmegaConf.create(
        {
            "framework": {
                "fastwam": {
                    "reverse_temporal_chunks": True,
                    "enable_video_prediction": True,
                }
            }
        }
    )
    runtime.contract = {"proprio_dim": 16}

    model_cfg = runtime._build_model_config()

    assert model_cfg["reverse_temporal_chunks"] is False


def test_attach_text_encoder_uses_low_cpu_memory_loader(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = FastWAMRuntime.__new__(FastWAMRuntime)
    runtime.params = {
        "load_text_encoder": True,
        "text_encoder_path": str(tmp_path / "umt5.pt"),
        "tokenizer_path": str(tmp_path / "tokenizer"),
    }
    model = types.SimpleNamespace(text_encoder=None, tokenizer=None)
    received = {}

    def load_text_encoder(*args, **kwargs):
        received["args"] = args
        received["kwargs"] = kwargs
        return object()

    loader_module = types.ModuleType("nmx_vla.model.modules.wam.helpers.loader")
    loader_module._load_registered_model = load_text_encoder
    tokenizer_module = types.ModuleType(
        "nmx_vla.model.modules.wam.wan_video_text_encoder"
    )
    tokenizer_module.HuggingfaceTokenizer = lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, "nmx_vla.model.modules.wam.helpers.loader", loader_module)
    monkeypatch.setitem(
        sys.modules,
        "nmx_vla.model.modules.wam.wan_video_text_encoder",
        tokenizer_module,
    )

    runtime._attach_text_encoder_if_configured(
        model,
        {"tokenizer_max_len": 128},
        torch.bfloat16,
        "cpu",
    )

    assert received["args"][1] == "wan_video_text_encoder"
    assert received["kwargs"]["low_cpu_mem_usage"] is True


def test_resolve_checkpoint_rejects_non_pt_file(tmp_path: Path) -> None:
    invalid = tmp_path / "checkpoint.yaml"
    invalid.write_text("not weights\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"\.pt suffix"):
        _resolve_checkpoint(str(invalid))
