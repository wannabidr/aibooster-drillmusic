"""Tests for JSON-RPC server and handlers."""

import json
from unittest.mock import MagicMock

import pytest

from src.interface.server import JsonRpcServer
from src.interface.handlers import register_handlers
from src.application.use_cases.analyze_track import AnalyzeTrack
from src.application.use_cases.batch_analyze import BatchAnalyze


@pytest.fixture
def server():
    return JsonRpcServer()


class TestJsonRpcServer:
    def test_register_and_call_method(self, server):
        server.register("echo", lambda msg: msg)
        response = server._handle({"jsonrpc": "2.0", "method": "echo", "params": {"msg": "hello"}, "id": 1})
        assert response["result"] == "hello"
        assert response["id"] == 1

    def test_method_not_found(self, server):
        response = server._handle({"jsonrpc": "2.0", "method": "nonexistent", "id": 1})
        assert response["error"]["code"] == -32601
        assert "not found" in response["error"]["message"]

    def test_method_exception_returns_error(self, server):
        def failing():
            raise ValueError("something broke")

        server.register("fail", failing)
        response = server._handle({"jsonrpc": "2.0", "method": "fail", "params": {}, "id": 1})
        assert response["error"]["code"] == -32000
        assert "something broke" in response["error"]["message"]

    def test_preserves_request_id(self, server):
        server.register("noop", lambda: "ok")
        response = server._handle({"jsonrpc": "2.0", "method": "noop", "params": {}, "id": 42})
        assert response["id"] == 42

    def test_null_id_in_error(self, server):
        response = server._error_response(None, -32700, "Parse error")
        assert response["id"] is None
        assert response["error"]["code"] == -32700

    def test_positional_params(self, server):
        server.register("add", lambda a, b: a + b)
        response = server._handle({"jsonrpc": "2.0", "method": "add", "params": [3, 4], "id": 1})
        assert response["result"] == 7

    def test_missing_params_defaults_to_empty(self, server):
        server.register("ping", lambda: "pong")
        response = server._handle({"jsonrpc": "2.0", "method": "ping", "id": 1})
        assert response["result"] == "pong"


class TestHandlerRegistration:
    def test_registers_all_methods(self):
        server = JsonRpcServer()
        mock_analyze = MagicMock(spec=AnalyzeTrack)
        mock_batch = MagicMock(spec=BatchAnalyze)

        register_handlers(server, mock_analyze, mock_batch)

        assert "analyze" in server._methods
        assert "batch_analyze" in server._methods
        assert "ping" in server._methods

    def test_ping_returns_pong(self):
        server = JsonRpcServer()
        mock_analyze = MagicMock(spec=AnalyzeTrack)
        mock_batch = MagicMock(spec=BatchAnalyze)

        register_handlers(server, mock_analyze, mock_batch)

        response = server._handle({"jsonrpc": "2.0", "method": "ping", "params": {}, "id": 1})
        assert response["result"] == "pong"
