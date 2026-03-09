"""JSON-RPC 2.0 server over stdio for Tauri sidecar communication."""

from __future__ import annotations

import json
import sys
from typing import Any


class JsonRpcServer:
    def __init__(self) -> None:
        self._methods: dict[str, Any] = {}

    def register(self, name: str, handler: Any) -> None:
        self._methods[name] = handler

    def run(self) -> None:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                request = json.loads(line)
                response = self._handle(request)
            except json.JSONDecodeError:
                response = self._error_response(None, -32700, "Parse error")
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()

    def _handle(self, request: dict) -> dict:
        req_id = request.get("id")
        method = request.get("method")
        params = request.get("params", {})

        if method not in self._methods:
            return self._error_response(req_id, -32601, f"Method not found: {method}")

        try:
            handler = self._methods[method]
            result = handler(**params) if isinstance(params, dict) else handler(*params)
            return {"jsonrpc": "2.0", "result": result, "id": req_id}
        except Exception as e:
            return self._error_response(req_id, -32000, str(e))

    @staticmethod
    def _error_response(req_id: Any, code: int, message: str) -> dict:
        return {
            "jsonrpc": "2.0",
            "error": {"code": code, "message": message},
            "id": req_id,
        }


def main() -> None:
    """Entry point for the sidecar binary."""
    import sys as _sys

    _sys.stderr.write("[sidecar-py] Starting JSON-RPC server...\n")
    _sys.stderr.flush()

    server = JsonRpcServer()

    # Register ping handler (always available)
    server.register("ping", lambda: "pong")

    # Try to wire up full handlers with real dependencies
    try:
        from src.infrastructure.analyzers.essentia_analyzer import EssentiaAnalyzer
        from src.infrastructure.fingerprint.chromaprint_fingerprinter import (
            ChromaprintFingerprinter,
        )
        from src.infrastructure.persistence.sqlite_track_repository import (
            SQLiteTrackRepository,
        )
        from src.infrastructure.persistence.sqlite_mix_history_repository import (
            SQLiteMixHistoryRepository,
        )
        from src.application.use_cases.analyze_track import AnalyzeTrack
        from src.application.use_cases.batch_analyze import BatchAnalyze
        from src.domain.services.history_scoring import HistoryScoring
        from src.interface.handlers import register_handlers

        repo = SQLiteTrackRepository()
        analyzer = EssentiaAnalyzer()
        fingerprinter = ChromaprintFingerprinter()
        analyze_track = AnalyzeTrack(
            analyzer=analyzer,
            fingerprinter=fingerprinter,
            repository=repo,
        )
        batch_analyze = BatchAnalyze(analyze_track=analyze_track)
        mix_repo = SQLiteMixHistoryRepository()
        history_scorer = HistoryScoring(repository=mix_repo)

        register_handlers(
            server,
            analyze_track=analyze_track,
            batch_analyze=batch_analyze,
            mix_history_repo=mix_repo,
            history_scorer=history_scorer,
        )
        _sys.stderr.write("[sidecar-py] Full handlers registered.\n")
    except ImportError as e:
        _sys.stderr.write(f"[sidecar-py] Warning: Could not load full handlers: {e}\n")
        _sys.stderr.write("[sidecar-py] Running with ping-only mode.\n")

    _sys.stderr.flush()
    server.run()


if __name__ == "__main__":
    main()
