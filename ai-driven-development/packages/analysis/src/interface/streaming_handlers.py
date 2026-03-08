"""JSON-RPC streaming method handlers.

Exposes streaming search and analysis use cases to the Tauri sidecar.
"""

from __future__ import annotations

from dataclasses import asdict

from src.application.use_cases.analyze_streaming_track import AnalyzeStreamingTrack
from src.application.use_cases.search_streaming import SearchStreaming
from src.domain.ports.streaming_provider import StreamingProvider
from src.interface.server import JsonRpcServer


def register_streaming_handlers(
    server: JsonRpcServer,
    search_streaming: SearchStreaming,
    analyze_streaming: AnalyzeStreamingTrack,
    providers: dict[str, StreamingProvider],
) -> None:
    """Register streaming-related JSON-RPC methods.

    Methods:
        streaming_search(query, provider_names?, offset?, limit?) -> UnifiedSearchResult
        streaming_analyze(provider_name, provider_track_id) -> StreamingAnalysisResult
        streaming_auth(provider_name, credentials) -> {authenticated: bool}
        streaming_providers() -> {providers: [{name, authenticated}]}
    """

    def handle_streaming_search(
        query: str,
        provider_names: list[str] | None = None,
        offset: int = 0,
        limit: int = 25,
    ) -> dict:
        result = search_streaming.execute(
            query=query,
            provider_names=provider_names,
            offset=offset,
            limit=limit,
        )
        return {
            "results": [asdict(t) for t in result.results],
            "total_by_provider": result.total_by_provider,
            "query": result.query,
        }

    def handle_streaming_analyze(
        provider_name: str,
        provider_track_id: str,
    ) -> dict:
        result = analyze_streaming.execute(
            provider_name=provider_name,
            provider_track_id=provider_track_id,
        )
        return asdict(result)

    def handle_streaming_auth(
        provider_name: str,
        credentials: dict[str, str],
    ) -> dict:
        provider = providers.get(provider_name)
        if provider is None:
            return {"authenticated": False, "error": f"Unknown provider: {provider_name}"}
        success = provider.authenticate(credentials)
        return {"authenticated": success}

    def handle_streaming_providers() -> dict:
        provider_list = [
            {"name": name, "authenticated": p.is_authenticated()}
            for name, p in providers.items()
        ]
        return {"providers": provider_list}

    server.register("streaming_search", handle_streaming_search)
    server.register("streaming_analyze", handle_streaming_analyze)
    server.register("streaming_auth", handle_streaming_auth)
    server.register("streaming_providers", handle_streaming_providers)
