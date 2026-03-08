"""Unit tests for streaming JSON-RPC handlers."""

from unittest.mock import MagicMock

from src.application.use_cases.analyze_streaming_track import StreamingAnalysisResult
from src.application.use_cases.search_streaming import UnifiedSearchResult
from src.domain.ports.streaming_provider import StreamingTrackMetadata
from src.interface.server import JsonRpcServer
from src.interface.streaming_handlers import register_streaming_handlers


def _setup_server():
    server = JsonRpcServer()
    search_use_case = MagicMock()
    analyze_use_case = MagicMock()

    spotify = MagicMock()
    spotify.is_authenticated.return_value = True
    beatport = MagicMock()
    beatport.is_authenticated.return_value = False

    providers = {"spotify": spotify, "beatport": beatport}

    register_streaming_handlers(server, search_use_case, analyze_use_case, providers)

    return server, search_use_case, analyze_use_case, providers


class TestStreamingHandlers:
    def test_streaming_search_registered(self):
        server, *_ = _setup_server()
        assert "streaming_search" in server._methods

    def test_streaming_analyze_registered(self):
        server, *_ = _setup_server()
        assert "streaming_analyze" in server._methods

    def test_streaming_auth_registered(self):
        server, *_ = _setup_server()
        assert "streaming_auth" in server._methods

    def test_streaming_providers_registered(self):
        server, *_ = _setup_server()
        assert "streaming_providers" in server._methods

    def test_streaming_search_handler(self):
        server, search_uc, *_ = _setup_server()
        track = StreamingTrackMetadata(
            provider="spotify",
            provider_track_id="sp1",
            title="Track",
            artist="Artist",
        )
        search_uc.execute.return_value = UnifiedSearchResult(
            results=[track],
            total_by_provider={"spotify": 1},
            query="test",
        )

        handler = server._methods["streaming_search"]
        result = handler(query="test")

        assert len(result["results"]) == 1
        assert result["results"][0]["title"] == "Track"
        assert result["total_by_provider"] == {"spotify": 1}
        assert result["query"] == "test"

    def test_streaming_analyze_handler(self):
        server, _, analyze_uc, _ = _setup_server()
        analyze_uc.execute.return_value = StreamingAnalysisResult(
            provider="beatport",
            provider_track_id="123",
            title="Analyzed",
            artist="DJ",
            bpm=128.0,
            key="Am",
            energy=75.0,
            bpm_source="provider",
            key_source="provider",
            preview_analyzed=False,
        )

        handler = server._methods["streaming_analyze"]
        result = handler(provider_name="beatport", provider_track_id="123")

        assert result["bpm"] == 128.0
        assert result["key"] == "Am"
        assert result["bpm_source"] == "provider"

    def test_streaming_auth_handler_success(self):
        server, _, _, providers = _setup_server()
        providers["spotify"].authenticate.return_value = True

        handler = server._methods["streaming_auth"]
        result = handler(
            provider_name="spotify",
            credentials={"client_id": "id", "client_secret": "secret"},
        )

        assert result["authenticated"] is True

    def test_streaming_auth_handler_unknown_provider(self):
        server, *_ = _setup_server()

        handler = server._methods["streaming_auth"]
        result = handler(provider_name="unknown", credentials={})

        assert result["authenticated"] is False
        assert "error" in result

    def test_streaming_providers_handler(self):
        server, *_ = _setup_server()

        handler = server._methods["streaming_providers"]
        result = handler()

        providers = result["providers"]
        assert len(providers) == 2

        spotify_info = next(p for p in providers if p["name"] == "spotify")
        beatport_info = next(p for p in providers if p["name"] == "beatport")

        assert spotify_info["authenticated"] is True
        assert beatport_info["authenticated"] is False

    def test_streaming_search_with_pagination(self):
        server, search_uc, *_ = _setup_server()
        search_uc.execute.return_value = UnifiedSearchResult(
            results=[],
            total_by_provider={},
            query="test",
        )

        handler = server._methods["streaming_search"]
        handler(query="test", offset=50, limit=10, provider_names=["spotify"])

        search_uc.execute.assert_called_once_with(
            query="test",
            provider_names=["spotify"],
            offset=50,
            limit=10,
        )
