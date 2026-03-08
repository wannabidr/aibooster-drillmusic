"""Unit tests for SearchStreaming use case."""

from unittest.mock import MagicMock

from src.application.use_cases.search_streaming import SearchStreaming
from src.domain.ports.streaming_provider import (
    StreamingSearchResult,
    StreamingTrackMetadata,
)


def _make_track(
    provider: str, track_id: str, title: str, isrc: str | None = None
) -> StreamingTrackMetadata:
    return StreamingTrackMetadata(
        provider=provider,
        provider_track_id=track_id,
        title=title,
        artist="Test Artist",
        isrc=isrc,
    )


def _make_provider(name: str, authenticated: bool = True) -> MagicMock:
    provider = MagicMock()
    provider.provider_name.return_value = name
    provider.is_authenticated.return_value = authenticated
    return provider


class TestSearchStreaming:
    def test_search_single_provider(self):
        provider = _make_provider("spotify")
        track = _make_track("spotify", "sp1", "Test Track")
        provider.search_tracks.return_value = StreamingSearchResult(
            tracks=[track], total=1, offset=0, limit=25
        )

        use_case = SearchStreaming(providers={"spotify": provider})
        result = use_case.execute("test query")

        assert len(result.results) == 1
        assert result.results[0].title == "Test Track"
        assert result.total_by_provider == {"spotify": 1}
        assert result.query == "test query"
        provider.search_tracks.assert_called_once_with("test query", offset=0, limit=25)

    def test_search_multiple_providers(self):
        spotify = _make_provider("spotify")
        beatport = _make_provider("beatport")

        sp_track = _make_track("spotify", "sp1", "Track A")
        bp_track = _make_track("beatport", "bp1", "Track B")

        spotify.search_tracks.return_value = StreamingSearchResult(
            tracks=[sp_track], total=50
        )
        beatport.search_tracks.return_value = StreamingSearchResult(
            tracks=[bp_track], total=30
        )

        use_case = SearchStreaming(providers={"spotify": spotify, "beatport": beatport})
        result = use_case.execute("house music")

        assert len(result.results) == 2
        assert result.total_by_provider == {"spotify": 50, "beatport": 30}

    def test_search_specific_providers(self):
        spotify = _make_provider("spotify")
        tidal = _make_provider("tidal")
        beatport = _make_provider("beatport")

        spotify.search_tracks.return_value = StreamingSearchResult(
            tracks=[_make_track("spotify", "1", "T1")], total=1
        )
        tidal.search_tracks.return_value = StreamingSearchResult(tracks=[], total=0)

        use_case = SearchStreaming(
            providers={"spotify": spotify, "tidal": tidal, "beatport": beatport}
        )
        result = use_case.execute("query", provider_names=["spotify", "tidal"])

        assert len(result.results) == 1
        beatport.search_tracks.assert_not_called()

    def test_skips_unauthenticated_providers(self):
        spotify = _make_provider("spotify", authenticated=True)
        tidal = _make_provider("tidal", authenticated=False)

        spotify.search_tracks.return_value = StreamingSearchResult(
            tracks=[_make_track("spotify", "1", "T1")], total=1
        )

        use_case = SearchStreaming(providers={"spotify": spotify, "tidal": tidal})
        result = use_case.execute("query")

        assert len(result.results) == 1
        tidal.search_tracks.assert_not_called()

    def test_deduplicates_by_isrc(self):
        spotify = _make_provider("spotify")
        beatport = _make_provider("beatport")

        # Same track on both services (same ISRC)
        sp_track = _make_track("spotify", "sp1", "Track (Spotify)", isrc="US123")
        bp_track = _make_track("beatport", "bp1", "Track (Beatport)", isrc="US123")

        spotify.search_tracks.return_value = StreamingSearchResult(
            tracks=[sp_track], total=1
        )
        beatport.search_tracks.return_value = StreamingSearchResult(
            tracks=[bp_track], total=1
        )

        use_case = SearchStreaming(providers={"spotify": spotify, "beatport": beatport})
        result = use_case.execute("track")

        # Should keep Beatport version (higher priority for DJ metadata)
        assert len(result.results) == 1
        assert result.results[0].provider == "beatport"

    def test_no_dedup_without_isrc(self):
        spotify = _make_provider("spotify")
        beatport = _make_provider("beatport")

        sp_track = _make_track("spotify", "sp1", "Track A")  # no ISRC
        bp_track = _make_track("beatport", "bp1", "Track B")  # no ISRC

        spotify.search_tracks.return_value = StreamingSearchResult(
            tracks=[sp_track], total=1
        )
        beatport.search_tracks.return_value = StreamingSearchResult(
            tracks=[bp_track], total=1
        )

        use_case = SearchStreaming(providers={"spotify": spotify, "beatport": beatport})
        result = use_case.execute("track")

        assert len(result.results) == 2

    def test_empty_query_returns_empty(self):
        use_case = SearchStreaming(providers={"spotify": _make_provider("spotify")})
        result = use_case.execute("")

        assert result.results == []
        assert result.total_by_provider == {}

    def test_whitespace_query_returns_empty(self):
        use_case = SearchStreaming(providers={"spotify": _make_provider("spotify")})
        result = use_case.execute("   ")

        assert result.results == []

    def test_pagination_passed_through(self):
        provider = _make_provider("spotify")
        provider.search_tracks.return_value = StreamingSearchResult(tracks=[], total=0)

        use_case = SearchStreaming(providers={"spotify": provider})
        use_case.execute("query", offset=50, limit=10)

        provider.search_tracks.assert_called_once_with("query", offset=50, limit=10)

    def test_unknown_provider_name_ignored(self):
        spotify = _make_provider("spotify")
        spotify.search_tracks.return_value = StreamingSearchResult(tracks=[], total=0)

        use_case = SearchStreaming(providers={"spotify": spotify})
        result = use_case.execute("query", provider_names=["nonexistent"])

        assert result.results == []
        spotify.search_tracks.assert_not_called()
