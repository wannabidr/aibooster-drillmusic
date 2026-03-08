"""Unit tests for StreamingProvider port and data classes."""

from src.domain.ports.streaming_provider import (
    StreamingSearchResult,
    StreamingTrackMetadata,
)


class TestStreamingTrackMetadata:
    def test_creation_with_required_fields(self):
        meta = StreamingTrackMetadata(
            provider="spotify",
            provider_track_id="abc123",
            title="Test Track",
            artist="Test Artist",
        )
        assert meta.provider == "spotify"
        assert meta.provider_track_id == "abc123"
        assert meta.title == "Test Track"
        assert meta.artist == "Test Artist"

    def test_default_optional_fields(self):
        meta = StreamingTrackMetadata(
            provider="beatport",
            provider_track_id="456",
            title="Track",
            artist="Artist",
        )
        assert meta.album == ""
        assert meta.duration_ms == 0
        assert meta.bpm is None
        assert meta.key is None
        assert meta.genre == ""
        assert meta.preview_url is None
        assert meta.artwork_url is None
        assert meta.isrc is None
        assert meta.release_date is None

    def test_full_metadata(self):
        meta = StreamingTrackMetadata(
            provider="beatport",
            provider_track_id="789",
            title="Deep House Vibes",
            artist="DJ Example",
            album="Summer EP",
            duration_ms=420000,
            bpm=124.0,
            key="Am",
            genre="Deep House",
            preview_url="https://example.com/preview.mp3",
            artwork_url="https://example.com/art.jpg",
            isrc="US1234567890",
            release_date="2025-06-15",
        )
        assert meta.bpm == 124.0
        assert meta.key == "Am"
        assert meta.isrc == "US1234567890"

    def test_frozen(self):
        meta = StreamingTrackMetadata(
            provider="tidal",
            provider_track_id="1",
            title="T",
            artist="A",
        )
        try:
            meta.title = "New"  # type: ignore[misc]
            assert False, "Should have raised FrozenInstanceError"
        except AttributeError:
            pass


class TestStreamingSearchResult:
    def test_empty_result(self):
        result = StreamingSearchResult()
        assert result.tracks == []
        assert result.total == 0
        assert result.offset == 0
        assert result.limit == 25
        assert result.has_more is False

    def test_with_tracks(self):
        track1 = StreamingTrackMetadata(
            provider="spotify", provider_track_id="1", title="T1", artist="A1"
        )
        track2 = StreamingTrackMetadata(
            provider="spotify", provider_track_id="2", title="T2", artist="A2"
        )
        result = StreamingSearchResult(
            tracks=[track1, track2],
            total=100,
            offset=0,
            limit=25,
            has_more=True,
        )
        assert len(result.tracks) == 2
        assert result.total == 100
        assert result.has_more is True

    def test_pagination(self):
        result = StreamingSearchResult(
            total=50,
            offset=25,
            limit=25,
            has_more=False,
        )
        assert result.offset == 25
        assert result.has_more is False
