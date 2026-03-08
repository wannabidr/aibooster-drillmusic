"""Unit tests for TidalAdapter."""

import sys
from unittest.mock import MagicMock

import pytest

from src.infrastructure.streaming.tidal_adapter import TidalAdapter


def _install_mock_tidalapi():
    """Install a mock tidalapi module so the adapter can import it."""
    mock_tidalapi = MagicMock()
    mock_media = MagicMock()
    mock_tidalapi.media = mock_media
    sys.modules["tidalapi"] = mock_tidalapi
    sys.modules["tidalapi.media"] = mock_media
    return mock_tidalapi


class TestTidalAdapter:
    def test_provider_name(self):
        adapter = TidalAdapter()
        assert adapter.provider_name() == "tidal"

    def test_not_authenticated_by_default(self):
        adapter = TidalAdapter()
        assert adapter.is_authenticated() is False

    def test_require_auth_raises_when_not_authenticated(self):
        adapter = TidalAdapter()
        with pytest.raises(RuntimeError, match="not authenticated"):
            adapter.search_tracks("test")

    def test_authenticate_missing_access_token(self):
        adapter = TidalAdapter()
        result = adapter.authenticate({})
        assert result is False

    def test_authenticate_success(self):
        mock_tidalapi = _install_mock_tidalapi()
        try:
            mock_session = MagicMock()
            mock_session.check_login.return_value = True
            mock_tidalapi.Session.return_value = mock_session

            adapter = TidalAdapter()
            result = adapter.authenticate(
                {"access_token": "tidal_token", "refresh_token": "tidal_refresh"}
            )

            assert result is True
            assert adapter.is_authenticated() is True
            mock_session.load_oauth_session.assert_called_once()
        finally:
            sys.modules.pop("tidalapi", None)
            sys.modules.pop("tidalapi.media", None)

    def test_authenticate_login_check_fails(self):
        mock_tidalapi = _install_mock_tidalapi()
        try:
            mock_session = MagicMock()
            mock_session.check_login.return_value = False
            mock_tidalapi.Session.return_value = mock_session

            adapter = TidalAdapter()
            result = adapter.authenticate({"access_token": "bad_token"})

            assert result is False
        finally:
            sys.modules.pop("tidalapi", None)
            sys.modules.pop("tidalapi.media", None)

    def _make_authenticated_adapter(self) -> TidalAdapter:
        adapter = TidalAdapter()
        adapter._session = MagicMock()
        adapter._authenticated = True
        # Mock _get_track_model to avoid importing tidalapi at runtime
        adapter._get_track_model = MagicMock(return_value="Track")
        return adapter

    def _make_mock_track(
        self,
        track_id: int = 123,
        name: str = "Test Track",
        artist_name: str = "DJ Test",
        album_name: str = "Test Album",
        duration: int = 300,
        isrc: str | None = "US1234567890",
    ) -> MagicMock:
        track = MagicMock()
        track.id = track_id
        track.name = name
        track.duration = duration
        track.isrc = isrc

        artist = MagicMock()
        artist.name = artist_name
        track.artist = artist
        track.artists = [artist]

        album = MagicMock()
        album.name = album_name
        album.image.return_value = "https://tidal.com/art.jpg"
        track.album = album

        return track

    def test_search_tracks(self):
        adapter = self._make_authenticated_adapter()
        mock_track = self._make_mock_track()

        adapter._session.search.return_value = {"tracks": [mock_track]}

        result = adapter.search_tracks("test query")

        assert len(result.tracks) == 1
        assert result.tracks[0].provider == "tidal"
        assert result.tracks[0].title == "Test Track"
        assert result.tracks[0].artist == "DJ Test"
        assert result.tracks[0].duration_ms == 300000
        assert result.tracks[0].isrc == "US1234567890"

    def test_get_track_metadata(self):
        adapter = self._make_authenticated_adapter()
        mock_track = self._make_mock_track(track_id=456, name="Single")

        adapter._session.track.return_value = mock_track

        meta = adapter.get_track_metadata("456")
        assert meta is not None
        assert meta.title == "Single"
        assert meta.provider_track_id == "456"

    def test_get_track_metadata_not_found(self):
        adapter = self._make_authenticated_adapter()
        adapter._session.track.return_value = None

        meta = adapter.get_track_metadata("999")
        assert meta is None

    def test_get_audio_features_returns_none(self):
        adapter = self._make_authenticated_adapter()
        features = adapter.get_audio_features("123")
        assert features is None

    def test_get_preview_url(self):
        adapter = self._make_authenticated_adapter()
        mock_track = self._make_mock_track()
        mock_track.get_url.return_value = "https://sp-pr-cf.audio.tidal.com/track.flac"
        adapter._session.track.return_value = mock_track

        url = adapter.get_preview_url("123")
        assert url == "https://sp-pr-cf.audio.tidal.com/track.flac"

    def test_get_preview_url_error(self):
        adapter = self._make_authenticated_adapter()
        adapter._session.track.side_effect = Exception("API error")

        url = adapter.get_preview_url("123")
        assert url is None

    def test_search_error_returns_empty(self):
        adapter = self._make_authenticated_adapter()
        adapter._session.search.side_effect = Exception("Timeout")

        result = adapter.search_tracks("test")
        assert result.tracks == []
        assert result.total == 0

    def test_parse_track_minimal(self):
        track = MagicMock()
        track.id = 1
        track.name = "Minimal"
        track.artist = None
        track.artists = []
        track.album = None
        track.duration = 0
        track.isrc = None

        meta = TidalAdapter._parse_track(track)
        assert meta is not None
        assert meta.title == "Minimal"
        assert meta.artist == ""
        assert meta.album == ""

    def test_get_track_metadata_error(self):
        adapter = self._make_authenticated_adapter()
        adapter._session.track.side_effect = Exception("Network error")

        meta = adapter.get_track_metadata("123")
        assert meta is None
