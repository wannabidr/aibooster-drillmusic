"""Unit tests for SpotifyAdapter."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from src.infrastructure.streaming.spotify_adapter import SpotifyAdapter


def _install_mock_spotipy():
    """Install a mock spotipy module so the adapter can import it."""
    mock_spotipy = MagicMock()
    mock_oauth = MagicMock()
    mock_spotipy.oauth2 = mock_oauth
    sys.modules["spotipy"] = mock_spotipy
    sys.modules["spotipy.oauth2"] = mock_oauth
    return mock_spotipy


class TestSpotifyAdapter:
    def test_provider_name(self):
        adapter = SpotifyAdapter()
        assert adapter.provider_name() == "spotify"

    def test_not_authenticated_by_default(self):
        adapter = SpotifyAdapter()
        assert adapter.is_authenticated() is False

    def test_require_auth_raises_when_not_authenticated(self):
        adapter = SpotifyAdapter()
        with pytest.raises(RuntimeError, match="not authenticated"):
            adapter.search_tracks("test")

    def test_authenticate_missing_credentials(self):
        adapter = SpotifyAdapter()
        result = adapter.authenticate({})
        assert result is False
        assert adapter.is_authenticated() is False

    def test_authenticate_success(self):
        mock_spotipy = _install_mock_spotipy()
        try:
            mock_client = MagicMock()
            mock_client.categories.return_value = {"categories": {"items": []}}
            mock_spotipy.Spotify.return_value = mock_client

            adapter = SpotifyAdapter()
            result = adapter.authenticate(
                {"client_id": "test_id", "client_secret": "test_secret"}
            )

            assert result is True
            assert adapter.is_authenticated() is True
        finally:
            sys.modules.pop("spotipy", None)
            sys.modules.pop("spotipy.oauth2", None)

    def _make_authenticated_adapter(self):
        """Create a SpotifyAdapter with mocked authentication."""
        adapter = SpotifyAdapter()
        adapter._client = MagicMock()
        adapter._authenticated = True
        return adapter

    def test_search_tracks(self):
        adapter = self._make_authenticated_adapter()
        adapter._client.search.return_value = {
            "tracks": {
                "items": [
                    {
                        "id": "sp123",
                        "name": "Test Track",
                        "artists": [{"name": "DJ Test"}],
                        "album": {
                            "name": "Test Album",
                            "images": [{"url": "https://img.example.com/art.jpg"}],
                            "release_date": "2025-01-01",
                        },
                        "duration_ms": 300000,
                        "preview_url": "https://p.scdn.co/preview.mp3",
                        "external_ids": {"isrc": "US1234567890"},
                    }
                ],
                "total": 100,
            }
        }

        result = adapter.search_tracks("test", limit=10)

        assert len(result.tracks) == 1
        assert result.tracks[0].title == "Test Track"
        assert result.tracks[0].artist == "DJ Test"
        assert result.tracks[0].provider == "spotify"
        assert result.tracks[0].provider_track_id == "sp123"
        assert result.tracks[0].duration_ms == 300000
        assert result.tracks[0].isrc == "US1234567890"
        assert result.total == 100

    def test_get_track_metadata(self):
        adapter = self._make_authenticated_adapter()
        adapter._client.track.return_value = {
            "id": "sp456",
            "name": "Single Track",
            "artists": [{"name": "Artist"}],
            "album": {"name": "Album", "images": [], "release_date": None},
            "duration_ms": 240000,
            "preview_url": None,
            "external_ids": {},
        }

        meta = adapter.get_track_metadata("sp456")

        assert meta is not None
        assert meta.title == "Single Track"
        assert meta.provider_track_id == "sp456"

    def test_get_track_metadata_not_found(self):
        adapter = self._make_authenticated_adapter()
        adapter._client.track.return_value = None

        meta = adapter.get_track_metadata("nonexistent")
        assert meta is None

    def test_get_audio_features(self):
        adapter = self._make_authenticated_adapter()
        adapter._client.audio_features.return_value = [
            {
                "tempo": 128.5,
                "energy": 0.85,
                "danceability": 0.72,
                "valence": 0.6,
                "loudness": -5.2,
                "key": 9,
                "mode": 0,
            }
        ]

        features = adapter.get_audio_features("sp789")

        assert features is not None
        assert features["bpm"] == 128.5
        assert features["energy"] == 85.0
        assert features["danceability"] == 72.0
        assert features["valence"] == 60.0
        assert features["loudness"] == -5.2

    def test_get_audio_features_none(self):
        adapter = self._make_authenticated_adapter()
        adapter._client.audio_features.return_value = [None]

        features = adapter.get_audio_features("sp000")
        assert features is None

    def test_get_preview_url(self):
        adapter = self._make_authenticated_adapter()
        adapter._client.track.return_value = {
            "preview_url": "https://p.scdn.co/preview.mp3"
        }

        url = adapter.get_preview_url("sp123")
        assert url == "https://p.scdn.co/preview.mp3"

    def test_search_limits_to_50(self):
        adapter = self._make_authenticated_adapter()
        adapter._client.search.return_value = {"tracks": {"items": [], "total": 0}}

        adapter.search_tracks("test", limit=100)

        # Should cap at 50
        adapter._client.search.assert_called_once_with(
            q="test", type="track", limit=50, offset=0
        )

    def test_search_error_returns_empty(self):
        adapter = self._make_authenticated_adapter()
        adapter._client.search.side_effect = Exception("API error")

        result = adapter.search_tracks("test")
        assert result.tracks == []
        assert result.total == 0

    def test_get_track_metadata_error_returns_none(self):
        adapter = self._make_authenticated_adapter()
        adapter._client.track.side_effect = Exception("Network error")

        meta = adapter.get_track_metadata("sp123")
        assert meta is None

    def test_get_audio_features_error_returns_none(self):
        adapter = self._make_authenticated_adapter()
        adapter._client.audio_features.side_effect = Exception("API error")

        features = adapter.get_audio_features("sp123")
        assert features is None

    def test_parse_track_no_artists(self):
        meta = SpotifyAdapter._parse_track({
            "id": "x",
            "name": "No Artists",
            "artists": [],
            "album": {"name": "", "images": [], "release_date": None},
            "duration_ms": 0,
            "preview_url": None,
            "external_ids": {},
        })
        assert meta.artist == ""

    def test_has_more_pagination(self):
        adapter = self._make_authenticated_adapter()
        adapter._client.search.return_value = {
            "tracks": {
                "items": [],
                "total": 100,
            }
        }

        result = adapter.search_tracks("test", offset=0, limit=25)
        assert result.has_more is True

        result2 = adapter.search_tracks("test", offset=75, limit=25)
        assert result2.has_more is False
