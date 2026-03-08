"""Unit tests for BeatportAdapter."""

import json
from unittest.mock import MagicMock, patch

import pytest

from src.infrastructure.streaming.beatport_adapter import BeatportAdapter


class TestBeatportAdapter:
    def test_provider_name(self):
        adapter = BeatportAdapter()
        assert adapter.provider_name() == "beatport"

    def test_not_authenticated_by_default(self):
        adapter = BeatportAdapter()
        assert adapter.is_authenticated() is False

    def test_require_auth_raises_when_not_authenticated(self):
        adapter = BeatportAdapter()
        with pytest.raises(RuntimeError, match="not authenticated"):
            adapter.search_tracks("test")

    def test_authenticate_missing_credentials(self):
        adapter = BeatportAdapter()
        result = adapter.authenticate({})
        assert result is False

    def _make_authenticated_adapter(self) -> BeatportAdapter:
        adapter = BeatportAdapter()
        adapter._access_token = "test_token"
        adapter._authenticated = True
        return adapter

    def test_search_tracks(self):
        adapter = self._make_authenticated_adapter()

        api_response = {
            "tracks": {
                "results": [
                    {
                        "id": 12345,
                        "name": "Deep House Track",
                        "artists": [{"name": "DJ Producer"}],
                        "release": {"name": "Summer EP"},
                        "bpm": 124,
                        "key": {"name": "Am", "camelot_number": "8A"},
                        "genre": {"name": "Deep House"},
                        "length_ms": 420000,
                        "sample_url": "https://geo-samples.beatport.com/track.mp3",
                        "image": {"uri": "https://geo-media.beatport.com/art.jpg"},
                        "isrc": "GB1234567890",
                        "publish_date": "2025-06-01",
                    }
                ],
                "count": 50,
            }
        }

        adapter._api_get = MagicMock(return_value=api_response)

        result = adapter.search_tracks("deep house", limit=25)

        assert len(result.tracks) == 1
        track = result.tracks[0]
        assert track.provider == "beatport"
        assert track.provider_track_id == "12345"
        assert track.title == "Deep House Track"
        assert track.artist == "DJ Producer"
        assert track.bpm == 124
        assert track.key == "Am"
        assert track.genre == "Deep House"
        assert track.duration_ms == 420000
        assert track.isrc == "GB1234567890"
        assert result.total == 50

    def test_get_track_metadata(self):
        adapter = self._make_authenticated_adapter()

        api_response = {
            "id": 99999,
            "name": "Single Track",
            "artists": [{"name": "Artist X"}],
            "release": {"name": "EP"},
            "bpm": 130,
            "key": {"name": "Cm"},
            "genre": {"name": "Techno"},
            "length_ms": 360000,
            "sample_url": None,
            "image": None,
            "isrc": None,
        }

        adapter._api_get = MagicMock(return_value=api_response)

        meta = adapter.get_track_metadata("99999")

        assert meta is not None
        assert meta.title == "Single Track"
        assert meta.bpm == 130
        assert meta.key == "Cm"

    def test_get_track_metadata_not_found(self):
        adapter = self._make_authenticated_adapter()
        adapter._api_get = MagicMock(return_value=None)

        meta = adapter.get_track_metadata("999")
        assert meta is None

    def test_get_audio_features(self):
        adapter = self._make_authenticated_adapter()

        api_response = {
            "id": 11111,
            "bpm": 128,
            "key": {"name": "Dm", "camelot_number": "7A"},
        }

        adapter._api_get = MagicMock(return_value=api_response)

        features = adapter.get_audio_features("11111")

        assert features is not None
        assert features["bpm"] == 128.0
        assert features["_has_key"] == 1.0

    def test_get_audio_features_no_data(self):
        adapter = self._make_authenticated_adapter()
        adapter._api_get = MagicMock(return_value=None)

        features = adapter.get_audio_features("999")
        assert features is None

    def test_get_audio_features_no_bpm_no_key(self):
        adapter = self._make_authenticated_adapter()
        adapter._api_get = MagicMock(return_value={"id": 1})

        features = adapter.get_audio_features("1")
        assert features is None

    def test_get_preview_url(self):
        adapter = self._make_authenticated_adapter()
        adapter._api_get = MagicMock(return_value={
            "sample_url": "https://geo-samples.beatport.com/track.mp3"
        })

        url = adapter.get_preview_url("123")
        assert url == "https://geo-samples.beatport.com/track.mp3"

    def test_get_preview_url_not_found(self):
        adapter = self._make_authenticated_adapter()
        adapter._api_get = MagicMock(return_value=None)

        url = adapter.get_preview_url("999")
        assert url is None

    def test_search_error_returns_empty(self):
        adapter = self._make_authenticated_adapter()
        adapter._api_get = MagicMock(side_effect=Exception("Timeout"))

        result = adapter.search_tracks("test")
        assert result.tracks == []
        assert result.total == 0

    def test_parse_track_no_artists(self):
        result = BeatportAdapter._parse_track({
            "id": 1,
            "name": "Track",
            "artists": [],
            "release": None,
            "bpm": None,
            "key": None,
            "genre": None,
            "length_ms": 0,
            "sample_url": None,
            "image": None,
        })
        assert result.artist == ""
        assert result.bpm is None

    def test_search_api_get_called_with_correct_url(self):
        adapter = self._make_authenticated_adapter()
        adapter._api_get = MagicMock(return_value={"tracks": {"results": [], "count": 0}})

        adapter.search_tracks("techno", offset=0, limit=10)

        call_url = adapter._api_get.call_args[0][0]
        assert "catalog/search" in call_url
        assert "q=techno" in call_url
        assert "per_page=10" in call_url

    def test_get_key_notation(self):
        adapter = self._make_authenticated_adapter()
        adapter._api_get = MagicMock(return_value={
            "key": {"name": "Am", "camelot_number": "8A"},
        })

        key = adapter.get_key_notation("123")
        assert key == "8A"

    def test_get_key_notation_no_key(self):
        adapter = self._make_authenticated_adapter()
        adapter._api_get = MagicMock(return_value={"key": None})

        key = adapter.get_key_notation("123")
        assert key is None
