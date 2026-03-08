"""Unit tests for SoundCloud tracklist parser."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.infrastructure.parsers.soundcloud_parser import SoundCloudParser

FIXTURE_PATH = Path(__file__).parents[3] / "fixtures" / "soundcloud_tracklist.json"


@pytest.fixture
def parser() -> SoundCloudParser:
    return SoundCloudParser()


class TestSoundCloudParser:
    def test_parse_tracks(self, parser: SoundCloudParser) -> None:
        result = parser.parse(str(FIXTURE_PATH))
        assert len(result.tracks) == 3

    def test_source_is_soundcloud(self, parser: SoundCloudParser) -> None:
        result = parser.parse(str(FIXTURE_PATH))
        assert result.source == "soundcloud"

    def test_track_metadata(self, parser: SoundCloudParser) -> None:
        result = parser.parse(str(FIXTURE_PATH))
        track = result.tracks[0]
        assert track["title"] == "Opening Anthem"
        assert track["artist"] == "Producer A"
        assert track["genre"] == "Techno"
        assert track["bpm"] == 130.0
        assert track["key"] == "Cm"
        assert track["duration_s"] == 300

    def test_duration_ms_to_seconds(self, parser: SoundCloudParser) -> None:
        result = parser.parse(str(FIXTURE_PATH))
        assert result.tracks[1]["duration_s"] == 420

    def test_empty_key(self, parser: SoundCloudParser) -> None:
        result = parser.parse(str(FIXTURE_PATH))
        assert result.tracks[2]["key"] == ""

    def test_play_history_from_tracklist(self, parser: SoundCloudParser) -> None:
        result = parser.parse(str(FIXTURE_PATH))
        assert len(result.play_history) == 2
        assert result.play_history[0]["track_a_id"] == "Opening Anthem"
        assert result.play_history[0]["track_b_id"] == "Mid Set Groove"

    def test_playlist_from_mix(self, parser: SoundCloudParser) -> None:
        result = parser.parse(str(FIXTURE_PATH))
        assert len(result.playlists) == 1
        assert result.playlists[0]["name"] == "Weekend Mix Vol. 3"
        assert len(result.playlists[0]["track_ids"]) == 3

    def test_nonexistent_file_raises(self, parser: SoundCloudParser) -> None:
        with pytest.raises(FileNotFoundError):
            parser.parse("/nonexistent/file.json")

    def test_empty_tracks(self, parser: SoundCloudParser, tmp_path: Path) -> None:
        data = {"title": "Empty Mix", "tracks": []}
        f = tmp_path / "empty.json"
        f.write_text(json.dumps(data))
        result = parser.parse(str(f))
        assert len(result.tracks) == 0

    def test_missing_bpm_defaults_zero(self, parser: SoundCloudParser, tmp_path: Path) -> None:
        data = {
            "title": "Mix",
            "tracks": [
                {"title": "Track", "user": {"full_name": "Artist"}, "duration": 300000}
            ],
        }
        f = tmp_path / "no_bpm.json"
        f.write_text(json.dumps(data))
        result = parser.parse(str(f))
        assert result.tracks[0]["bpm"] == 0.0
        assert result.tracks[0]["key"] == ""

    def test_invalid_json_raises(self, parser: SoundCloudParser, tmp_path: Path) -> None:
        f = tmp_path / "bad.json"
        f.write_text("not json {{{")
        with pytest.raises(ValueError, match="Invalid JSON"):
            parser.parse(str(f))
