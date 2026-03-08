"""Unit tests for Mixcloud tracklist parser."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.infrastructure.parsers.mixcloud_parser import MixcloudParser

FIXTURE_PATH = Path(__file__).parents[3] / "fixtures" / "mixcloud_tracklist.json"


@pytest.fixture
def parser() -> MixcloudParser:
    return MixcloudParser()


class TestMixcloudParser:
    def test_parse_tracks(self, parser: MixcloudParser) -> None:
        result = parser.parse(str(FIXTURE_PATH))
        assert len(result.tracks) == 3

    def test_source_is_mixcloud(self, parser: MixcloudParser) -> None:
        result = parser.parse(str(FIXTURE_PATH))
        assert result.source == "mixcloud"

    def test_track_metadata(self, parser: MixcloudParser) -> None:
        result = parser.parse(str(FIXTURE_PATH))
        track = result.tracks[0]
        assert track["title"] == "Techno Track"
        assert track["artist"] == "DJ Alpha"

    def test_track_start_time(self, parser: MixcloudParser) -> None:
        result = parser.parse(str(FIXTURE_PATH))
        assert result.tracks[0]["start_time_s"] == 0
        assert result.tracks[1]["start_time_s"] == 360

    def test_play_history_from_sections(self, parser: MixcloudParser) -> None:
        result = parser.parse(str(FIXTURE_PATH))
        assert len(result.play_history) == 2
        assert result.play_history[0]["track_a_id"] == "Techno Track"
        assert result.play_history[0]["track_b_id"] == "Deep House Vibes"

    def test_playlist_from_mix(self, parser: MixcloudParser) -> None:
        result = parser.parse(str(FIXTURE_PATH))
        assert len(result.playlists) == 1
        assert result.playlists[0]["name"] == "Friday Night Techno Session"
        assert len(result.playlists[0]["track_ids"]) == 3

    def test_nonexistent_file_raises(self, parser: MixcloudParser) -> None:
        with pytest.raises(FileNotFoundError):
            parser.parse("/nonexistent/file.json")

    def test_empty_sections(self, parser: MixcloudParser, tmp_path: Path) -> None:
        data = {"name": "Empty Mix", "sections": []}
        f = tmp_path / "empty.json"
        f.write_text(json.dumps(data))
        result = parser.parse(str(f))
        assert len(result.tracks) == 0
        assert len(result.play_history) == 0

    def test_missing_artist(self, parser: MixcloudParser, tmp_path: Path) -> None:
        data = {
            "name": "Mix",
            "sections": [
                {"track": {"name": "No Artist"}, "start_time": 0, "section_type": "track"}
            ],
        }
        f = tmp_path / "no_artist.json"
        f.write_text(json.dumps(data))
        result = parser.parse(str(f))
        assert result.tracks[0]["artist"] == ""

    def test_non_track_sections_skipped(self, parser: MixcloudParser, tmp_path: Path) -> None:
        data = {
            "name": "Mix",
            "sections": [
                {"track": {"name": "Track"}, "start_time": 0, "section_type": "track"},
                {"start_time": 300, "section_type": "chapter"},
            ],
        }
        f = tmp_path / "with_chapters.json"
        f.write_text(json.dumps(data))
        result = parser.parse(str(f))
        assert len(result.tracks) == 1

    def test_invalid_json_raises(self, parser: MixcloudParser, tmp_path: Path) -> None:
        f = tmp_path / "bad.json"
        f.write_text("not json {{{")
        with pytest.raises(ValueError, match="Invalid JSON"):
            parser.parse(str(f))
