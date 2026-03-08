"""Unit tests for Traktor NML parser."""

from pathlib import Path

import pytest

FIXTURE_PATH = Path(__file__).parents[3] / "fixtures" / "traktor_collection.nml"


@pytest.fixture
def parser():
    from src.infrastructure.parsers.traktor_parser import TraktorParser

    return TraktorParser()


class TestTraktorParser:
    def test_parse_tracks(self, parser):
        result = parser.parse(str(FIXTURE_PATH))
        assert len(result.tracks) == 2

    def test_track_metadata(self, parser):
        result = parser.parse(str(FIXTURE_PATH))
        track = result.tracks[0]
        assert track["title"] == "Techno Track"
        assert track["artist"] == "DJ Alpha"
        assert track["genre"] == "Techno"
        assert track["bpm"] == 128.0
        assert track["key"] == "Am"

    def test_track_file_path(self, parser):
        result = parser.parse(str(FIXTURE_PATH))
        track = result.tracks[0]
        assert track["file_path"] == "/Users/dj/Music/techno_track.mp3"

    def test_parse_playlists(self, parser):
        result = parser.parse(str(FIXTURE_PATH))
        assert len(result.playlists) == 1
        assert result.playlists[0]["name"] == "Main Set"

    def test_handles_nonexistent_file(self, parser):
        with pytest.raises(FileNotFoundError):
            parser.parse("/nonexistent/file.nml")
