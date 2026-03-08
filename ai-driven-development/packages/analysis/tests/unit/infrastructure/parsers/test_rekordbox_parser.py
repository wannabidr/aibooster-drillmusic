"""Unit tests for Rekordbox XML parser."""

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

FIXTURE_PATH = Path(__file__).parents[3] / "fixtures" / "rekordbox_export.xml"


@pytest.fixture
def parser():
    from src.infrastructure.parsers.rekordbox_parser import RekordboxParser

    return RekordboxParser()


class TestRekordboxParser:
    def test_parse_tracks(self, parser):
        result = parser.parse(str(FIXTURE_PATH))
        assert len(result.tracks) == 3

    def test_track_metadata(self, parser):
        result = parser.parse(str(FIXTURE_PATH))
        track = result.tracks[0]
        assert track["title"] == "Techno Track"
        assert track["artist"] == "DJ Alpha"
        assert track["genre"] == "Techno"
        assert track["bpm"] == 128.0
        assert track["key"] == "Am"
        assert track["rating"] == 5

    def test_track_file_path(self, parser):
        result = parser.parse(str(FIXTURE_PATH))
        track = result.tracks[0]
        assert track["file_path"] == "/Users/dj/Music/techno_track.mp3"

    def test_parse_playlists(self, parser):
        result = parser.parse(str(FIXTURE_PATH))
        assert len(result.playlists) == 2
        assert result.playlists[0]["name"] == "Friday Set"
        assert len(result.playlists[0]["track_ids"]) == 2

    def test_handles_missing_fields(self, parser):
        result = parser.parse(str(FIXTURE_PATH))
        # Track 3 has empty album
        track = result.tracks[2]
        assert track["album"] == ""

    def test_handles_nonexistent_file(self, parser):
        with pytest.raises(FileNotFoundError):
            parser.parse("/nonexistent/file.xml")

    def test_handles_invalid_xml(self, parser, tmp_path):
        bad_file = tmp_path / "bad.xml"
        bad_file.write_text("not valid xml <<<")
        with pytest.raises(ET.ParseError):
            parser.parse(str(bad_file))
