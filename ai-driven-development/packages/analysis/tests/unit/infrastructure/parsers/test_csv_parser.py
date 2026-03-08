"""Unit tests for CSV library parser."""

from __future__ import annotations

from pathlib import Path

import pytest
from src.infrastructure.parsers.csv_parser import CsvParser

FIXTURE_PATH = Path(__file__).parents[3] / "fixtures" / "library_export.csv"


@pytest.fixture
def parser() -> CsvParser:
    return CsvParser()


class TestCsvParser:
    def test_parse_tracks(self, parser: CsvParser) -> None:
        result = parser.parse(str(FIXTURE_PATH))
        assert len(result.tracks) == 3

    def test_source_is_csv(self, parser: CsvParser) -> None:
        result = parser.parse(str(FIXTURE_PATH))
        assert result.source == "csv"

    def test_track_metadata(self, parser: CsvParser) -> None:
        result = parser.parse(str(FIXTURE_PATH))
        track = result.tracks[0]
        assert track["title"] == "Techno Track"
        assert track["artist"] == "DJ Alpha"
        assert track["album"] == "Night Sessions"
        assert track["genre"] == "Techno"
        assert track["bpm"] == 128.0
        assert track["key"] == "Am"
        assert track["rating"] == 5
        assert track["duration_s"] == 360
        assert track["file_path"] == "/Users/dj/Music/techno_track.mp3"
        assert track["date_added"] == "2025-01-15"

    def test_empty_fields_default(self, parser: CsvParser) -> None:
        result = parser.parse(str(FIXTURE_PATH))
        # Track 2 has empty album
        track = result.tracks[1]
        assert track["album"] == ""

    def test_numeric_conversion(self, parser: CsvParser) -> None:
        result = parser.parse(str(FIXTURE_PATH))
        track = result.tracks[1]
        assert track["bpm"] == 122.5
        assert track["rating"] == 4
        assert track["duration_s"] == 420

    def test_playlists_empty(self, parser: CsvParser) -> None:
        result = parser.parse(str(FIXTURE_PATH))
        assert result.playlists == []

    def test_play_history_empty(self, parser: CsvParser) -> None:
        result = parser.parse(str(FIXTURE_PATH))
        assert result.play_history == []

    def test_nonexistent_file_raises(self, parser: CsvParser) -> None:
        with pytest.raises(FileNotFoundError):
            parser.parse("/nonexistent/file.csv")

    def test_empty_csv(self, parser: CsvParser, tmp_path: Path) -> None:
        empty_file = tmp_path / "empty.csv"
        empty_file.write_text("title,artist,bpm\n")
        result = parser.parse(str(empty_file))
        assert len(result.tracks) == 0

    def test_minimal_columns(self, parser: CsvParser, tmp_path: Path) -> None:
        minimal = tmp_path / "minimal.csv"
        minimal.write_text("title,artist\nSome Track,Some Artist\n")
        result = parser.parse(str(minimal))
        assert len(result.tracks) == 1
        assert result.tracks[0]["title"] == "Some Track"
        assert result.tracks[0]["artist"] == "Some Artist"
        assert result.tracks[0]["bpm"] == 0.0
        assert result.tracks[0]["rating"] == 0

    def test_bad_numeric_values(self, parser: CsvParser, tmp_path: Path) -> None:
        bad = tmp_path / "bad_nums.csv"
        bad.write_text("title,bpm,rating,duration_s\nTrack,not_a_number,bad,nope\n")
        result = parser.parse(str(bad))
        track = result.tracks[0]
        assert track["bpm"] == 0.0
        assert track["rating"] == 0
        assert track["duration_s"] == 0

    def test_semicolon_delimiter(self, parser: CsvParser, tmp_path: Path) -> None:
        semi = tmp_path / "semi.csv"
        semi.write_text("title;artist;bpm\nTrack A;DJ One;125.0\n")
        result = parser.parse(str(semi))
        assert len(result.tracks) == 1
        assert result.tracks[0]["title"] == "Track A"
        assert result.tracks[0]["bpm"] == 125.0

    def test_tab_delimiter(self, parser: CsvParser, tmp_path: Path) -> None:
        tsv = tmp_path / "tracks.tsv"
        tsv.write_text("title\tartist\tbpm\nTrack B\tDJ Two\t130.0\n")
        result = parser.parse(str(tsv))
        assert len(result.tracks) == 1
        assert result.tracks[0]["title"] == "Track B"
