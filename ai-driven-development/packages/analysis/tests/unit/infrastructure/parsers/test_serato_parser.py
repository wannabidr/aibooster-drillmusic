"""Tests for Serato parser."""

import struct
import tempfile
from pathlib import Path

import pytest

from src.infrastructure.parsers.serato_parser import SeratoParser


@pytest.fixture
def parser():
    return SeratoParser()


def _build_crate_bytes(track_paths: list[str]) -> bytes:
    """Build a minimal Serato .crate binary file."""
    data = b""
    for path in track_paths:
        path_bytes = path.encode("utf-16-be")
        ptrk = b"ptrk" + struct.pack(">I", len(path_bytes)) + path_bytes
        otrk = b"otrk" + struct.pack(">I", len(ptrk)) + ptrk
        data += otrk
    return data


class TestSeratoParser:
    def test_parse_nonexistent_directory(self, parser):
        with pytest.raises(FileNotFoundError):
            parser.parse("/nonexistent/path")

    def test_parse_empty_directory(self, parser, tmp_path):
        result = parser.parse(str(tmp_path))
        assert result.source == "serato"
        assert len(result.tracks) == 0
        assert len(result.playlists) == 0

    def test_parse_crate_file(self, parser, tmp_path):
        # Create Serato directory structure
        subcrates = tmp_path / "_Serato_" / "Subcrates"
        subcrates.mkdir(parents=True)

        # Write a crate file with track paths
        crate_data = _build_crate_bytes(["/music/track1.mp3", "/music/track2.wav"])
        (subcrates / "MyPlaylist.crate").write_bytes(crate_data)

        result = parser.parse(str(tmp_path))
        assert result.source == "serato"
        assert len(result.playlists) == 1
        assert result.playlists[0]["name"] == "MyPlaylist"
        assert len(result.playlists[0]["track_ids"]) == 2

    def test_parse_multiple_crates(self, parser, tmp_path):
        subcrates = tmp_path / "_Serato_" / "Subcrates"
        subcrates.mkdir(parents=True)

        (subcrates / "House.crate").write_bytes(_build_crate_bytes(["/music/house1.mp3"]))
        (subcrates / "Techno.crate").write_bytes(_build_crate_bytes(["/music/techno1.mp3", "/music/techno2.mp3"]))

        result = parser.parse(str(tmp_path))
        assert len(result.playlists) == 2
        names = {p["name"] for p in result.playlists}
        assert "House" in names
        assert "Techno" in names

    def test_parse_corrupt_crate_gracefully(self, parser, tmp_path):
        subcrates = tmp_path / "_Serato_" / "Subcrates"
        subcrates.mkdir(parents=True)
        (subcrates / "Bad.crate").write_bytes(b"\x00\x01\x02\x03")

        result = parser.parse(str(tmp_path))
        # Should not crash, just return empty tracks for corrupt crate
        assert result.source == "serato"

    def test_no_database_returns_empty_tracks(self, parser, tmp_path):
        # Serato dir exists but no database V2
        (tmp_path / "_Serato_").mkdir()
        result = parser.parse(str(tmp_path))
        assert len(result.tracks) == 0
