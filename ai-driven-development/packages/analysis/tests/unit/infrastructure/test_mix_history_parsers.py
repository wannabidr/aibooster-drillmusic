"""Tests for history extraction from DJ software parsers."""

from pathlib import Path

import pytest

FIXTURES = Path(__file__).parents[2] / "fixtures"
REKORDBOX_FIXTURE = FIXTURES / "rekordbox_export.xml"
TRAKTOR_FIXTURE = FIXTURES / "traktor_collection.nml"


class TestRekordboxHistoryExtraction:
    @pytest.fixture
    def parser(self):
        from src.infrastructure.parsers.rekordbox_parser import RekordboxParser

        return RekordboxParser()

    def test_parse_returns_play_history(self, parser):
        result = parser.parse(str(REKORDBOX_FIXTURE))
        assert hasattr(result, "play_history")
        assert len(result.play_history) > 0

    def test_history_contains_track_pairs(self, parser):
        result = parser.parse(str(REKORDBOX_FIXTURE))
        # First session: tracks 1->2->3, so pairs (1,2) and (2,3)
        pair = result.play_history[0]
        assert "track_a_id" in pair
        assert "track_b_id" in pair
        assert "timestamp" in pair

    def test_history_pair_count(self, parser):
        result = parser.parse(str(REKORDBOX_FIXTURE))
        # Session 1: 1->2, 2->3 = 2 pairs
        # Session 2: 2->1 = 1 pair
        # Total = 3 pairs
        assert len(result.play_history) == 3

    def test_history_timestamps(self, parser):
        result = parser.parse(str(REKORDBOX_FIXTURE))
        # First session date is 2025-01-20
        assert result.play_history[0]["timestamp"] == "2025-01-20"

    def test_existing_tracks_still_parsed(self, parser):
        """Ensure adding history doesn't break existing track parsing."""
        result = parser.parse(str(REKORDBOX_FIXTURE))
        assert len(result.tracks) == 3
        assert result.tracks[0]["title"] == "Techno Track"

    def test_existing_playlists_still_parsed(self, parser):
        result = parser.parse(str(REKORDBOX_FIXTURE))
        assert len(result.playlists) == 2


class TestTraktorHistoryExtraction:
    @pytest.fixture
    def parser(self):
        from src.infrastructure.parsers.traktor_parser import TraktorParser

        return TraktorParser()

    def test_parse_returns_play_history(self, parser):
        result = parser.parse(str(TRAKTOR_FIXTURE))
        assert hasattr(result, "play_history")
        assert len(result.play_history) > 0

    def test_history_contains_track_pairs(self, parser):
        result = parser.parse(str(TRAKTOR_FIXTURE))
        pair = result.play_history[0]
        assert "track_a_id" in pair
        assert "track_b_id" in pair

    def test_history_pair_count(self, parser):
        result = parser.parse(str(TRAKTOR_FIXTURE))
        # One set with 2 tracks -> 1 pair
        assert len(result.play_history) == 1

    def test_existing_tracks_still_parsed(self, parser):
        result = parser.parse(str(TRAKTOR_FIXTURE))
        assert len(result.tracks) == 2
        assert result.tracks[0]["title"] == "Techno Track"

    def test_existing_playlists_still_parsed(self, parser):
        result = parser.parse(str(TRAKTOR_FIXTURE))
        assert len(result.playlists) == 1


class TestSeratoHistoryExtraction:
    @pytest.fixture
    def parser(self):
        from src.infrastructure.parsers.serato_parser import SeratoParser

        return SeratoParser()

    def test_parse_returns_play_history_field(self, parser, tmp_path):
        """Serato parser returns empty play_history (binary format, no fixture)."""
        serato_dir = tmp_path / "_Serato_" / "Subcrates"
        serato_dir.mkdir(parents=True)
        result = parser.parse(str(tmp_path))
        assert hasattr(result, "play_history")
        assert isinstance(result.play_history, list)
