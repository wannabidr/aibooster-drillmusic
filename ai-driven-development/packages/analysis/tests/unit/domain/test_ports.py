"""Unit tests for domain port interfaces (abstract base classes)."""

import pytest


class TestTrackRepository:
    """Tests for TrackRepository port interface."""

    def test_is_abstract(self):
        from src.domain.ports.track_repository import TrackRepository

        with pytest.raises(TypeError):
            TrackRepository()

    def test_has_required_methods(self):
        from src.domain.ports.track_repository import TrackRepository

        assert hasattr(TrackRepository, "save")
        assert hasattr(TrackRepository, "find_by_id")
        assert hasattr(TrackRepository, "find_by_hash")
        assert hasattr(TrackRepository, "find_all")
        assert hasattr(TrackRepository, "delete")


class TestAnalyzerPort:
    """Tests for AudioAnalyzer port interface."""

    def test_is_abstract(self):
        from src.domain.ports.audio_analyzer import AudioAnalyzer

        with pytest.raises(TypeError):
            AudioAnalyzer()

    def test_has_analyze_method(self):
        from src.domain.ports.audio_analyzer import AudioAnalyzer

        assert hasattr(AudioAnalyzer, "analyze")


class TestFingerprintPort:
    """Tests for AudioFingerprinter port interface."""

    def test_is_abstract(self):
        from src.domain.ports.audio_fingerprinter import AudioFingerprinter

        with pytest.raises(TypeError):
            AudioFingerprinter()

    def test_has_fingerprint_method(self):
        from src.domain.ports.audio_fingerprinter import AudioFingerprinter

        assert hasattr(AudioFingerprinter, "generate_fingerprint")
