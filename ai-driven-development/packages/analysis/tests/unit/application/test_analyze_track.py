"""Unit tests for AnalyzeTrack use case."""

import uuid
from unittest.mock import MagicMock

import pytest
from src.domain.entities.analysis_result import AnalysisResult
from src.domain.entities.audio_track import AudioTrack
from src.domain.value_objects.bpm_value import BPMValue
from src.domain.value_objects.energy_profile import EnergyProfile
from src.domain.value_objects.key_signature import KeySignature


def _make_track(file_path: str = "/music/track.mp3", file_hash: str = "abc123") -> AudioTrack:
    return AudioTrack(id=uuid.uuid4(), file_path=file_path, file_hash=file_hash)


def _make_result(track: AudioTrack) -> AnalysisResult:
    return AnalysisResult(
        id=uuid.uuid4(),
        track_id=track.id,
        bpm=BPMValue(128.0),
        key=KeySignature("Am"),
        energy=EnergyProfile(overall=75.0),
        fingerprint="AQAA...",
    )


class TestAnalyzeTrack:
    """Tests for the AnalyzeTrack use case."""

    def test_analyzes_new_track(self):
        from src.application.use_cases.analyze_track import AnalyzeTrack

        track = _make_track()
        result = _make_result(track)

        repo = MagicMock()
        repo.find_by_hash.return_value = None

        analyzer = MagicMock()
        analyzer.analyze.return_value = result

        fingerprinter = MagicMock()
        fingerprinter.generate_fingerprint.return_value = "AQAA..."

        use_case = AnalyzeTrack(
            track_repository=repo,
            analyzer=analyzer,
            fingerprinter=fingerprinter,
        )
        response = use_case.execute(file_path="/music/track.mp3", file_hash="abc123")

        assert response.bpm == 128.0
        assert response.key == "Am"
        assert response.cached is False
        analyzer.analyze.assert_called_once()
        repo.save.assert_called()

    def test_returns_cached_result(self):
        from src.application.use_cases.analyze_track import AnalyzeTrack

        track = _make_track()
        result = _make_result(track)

        repo = MagicMock()
        repo.find_by_hash.return_value = track
        repo.find_analysis_by_track_id.return_value = result

        analyzer = MagicMock()
        fingerprinter = MagicMock()

        use_case = AnalyzeTrack(
            track_repository=repo,
            analyzer=analyzer,
            fingerprinter=fingerprinter,
        )
        response = use_case.execute(file_path="/music/track.mp3", file_hash="abc123")

        assert response.cached is True
        assert response.bpm == 128.0
        analyzer.analyze.assert_not_called()

    def test_force_reanalyze_bypasses_cache(self):
        from src.application.use_cases.analyze_track import AnalyzeTrack

        track = _make_track()
        result = _make_result(track)

        repo = MagicMock()
        repo.find_by_hash.return_value = track

        analyzer = MagicMock()
        analyzer.analyze.return_value = result

        fingerprinter = MagicMock()
        fingerprinter.generate_fingerprint.return_value = "AQAA..."

        use_case = AnalyzeTrack(
            track_repository=repo,
            analyzer=analyzer,
            fingerprinter=fingerprinter,
        )
        response = use_case.execute(file_path="/music/track.mp3", file_hash="abc123", force=True)

        assert response.cached is False
        analyzer.analyze.assert_called_once()

    def test_handles_analysis_failure(self):
        from src.application.use_cases.analyze_track import AnalysisError, AnalyzeTrack

        repo = MagicMock()
        repo.find_by_hash.return_value = None

        analyzer = MagicMock()
        analyzer.analyze.side_effect = RuntimeError("Unsupported format")

        fingerprinter = MagicMock()

        use_case = AnalyzeTrack(
            track_repository=repo,
            analyzer=analyzer,
            fingerprinter=fingerprinter,
        )

        with pytest.raises(AnalysisError, match="Unsupported format"):
            use_case.execute(file_path="/music/track.midi", file_hash="xyz")
