"""Tests for BatchAnalyze use case."""

import os
import tempfile
from unittest.mock import MagicMock

import pytest

from src.application.dto.analysis_response import AnalysisResponse
from src.application.use_cases.analyze_track import AnalysisError, AnalyzeTrack
from src.application.use_cases.batch_analyze import (
    SUPPORTED_EXTENSIONS,
    BatchAnalyze,
    BatchResult,
)


@pytest.fixture
def mock_analyze_track():
    return MagicMock(spec=AnalyzeTrack)


@pytest.fixture
def batch_analyze(mock_analyze_track):
    return BatchAnalyze(mock_analyze_track)


class TestBatchAnalyze:
    def test_scans_supported_audio_files(self, batch_analyze, mock_analyze_track):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create files with supported and unsupported extensions
            for ext in [".wav", ".mp3", ".flac", ".txt", ".jpg"]:
                (open(os.path.join(tmpdir, f"test{ext}"), "w")).close()

            mock_analyze_track.execute.return_value = AnalysisResponse(
                track_id="id", file_path="/tmp/test.wav", bpm=128.0,
                bpm_confidence=0.9, key="Am", key_camelot="8A",
                key_confidence=0.8, energy_overall=75.0,
                energy_trajectory="maintain", fingerprint=None, cached=False,
            )

            result = batch_analyze.execute(tmpdir)
            # Should only process .wav, .mp3, .flac (not .txt, .jpg)
            assert mock_analyze_track.execute.call_count == 3

    def test_returns_batch_result(self, batch_analyze, mock_analyze_track):
        with tempfile.TemporaryDirectory() as tmpdir:
            open(os.path.join(tmpdir, "track.wav"), "w").close()

            mock_analyze_track.execute.return_value = AnalysisResponse(
                track_id="id", file_path="/tmp/test.wav", bpm=128.0,
                bpm_confidence=0.9, key="Am", key_camelot="8A",
                key_confidence=0.8, energy_overall=75.0,
                energy_trajectory="maintain", fingerprint=None, cached=False,
            )

            result = batch_analyze.execute(tmpdir)
            assert isinstance(result, BatchResult)
            assert len(result.succeeded) == 1
            assert len(result.failed) == 0
            assert result.skipped == 0

    def test_counts_cached_as_skipped(self, batch_analyze, mock_analyze_track):
        with tempfile.TemporaryDirectory() as tmpdir:
            open(os.path.join(tmpdir, "track.wav"), "w").close()

            mock_analyze_track.execute.return_value = AnalysisResponse(
                track_id="id", file_path="/tmp/test.wav", bpm=128.0,
                bpm_confidence=0.9, key="Am", key_camelot="8A",
                key_confidence=0.8, energy_overall=75.0,
                energy_trajectory="maintain", fingerprint=None, cached=True,
            )

            result = batch_analyze.execute(tmpdir)
            assert len(result.succeeded) == 0
            assert result.skipped == 1

    def test_captures_failures(self, batch_analyze, mock_analyze_track):
        with tempfile.TemporaryDirectory() as tmpdir:
            open(os.path.join(tmpdir, "bad.mp3"), "w").close()

            mock_analyze_track.execute.side_effect = AnalysisError("corrupt file")

            result = batch_analyze.execute(tmpdir)
            assert len(result.failed) == 1
            assert result.failed[0][1] == "corrupt file"

    def test_progress_callback(self, batch_analyze, mock_analyze_track):
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(3):
                open(os.path.join(tmpdir, f"track{i}.wav"), "w").close()

            mock_analyze_track.execute.return_value = AnalysisResponse(
                track_id="id", file_path="/tmp/test.wav", bpm=128.0,
                bpm_confidence=0.9, key="Am", key_camelot="8A",
                key_confidence=0.8, energy_overall=75.0,
                energy_trajectory="maintain", fingerprint=None, cached=False,
            )

            progress_calls = []
            batch_analyze.execute(tmpdir, progress_callback=lambda cur, total: progress_calls.append((cur, total)))

            assert len(progress_calls) == 3
            assert progress_calls[0] == (1, 3)
            assert progress_calls[2] == (3, 3)

    def test_empty_directory(self, batch_analyze):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = batch_analyze.execute(tmpdir)
            assert len(result.succeeded) == 0
            assert len(result.failed) == 0
            assert result.skipped == 0

    def test_supported_extensions_constant(self):
        assert ".wav" in SUPPORTED_EXTENSIONS
        assert ".mp3" in SUPPORTED_EXTENSIONS
        assert ".flac" in SUPPORTED_EXTENSIONS
        assert ".aiff" in SUPPORTED_EXTENSIONS
        assert ".txt" not in SUPPORTED_EXTENSIONS
