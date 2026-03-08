"""AnalyzeTrack use case -- orchestrates analysis of a single audio file."""

from __future__ import annotations

import uuid

from src.application.dto.analysis_response import AnalysisResponse
from src.domain.entities.audio_track import AudioTrack
from src.domain.ports.audio_analyzer import AudioAnalyzer
from src.domain.ports.audio_fingerprinter import AudioFingerprinter
from src.domain.ports.genre_classifier import GenreClassifier
from src.domain.ports.track_repository import TrackRepository


class AnalysisError(Exception):
    """Raised when audio analysis fails."""


class AnalyzeTrack:
    def __init__(
        self,
        track_repository: TrackRepository,
        analyzer: AudioAnalyzer,
        fingerprinter: AudioFingerprinter,
        genre_classifier: GenreClassifier | None = None,
    ) -> None:
        self._repo = track_repository
        self._analyzer = analyzer
        self._fingerprinter = fingerprinter
        self._genre_classifier = genre_classifier

    def execute(
        self,
        file_path: str,
        file_hash: str,
        force: bool = False,
    ) -> AnalysisResponse:
        # Check cache
        if not force:
            existing = self._repo.find_by_hash(file_hash)
            if existing is not None:
                cached_result = self._repo.find_analysis_by_track_id(existing.id)
                if cached_result is not None:
                    return self._to_response(track=existing, result=cached_result, cached=True)

        # Create or reuse track
        existing = self._repo.find_by_hash(file_hash)
        if existing is not None:
            track = existing
        else:
            track = AudioTrack(
                id=uuid.uuid4(),
                file_path=file_path,
                file_hash=file_hash,
            )
            self._repo.save(track)

        # Run analysis
        try:
            result = self._analyzer.analyze(track)
        except Exception as e:
            failed = track.mark_as_failed(str(e))
            self._repo.save(failed)
            raise AnalysisError(str(e)) from e

        # Save results
        analyzed = track.mark_as_analyzed()
        self._repo.save(analyzed)
        self._repo.save_analysis(result)

        return self._to_response(track=analyzed, result=result, cached=False)

    @staticmethod
    def _to_response(track, result, cached: bool) -> AnalysisResponse:
        return AnalysisResponse(
            track_id=str(track.id),
            file_path=track.file_path,
            bpm=result.bpm.value,
            bpm_confidence=0.0,
            key=f"{result.key.root}{'m' if result.key.mode == 'minor' else ''}",
            key_camelot=result.key.to_camelot(),
            key_confidence=0.0,
            energy_overall=result.energy.overall,
            energy_trajectory=result.energy.trajectory,
            fingerprint=result.fingerprint,
            cached=cached,
            genre_embedding=result.genre_embedding,
        )
