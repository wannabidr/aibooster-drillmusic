"""Audio analyzer port (abstract interface)."""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.domain.entities.analysis_result import AnalysisResult
from src.domain.entities.audio_track import AudioTrack


class AudioAnalyzer(ABC):
    @abstractmethod
    def analyze(self, track: AudioTrack) -> AnalysisResult: ...
