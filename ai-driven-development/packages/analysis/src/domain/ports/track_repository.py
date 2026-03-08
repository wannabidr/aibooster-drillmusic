"""Track repository port (abstract interface)."""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod

from src.domain.entities.analysis_result import AnalysisResult
from src.domain.entities.audio_track import AudioTrack


class TrackRepository(ABC):
    @abstractmethod
    def save(self, track: AudioTrack) -> None: ...

    @abstractmethod
    def find_by_id(self, track_id: uuid.UUID) -> AudioTrack | None: ...

    @abstractmethod
    def find_by_hash(self, file_hash: str) -> AudioTrack | None: ...

    @abstractmethod
    def find_all(self) -> list[AudioTrack]: ...

    @abstractmethod
    def delete(self, track_id: uuid.UUID) -> None: ...

    @abstractmethod
    def save_analysis(self, result: AnalysisResult) -> None: ...

    @abstractmethod
    def find_analysis_by_track_id(self, track_id: uuid.UUID) -> AnalysisResult | None: ...
