"""AudioTrack entity."""

from __future__ import annotations

import uuid
from dataclasses import dataclass


@dataclass(frozen=True)
class AudioTrack:
    id: uuid.UUID
    file_path: str
    file_hash: str
    title: str | None = None
    artist: str | None = None
    duration_ms: int | None = None
    analysis_status: str = "pending"
    failure_reason: str | None = None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AudioTrack):
            return NotImplemented
        return self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)

    def mark_as_analyzed(self) -> AudioTrack:
        return AudioTrack(
            id=self.id,
            file_path=self.file_path,
            file_hash=self.file_hash,
            title=self.title,
            artist=self.artist,
            duration_ms=self.duration_ms,
            analysis_status="analyzed",
        )

    def mark_as_failed(self, reason: str) -> AudioTrack:
        return AudioTrack(
            id=self.id,
            file_path=self.file_path,
            file_hash=self.file_hash,
            title=self.title,
            artist=self.artist,
            duration_ms=self.duration_ms,
            analysis_status="failed",
            failure_reason=reason,
        )
