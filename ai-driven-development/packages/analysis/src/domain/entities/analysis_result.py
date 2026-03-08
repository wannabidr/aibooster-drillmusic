"""AnalysisResult entity."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime

from src.domain.value_objects.bpm_value import BPMValue
from src.domain.value_objects.energy_profile import EnergyProfile
from src.domain.value_objects.key_signature import KeySignature


@dataclass(frozen=True)
class AnalysisResult:
    id: uuid.UUID
    track_id: uuid.UUID
    bpm: BPMValue
    key: KeySignature
    energy: EnergyProfile
    fingerprint: str | None = None
    genre_embedding: list[float] | None = None
    analyzed_at: datetime | None = None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AnalysisResult):
            return NotImplemented
        return self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)
