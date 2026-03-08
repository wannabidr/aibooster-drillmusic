"""Analysis response DTO."""

from dataclasses import dataclass


@dataclass(frozen=True)
class AnalysisResponse:
    track_id: str
    file_path: str
    bpm: float
    bpm_confidence: float
    key: str
    key_camelot: str
    key_confidence: float
    energy_overall: float
    energy_trajectory: str | None
    fingerprint: str | None
    cached: bool
    genre_embedding: list[float] | None = None
