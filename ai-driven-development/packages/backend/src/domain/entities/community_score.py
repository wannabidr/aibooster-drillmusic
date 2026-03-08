"""CommunityScore entity -- aggregated transition frequency."""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class CommunityScore:
    track_a_fingerprint: str
    track_b_fingerprint: str
    frequency: int

    def increment(self, amount: int = 1) -> CommunityScore:
        return replace(self, frequency=self.frequency + amount)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CommunityScore):
            return NotImplemented
        return (
            self.track_a_fingerprint == other.track_a_fingerprint
            and self.track_b_fingerprint == other.track_b_fingerprint
        )

    def __hash__(self) -> int:
        return hash((self.track_a_fingerprint, self.track_b_fingerprint))
