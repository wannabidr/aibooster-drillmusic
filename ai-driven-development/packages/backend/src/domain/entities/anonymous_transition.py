"""AnonymousTransition entity -- community transition data with no PII."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class AnonymousTransition:
    id: uuid.UUID
    track_a_fingerprint: str
    track_b_fingerprint: str
    contributed_at: datetime

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AnonymousTransition):
            return NotImplemented
        return self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)
