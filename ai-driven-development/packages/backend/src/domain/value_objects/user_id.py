"""UserId value object."""

from __future__ import annotations

import uuid
from dataclasses import dataclass


@dataclass(frozen=True)
class UserId:
    value: uuid.UUID

    @classmethod
    def generate(cls) -> UserId:
        return cls(value=uuid.uuid4())

    @classmethod
    def from_str(cls, s: str) -> UserId:
        return cls(value=uuid.UUID(s))

    def __str__(self) -> str:
        return str(self.value)
