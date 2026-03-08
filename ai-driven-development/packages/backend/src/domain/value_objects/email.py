"""Email value object with validation."""

from __future__ import annotations

import re
from dataclasses import dataclass

_EMAIL_REGEX = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")
_MAX_LENGTH = 255


@dataclass(frozen=True)
class Email:
    value: str

    def __post_init__(self) -> None:
        if not self.value or len(self.value) > _MAX_LENGTH:
            raise ValueError(f"Email must be 1-{_MAX_LENGTH} characters")
        normalized = self.value.lower()
        if normalized != self.value:
            object.__setattr__(self, "value", normalized)
        if not _EMAIL_REGEX.match(self.value):
            raise ValueError(f"Invalid email format: {self.value}")
