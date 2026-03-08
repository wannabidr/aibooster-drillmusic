"""Shared result type for library parsers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class LibraryParseResult:
    tracks: list[dict[str, Any]] = field(default_factory=list)
    playlists: list[dict[str, Any]] = field(default_factory=list)
    play_history: list[dict[str, Any]] = field(default_factory=list)
    source: str = ""
