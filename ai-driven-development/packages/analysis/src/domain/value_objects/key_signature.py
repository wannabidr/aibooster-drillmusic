"""Musical key signature value object with Camelot wheel support."""

from __future__ import annotations

import re
from dataclasses import dataclass

# Camelot wheel mapping: camelot_code -> (root, mode)
_CAMELOT_TO_KEY: dict[str, tuple[str, str]] = {
    "1A": ("Ab", "minor"),
    "2A": ("Eb", "minor"),
    "3A": ("Bb", "minor"),
    "4A": ("F", "minor"),
    "5A": ("C", "minor"),
    "6A": ("G", "minor"),
    "7A": ("D", "minor"),
    "8A": ("A", "minor"),
    "9A": ("E", "minor"),
    "10A": ("B", "minor"),
    "11A": ("F#", "minor"),
    "12A": ("Db", "minor"),
    "1B": ("B", "major"),
    "2B": ("F#", "major"),
    "3B": ("Db", "major"),
    "4B": ("Ab", "major"),
    "5B": ("Eb", "major"),
    "6B": ("Bb", "major"),
    "7B": ("F", "major"),
    "8B": ("C", "major"),
    "9B": ("G", "major"),
    "10B": ("D", "major"),
    "11B": ("A", "major"),
    "12B": ("E", "major"),
}

# Reverse mapping: (root, mode) -> camelot_code
_KEY_TO_CAMELOT: dict[tuple[str, str], str] = {v: k for k, v in _CAMELOT_TO_KEY.items()}

# Valid roots
_VALID_ROOTS = {"C", "D", "E", "F", "G", "A", "B", "Db", "Eb", "F#", "Ab", "Bb"}

# Standard notation patterns: "Am", "Cm", "C", "F#m", "Bbm", etc.
_KEY_PATTERN = re.compile(r"^([A-G][b#]?)(m(?:inor)?|maj(?:or)?)?$")

# Enharmonic equivalents
_ENHARMONIC: dict[str, str] = {
    "C#": "Db",
    "D#": "Eb",
    "Gb": "F#",
    "G#": "Ab",
    "A#": "Bb",
}


@dataclass(frozen=True)
class KeySignature:
    root: str
    mode: str  # "major" or "minor"

    def __init__(self, notation: str) -> None:
        match = _KEY_PATTERN.match(notation)
        if not match:
            raise ValueError(f"Invalid key notation: {notation}")

        root = match.group(1)
        mode_str = match.group(2)

        # Normalize enharmonic equivalents
        root = _ENHARMONIC.get(root, root)

        if root not in _VALID_ROOTS:
            raise ValueError(f"Invalid root note: {root}")

        mode = "minor" if mode_str and mode_str.startswith("m") else "major"

        object.__setattr__(self, "root", root)
        object.__setattr__(self, "mode", mode)

    @classmethod
    def from_camelot(cls, camelot: str) -> KeySignature:
        if camelot not in _CAMELOT_TO_KEY:
            raise ValueError(f"Invalid Camelot notation: {camelot}")
        root, mode = _CAMELOT_TO_KEY[camelot]
        # Build standard notation
        notation = f"{root}m" if mode == "minor" else root
        return cls(notation)

    def to_camelot(self) -> str:
        key = (self.root, self.mode)
        if key not in _KEY_TO_CAMELOT:
            raise ValueError(f"No Camelot mapping for {self.root} {self.mode}")
        return _KEY_TO_CAMELOT[key]
