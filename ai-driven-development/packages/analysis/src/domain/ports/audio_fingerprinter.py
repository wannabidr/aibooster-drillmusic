"""Audio fingerprinter port (abstract interface)."""

from __future__ import annotations

from abc import ABC, abstractmethod


class AudioFingerprinter(ABC):
    @abstractmethod
    def generate_fingerprint(self, file_path: str) -> str: ...
