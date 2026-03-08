"""Chromaprint-based audio fingerprinter implementation."""

from __future__ import annotations

import hashlib
import subprocess

from src.domain.ports.audio_fingerprinter import AudioFingerprinter


class ChromaprintFingerprinter(AudioFingerprinter):
    def generate_fingerprint(self, file_path: str) -> str:
        try:
            result = subprocess.run(
                ["fpcalc", "-raw", file_path],
                capture_output=True,
                text=True,
                timeout=30,
                check=True,
            )
            for line in result.stdout.strip().split("\n"):
                if line.startswith("FINGERPRINT="):
                    return line.split("=", 1)[1]
            raise RuntimeError(f"fpcalc returned no fingerprint for {file_path}")
        except FileNotFoundError:
            raise RuntimeError(
                "fpcalc not found. Install Chromaprint: "
                "brew install chromaprint (macOS) or "
                "apt install libchromaprint-tools (Linux)"
            ) from None
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"fpcalc failed for {file_path}: {e.stderr}") from e

    @staticmethod
    def compute_file_hash(file_path: str) -> str:
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
