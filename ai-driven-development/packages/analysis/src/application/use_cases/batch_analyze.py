"""BatchAnalyze use case -- analyze multiple audio files."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from src.application.dto.analysis_response import AnalysisResponse
from src.application.use_cases.analyze_track import AnalysisError, AnalyzeTrack

SUPPORTED_EXTENSIONS = {".wav", ".aiff", ".aif", ".mp3", ".flac", ".ogg", ".m4a"}


@dataclass(frozen=True)
class BatchResult:
    succeeded: list[AnalysisResponse]
    failed: list[tuple[str, str]]  # (file_path, error_message)
    skipped: int


class BatchAnalyze:
    def __init__(self, analyze_track: AnalyzeTrack) -> None:
        self._analyze_track = analyze_track

    def execute(
        self,
        directory: str,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> BatchResult:
        audio_files = self._scan_directory(directory)
        total = len(audio_files)
        succeeded: list[AnalysisResponse] = []
        failed: list[tuple[str, str]] = []
        skipped = 0

        for i, file_path in enumerate(audio_files):
            if progress_callback:
                progress_callback(i + 1, total)
            try:
                file_hash = self._compute_hash(file_path)
                response = self._analyze_track.execute(file_path=file_path, file_hash=file_hash)
                if response.cached:
                    skipped += 1
                else:
                    succeeded.append(response)
            except AnalysisError as e:
                failed.append((file_path, str(e)))

        return BatchResult(succeeded=succeeded, failed=failed, skipped=skipped)

    @staticmethod
    def _scan_directory(directory: str) -> list[str]:
        files = []
        for entry in sorted(Path(directory).rglob("*")):
            if entry.is_file() and entry.suffix.lower() in SUPPORTED_EXTENSIONS:
                files.append(str(entry))
        return files

    @staticmethod
    def _compute_hash(file_path: str) -> str:
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
