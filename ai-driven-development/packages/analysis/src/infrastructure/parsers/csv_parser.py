"""CSV library parser -- imports tracks from a CSV/TSV file."""

from __future__ import annotations

import csv
import io
from pathlib import Path

from src.infrastructure.parsers.parse_result import LibraryParseResult

# Columns we know how to map.
_FLOAT_COLS = {"bpm"}
_INT_COLS = {"rating", "duration_s"}
_STR_COLS = {"title", "artist", "album", "genre", "key", "file_path", "date_added", "id"}


class CsvParser:
    def parse(self, file_path: str) -> LibraryParseResult:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        text = path.read_text(encoding="utf-8-sig")
        dialect = self._detect_dialect(text)
        reader = csv.DictReader(io.StringIO(text), dialect=dialect)

        tracks: list[dict] = []
        for row in reader:
            track = self._row_to_track(row)
            tracks.append(track)

        return LibraryParseResult(
            tracks=tracks,
            playlists=[],
            play_history=[],
            source="csv",
        )

    @staticmethod
    def _row_to_track(row: dict[str, str]) -> dict:
        track: dict = {}
        for col in _STR_COLS:
            track[col] = row.get(col, "")
        for col in _FLOAT_COLS:
            track[col] = _safe_float(row.get(col, ""))
        for col in _INT_COLS:
            track[col] = _safe_int(row.get(col, ""))
        return track

    @staticmethod
    def _detect_dialect(text: str) -> csv.Dialect:
        try:
            return csv.Sniffer().sniff(text[:2048], delimiters=",;\t|")
        except csv.Error:
            return csv.excel  # type: ignore[return-value]


def _safe_float(value: str) -> float:
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.0


def _safe_int(value: str) -> int:
    try:
        return int(value)
    except (ValueError, TypeError):
        return 0
