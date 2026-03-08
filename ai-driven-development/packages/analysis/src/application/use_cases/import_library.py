"""ImportLibrary use case -- import tracks from DJ software."""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from src.domain.entities.audio_track import AudioTrack
from src.domain.ports.track_repository import TrackRepository
from src.infrastructure.parsers.csv_parser import CsvParser
from src.infrastructure.parsers.mixcloud_parser import MixcloudParser
from src.infrastructure.parsers.parse_result import LibraryParseResult
from src.infrastructure.parsers.rekordbox_parser import RekordboxParser
from src.infrastructure.parsers.serato_parser import SeratoParser
from src.infrastructure.parsers.soundcloud_parser import SoundCloudParser
from src.infrastructure.parsers.traktor_parser import TraktorParser


@dataclass(frozen=True)
class ImportResult:
    imported: int
    skipped: int
    source: str
    playlists: int


class ImportLibrary:
    def __init__(self, track_repository: TrackRepository) -> None:
        self._repo = track_repository
        self._parsers = {
            "rekordbox": RekordboxParser(),
            "traktor": TraktorParser(),
            "serato": SeratoParser(),
            "csv": CsvParser(),
            "mixcloud": MixcloudParser(),
            "soundcloud": SoundCloudParser(),
        }

    def execute(
        self,
        file_path: str,
        source: str | None = None,
    ) -> ImportResult:
        if source is None:
            source = self._detect_source(file_path)

        parser = self._parsers.get(source)
        if parser is None:
            raise ValueError(f"Unsupported source: {source}")

        result: LibraryParseResult = parser.parse(file_path)
        imported = 0
        skipped = 0

        for track_data in result.tracks:
            file_hash = track_data.get("file_path", "")
            existing = self._repo.find_by_hash(file_hash)
            if existing is not None:
                skipped += 1
                continue

            track = AudioTrack(
                id=uuid.uuid4(),
                file_path=track_data.get("file_path", ""),
                file_hash=file_hash,
                title=track_data.get("title"),
                artist=track_data.get("artist"),
            )
            self._repo.save(track)
            imported += 1

        return ImportResult(
            imported=imported,
            skipped=skipped,
            source=result.source,
            playlists=len(result.playlists),
        )

    @staticmethod
    def _detect_source(file_path: str) -> str:
        lower = file_path.lower()
        if lower.endswith(".xml"):
            return "rekordbox"
        elif lower.endswith(".nml"):
            return "traktor"
        elif lower.endswith((".csv", ".tsv")):
            return "csv"
        elif lower.endswith(".json"):
            return ImportLibrary._detect_json_source(file_path)
        else:
            return "serato"

    @staticmethod
    def _detect_json_source(file_path: str) -> str:
        import json

        try:
            with open(file_path, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return "mixcloud"

        if "sections" in data:
            return "mixcloud"
        elif "tracks" in data:
            return "soundcloud"
        return "mixcloud"
