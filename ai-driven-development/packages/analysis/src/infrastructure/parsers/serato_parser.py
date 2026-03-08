"""Serato library parser (crate files)."""

from __future__ import annotations

import struct
from pathlib import Path

from src.infrastructure.parsers.parse_result import LibraryParseResult


class SeratoParser:
    def parse(self, crate_dir: str) -> LibraryParseResult:
        path = Path(crate_dir)
        if not path.exists():
            raise FileNotFoundError(f"Directory not found: {crate_dir}")

        tracks: list[dict] = []
        playlists: list[dict] = []

        # Serato stores crates in _Serato_/Subcrates/
        subcrates = path / "_Serato_" / "Subcrates"
        if subcrates.exists():
            for crate_file in sorted(subcrates.glob("*.crate")):
                crate_tracks = self._parse_crate(crate_file)
                playlists.append(
                    {
                        "name": crate_file.stem,
                        "track_ids": crate_tracks,
                    }
                )

        # Parse main database for track metadata
        database = path / "_Serato_" / "database V2"
        if database.exists():
            tracks = self._parse_database(database)

        return LibraryParseResult(tracks=tracks, playlists=playlists, source="serato")

    def _parse_crate(self, crate_path: Path) -> list[str]:
        track_paths = []
        try:
            data = crate_path.read_bytes()
            offset = 0
            while offset < len(data):
                if offset + 8 > len(data):
                    break
                tag = data[offset : offset + 4].decode("ascii", errors="ignore")
                length = struct.unpack(">I", data[offset + 4 : offset + 8])[0]
                offset += 8
                if tag == "otrk":
                    # Track entry contains ptrk sub-tag
                    end = offset + length
                    while offset < end:
                        if offset + 8 > end:
                            break
                        sub_tag = data[offset : offset + 4].decode("ascii", errors="ignore")
                        sub_len = struct.unpack(">I", data[offset + 4 : offset + 8])[0]
                        offset += 8
                        if sub_tag == "ptrk":
                            path_bytes = data[offset : offset + sub_len]
                            file_path = path_bytes.decode("utf-16-be", errors="ignore")
                            track_paths.append(file_path)
                        offset += sub_len
                else:
                    offset += length
        except Exception:
            pass
        return track_paths

    def _parse_database(self, db_path: Path) -> list[dict]:
        # Serato database V2 is a binary format
        # For now, return empty -- full binary parsing is complex
        # and will be refined in later iterations
        return []
