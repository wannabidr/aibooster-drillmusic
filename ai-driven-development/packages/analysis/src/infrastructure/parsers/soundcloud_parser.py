"""SoundCloud tracklist JSON parser."""

from __future__ import annotations

import json
from pathlib import Path

from src.infrastructure.parsers.parse_result import LibraryParseResult


class SoundCloudParser:
    def parse(self, file_path: str) -> LibraryParseResult:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}") from e

        mix_title = data.get("title", "")
        raw_tracks = data.get("tracks", [])

        tracks = []
        track_titles = []

        for t in raw_tracks:
            user = t.get("user", {})
            title = t.get("title", "")
            duration_ms = t.get("duration", 0)

            track = {
                "title": title,
                "artist": user.get("full_name", "") if isinstance(user, dict) else "",
                "genre": t.get("genre", ""),
                "bpm": self._parse_float(t.get("bpm", 0)),
                "key": t.get("key_signature", ""),
                "duration_s": duration_ms // 1000 if duration_ms else 0,
            }
            tracks.append(track)
            track_titles.append(title)

        play_history = []
        for i in range(len(track_titles) - 1):
            play_history.append({
                "track_a_id": track_titles[i],
                "track_b_id": track_titles[i + 1],
                "timestamp": data.get("created_at", ""),
            })

        playlists = []
        if tracks:
            playlists.append({
                "name": mix_title,
                "track_ids": track_titles,
            })

        return LibraryParseResult(
            tracks=tracks,
            playlists=playlists,
            play_history=play_history,
            source="soundcloud",
        )

    @staticmethod
    def _parse_float(value: object) -> float:
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0
