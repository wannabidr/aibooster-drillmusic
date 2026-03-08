"""Mixcloud tracklist JSON parser."""

from __future__ import annotations

import json
from pathlib import Path

from src.infrastructure.parsers.parse_result import LibraryParseResult


class MixcloudParser:
    def parse(self, file_path: str) -> LibraryParseResult:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}") from e

        sections = data.get("sections", [])
        mix_name = data.get("name", "")

        tracks = []
        track_titles = []

        for section in sections:
            if section.get("section_type") != "track":
                continue
            track_data = section.get("track", {})
            artist_data = track_data.get("artist", {})

            title = track_data.get("name", "")
            track = {
                "title": title,
                "artist": artist_data.get("name", "") if isinstance(artist_data, dict) else "",
                "start_time_s": section.get("start_time", 0),
            }
            tracks.append(track)
            track_titles.append(title)

        play_history = []
        for i in range(len(track_titles) - 1):
            play_history.append({
                "track_a_id": track_titles[i],
                "track_b_id": track_titles[i + 1],
                "timestamp": data.get("created_time", ""),
            })

        playlists = []
        if tracks:
            playlists.append({
                "name": mix_name,
                "track_ids": track_titles,
            })

        return LibraryParseResult(
            tracks=tracks,
            playlists=playlists,
            play_history=play_history,
            source="mixcloud",
        )
