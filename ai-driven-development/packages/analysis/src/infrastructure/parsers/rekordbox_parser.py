"""Rekordbox XML library parser."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import unquote, urlparse

from src.infrastructure.parsers.parse_result import LibraryParseResult


class RekordboxParser:
    def parse(self, file_path: str) -> LibraryParseResult:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        tree = ET.parse(file_path)
        root = tree.getroot()

        tracks = self._parse_tracks(root)
        playlists = self._parse_playlists(root)
        play_history = self._parse_history(root)

        return LibraryParseResult(
            tracks=tracks, playlists=playlists, play_history=play_history, source="rekordbox"
        )

    def _parse_tracks(self, root: ET.Element) -> list[dict]:
        tracks = []
        collection = root.find("COLLECTION")
        if collection is None:
            return tracks

        for track_el in collection.findall("TRACK"):
            track = {
                "id": track_el.get("TrackID", ""),
                "title": track_el.get("Name", ""),
                "artist": track_el.get("Artist", ""),
                "album": track_el.get("Album", ""),
                "genre": track_el.get("Genre", ""),
                "bpm": self._parse_float(track_el.get("AverageBpm", "0")),
                "key": track_el.get("Tonality", ""),
                "rating": self._parse_int(track_el.get("Rating", "0")),
                "duration_s": self._parse_int(track_el.get("TotalTime", "0")),
                "file_path": self._parse_location(track_el.get("Location", "")),
                "date_added": track_el.get("DateAdded", ""),
            }
            tracks.append(track)

        return tracks

    def _parse_playlists(self, root: ET.Element) -> list[dict]:
        playlists_el = root.find("PLAYLISTS")
        if playlists_el is None:
            return []

        result = []
        self._walk_playlist_nodes(playlists_el, result)
        return result

    def _walk_playlist_nodes(self, node: ET.Element, result: list[dict]) -> None:
        for child in node.findall("NODE"):
            node_type = child.get("Type", "")
            if node_type == "1":  # Playlist
                track_ids = [t.get("Key", "") for t in child.findall("TRACK")]
                result.append(
                    {
                        "name": child.get("Name", ""),
                        "track_ids": track_ids,
                    }
                )
            elif node_type == "0":  # Folder
                self._walk_playlist_nodes(child, result)

    def _parse_history(self, root: ET.Element) -> list[dict]:
        history = []
        histories_el = root.find("HISTORIES")
        if histories_el is None:
            return history

        for session in histories_el.findall("HISTORY"):
            date = session.get("Date", "")
            track_keys = [t.get("Key", "") for t in session.findall("TRACK")]
            for i in range(len(track_keys) - 1):
                history.append(
                    {
                        "track_a_id": track_keys[i],
                        "track_b_id": track_keys[i + 1],
                        "timestamp": date,
                    }
                )
        return history

    @staticmethod
    def _parse_location(location: str) -> str:
        if not location:
            return ""
        parsed = urlparse(location)
        return unquote(parsed.path)

    @staticmethod
    def _parse_float(value: str) -> float:
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0

    @staticmethod
    def _parse_int(value: str) -> int:
        try:
            return int(value)
        except (ValueError, TypeError):
            return 0
