"""Traktor NML library parser."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from src.infrastructure.parsers.parse_result import LibraryParseResult


class TraktorParser:
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
            tracks=tracks, playlists=playlists, play_history=play_history, source="traktor"
        )

    def _parse_tracks(self, root: ET.Element) -> list[dict]:
        tracks = []
        collection = root.find("COLLECTION")
        if collection is None:
            return tracks

        for entry in collection.findall("ENTRY"):
            location = entry.find("LOCATION")
            info = entry.find("INFO")
            tempo = entry.find("TEMPO")

            file_path = self._build_path(location) if location is not None else ""
            bpm = 0.0
            if tempo is not None:
                bpm = self._parse_float(tempo.get("BPM", "0"))

            track = {
                "title": entry.get("TITLE", ""),
                "artist": entry.get("ARTIST", ""),
                "genre": info.get("GENRE", "") if info is not None else "",
                "key": info.get("KEY", "") if info is not None else "",
                "bpm": round(bpm, 1),
                "duration_s": self._parse_int(
                    info.get("PLAYTIME", "0") if info is not None else "0"
                ),
                "file_path": file_path,
                "date_added": (info.get("IMPORT_DATE", "") if info is not None else ""),
            }
            tracks.append(track)

        return tracks

    def _parse_playlists(self, root: ET.Element) -> list[dict]:
        playlists_el = root.find("PLAYLISTS")
        if playlists_el is None:
            return []

        result = []
        self._walk_nodes(playlists_el, result)
        return result

    def _walk_nodes(self, node: ET.Element, result: list[dict]) -> None:
        for child in node.findall("NODE"):
            node_type = child.get("TYPE", "")
            if node_type == "PLAYLIST":
                playlist_el = child.find("PLAYLIST")
                track_keys = []
                if playlist_el is not None:
                    for entry in playlist_el.findall("ENTRY"):
                        pk = entry.find("PRIMARYKEY")
                        if pk is not None:
                            track_keys.append(pk.get("KEY", ""))
                result.append(
                    {
                        "name": child.get("NAME", ""),
                        "track_ids": track_keys,
                    }
                )
            elif node_type == "FOLDER":
                subnodes = child.find("SUBNODES")
                if subnodes is not None:
                    self._walk_nodes(subnodes, result)

    def _parse_history(self, root: ET.Element) -> list[dict]:
        history = []
        sets_el = root.find("SETS")
        if sets_el is None:
            return history

        for set_entry in sets_el.findall("ENTRY"):
            tracks_el = set_entry.find("TRACKS")
            if tracks_el is None:
                continue
            track_keys = []
            for entry in tracks_el.findall("ENTRY"):
                pk = entry.find("PRIMARYKEY")
                if pk is not None:
                    track_keys.append(pk.get("KEY", ""))

            title = set_entry.get("TITLE", "")
            for i in range(len(track_keys) - 1):
                history.append(
                    {
                        "track_a_id": track_keys[i],
                        "track_b_id": track_keys[i + 1],
                        "timestamp": title,
                    }
                )
        return history

    @staticmethod
    def _build_path(location: ET.Element) -> str:
        directory = location.get("DIR", "")
        filename = location.get("FILE", "")

        # Traktor uses /: as separator
        path = directory.replace("/:", "/") + filename
        if not path.startswith("/"):
            path = "/" + path
        return path

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
