"""SQLite implementation of the TrackRepository port."""

from __future__ import annotations

import json
import sqlite3
import uuid

from src.domain.entities.analysis_result import AnalysisResult
from src.domain.entities.audio_track import AudioTrack
from src.domain.ports.track_repository import TrackRepository
from src.domain.value_objects.bpm_value import BPMValue
from src.domain.value_objects.energy_profile import EnergyProfile
from src.domain.value_objects.key_signature import KeySignature

_TRACK_COLS = (
    "id, file_path, file_hash, title, artist," " duration_ms, analysis_status, failure_reason"
)

_TRACKS_DDL = """
CREATE TABLE IF NOT EXISTS tracks (
    id TEXT PRIMARY KEY,
    file_path TEXT NOT NULL,
    file_hash TEXT NOT NULL UNIQUE,
    title TEXT,
    artist TEXT,
    duration_ms INTEGER,
    analysis_status TEXT NOT NULL DEFAULT 'pending',
    failure_reason TEXT
)
"""

_ANALYSIS_DDL = """
CREATE TABLE IF NOT EXISTS analysis_results (
    id TEXT PRIMARY KEY,
    track_id TEXT NOT NULL UNIQUE,
    bpm REAL NOT NULL,
    key_root TEXT NOT NULL,
    key_mode TEXT NOT NULL,
    energy_overall REAL NOT NULL,
    energy_segments TEXT,
    energy_trajectory TEXT,
    fingerprint TEXT,
    genre_embedding TEXT,
    analyzed_at TEXT,
    FOREIGN KEY (track_id) REFERENCES tracks(id) ON DELETE CASCADE
)
"""


class SQLiteTrackRepository(TrackRepository):
    def __init__(self, db_path: str) -> None:
        self._conn = sqlite3.connect(db_path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute(_TRACKS_DDL)
        self._conn.execute(_ANALYSIS_DDL)
        self._conn.commit()

    def save(self, track: AudioTrack) -> None:
        self._conn.execute(
            f"INSERT OR REPLACE INTO tracks ({_TRACK_COLS})" " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                str(track.id),
                track.file_path,
                track.file_hash,
                track.title,
                track.artist,
                track.duration_ms,
                track.analysis_status,
                track.failure_reason,
            ),
        )
        self._conn.commit()

    def find_by_id(self, track_id: uuid.UUID) -> AudioTrack | None:
        row = self._conn.execute(
            f"SELECT {_TRACK_COLS} FROM tracks WHERE id = ?",
            (str(track_id),),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_track(row)

    def find_by_hash(self, file_hash: str) -> AudioTrack | None:
        row = self._conn.execute(
            f"SELECT {_TRACK_COLS} FROM tracks WHERE file_hash = ?",
            (file_hash,),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_track(row)

    def find_all(self) -> list[AudioTrack]:
        rows = self._conn.execute(f"SELECT {_TRACK_COLS} FROM tracks").fetchall()
        return [self._row_to_track(row) for row in rows]

    def delete(self, track_id: uuid.UUID) -> None:
        self._conn.execute("DELETE FROM tracks WHERE id = ?", (str(track_id),))
        self._conn.commit()

    def save_analysis(self, result: AnalysisResult) -> None:
        segments_json = json.dumps(result.energy.segments) if result.energy.segments else None
        embedding_json = json.dumps(result.genre_embedding) if result.genre_embedding else None
        self._conn.execute(
            """INSERT OR REPLACE INTO analysis_results
               (id, track_id, bpm, key_root, key_mode, energy_overall, energy_segments,
                energy_trajectory, fingerprint, genre_embedding, analyzed_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                str(result.id),
                str(result.track_id),
                result.bpm.value,
                result.key.root,
                result.key.mode,
                result.energy.overall,
                segments_json,
                result.energy.trajectory,
                result.fingerprint,
                embedding_json,
                result.analyzed_at.isoformat() if result.analyzed_at else None,
            ),
        )
        self._conn.commit()

    def find_analysis_by_track_id(self, track_id: uuid.UUID) -> AnalysisResult | None:
        row = self._conn.execute(
            """SELECT id, track_id, bpm, key_root, key_mode, energy_overall,
                      energy_segments, energy_trajectory, fingerprint, genre_embedding, analyzed_at
               FROM analysis_results WHERE track_id = ?""",
            (str(track_id),),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_result(row)

    @staticmethod
    def _row_to_track(row: tuple) -> AudioTrack:
        return AudioTrack(
            id=uuid.UUID(row[0]),
            file_path=row[1],
            file_hash=row[2],
            title=row[3],
            artist=row[4],
            duration_ms=row[5],
            analysis_status=row[6],
            failure_reason=row[7],
        )

    @staticmethod
    def _row_to_result(row: tuple) -> AnalysisResult:
        segments = json.loads(row[6]) if row[6] else []
        embedding = json.loads(row[9]) if row[9] else None
        key_notation = f"{row[3]}m" if row[4] == "minor" else row[3]
        return AnalysisResult(
            id=uuid.UUID(row[0]),
            track_id=uuid.UUID(row[1]),
            bpm=BPMValue(row[2]),
            key=KeySignature(key_notation),
            energy=EnergyProfile(overall=row[5], segments=segments, trajectory=row[7]),
            fingerprint=row[8],
            genre_embedding=embedding,
        )
