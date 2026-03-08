"""SQLite implementation of the MixHistoryRepository port."""

from __future__ import annotations

import sqlite3
from datetime import datetime

from src.domain.entities.mix_transition import MixTransition
from src.domain.ports.mix_history_repository import MixHistoryRepository

_MIX_HISTORY_DDL = """
CREATE TABLE IF NOT EXISTS mix_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    track_a_hash TEXT NOT NULL,
    track_b_hash TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    source TEXT NOT NULL
)
"""

_MIX_HISTORY_INDEX = """
CREATE INDEX IF NOT EXISTS idx_mix_history_track_a ON mix_history (track_a_hash)
"""


class SQLiteMixHistoryRepository(MixHistoryRepository):
    def __init__(self, db_path: str) -> None:
        self._conn = sqlite3.connect(db_path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(_MIX_HISTORY_DDL)
        self._conn.execute(_MIX_HISTORY_INDEX)
        self._conn.commit()

    def save_transition(self, transition: MixTransition) -> None:
        self._conn.execute(
            "INSERT INTO mix_history (track_a_hash, track_b_hash, timestamp, source)"
            " VALUES (?, ?, ?, ?)",
            (
                transition.track_a_hash,
                transition.track_b_hash,
                transition.timestamp.isoformat(),
                transition.source,
            ),
        )
        self._conn.commit()

    def save_transitions(self, transitions: list[MixTransition]) -> None:
        self._conn.executemany(
            "INSERT INTO mix_history (track_a_hash, track_b_hash, timestamp, source)"
            " VALUES (?, ?, ?, ?)",
            [
                (t.track_a_hash, t.track_b_hash, t.timestamp.isoformat(), t.source)
                for t in transitions
            ],
        )
        self._conn.commit()

    def find_transitions_from(self, track_hash: str) -> list[MixTransition]:
        rows = self._conn.execute(
            "SELECT track_a_hash, track_b_hash, timestamp, source"
            " FROM mix_history WHERE track_a_hash = ?",
            (track_hash,),
        ).fetchall()
        return [self._row_to_transition(row) for row in rows]

    def get_pair_frequency(self, track_a_hash: str, track_b_hash: str) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM mix_history" " WHERE track_a_hash = ? AND track_b_hash = ?",
            (track_a_hash, track_b_hash),
        ).fetchone()
        return row[0] if row else 0

    def get_top_successors(self, track_hash: str, limit: int = 10) -> list[tuple[str, int]]:
        rows = self._conn.execute(
            "SELECT track_b_hash, COUNT(*) as freq FROM mix_history"
            " WHERE track_a_hash = ? GROUP BY track_b_hash"
            " ORDER BY freq DESC LIMIT ?",
            (track_hash, limit),
        ).fetchall()
        return [(row[0], row[1]) for row in rows]

    def count_all(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM mix_history").fetchone()
        return row[0] if row else 0

    @staticmethod
    def _row_to_transition(row: tuple) -> MixTransition:
        return MixTransition(
            track_a_hash=row[0],
            track_b_hash=row[1],
            timestamp=datetime.fromisoformat(row[2]),
            source=row[3],
        )
