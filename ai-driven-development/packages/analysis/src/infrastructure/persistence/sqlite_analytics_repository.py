"""SQLite implementation of the AnalyticsRepository port."""

from __future__ import annotations

import json
import sqlite3
import uuid
from collections import Counter
from datetime import UTC, datetime, timedelta

from src.domain.entities.session_analytics import (
    AggregateStats,
    SessionAnalytics,
    TrackPlayEvent,
)
from src.domain.ports.analytics_repository import AnalyticsRepository

_SESSIONS_DDL = """
CREATE TABLE IF NOT EXISTS analytics_sessions (
    session_id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    tracks_played TEXT NOT NULL,
    energy_curve TEXT NOT NULL,
    genre_distribution TEXT NOT NULL,
    bpm_min REAL NOT NULL,
    bpm_max REAL NOT NULL,
    key_transitions TEXT NOT NULL,
    avg_transition_quality REAL NOT NULL
)
"""

_SESSIONS_INDEX_DDL = """
CREATE INDEX IF NOT EXISTS idx_sessions_timestamp
ON analytics_sessions (timestamp)
"""


class SQLiteAnalyticsRepository(AnalyticsRepository):
    def __init__(self, db_path: str) -> None:
        self._conn = sqlite3.connect(db_path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute(_SESSIONS_DDL)
        self._conn.execute(_SESSIONS_INDEX_DDL)
        self._conn.commit()

    def save_session(self, session: SessionAnalytics) -> None:
        tracks_json = json.dumps(
            [
                {
                    "track_id": str(e.track_id),
                    "played_at": e.played_at.isoformat(),
                    "energy": e.energy,
                    "genre": e.genre,
                    "bpm": e.bpm,
                    "key": e.key,
                }
                for e in session.tracks_played
            ]
        )
        energy_json = json.dumps(session.energy_curve)
        genre_json = json.dumps(session.genre_distribution)
        key_trans_json = json.dumps(session.key_transitions)

        self._conn.execute(
            """INSERT OR REPLACE INTO analytics_sessions
               (session_id, timestamp, tracks_played, energy_curve,
                genre_distribution, bpm_min, bpm_max, key_transitions,
                avg_transition_quality)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                str(session.session_id),
                session.timestamp.isoformat(),
                tracks_json,
                energy_json,
                genre_json,
                session.bpm_range[0],
                session.bpm_range[1],
                key_trans_json,
                session.avg_transition_quality,
            ),
        )
        self._conn.commit()

    def get_sessions(
        self, from_date: datetime, to_date: datetime
    ) -> list[SessionAnalytics]:
        rows = self._conn.execute(
            """SELECT session_id, timestamp, tracks_played, energy_curve,
                      genre_distribution, bpm_min, bpm_max, key_transitions,
                      avg_transition_quality
               FROM analytics_sessions
               WHERE timestamp >= ? AND timestamp <= ?
               ORDER BY timestamp DESC""",
            (from_date.isoformat(), to_date.isoformat()),
        ).fetchall()
        return [self._row_to_session(row) for row in rows]

    def get_aggregate_stats(self, days: int) -> AggregateStats:
        now = datetime.now(UTC)
        from_date = now - timedelta(days=days)
        sessions = self.get_sessions(from_date, now)

        if not sessions:
            return AggregateStats(
                total_sessions=0,
                total_tracks_played=0,
                avg_session_length=0.0,
                top_genres=[],
                avg_bpm=0.0,
                avg_energy=0.0,
                avg_transition_quality=0.0,
                most_common_keys=[],
            )

        total_tracks = sum(s.track_count for s in sessions)
        avg_length = sum(s.duration_minutes for s in sessions) / len(sessions)

        genre_counter: Counter[str] = Counter()
        key_counter: Counter[str] = Counter()
        all_bpms: list[float] = []
        all_energies: list[float] = []
        quality_sum = 0.0

        for session in sessions:
            for genre, count in session.genre_distribution.items():
                genre_counter[genre] += count
            for event in session.tracks_played:
                all_bpms.append(event.bpm)
                all_energies.append(event.energy)
                key_counter[event.key] += 1
            quality_sum += session.avg_transition_quality

        return AggregateStats(
            total_sessions=len(sessions),
            total_tracks_played=total_tracks,
            avg_session_length=round(avg_length, 1),
            top_genres=genre_counter.most_common(10),
            avg_bpm=round(sum(all_bpms) / len(all_bpms), 1) if all_bpms else 0.0,
            avg_energy=round(sum(all_energies) / len(all_energies), 3) if all_energies else 0.0,
            avg_transition_quality=round(quality_sum / len(sessions), 3),
            most_common_keys=key_counter.most_common(10),
        )

    def delete_session(self, session_id: str) -> None:
        self._conn.execute(
            "DELETE FROM analytics_sessions WHERE session_id = ?",
            (session_id,),
        )
        self._conn.commit()

    def count_sessions(self) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM analytics_sessions"
        ).fetchone()
        return row[0] if row else 0

    @staticmethod
    def _row_to_session(row: tuple) -> SessionAnalytics:
        tracks_data = json.loads(row[2])
        tracks = [
            TrackPlayEvent(
                track_id=uuid.UUID(t["track_id"]),
                played_at=datetime.fromisoformat(t["played_at"]),
                energy=t["energy"],
                genre=t["genre"],
                bpm=t["bpm"],
                key=t["key"],
            )
            for t in tracks_data
        ]

        energy_curve = json.loads(row[3])
        genre_distribution = json.loads(row[4])
        key_transitions = [tuple(kt) for kt in json.loads(row[7])]

        return SessionAnalytics(
            session_id=uuid.UUID(row[0]),
            timestamp=datetime.fromisoformat(row[1]),
            tracks_played=tracks,
            energy_curve=energy_curve,
            genre_distribution=genre_distribution,
            bpm_range=(row[5], row[6]),
            key_transitions=key_transitions,
            avg_transition_quality=row[8],
        )
