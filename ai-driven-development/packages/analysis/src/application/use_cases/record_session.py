"""RecordSession use case -- save a completed DJ session for analytics."""

from __future__ import annotations

from src.domain.entities.session_analytics import SessionAnalytics
from src.domain.ports.analytics_repository import AnalyticsRepository


class RecordSession:
    def __init__(self, analytics_repo: AnalyticsRepository) -> None:
        self._repo = analytics_repo

    def execute(self, session: SessionAnalytics) -> None:
        if not session.tracks_played:
            return
        self._repo.save_session(session)
