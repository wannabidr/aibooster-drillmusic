"""Analytics repository port (abstract interface)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

from src.domain.entities.session_analytics import AggregateStats, SessionAnalytics


class AnalyticsRepository(ABC):
    @abstractmethod
    def save_session(self, session: SessionAnalytics) -> None: ...

    @abstractmethod
    def get_sessions(
        self, from_date: datetime, to_date: datetime
    ) -> list[SessionAnalytics]: ...

    @abstractmethod
    def get_aggregate_stats(self, days: int) -> AggregateStats: ...

    @abstractmethod
    def delete_session(self, session_id: str) -> None: ...

    @abstractmethod
    def count_sessions(self) -> int: ...
