"""GenerateAnalytics use case -- build dashboard data from session history."""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime, timedelta

from src.application.dto.analytics_dashboard_dto import (
    AnalyticsDashboardDTO,
    EnergyReportDTO,
    GenreDistributionDTO,
    MixingPatternsDTO,
    SessionTimelineItemDTO,
)
from src.domain.ports.analytics_repository import AnalyticsRepository


class GenerateAnalytics:
    def __init__(self, analytics_repo: AnalyticsRepository) -> None:
        self._repo = analytics_repo

    def execute(self, days: int = 30) -> AnalyticsDashboardDTO:
        now = datetime.now(UTC)
        from_date = now - timedelta(days=days)
        sessions = self._repo.get_sessions(from_date, now)

        energy_reports = []
        genre_counter: Counter[str] = Counter()
        all_bpms: list[float] = []
        all_energies: list[float] = []
        key_counter: Counter[str] = Counter()
        transition_counter: Counter[tuple[str, str]] = Counter()
        total_tracks = 0
        quality_sum = 0.0
        timeline: list[SessionTimelineItemDTO] = []

        for session in sessions:
            # Energy reports
            curve = session.energy_curve
            if curve:
                energy_reports.append(
                    EnergyReportDTO(
                        session_id=str(session.session_id),
                        timestamp=session.timestamp.isoformat(),
                        energy_curve=curve,
                        peak_energy=max(curve),
                        valley_energy=min(curve),
                        avg_energy=sum(curve) / len(curve),
                    )
                )

            # Genre distribution
            for genre, count in session.genre_distribution.items():
                genre_counter[genre] += count

            # Track-level data
            for event in session.tracks_played:
                all_bpms.append(event.bpm)
                all_energies.append(event.energy)
                key_counter[event.key] += 1

            # Key transitions
            for a_key, b_key in session.key_transitions:
                transition_counter[(a_key, b_key)] += 1

            total_tracks += session.track_count
            quality_sum += session.avg_transition_quality

            # Timeline
            top_genre = max(session.genre_distribution, key=session.genre_distribution.get) if session.genre_distribution else "unknown"
            avg_e = sum(curve) / len(curve) if curve else 0.0
            timeline.append(
                SessionTimelineItemDTO(
                    session_id=str(session.session_id),
                    timestamp=session.timestamp.isoformat(),
                    track_count=session.track_count,
                    duration_minutes=session.duration_minutes,
                    avg_energy=round(avg_e, 3),
                    top_genre=top_genre,
                )
            )

        # Build mixing patterns
        avg_bpm = sum(all_bpms) / len(all_bpms) if all_bpms else 0.0
        bpm_range = (min(all_bpms), max(all_bpms)) if all_bpms else (0.0, 0.0)
        avg_quality = quality_sum / len(sessions) if sessions else 0.0

        most_common_transitions = [
            (a, b, c) for (a, b), c in transition_counter.most_common(10)
        ]
        key_prefs = key_counter.most_common(10)

        return AnalyticsDashboardDTO(
            energy_reports=energy_reports,
            genre_distribution=GenreDistributionDTO(
                genres=dict(genre_counter.most_common()),
                total_tracks=total_tracks,
            ),
            mixing_patterns=MixingPatternsDTO(
                most_common_transitions=most_common_transitions,
                preferred_bpm_range=bpm_range,
                avg_bpm=round(avg_bpm, 1),
                key_preferences=list(key_prefs),
                avg_transition_quality=round(avg_quality, 3),
            ),
            session_timeline=timeline,
            total_sessions=len(sessions),
            total_tracks_played=total_tracks,
        )
