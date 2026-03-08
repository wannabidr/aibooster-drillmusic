"""Integration test: history signal demonstrably affects recommendation ranking."""

from datetime import datetime

import pytest

from src.domain.entities.mix_transition import MixTransition
from src.domain.services.history_scoring import HistoryScoring
from src.infrastructure.persistence.sqlite_mix_history_repository import (
    SQLiteMixHistoryRepository,
)


@pytest.fixture
def repo(tmp_path):
    db_path = str(tmp_path / "test_reco.db")
    return SQLiteMixHistoryRepository(db_path)


class TestHistoryImprovesRecommendations:
    def test_history_changes_ranking_order(self, repo):
        """When DJ has played A->B multiple times, B should rank higher than C."""
        transitions = [
            MixTransition("current", "track_b", datetime(2025, 1, 1), "rekordbox"),
            MixTransition("current", "track_b", datetime(2025, 1, 8), "rekordbox"),
            MixTransition("current", "track_b", datetime(2025, 1, 15), "rekordbox"),
            MixTransition("current", "track_c", datetime(2025, 1, 2), "rekordbox"),
        ]
        repo.save_transitions(transitions)

        scorer = HistoryScoring(repo)
        scores = scorer.score_batch("current", ["track_b", "track_c", "track_d"])

        # track_b played 3 times after current -> highest
        # track_c played 1 time -> medium
        # track_d never played -> lowest (penalty)
        assert scores["track_b"] > scores["track_c"] > scores["track_d"]

    def test_no_history_gives_neutral_scores(self, repo):
        scorer = HistoryScoring(repo)
        scores = scorer.score_batch("any_track", ["a", "b", "c"])
        assert all(s == 0.5 for s in scores.values())

    def test_history_score_range(self, repo):
        """All history scores should be in [0, 1]."""
        transitions = [
            MixTransition("x", "y", datetime(2025, i, 1), "rekordbox") for i in range(1, 13)
        ]
        repo.save_transitions(transitions)
        scorer = HistoryScoring(repo)
        score = scorer.score("x", "y")
        assert 0.0 <= score <= 1.0

    def test_performance_under_load(self, repo):
        """Scoring should complete quickly even with many transitions."""
        import time

        transitions = [
            MixTransition("current", f"track_{i}", datetime(2025, 1, 1), "test")
            for i in range(500)
        ]
        repo.save_transitions(transitions)

        candidates = [f"track_{i}" for i in range(100)]
        scorer = HistoryScoring(repo)

        start = time.monotonic()
        scores = scorer.score_batch("current", candidates)
        elapsed_ms = (time.monotonic() - start) * 1000

        assert len(scores) == 100
        assert elapsed_ms < 200, f"Scoring took {elapsed_ms:.1f}ms, should be < 200ms"
