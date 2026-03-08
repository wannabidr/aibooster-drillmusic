"""Unit tests for HistoryScoring service."""

from __future__ import annotations

from src.domain.ports.mix_history_repository import MixHistoryRepository
from src.domain.services.history_scoring import HistoryScoring
from src.domain.entities.mix_transition import MixTransition


class FakeMixHistoryRepo(MixHistoryRepository):
    def __init__(self, successors: list[tuple[str, int]] | None = None):
        self._successors = successors or []

    def save_transition(self, transition: MixTransition) -> None:
        pass

    def save_transitions(self, transitions: list[MixTransition]) -> None:
        pass

    def find_transitions_from(self, track_hash: str) -> list[MixTransition]:
        return []

    def get_pair_frequency(self, track_a_hash: str, track_b_hash: str) -> int:
        return 0

    def get_top_successors(self, track_hash: str, limit: int = 10) -> list[tuple[str, int]]:
        return self._successors[:limit]

    def count_all(self) -> int:
        return 0


class TestHistoryScoring:
    def test_neutral_when_no_history(self):
        repo = FakeMixHistoryRepo(successors=[])
        scorer = HistoryScoring(repo)
        assert scorer.score("track_a", "track_b") == 0.5

    def test_boost_for_known_pair(self):
        repo = FakeMixHistoryRepo(successors=[("track_b", 5), ("track_c", 2)])
        scorer = HistoryScoring(repo)
        score = scorer.score("track_a", "track_b")
        # track_b has max freq (5), so score = 0.5 + 0.5 * (5/5) = 1.0
        assert score == 1.0

    def test_partial_boost_for_less_frequent(self):
        repo = FakeMixHistoryRepo(successors=[("track_b", 10), ("track_c", 4)])
        scorer = HistoryScoring(repo)
        score = scorer.score("track_a", "track_c")
        # track_c: freq=4, max=10 -> 0.5 + 0.5*(4/10) = 0.7
        assert abs(score - 0.7) < 0.001

    def test_penalty_for_unknown_candidate(self):
        repo = FakeMixHistoryRepo(successors=[("track_b", 5)])
        scorer = HistoryScoring(repo)
        score = scorer.score("track_a", "track_x")
        # History exists but track_x not in it -> 0.3
        assert score == 0.3

    def test_score_batch(self):
        repo = FakeMixHistoryRepo(successors=[("b", 10), ("c", 5)])
        scorer = HistoryScoring(repo)
        scores = scorer.score_batch("a", ["b", "c", "d"])
        assert scores["b"] == 1.0
        assert abs(scores["c"] - 0.75) < 0.001
        assert scores["d"] == 0.3

    def test_score_batch_no_history(self):
        repo = FakeMixHistoryRepo(successors=[])
        scorer = HistoryScoring(repo)
        scores = scorer.score_batch("a", ["b", "c"])
        assert scores["b"] == 0.5
        assert scores["c"] == 0.5

    def test_history_affects_ranking(self):
        """Integration-style: history signal changes which track ranks higher."""
        repo = FakeMixHistoryRepo(successors=[("track_b", 8), ("track_c", 2)])
        scorer = HistoryScoring(repo)

        score_b = scorer.score("current", "track_b")
        score_c = scorer.score("current", "track_c")
        score_x = scorer.score("current", "track_x")

        # track_b (most played after current) > track_c > track_x (never played)
        assert score_b > score_c > score_x
