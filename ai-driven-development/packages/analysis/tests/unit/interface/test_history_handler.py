"""Tests for mix history JSON-RPC handler."""

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from src.domain.entities.mix_transition import MixTransition
from src.domain.ports.mix_history_repository import MixHistoryRepository
from src.domain.services.history_scoring import HistoryScoring
from src.interface.handlers import register_handlers
from src.interface.server import JsonRpcServer


class FakeHistoryRepo(MixHistoryRepository):
    def __init__(self):
        self._transitions: list[MixTransition] = []

    def save_transition(self, transition: MixTransition) -> None:
        self._transitions.append(transition)

    def save_transitions(self, transitions: list[MixTransition]) -> None:
        self._transitions.extend(transitions)

    def find_transitions_from(self, track_hash: str) -> list[MixTransition]:
        return [t for t in self._transitions if t.track_a_hash == track_hash]

    def get_pair_frequency(self, track_a_hash: str, track_b_hash: str) -> int:
        return sum(
            1
            for t in self._transitions
            if t.track_a_hash == track_a_hash and t.track_b_hash == track_b_hash
        )

    def get_top_successors(self, track_hash: str, limit: int = 10) -> list[tuple[str, int]]:
        freq: dict[str, int] = {}
        for t in self._transitions:
            if t.track_a_hash == track_hash:
                freq[t.track_b_hash] = freq.get(t.track_b_hash, 0) + 1
        sorted_pairs = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        return sorted_pairs[:limit]

    def count_all(self) -> int:
        return len(self._transitions)


class TestHistoryHandlers:
    @pytest.fixture
    def setup(self):
        server = JsonRpcServer()
        analyze = MagicMock()
        batch = MagicMock()
        history_repo = FakeHistoryRepo()
        scorer = HistoryScoring(history_repo)
        register_handlers(
            server,
            analyze,
            batch,
            mix_history_repo=history_repo,
            history_scorer=scorer,
        )
        return server, history_repo

    def test_import_history_handler(self, setup):
        server, repo = setup
        handler = server._methods["import_history"]
        result = handler(
            transitions=[
                {"track_a_hash": "a", "track_b_hash": "b", "timestamp": "2025-01-20", "source": "rekordbox"},
                {"track_a_hash": "b", "track_b_hash": "c", "timestamp": "2025-01-20", "source": "rekordbox"},
            ]
        )
        assert result["imported"] == 2
        assert repo.count_all() == 2

    def test_score_history_handler(self, setup):
        server, repo = setup
        # Add some history first
        repo.save_transitions([
            MixTransition("a", "b", datetime(2025, 1, 20), "rekordbox"),
            MixTransition("a", "b", datetime(2025, 2, 15), "rekordbox"),
            MixTransition("a", "c", datetime(2025, 1, 21), "rekordbox"),
        ])
        handler = server._methods["score_history"]
        result = handler(current_track_hash="a", candidate_hashes=["b", "c", "d"])
        assert result["b"] > result["c"]
        assert result["c"] > result["d"]
