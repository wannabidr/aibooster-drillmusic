"""Integration tests for SQLite MixHistoryRepository."""

from datetime import datetime

import pytest

from src.domain.entities.mix_transition import MixTransition
from src.infrastructure.persistence.sqlite_mix_history_repository import (
    SQLiteMixHistoryRepository,
)


@pytest.fixture
def repo(tmp_path):
    db_path = str(tmp_path / "test_history.db")
    return SQLiteMixHistoryRepository(db_path)


class TestSQLiteMixHistoryRepository:
    def test_save_and_count(self, repo):
        t = MixTransition("hash_a", "hash_b", datetime(2025, 1, 20), "rekordbox")
        repo.save_transition(t)
        assert repo.count_all() == 1

    def test_save_multiple(self, repo):
        transitions = [
            MixTransition("a", "b", datetime(2025, 1, 20), "rekordbox"),
            MixTransition("b", "c", datetime(2025, 1, 20), "rekordbox"),
            MixTransition("a", "b", datetime(2025, 2, 15), "rekordbox"),
        ]
        repo.save_transitions(transitions)
        assert repo.count_all() == 3

    def test_find_transitions_from(self, repo):
        transitions = [
            MixTransition("a", "b", datetime(2025, 1, 20), "rekordbox"),
            MixTransition("a", "c", datetime(2025, 1, 21), "rekordbox"),
            MixTransition("b", "c", datetime(2025, 1, 22), "rekordbox"),
        ]
        repo.save_transitions(transitions)
        from_a = repo.find_transitions_from("a")
        assert len(from_a) == 2

    def test_get_pair_frequency(self, repo):
        transitions = [
            MixTransition("a", "b", datetime(2025, 1, 20), "rekordbox"),
            MixTransition("a", "b", datetime(2025, 2, 15), "rekordbox"),
            MixTransition("a", "c", datetime(2025, 3, 1), "rekordbox"),
        ]
        repo.save_transitions(transitions)
        assert repo.get_pair_frequency("a", "b") == 2
        assert repo.get_pair_frequency("a", "c") == 1
        assert repo.get_pair_frequency("a", "x") == 0

    def test_get_top_successors(self, repo):
        transitions = [
            MixTransition("a", "b", datetime(2025, 1, 20), "rekordbox"),
            MixTransition("a", "b", datetime(2025, 2, 15), "rekordbox"),
            MixTransition("a", "b", datetime(2025, 3, 1), "rekordbox"),
            MixTransition("a", "c", datetime(2025, 1, 21), "rekordbox"),
        ]
        repo.save_transitions(transitions)
        top = repo.get_top_successors("a", limit=10)
        assert len(top) == 2
        assert top[0] == ("b", 3)
        assert top[1] == ("c", 1)

    def test_top_successors_respects_limit(self, repo):
        transitions = [
            MixTransition("a", "b", datetime(2025, 1, 20), "rekordbox"),
            MixTransition("a", "c", datetime(2025, 1, 21), "rekordbox"),
            MixTransition("a", "d", datetime(2025, 1, 22), "rekordbox"),
        ]
        repo.save_transitions(transitions)
        top = repo.get_top_successors("a", limit=2)
        assert len(top) == 2

    def test_empty_results(self, repo):
        assert repo.find_transitions_from("nonexistent") == []
        assert repo.get_pair_frequency("a", "b") == 0
        assert repo.get_top_successors("a") == []
        assert repo.count_all() == 0

    def test_duplicate_transitions_stored_separately(self, repo):
        """Same pair at different times should both be stored."""
        t1 = MixTransition("a", "b", datetime(2025, 1, 20), "rekordbox")
        t2 = MixTransition("a", "b", datetime(2025, 2, 15), "rekordbox")
        repo.save_transition(t1)
        repo.save_transition(t2)
        assert repo.count_all() == 2
        assert repo.get_pair_frequency("a", "b") == 2
