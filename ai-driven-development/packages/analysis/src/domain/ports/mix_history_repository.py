"""MixHistoryRepository port (abstract interface)."""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.domain.entities.mix_transition import MixTransition


class MixHistoryRepository(ABC):
    @abstractmethod
    def save_transition(self, transition: MixTransition) -> None: ...

    @abstractmethod
    def save_transitions(self, transitions: list[MixTransition]) -> None: ...

    @abstractmethod
    def find_transitions_from(self, track_hash: str) -> list[MixTransition]: ...

    @abstractmethod
    def get_pair_frequency(self, track_a_hash: str, track_b_hash: str) -> int: ...

    @abstractmethod
    def get_top_successors(self, track_hash: str, limit: int = 10) -> list[tuple[str, int]]: ...

    @abstractmethod
    def count_all(self) -> int: ...
