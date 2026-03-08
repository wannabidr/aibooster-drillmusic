"""History-based scoring service for mix recommendations."""

from __future__ import annotations

from src.domain.ports.mix_history_repository import MixHistoryRepository


class HistoryScoring:
    """Scores candidate tracks based on personal mix history.

    If the DJ has previously played A->B, B gets a boosted score when A is current.
    Score is based on frequency of the pair relative to the most frequent successor.
    """

    def __init__(self, history_repo: MixHistoryRepository) -> None:
        self._repo = history_repo

    def score(self, current_track_hash: str, candidate_hash: str) -> float:
        """Return a score between 0.0 and 1.0 for the candidate given the current track.

        Returns 0.5 (neutral) if no history exists for the current track.
        """
        top = self._repo.get_top_successors(current_track_hash, limit=50)
        if not top:
            return 0.5

        max_freq = top[0][1]
        for successor_hash, freq in top:
            if successor_hash == candidate_hash:
                return min(0.5 + 0.5 * (freq / max_freq), 1.0)

        return 0.3

    def score_batch(self, current_track_hash: str, candidate_hashes: list[str]) -> dict[str, float]:
        """Score multiple candidates at once (more efficient)."""
        top = self._repo.get_top_successors(current_track_hash, limit=50)
        if not top:
            return {h: 0.5 for h in candidate_hashes}

        max_freq = top[0][1]
        freq_map = dict(top)

        result = {}
        for h in candidate_hashes:
            if h in freq_map:
                result[h] = min(0.5 + 0.5 * (freq_map[h] / max_freq), 1.0)
            else:
                result[h] = 0.3
        return result
