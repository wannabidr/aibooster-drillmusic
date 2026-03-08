"""Unit tests for MixTransition entity."""

from datetime import datetime

from src.domain.entities.mix_transition import MixTransition


class TestMixTransition:
    def test_create(self):
        t = MixTransition(
            track_a_hash="hash_a",
            track_b_hash="hash_b",
            timestamp=datetime(2025, 1, 20, 22, 0),
            source="rekordbox",
        )
        assert t.track_a_hash == "hash_a"
        assert t.track_b_hash == "hash_b"
        assert t.source == "rekordbox"

    def test_equality(self):
        ts = datetime(2025, 1, 20, 22, 0)
        t1 = MixTransition("a", "b", ts, "rekordbox")
        t2 = MixTransition("a", "b", ts, "traktor")
        assert t1 == t2

    def test_inequality(self):
        ts = datetime(2025, 1, 20, 22, 0)
        t1 = MixTransition("a", "b", ts, "rekordbox")
        t2 = MixTransition("a", "c", ts, "rekordbox")
        assert t1 != t2

    def test_frozen(self):
        t = MixTransition("a", "b", datetime.now(), "rekordbox")
        try:
            t.track_a_hash = "x"  # type: ignore[misc]
            assert False, "Should be frozen"
        except AttributeError:
            pass

    def test_hashable(self):
        ts = datetime(2025, 1, 20)
        t1 = MixTransition("a", "b", ts, "rekordbox")
        t2 = MixTransition("a", "b", ts, "traktor")
        assert hash(t1) == hash(t2)
        assert len({t1, t2}) == 1
