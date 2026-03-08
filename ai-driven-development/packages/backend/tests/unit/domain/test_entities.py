"""Tests for domain entities."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest
from src.domain.entities.anonymous_transition import AnonymousTransition
from src.domain.entities.community_score import CommunityScore
from src.domain.entities.session import Session
from src.domain.entities.user import User
from src.domain.value_objects.email import Email
from src.domain.value_objects.subscription_tier import SubscriptionTier


class TestUser:
    def test_mark_deleted(self, sample_user: User) -> None:
        now = datetime.now(UTC)
        deleted = sample_user.mark_deleted(now)
        assert deleted.is_deleted
        assert deleted.deleted_at == now
        assert not sample_user.is_deleted  # original unchanged

    def test_mark_contributed(self, sample_user: User) -> None:
        contributed = sample_user.mark_contributed()
        assert contributed.has_contributed
        assert not sample_user.has_contributed

    def test_upgrade_tier(self, sample_user: User) -> None:
        pro = sample_user.upgrade_tier(SubscriptionTier.PRO)
        assert pro.subscription_tier == SubscriptionTier.PRO
        assert sample_user.subscription_tier == SubscriptionTier.FREE

    def test_is_not_deleted_by_default(self, sample_user: User) -> None:
        assert not sample_user.is_deleted

    def test_user_is_frozen(self, sample_user: User) -> None:
        with pytest.raises(AttributeError):
            sample_user.email = Email("other@example.com")  # type: ignore[misc]


class TestSession:
    def test_is_not_expired(self, sample_session: Session) -> None:
        assert not sample_session.is_expired(datetime.now(UTC))

    def test_is_expired(self, expired_session: Session) -> None:
        assert expired_session.is_expired(datetime.now(UTC))

    def test_revoke(self, sample_session: Session) -> None:
        revoked = sample_session.revoke()
        assert revoked.revoked
        assert not sample_session.revoked

    def test_session_is_frozen(self, sample_session: Session) -> None:
        with pytest.raises(AttributeError):
            sample_session.revoked = True  # type: ignore[misc]


class TestAnonymousTransition:
    def test_creation(self, sample_transition: AnonymousTransition) -> None:
        assert sample_transition.track_a_fingerprint == "fp-track-a"
        assert sample_transition.track_b_fingerprint == "fp-track-b"

    def test_equality_by_id(self) -> None:
        uid = uuid.uuid4()
        now = datetime.now(UTC)
        a = AnonymousTransition(
            id=uid,
            track_a_fingerprint="x",
            track_b_fingerprint="y",
            contributed_at=now,
        )
        b = AnonymousTransition(
            id=uid,
            track_a_fingerprint="x",
            track_b_fingerprint="y",
            contributed_at=now,
        )
        assert a == b

    def test_inequality(self) -> None:
        now = datetime.now(UTC)
        a = AnonymousTransition(
            id=uuid.uuid4(),
            track_a_fingerprint="x",
            track_b_fingerprint="y",
            contributed_at=now,
        )
        b = AnonymousTransition(
            id=uuid.uuid4(),
            track_a_fingerprint="x",
            track_b_fingerprint="y",
            contributed_at=now,
        )
        assert a != b

    def test_frozen(self, sample_transition: AnonymousTransition) -> None:
        with pytest.raises(AttributeError):
            sample_transition.track_a_fingerprint = "z"  # type: ignore[misc]


class TestCommunityScore:
    def test_increment(self, sample_score: CommunityScore) -> None:
        incremented = sample_score.increment()
        assert incremented.frequency == 6
        assert sample_score.frequency == 5

    def test_increment_by_amount(self, sample_score: CommunityScore) -> None:
        incremented = sample_score.increment(3)
        assert incremented.frequency == 8

    def test_equality_by_fingerprints(self) -> None:
        a = CommunityScore(track_a_fingerprint="x", track_b_fingerprint="y", frequency=1)
        b = CommunityScore(track_a_fingerprint="x", track_b_fingerprint="y", frequency=10)
        assert a == b

    def test_inequality(self) -> None:
        a = CommunityScore(track_a_fingerprint="x", track_b_fingerprint="y", frequency=1)
        b = CommunityScore(track_a_fingerprint="x", track_b_fingerprint="z", frequency=1)
        assert a != b
