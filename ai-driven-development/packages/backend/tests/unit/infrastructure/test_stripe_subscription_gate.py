"""Tests for StripeSubscriptionGate feature access matrix.

These tests verify the give-to-get model and subscription tier gating
without requiring Stripe API access.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest
from src.domain.entities.user import User
from src.domain.value_objects.email import Email
from src.domain.value_objects.oauth_provider import OAuthProvider
from src.domain.value_objects.subscription_tier import SubscriptionTier
from src.domain.value_objects.user_id import UserId
from src.infrastructure.stripe.stripe_subscription_gate import (
    FEATURE_ACCESS,
    StripeSubscriptionGate,
)


def _make_user(
    tier: SubscriptionTier = SubscriptionTier.FREE,
    has_contributed: bool = False,
) -> User:
    """Create a test user with the given tier and contribution status."""
    return User(
        id=UserId(uuid.uuid4()),
        email=Email("dj@example.com"),
        display_name="Test DJ",
        oauth_provider=OAuthProvider.GOOGLE,
        oauth_subject="google-123",
        subscription_tier=tier,
        has_contributed=has_contributed,
        created_at=datetime.now(UTC),
        deleted_at=None,
    )


@pytest.fixture
def gate() -> StripeSubscriptionGate:
    return StripeSubscriptionGate(stripe_api_key="sk_test_fake")


class TestFreeNoContribution:
    """FREE tier user who has NOT contributed mix history."""

    def test_can_access_local_analysis(self, gate: StripeSubscriptionGate) -> None:
        user = _make_user(SubscriptionTier.FREE, has_contributed=False)
        assert gate.can_access_feature(user, "local_analysis") is True

    def test_can_access_basic_recommendations(self, gate: StripeSubscriptionGate) -> None:
        user = _make_user(SubscriptionTier.FREE, has_contributed=False)
        assert gate.can_access_feature(user, "basic_recommendations") is True

    def test_can_access_crossfade_preview(self, gate: StripeSubscriptionGate) -> None:
        user = _make_user(SubscriptionTier.FREE, has_contributed=False)
        assert gate.can_access_feature(user, "crossfade_preview") is True

    def test_cannot_access_community_scores(self, gate: StripeSubscriptionGate) -> None:
        user = _make_user(SubscriptionTier.FREE, has_contributed=False)
        assert gate.can_access_feature(user, "community_scores") is False

    def test_cannot_access_ai_blend_styles(self, gate: StripeSubscriptionGate) -> None:
        user = _make_user(SubscriptionTier.FREE, has_contributed=False)
        assert gate.can_access_feature(user, "ai_blend_styles") is False

    def test_cannot_access_confidence_scoring(self, gate: StripeSubscriptionGate) -> None:
        user = _make_user(SubscriptionTier.FREE, has_contributed=False)
        assert gate.can_access_feature(user, "confidence_scoring") is False


class TestFreeWithContribution:
    """FREE tier user who HAS contributed (give-to-get unlocked)."""

    def test_can_access_community_scores(self, gate: StripeSubscriptionGate) -> None:
        user = _make_user(SubscriptionTier.FREE, has_contributed=True)
        assert gate.can_access_feature(user, "community_scores") is True

    def test_cannot_access_ai_blend_styles(self, gate: StripeSubscriptionGate) -> None:
        user = _make_user(SubscriptionTier.FREE, has_contributed=True)
        assert gate.can_access_feature(user, "ai_blend_styles") is False

    def test_cannot_access_confidence_scoring(self, gate: StripeSubscriptionGate) -> None:
        user = _make_user(SubscriptionTier.FREE, has_contributed=True)
        assert gate.can_access_feature(user, "confidence_scoring") is False


class TestProTier:
    """PRO tier user -- full access to everything."""

    def test_can_access_all_features(self, gate: StripeSubscriptionGate) -> None:
        user = _make_user(SubscriptionTier.PRO, has_contributed=False)
        for feature in FEATURE_ACCESS:
            assert (
                gate.can_access_feature(user, feature) is True
            ), f"PRO user should access {feature}"

    def test_pro_with_contribution_has_full_access(self, gate: StripeSubscriptionGate) -> None:
        user = _make_user(SubscriptionTier.PRO, has_contributed=True)
        for feature in FEATURE_ACCESS:
            assert gate.can_access_feature(user, feature) is True


class TestUnknownFeature:
    """Unknown feature names should be denied."""

    def test_unknown_feature_denied(self, gate: StripeSubscriptionGate) -> None:
        user = _make_user(SubscriptionTier.PRO)
        assert gate.can_access_feature(user, "nonexistent_feature") is False
