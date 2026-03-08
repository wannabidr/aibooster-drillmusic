"""Tests for User entity."""

from __future__ import annotations

from datetime import UTC, datetime

from src.domain.value_objects.subscription_tier import SubscriptionTier

from tests.conftest import make_user


def test_user_creation():
    user = make_user()
    assert user.email.value == "test@example.com"
    assert user.is_deleted is False


def test_mark_deleted():
    user = make_user()
    now = datetime.now(UTC)
    deleted = user.mark_deleted(now)
    assert deleted.is_deleted is True
    assert deleted.deleted_at == now


def test_mark_contributed():
    user = make_user(has_contributed=False)
    contributed = user.mark_contributed()
    assert contributed.has_contributed is True


def test_upgrade_tier():
    user = make_user(subscription_tier=SubscriptionTier.FREE)
    upgraded = user.upgrade_tier(SubscriptionTier.PRO)
    assert upgraded.subscription_tier == SubscriptionTier.PRO


def test_set_stripe_customer_id():
    user = make_user()
    assert user.stripe_customer_id is None
    updated = user.set_stripe_customer_id("cus_123")
    assert updated.stripe_customer_id == "cus_123"
