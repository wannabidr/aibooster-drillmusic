"""Tests for CheckFeatureAccess use case."""

from __future__ import annotations

import pytest
from src.application.use_cases.check_feature_access import CheckFeatureAccess
from src.domain.value_objects.subscription_tier import SubscriptionTier

from tests.conftest import make_user


@pytest.fixture
def check():
    return CheckFeatureAccess()


def test_free_can_access_local_analysis(check):
    user = make_user(subscription_tier=SubscriptionTier.FREE, has_contributed=False)
    assert check.execute(user, "local_analysis") is True


def test_free_no_contribution_blocked_community(check):
    user = make_user(subscription_tier=SubscriptionTier.FREE, has_contributed=False)
    assert check.execute(user, "community_scores") is False


def test_free_contributed_can_access_community(check):
    user = make_user(subscription_tier=SubscriptionTier.FREE, has_contributed=True)
    assert check.execute(user, "community_scores") is True


def test_free_contributed_blocked_ai_blend(check):
    user = make_user(subscription_tier=SubscriptionTier.FREE, has_contributed=True)
    assert check.execute(user, "ai_blend_styles") is False


def test_pro_can_access_all(check):
    user = make_user(subscription_tier=SubscriptionTier.PRO)
    assert check.execute(user, "ai_blend_styles") is True
    assert check.execute(user, "community_scores") is True
    assert check.execute(user, "confidence_scoring") is True


def test_unknown_feature_raises(check):
    user = make_user()
    with pytest.raises(ValueError, match="Unknown feature"):
        check.execute(user, "nonexistent_feature")
