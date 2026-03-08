"""Tests for other value objects."""

from __future__ import annotations

import uuid

from src.domain.value_objects.oauth_provider import OAuthProvider
from src.domain.value_objects.subscription_tier import SubscriptionTier
from src.domain.value_objects.token_pair import TokenPair
from src.domain.value_objects.user_id import UserId


def test_user_id_generate():
    uid = UserId.generate()
    assert isinstance(uid.value, uuid.UUID)


def test_user_id_from_str():
    raw = str(uuid.uuid4())
    uid = UserId.from_str(raw)
    assert str(uid) == raw


def test_token_pair_immutable():
    tp = TokenPair(access_token="a", refresh_token="r", expires_in=900)
    assert tp.access_token == "a"
    assert tp.expires_in == 900


def test_oauth_provider_values():
    assert OAuthProvider.GOOGLE.value == "google"
    assert OAuthProvider.APPLE.value == "apple"


def test_subscription_tier_values():
    assert SubscriptionTier.FREE.value == "free"
    assert SubscriptionTier.PRO.value == "pro"
