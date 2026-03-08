"""Tests for StripeWebhookHandler.

Tests verify that webhook events correctly update user subscription tiers.
All Stripe API calls and repository interactions are mocked.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from src.domain.entities.user import User
from src.domain.value_objects.email import Email
from src.domain.value_objects.oauth_provider import OAuthProvider
from src.domain.value_objects.subscription_tier import SubscriptionTier
from src.domain.value_objects.user_id import UserId
from src.infrastructure.stripe.stripe_webhook_handler import StripeWebhookHandler


def _make_user(
    tier: SubscriptionTier = SubscriptionTier.FREE,
    stripe_customer_id: str = "cus_test123",
) -> User:
    return User(
        id=UserId(uuid.uuid4()),
        email=Email("dj@example.com"),
        display_name="Test DJ",
        oauth_provider=OAuthProvider.GOOGLE,
        oauth_subject="google-123",
        subscription_tier=tier,
        has_contributed=False,
        created_at=datetime.now(UTC),
        deleted_at=None,
        stripe_customer_id=stripe_customer_id,
    )


@pytest.fixture
def user_repo() -> AsyncMock:
    repo = AsyncMock()
    return repo


@pytest.fixture
def handler(user_repo: AsyncMock) -> StripeWebhookHandler:
    return StripeWebhookHandler(
        user_repo=user_repo,
        webhook_secret="whsec_test",
    )


class TestCheckoutCompleted:
    @pytest.mark.asyncio
    async def test_upgrades_user_to_pro(
        self, handler: StripeWebhookHandler, user_repo: AsyncMock
    ) -> None:
        user = _make_user(SubscriptionTier.FREE)
        user_repo.find_by_stripe_customer.return_value = user

        session = MagicMock()
        session.customer = "cus_test123"

        await handler._handle_checkout_completed(session)

        user_repo.save.assert_called_once()
        saved_user = user_repo.save.call_args[0][0]
        assert saved_user.subscription_tier == SubscriptionTier.PRO

    @pytest.mark.asyncio
    async def test_unknown_customer_logs_error(
        self, handler: StripeWebhookHandler, user_repo: AsyncMock
    ) -> None:
        user_repo.find_by_stripe_customer.return_value = None

        session = MagicMock()
        session.customer = "cus_unknown"

        await handler._handle_checkout_completed(session)
        user_repo.save.assert_not_called()


class TestSubscriptionUpdated:
    @pytest.mark.asyncio
    async def test_active_subscription_sets_pro(
        self, handler: StripeWebhookHandler, user_repo: AsyncMock
    ) -> None:
        user = _make_user(SubscriptionTier.FREE)
        user_repo.find_by_stripe_customer.return_value = user

        subscription = MagicMock()
        subscription.customer = "cus_test123"
        subscription.status = "active"

        await handler._handle_subscription_updated(subscription)

        saved_user = user_repo.save.call_args[0][0]
        assert saved_user.subscription_tier == SubscriptionTier.PRO

    @pytest.mark.asyncio
    async def test_trialing_subscription_sets_pro(
        self, handler: StripeWebhookHandler, user_repo: AsyncMock
    ) -> None:
        user = _make_user(SubscriptionTier.FREE)
        user_repo.find_by_stripe_customer.return_value = user

        subscription = MagicMock()
        subscription.customer = "cus_test123"
        subscription.status = "trialing"

        await handler._handle_subscription_updated(subscription)

        saved_user = user_repo.save.call_args[0][0]
        assert saved_user.subscription_tier == SubscriptionTier.PRO

    @pytest.mark.asyncio
    async def test_past_due_subscription_downgrades(
        self, handler: StripeWebhookHandler, user_repo: AsyncMock
    ) -> None:
        user = _make_user(SubscriptionTier.PRO)
        user_repo.find_by_stripe_customer.return_value = user

        subscription = MagicMock()
        subscription.customer = "cus_test123"
        subscription.status = "past_due"

        await handler._handle_subscription_updated(subscription)

        saved_user = user_repo.save.call_args[0][0]
        assert saved_user.subscription_tier == SubscriptionTier.FREE


class TestSubscriptionDeleted:
    @pytest.mark.asyncio
    async def test_downgrades_to_free(
        self, handler: StripeWebhookHandler, user_repo: AsyncMock
    ) -> None:
        user = _make_user(SubscriptionTier.PRO)
        user_repo.find_by_stripe_customer.return_value = user

        subscription = MagicMock()
        subscription.customer = "cus_test123"

        await handler._handle_subscription_deleted(subscription)

        saved_user = user_repo.save.call_args[0][0]
        assert saved_user.subscription_tier == SubscriptionTier.FREE
