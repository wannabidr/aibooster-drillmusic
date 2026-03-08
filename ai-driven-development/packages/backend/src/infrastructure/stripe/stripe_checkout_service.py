"""Stripe checkout and customer portal service.

Handles creation of Stripe checkout sessions for new subscriptions
and customer portal sessions for managing existing subscriptions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import stripe

from ...domain.entities.user import User

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StripeConfig:
    """Stripe configuration loaded from environment."""

    api_key: str
    webhook_secret: str
    pro_monthly_price_id: str
    pro_annual_price_id: str
    success_url: str  # e.g. "aidj://subscription/success"
    cancel_url: str  # e.g. "aidj://subscription/cancel"
    trial_period_days: int = 14


class StripeCheckoutService:
    """Creates Stripe checkout sessions and customer portal links."""

    def __init__(self, config: StripeConfig) -> None:
        self._config = config
        stripe.api_key = config.api_key

    async def create_checkout_session(
        self,
        user: User,
        plan: str,  # "monthly" or "annual"
    ) -> str:
        """Create a Stripe Checkout session and return the URL.

        If the user doesn't have a Stripe customer ID yet, one is created.
        The 14-day free trial is applied automatically for first-time subscribers.
        """
        customer_id = await self._ensure_customer(user)

        price_id = (
            self._config.pro_monthly_price_id
            if plan == "monthly"
            else self._config.pro_annual_price_id
        )

        session_params: dict = {
            "customer": customer_id,
            "payment_method_types": ["card"],
            "line_items": [{"price": price_id, "quantity": 1}],
            "mode": "subscription",
            "success_url": self._config.success_url,
            "cancel_url": self._config.cancel_url,
            "metadata": {"user_id": str(user.id)},
        }

        # Apply trial for first-time subscribers
        if not await self._has_previous_subscription(customer_id):
            session_params["subscription_data"] = {
                "trial_period_days": self._config.trial_period_days,
            }

        session = stripe.checkout.Session.create(**session_params)
        logger.info(
            "Created checkout session %s for user %s (plan=%s)",
            session.id,
            user.id,
            plan,
        )
        return session.url

    async def create_portal_session(self, user: User) -> str:
        """Create a Stripe Customer Portal session for subscription management.

        The portal allows users to:
        - Update payment method
        - Switch between monthly/annual plans
        - Cancel subscription
        - View invoice history
        """
        if not user.stripe_customer_id:
            raise ValueError("User has no Stripe customer ID")

        session = stripe.billing_portal.Session.create(
            customer=user.stripe_customer_id,
            return_url=self._config.success_url,
        )
        return session.url

    async def _ensure_customer(self, user: User) -> str:
        """Get or create a Stripe customer for the user."""
        if user.stripe_customer_id:
            return user.stripe_customer_id

        customer = stripe.Customer.create(
            email=str(user.email),
            metadata={"user_id": str(user.id)},
        )
        logger.info("Created Stripe customer %s for user %s", customer.id, user.id)
        # Caller is responsible for persisting customer.id on the User entity
        return customer.id

    async def _has_previous_subscription(self, customer_id: str) -> bool:
        """Check if the customer has ever had a subscription (no double trials)."""
        subs = stripe.Subscription.list(
            customer=customer_id,
            limit=1,
        )
        return bool(subs.data)
