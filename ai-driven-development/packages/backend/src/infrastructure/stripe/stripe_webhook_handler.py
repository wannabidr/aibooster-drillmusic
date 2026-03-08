"""Stripe webhook handler for subscription lifecycle events.

Processes Stripe webhook events to keep user subscription status
in sync with Stripe's billing system.
"""

from __future__ import annotations

import logging
from typing import Any

import stripe

from ...domain.entities.user import User
from ...domain.ports.user_repository import UserRepository
from ...domain.value_objects.subscription_tier import SubscriptionTier

logger = logging.getLogger(__name__)

# Stripe product IDs (set via environment config)
PRO_MONTHLY_PRICE_ID = ""  # Populated from config
PRO_ANNUAL_PRICE_ID = ""  # Populated from config


class StripeWebhookHandler:
    """Handles Stripe webhook events for subscription management."""

    def __init__(
        self,
        user_repo: UserRepository,
        webhook_secret: str,
    ) -> None:
        self._user_repo = user_repo
        self._webhook_secret = webhook_secret

    def verify_and_parse(self, payload: bytes, sig_header: str) -> stripe.Event:
        """Verify webhook signature and parse the event."""
        return stripe.Webhook.construct_event(payload, sig_header, self._webhook_secret)

    async def handle_event(self, event: stripe.Event) -> None:
        """Route a verified Stripe event to the appropriate handler."""
        handlers = {
            "checkout.session.completed": self._handle_checkout_completed,
            "customer.subscription.updated": self._handle_subscription_updated,
            "customer.subscription.deleted": self._handle_subscription_deleted,
            "invoice.payment_failed": self._handle_payment_failed,
        }

        handler = handlers.get(event.type)
        if handler:
            await handler(event.data.object)
        else:
            logger.debug("Unhandled Stripe event type: %s", event.type)

    async def _handle_checkout_completed(self, session: Any) -> None:
        """Handle successful checkout -- activate Pro subscription."""
        customer_id = session.customer
        user = await self._find_user_by_stripe_customer(customer_id)
        if not user:
            logger.error("Checkout completed for unknown Stripe customer: %s", customer_id)
            return

        upgraded = user.upgrade_tier(SubscriptionTier.PRO)
        await self._user_repo.save(upgraded)
        logger.info("User %s upgraded to PRO via checkout", user.id)

    async def _handle_subscription_updated(self, subscription: Any) -> None:
        """Handle subscription status changes (active, past_due, trialing)."""
        customer_id = subscription.customer
        user = await self._find_user_by_stripe_customer(customer_id)
        if not user:
            return

        status = subscription.status
        if status in ("active", "trialing"):
            updated = user.upgrade_tier(SubscriptionTier.PRO)
        else:
            updated = user.upgrade_tier(SubscriptionTier.FREE)

        await self._user_repo.save(updated)
        logger.info(
            "User %s subscription updated: status=%s, tier=%s",
            user.id,
            status,
            updated.subscription_tier,
        )

    async def _handle_subscription_deleted(self, subscription: Any) -> None:
        """Handle subscription cancellation -- downgrade to Free."""
        customer_id = subscription.customer
        user = await self._find_user_by_stripe_customer(customer_id)
        if not user:
            return

        downgraded = user.upgrade_tier(SubscriptionTier.FREE)
        await self._user_repo.save(downgraded)
        logger.info("User %s downgraded to FREE (subscription deleted)", user.id)

    async def _handle_payment_failed(self, invoice: Any) -> None:
        """Handle failed payment -- log for monitoring, no immediate downgrade.

        Stripe's built-in dunning handles retries. We only downgrade
        when the subscription is actually deleted (after all retries fail).
        """
        customer_id = invoice.customer
        logger.warning(
            "Payment failed for Stripe customer %s, invoice %s",
            customer_id,
            invoice.id,
        )

    async def _find_user_by_stripe_customer(self, customer_id: str) -> User | None:
        """Look up a user by their Stripe customer ID."""
        user = await self._user_repo.find_by_stripe_customer(customer_id)
        if not user:
            logger.warning("No user found for Stripe customer: %s", customer_id)
        return user
