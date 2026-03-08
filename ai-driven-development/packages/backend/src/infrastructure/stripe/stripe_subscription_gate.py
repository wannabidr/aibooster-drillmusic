"""Stripe-based implementation of the SubscriptionGate port.

This adapter checks a user's subscription status via Stripe
and determines feature access based on their tier.
"""

from __future__ import annotations

import logging

import stripe

from ...domain.entities.user import User
from ...domain.ports.subscription_gate import SubscriptionGate
from ...domain.value_objects.subscription_tier import SubscriptionTier

logger = logging.getLogger(__name__)

# Feature access matrix:
# FREE (no contribution): Local analysis, basic recommendations, crossfade preview
# FREE (contributed):     + Community scores
# PRO:                    + AI blend styles, confidence scoring
FEATURE_ACCESS: dict[str, dict[str, bool]] = {
    "local_analysis": {"free": True, "free_contributed": True, "pro": True},
    "basic_recommendations": {"free": True, "free_contributed": True, "pro": True},
    "crossfade_preview": {"free": True, "free_contributed": True, "pro": True},
    "community_scores": {"free": False, "free_contributed": True, "pro": True},
    "ai_blend_styles": {"free": False, "free_contributed": False, "pro": True},
    "confidence_scoring": {"free": False, "free_contributed": False, "pro": True},
}


class StripeSubscriptionGate(SubscriptionGate):
    """Checks feature access using Stripe subscription data and contribution status."""

    def __init__(self, stripe_api_key: str) -> None:
        stripe.api_key = stripe_api_key

    def can_access_feature(self, user: User, feature: str) -> bool:
        """Check if a user can access a given feature.

        Access is determined by the combination of:
        1. Subscription tier (FREE or PRO)
        2. Contribution status (has_contributed flag for give-to-get)
        """
        if feature not in FEATURE_ACCESS:
            logger.warning("Unknown feature requested: %s", feature)
            return False

        matrix = FEATURE_ACCESS[feature]

        if user.subscription_tier == SubscriptionTier.PRO:
            return matrix["pro"]

        if user.has_contributed:
            return matrix["free_contributed"]

        return matrix["free"]

    async def sync_subscription_from_stripe(self, user: User) -> SubscriptionTier:
        """Query Stripe for the user's current subscription status.

        Returns the subscription tier based on active Stripe subscriptions.
        This should be called periodically or on login to keep the local
        tier in sync with Stripe.
        """
        if not user.stripe_customer_id:
            return SubscriptionTier.FREE

        try:
            subscriptions = stripe.Subscription.list(
                customer=user.stripe_customer_id,
                status="active",
                limit=1,
            )

            if subscriptions.data:
                return SubscriptionTier.PRO

            # Check for trialing subscriptions
            trial_subs = stripe.Subscription.list(
                customer=user.stripe_customer_id,
                status="trialing",
                limit=1,
            )

            if trial_subs.data:
                return SubscriptionTier.PRO

        except stripe.error.StripeError as e:
            logger.error("Stripe API error while syncing subscription: %s", e)
            # Fall back to current tier on Stripe errors
            return user.subscription_tier

        return SubscriptionTier.FREE
