"""User entity."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime

from src.domain.value_objects.email import Email
from src.domain.value_objects.oauth_provider import OAuthProvider
from src.domain.value_objects.subscription_tier import SubscriptionTier
from src.domain.value_objects.user_id import UserId


@dataclass(frozen=True)
class User:
    id: UserId
    email: Email
    display_name: str | None
    oauth_provider: OAuthProvider
    oauth_subject: str
    subscription_tier: SubscriptionTier
    has_contributed: bool
    created_at: datetime
    deleted_at: datetime | None = None
    stripe_customer_id: str | None = None

    def mark_deleted(self, now: datetime) -> User:
        return replace(self, deleted_at=now)

    def mark_contributed(self) -> User:
        return replace(self, has_contributed=True)

    def upgrade_tier(self, tier: SubscriptionTier) -> User:
        return replace(self, subscription_tier=tier)

    def set_stripe_customer_id(self, customer_id: str) -> User:
        return replace(self, stripe_customer_id=customer_id)

    @property
    def is_deleted(self) -> bool:
        return self.deleted_at is not None
