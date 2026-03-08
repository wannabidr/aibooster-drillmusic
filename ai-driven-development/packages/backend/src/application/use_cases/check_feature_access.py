"""CheckFeatureAccess use case -- give-to-get + subscription gating."""

from __future__ import annotations

from src.domain.entities.user import User
from src.domain.value_objects.subscription_tier import SubscriptionTier

FEATURE_ACCESS: dict[str, dict[str, bool]] = {
    "local_analysis": {"free": True, "free_contributed": True, "pro": True},
    "basic_recommendations": {"free": True, "free_contributed": True, "pro": True},
    "crossfade_preview": {"free": True, "free_contributed": True, "pro": True},
    "community_scores": {"free": False, "free_contributed": True, "pro": True},
    "ai_blend_styles": {"free": False, "free_contributed": False, "pro": True},
    "confidence_scoring": {"free": False, "free_contributed": False, "pro": True},
}


class FeatureAccessDeniedError(Exception):
    """Raised when a user cannot access a feature."""

    def __init__(self, feature: str, reason: str) -> None:
        self.feature = feature
        self.reason = reason
        super().__init__(f"Access denied to '{feature}': {reason}")


class CheckFeatureAccess:
    def execute(self, user: User, feature: str) -> bool:
        if feature not in FEATURE_ACCESS:
            raise ValueError(f"Unknown feature: {feature}")

        matrix = FEATURE_ACCESS[feature]

        if user.subscription_tier == SubscriptionTier.PRO:
            return matrix["pro"]

        if user.has_contributed:
            return matrix["free_contributed"]

        return matrix["free"]
