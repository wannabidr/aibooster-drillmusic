"""B2B API tier enum."""

from __future__ import annotations

from enum import StrEnum


class ApiTier(StrEnum):
    BASIC = "basic"  # 100 req/min, genre trends only
    PRO = "pro"  # 1000 req/min, + regional data
    ENTERPRISE = "enterprise"  # 5000 req/min, + raw anonymized sets

    @property
    def rate_limit(self) -> int:
        """Requests per minute for this tier."""
        limits = {
            ApiTier.BASIC: 100,
            ApiTier.PRO: 1000,
            ApiTier.ENTERPRISE: 5000,
        }
        return limits[self]
