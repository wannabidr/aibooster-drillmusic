"""Subscription gate port."""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.domain.entities.user import User


class SubscriptionGate(ABC):
    @abstractmethod
    def can_access_feature(self, user: User, feature: str) -> bool: ...
