"""OAuth provider enum."""

from __future__ import annotations

from enum import StrEnum


class OAuthProvider(StrEnum):
    GOOGLE = "google"
    APPLE = "apple"
