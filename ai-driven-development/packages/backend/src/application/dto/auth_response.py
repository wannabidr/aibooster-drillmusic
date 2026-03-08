"""Auth response DTOs."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TokenResponse:
    access_token: str
    refresh_token: str
    expires_in: int


@dataclass(frozen=True)
class UserResponse:
    id: str
    email: str
    display_name: str | None
    tier: str
    has_contributed: bool


@dataclass(frozen=True)
class LoginResponse:
    token: TokenResponse
    user: UserResponse


@dataclass(frozen=True)
class ScoreResponse:
    track_a_fingerprint: str
    track_b_fingerprint: str
    frequency: int
