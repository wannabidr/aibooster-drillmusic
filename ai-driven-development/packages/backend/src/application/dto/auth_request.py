"""Auth request DTOs."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LoginRequest:
    provider: str
    code: str
    redirect_uri: str


@dataclass(frozen=True)
class RefreshRequest:
    refresh_token: str


@dataclass(frozen=True)
class SyncRequest:
    transitions: list[TransitionData]


@dataclass(frozen=True)
class TransitionData:
    track_a_fingerprint: str
    track_b_fingerprint: str
