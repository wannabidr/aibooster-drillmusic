"""Shared test fixtures."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from src.domain.entities.anonymous_transition import AnonymousTransition
from src.domain.entities.community_score import CommunityScore
from src.domain.entities.session import Session
from src.domain.entities.user import User
from src.domain.ports.auth_provider import AuthProvider, OAuthUserInfo
from src.domain.ports.session_repository import SessionRepository
from src.domain.ports.token_service import TokenService
from src.domain.ports.user_repository import UserRepository
from src.domain.value_objects.email import Email
from src.domain.value_objects.oauth_provider import OAuthProvider
from src.domain.value_objects.subscription_tier import SubscriptionTier
from src.domain.value_objects.token_pair import TokenPair
from src.domain.value_objects.user_id import UserId


@pytest.fixture
def rsa_keys() -> tuple[str, str]:
    """Generate an RSA key pair for testing."""
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private_pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode()
    public_pem = key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode()
    return private_pem, public_pem


def make_user(**overrides: Any) -> User:
    """Factory for User entities."""
    defaults = {
        "id": UserId.generate(),
        "email": Email("test@example.com"),
        "display_name": "Test User",
        "oauth_provider": OAuthProvider.GOOGLE,
        "oauth_subject": "google-sub-123",
        "subscription_tier": SubscriptionTier.FREE,
        "has_contributed": False,
        "created_at": datetime.now(UTC),
    }
    defaults.update(overrides)
    return User(**defaults)


def make_session(user: User | None = None, **overrides: Any) -> Session:
    """Factory for Session entities."""
    uid = user.id if user else UserId.generate()
    defaults = {
        "id": uuid.uuid4(),
        "user_id": uid,
        "refresh_token_hash": "hashed-token",
        "expires_at": datetime.now(UTC) + timedelta(days=30),
        "revoked": False,
        "created_at": datetime.now(UTC),
    }
    defaults.update(overrides)
    return Session(**defaults)


class MockUserRepository(UserRepository):
    def __init__(self) -> None:
        self.users: dict[str, User] = {}

    async def save(self, user: User) -> None:
        self.users[str(user.id)] = user

    async def find_by_id(self, user_id: UserId) -> User | None:
        return self.users.get(str(user_id))

    async def find_by_oauth(
        self, provider: OAuthProvider, subject: str
    ) -> User | None:
        for u in self.users.values():
            if u.oauth_provider == provider and u.oauth_subject == subject:
                return u
        return None

    async def find_by_stripe_customer(self, customer_id: str) -> User | None:
        for u in self.users.values():
            if u.stripe_customer_id == customer_id:
                return u
        return None

    async def delete(self, user_id: UserId) -> None:
        self.users.pop(str(user_id), None)


class MockSessionRepository(SessionRepository):
    def __init__(self) -> None:
        self.sessions: dict[str, Session] = {}

    async def save(self, session: Session) -> None:
        self.sessions[str(session.id)] = session

    async def find_by_refresh_hash(self, token_hash: str) -> Session | None:
        for s in self.sessions.values():
            if s.refresh_token_hash == token_hash:
                return s
        return None

    async def revoke_all_for_user(self, user_id: UserId) -> None:
        for key, s in list(self.sessions.items()):
            if s.user_id == user_id and not s.revoked:
                self.sessions[key] = s.revoke()


class MockTokenService(TokenService):
    def __init__(self) -> None:
        self._counter = 0

    def create_token_pair(self, user_id: UserId) -> TokenPair:
        self._counter += 1
        return TokenPair(
            access_token=f"access-{user_id}-{self._counter}",
            refresh_token=f"refresh-{user_id}-{self._counter}",
            expires_in=900,
        )

    def verify_access_token(self, token: str) -> UserId:
        # Extract user_id from "access-UserId(...)-N"
        parts = token.split("-", 1)
        if len(parts) < 2 or not parts[1]:
            raise PermissionError("Invalid token")
        uid_str = parts[1].rsplit("-", 1)[0]
        try:
            return UserId.from_str(uid_str)
        except ValueError as e:
            raise PermissionError(f"Invalid token: {e}") from e

    def hash_refresh_token(self, token: str) -> str:
        return f"hash:{token}"

    def verify_refresh_token_hash(self, token: str, hashed: str) -> bool:
        return hashed == f"hash:{token}"


class MockAuthProvider(AuthProvider):
    def __init__(
        self,
        email: str = "user@example.com",
        name: str = "OAuth User",
        provider: OAuthProvider = OAuthProvider.GOOGLE,
        subject: str = "oauth-sub-1",
        should_fail: bool = False,
    ) -> None:
        self._email = email
        self._name = name
        self._provider = provider
        self._subject = subject
        self._should_fail = should_fail

    async def exchange_code(self, code: str, redirect_uri: str) -> OAuthUserInfo:
        if self._should_fail:
            raise ValueError("Invalid authorization code")
        return OAuthUserInfo(
            email=self._email,
            name=self._name,
            provider=self._provider,
            subject=self._subject,
        )


# ---------------------------------------------------------------------------
# Entity fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_user_id() -> UserId:
    return UserId(value=uuid.UUID("12345678-1234-5678-1234-567812345678"))


@pytest.fixture
def sample_user(sample_user_id: UserId) -> User:
    return make_user(
        id=sample_user_id,
        email=Email("dj@example.com"),
        display_name="DJ Test",
    )


@pytest.fixture
def contributed_user(sample_user: User) -> User:
    return sample_user.mark_contributed()


@pytest.fixture
def pro_user(sample_user: User) -> User:
    return sample_user.upgrade_tier(SubscriptionTier.PRO)


@pytest.fixture
def sample_token_pair() -> TokenPair:
    return TokenPair(
        access_token="access.token.here",
        refresh_token="refresh-token-hex",
        expires_in=900,
    )


@pytest.fixture
def sample_session(sample_user: User) -> Session:
    return make_session(sample_user)


@pytest.fixture
def expired_session(sample_user: User) -> Session:
    return make_session(
        sample_user,
        expires_at=datetime.now(UTC) - timedelta(hours=1),
    )


@pytest.fixture
def sample_transition() -> AnonymousTransition:
    return AnonymousTransition(
        id=uuid.uuid4(),
        track_a_fingerprint="fp-track-a",
        track_b_fingerprint="fp-track-b",
        contributed_at=datetime.now(UTC),
    )


@pytest.fixture
def sample_score() -> CommunityScore:
    return CommunityScore(
        track_a_fingerprint="fp-track-a",
        track_b_fingerprint="fp-track-b",
        frequency=5,
    )


# ---------------------------------------------------------------------------
# AsyncMock / MagicMock fixtures for use-case tests
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_user_repo() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def mock_session_repo() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def mock_token_service() -> MagicMock:
    """MagicMock because create_token_pair / hash_refresh_token are sync."""
    mock = MagicMock()
    mock.create_token_pair.return_value = TokenPair(
        access_token="new-access-token",
        refresh_token="new-refresh-token",
        expires_in=900,
    )
    mock.hash_refresh_token.return_value = "hashed-refresh-token"
    mock.verify_access_token.return_value = UserId(
        value=uuid.UUID("12345678-1234-5678-1234-567812345678")
    )
    mock.verify_refresh_token_hash.return_value = True
    return mock


@pytest.fixture
def mock_auth_provider() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def mock_community_repo() -> AsyncMock:
    return AsyncMock()
