"""LoginWithOAuth use case."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

from src.application.dto.auth_request import LoginRequest
from src.application.dto.auth_response import LoginResponse, TokenResponse, UserResponse
from src.domain.entities.session import Session
from src.domain.entities.user import User
from src.domain.ports.auth_provider import AuthProvider
from src.domain.ports.session_repository import SessionRepository
from src.domain.ports.token_service import TokenService
from src.domain.ports.user_repository import UserRepository
from src.domain.value_objects.email import Email
from src.domain.value_objects.subscription_tier import SubscriptionTier
from src.domain.value_objects.user_id import UserId

REFRESH_TOKEN_DAYS = 30


class LoginWithOAuth:
    def __init__(
        self,
        auth_providers: dict[str, AuthProvider],
        user_repo: UserRepository,
        session_repo: SessionRepository,
        token_service: TokenService,
    ) -> None:
        self._auth_providers = auth_providers
        self._user_repo = user_repo
        self._session_repo = session_repo
        self._token_service = token_service

    async def execute(self, request: LoginRequest) -> LoginResponse:
        provider = self._auth_providers.get(request.provider)
        if provider is None:
            raise ValueError(f"Unsupported provider: {request.provider}")

        info = await provider.exchange_code(request.code, request.redirect_uri)

        user = await self._user_repo.find_by_oauth(info.provider, info.subject)
        if user is None:
            user = User(
                id=UserId.generate(),
                email=Email(info.email),
                display_name=info.name,
                oauth_provider=info.provider,
                oauth_subject=info.subject,
                subscription_tier=SubscriptionTier.FREE,
                has_contributed=False,
                created_at=datetime.now(UTC),
            )
            await self._user_repo.save(user)

        token_pair = self._token_service.create_token_pair(user.id)
        now = datetime.now(UTC)

        session = Session(
            id=uuid.uuid4(),
            user_id=user.id,
            refresh_token_hash=self._token_service.hash_refresh_token(token_pair.refresh_token),
            expires_at=now + timedelta(days=REFRESH_TOKEN_DAYS),
            created_at=now,
        )
        await self._session_repo.save(session)

        return LoginResponse(
            token=TokenResponse(
                access_token=token_pair.access_token,
                refresh_token=token_pair.refresh_token,
                expires_in=token_pair.expires_in,
            ),
            user=UserResponse(
                id=str(user.id),
                email=user.email.value,
                display_name=user.display_name,
                tier=user.subscription_tier.value,
                has_contributed=user.has_contributed,
            ),
        )
