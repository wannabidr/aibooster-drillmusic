"""FastAPI application factory."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.application.use_cases.delete_account import DeleteAccount
from src.application.use_cases.login_with_oauth import LoginWithOAuth
from src.application.use_cases.logout import Logout
from src.application.use_cases.query_community_scores import QueryCommunityScores
from src.application.use_cases.refresh_access_token import RefreshAccessToken
from src.application.use_cases.sync_mix_history import SyncMixHistory
from src.application.use_cases.query_trends import QueryTrends
from src.infrastructure.auth.api_key_authenticator import ApiKeyAuthenticator
from src.infrastructure.auth.apple_oauth_provider import AppleOAuthProvider
from src.infrastructure.auth.google_oauth_provider import GoogleOAuthProvider
from src.infrastructure.auth.jwt_token_service import JWTTokenService
from src.infrastructure.config import Settings
from src.infrastructure.persistence.postgres_api_client_repository import PostgresApiClientRepository
from src.infrastructure.persistence.postgres_community_repository import PostgresCommunityRepository
from src.infrastructure.persistence.postgres_trend_repository import PostgresTrendRepository
from src.infrastructure.persistence.postgres_session_repository import PostgresSessionRepository
from src.infrastructure.persistence.postgres_user_repository import PostgresUserRepository
from src.infrastructure.stripe.stripe_checkout_service import StripeCheckoutService, StripeConfig
from src.infrastructure.stripe.stripe_subscription_gate import StripeSubscriptionGate
from src.infrastructure.stripe.stripe_webhook_handler import StripeWebhookHandler
from src.interface.middleware import api_key_middleware, auth_middleware
from src.interface.routes import (
    auth_routes,
    b2b_routes,
    community_routes,
    health_routes,
    subscription_routes,
)


def create_app(settings: Settings | None = None) -> FastAPI:
    if settings is None:
        settings = Settings()  # type: ignore[call-arg]

    engine = create_async_engine(settings.database_url, echo=False)
    async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    private_key = settings.load_jwt_private_key()
    public_key = settings.load_jwt_public_key()
    token_service = JWTTokenService(private_key, public_key)

    providers = {}
    if settings.google_client_id:
        providers["google"] = GoogleOAuthProvider(
            settings.google_client_id, settings.google_client_secret
        )
    if settings.apple_client_id:
        providers["apple"] = AppleOAuthProvider(settings.apple_client_id)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        yield
        await engine.dispose()

    app = FastAPI(title="AI DJ Assist API", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins.split(","),
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-Api-Key"],
    )

    # Wire dependencies
    def get_db_session() -> AsyncSession:
        return async_session()

    def get_token_service() -> JWTTokenService:
        return token_service

    async def get_user_repo() -> PostgresUserRepository:
        return PostgresUserRepository(await get_db_session().__aenter__())

    async def get_login_use_case() -> LoginWithOAuth:
        db = get_db_session()
        session = await db.__aenter__()
        return LoginWithOAuth(
            auth_providers=providers,
            user_repo=PostgresUserRepository(session),
            session_repo=PostgresSessionRepository(session),
            token_service=token_service,
        )

    async def get_refresh_use_case() -> RefreshAccessToken:
        db = get_db_session()
        session = await db.__aenter__()
        return RefreshAccessToken(
            session_repo=PostgresSessionRepository(session),
            token_service=token_service,
        )

    async def get_logout_use_case() -> Logout:
        db = get_db_session()
        session = await db.__aenter__()
        return Logout(
            session_repo=PostgresSessionRepository(session),
            token_service=token_service,
        )

    async def get_delete_use_case() -> DeleteAccount:
        db = get_db_session()
        session = await db.__aenter__()
        return DeleteAccount(
            user_repo=PostgresUserRepository(session),
            session_repo=PostgresSessionRepository(session),
        )

    # Override dependency stubs
    app.dependency_overrides[auth_middleware._get_token_service] = get_token_service
    app.dependency_overrides[auth_middleware._get_user_repo] = get_user_repo
    app.dependency_overrides[auth_routes._get_login_use_case] = get_login_use_case
    app.dependency_overrides[auth_routes._get_refresh_use_case] = get_refresh_use_case
    app.dependency_overrides[auth_routes._get_logout_use_case] = get_logout_use_case
    app.dependency_overrides[auth_routes._get_delete_use_case] = get_delete_use_case

    # Community use cases (give-to-get contribution tracking)
    async def get_sync_use_case() -> SyncMixHistory:
        db = get_db_session()
        session = await db.__aenter__()
        return SyncMixHistory(
            community_repo=PostgresCommunityRepository(session),
            user_repo=PostgresUserRepository(session),
        )

    async def get_scores_use_case() -> QueryCommunityScores:
        db = get_db_session()
        session = await db.__aenter__()
        return QueryCommunityScores(
            community_repo=PostgresCommunityRepository(session),
        )

    app.dependency_overrides[community_routes._get_sync_use_case] = get_sync_use_case
    app.dependency_overrides[community_routes._get_scores_use_case] = get_scores_use_case

    app.state.token_service = token_service

    # Stripe integration (only if configured)
    if settings.stripe_api_key:
        stripe_config = StripeConfig(
            api_key=settings.stripe_api_key,
            webhook_secret=settings.stripe_webhook_secret,
            pro_monthly_price_id=settings.stripe_pro_monthly_price_id,
            pro_annual_price_id=settings.stripe_pro_annual_price_id,
            success_url="aidj://subscription/success",
            cancel_url="aidj://subscription/cancel",
        )
        checkout_service = StripeCheckoutService(stripe_config)

        def get_checkout_service() -> StripeCheckoutService:
            return checkout_service

        async def get_webhook_handler() -> StripeWebhookHandler:
            db = get_db_session()
            session = await db.__aenter__()
            return StripeWebhookHandler(
                user_repo=PostgresUserRepository(session),
                webhook_secret=settings.stripe_webhook_secret,
            )

        app.dependency_overrides[subscription_routes._get_checkout_service] = get_checkout_service
        app.dependency_overrides[subscription_routes._get_webhook_handler] = get_webhook_handler

        app.state.subscription_gate = StripeSubscriptionGate(settings.stripe_api_key)

    # B2B API (always enabled -- API key auth is self-contained)
    async def get_api_key_authenticator() -> ApiKeyAuthenticator:
        db = get_db_session()
        session = await db.__aenter__()
        return ApiKeyAuthenticator(PostgresApiClientRepository(session))

    async def get_query_trends() -> QueryTrends:
        db = get_db_session()
        session = await db.__aenter__()
        return QueryTrends(trend_aggregator=PostgresTrendRepository(session))

    app.dependency_overrides[api_key_middleware._get_authenticator] = get_api_key_authenticator
    app.dependency_overrides[b2b_routes._get_query_trends] = get_query_trends

    app.include_router(auth_routes.router, prefix="/auth")
    app.include_router(community_routes.router)
    app.include_router(health_routes.router)
    app.include_router(subscription_routes.router)
    app.include_router(b2b_routes.router)

    return app
