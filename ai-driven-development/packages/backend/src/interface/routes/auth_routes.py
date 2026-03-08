"""Auth routes -- login, refresh, logout, me, delete account."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from src.application.dto.auth_request import LoginRequest, RefreshRequest
from src.application.use_cases.delete_account import DeleteAccount
from src.application.use_cases.login_with_oauth import LoginWithOAuth
from src.application.use_cases.logout import Logout
from src.application.use_cases.refresh_access_token import RefreshAccessToken
from src.domain.entities.user import User
from src.interface.middleware.auth_middleware import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter()


class LoginBody(BaseModel):
    provider: str
    code: str
    redirect_uri: str


class RefreshBody(BaseModel):
    refresh_token: str


class LogoutBody(BaseModel):
    refresh_token: str


class TokenOut(BaseModel):
    access_token: str
    refresh_token: str
    expires_in: int


class UserOut(BaseModel):
    id: str
    email: str
    display_name: str | None
    tier: str
    has_contributed: bool


class LoginOut(BaseModel):
    token: TokenOut
    user: UserOut


def _get_login_use_case() -> LoginWithOAuth:
    raise HTTPException(status_code=501, detail="OAuth login not configured")


def _get_refresh_use_case() -> RefreshAccessToken:
    raise HTTPException(status_code=501, detail="Token refresh not configured")


def _get_logout_use_case() -> Logout:
    raise HTTPException(status_code=501, detail="Logout not configured")


def _get_delete_use_case() -> DeleteAccount:
    raise HTTPException(status_code=501, detail="Account deletion not configured")


@router.post("/login", response_model=LoginOut)
async def login(
    body: LoginBody,
    use_case: LoginWithOAuth = Depends(_get_login_use_case),
) -> LoginOut:
    try:
        result = await use_case.execute(
            LoginRequest(
                provider=body.provider,
                code=body.code,
                redirect_uri=body.redirect_uri,
            )
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    return LoginOut(
        token=TokenOut(
            access_token=result.token.access_token,
            refresh_token=result.token.refresh_token,
            expires_in=result.token.expires_in,
        ),
        user=UserOut(
            id=result.user.id,
            email=result.user.email,
            display_name=result.user.display_name,
            tier=result.user.tier,
            has_contributed=result.user.has_contributed,
        ),
    )


@router.post("/refresh", response_model=TokenOut)
async def refresh(
    body: RefreshBody,
    use_case: RefreshAccessToken = Depends(_get_refresh_use_case),
) -> TokenOut:
    try:
        result = await use_case.execute(RefreshRequest(refresh_token=body.refresh_token))
    except PermissionError as e:
        raise HTTPException(status_code=401, detail=str(e)) from e

    return TokenOut(
        access_token=result.access_token,
        refresh_token=result.refresh_token,
        expires_in=result.expires_in,
    )


@router.post("/logout", status_code=204)
async def logout(
    body: LogoutBody,
    use_case: Logout = Depends(_get_logout_use_case),
) -> None:
    await use_case.execute(body.refresh_token)


@router.get("/me", response_model=UserOut)
async def me(
    current_user: User = Depends(get_current_user),
) -> UserOut:
    return UserOut(
        id=str(current_user.id),
        email=current_user.email.value,
        display_name=current_user.display_name,
        tier=current_user.subscription_tier.value,
        has_contributed=current_user.has_contributed,
    )


@router.delete("/account", status_code=204)
async def delete_account(
    current_user: User = Depends(get_current_user),
    use_case: DeleteAccount = Depends(_get_delete_use_case),
) -> None:
    await use_case.execute(current_user.id)
