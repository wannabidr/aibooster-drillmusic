"""JWT-based token service implementation (RS256)."""

from __future__ import annotations

import secrets
from datetime import UTC, datetime, timedelta

import bcrypt as _bcrypt
import jwt

from src.domain.ports.token_service import TokenService
from src.domain.value_objects.token_pair import TokenPair
from src.domain.value_objects.user_id import UserId

ACCESS_TOKEN_MINUTES = 15
REFRESH_TOKEN_BYTES = 64
ISSUER = "ai-dj-assist"


class JWTTokenService(TokenService):
    def __init__(self, private_key: str, public_key: str) -> None:
        self._private_key = private_key
        self._public_key = public_key

    def create_token_pair(self, user_id: UserId) -> TokenPair:
        now = datetime.now(UTC)
        expires_in = ACCESS_TOKEN_MINUTES * 60

        payload = {
            "sub": str(user_id),
            "iat": now,
            "exp": now + timedelta(minutes=ACCESS_TOKEN_MINUTES),
            "iss": ISSUER,
        }
        access_token = jwt.encode(payload, self._private_key, algorithm="RS256")
        refresh_token = secrets.token_hex(REFRESH_TOKEN_BYTES)

        return TokenPair(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=expires_in,
        )

    def verify_access_token(self, token: str) -> UserId:
        try:
            payload = jwt.decode(
                token,
                self._public_key,
                algorithms=["RS256"],
                issuer=ISSUER,
                options={"require": ["sub", "exp", "iat", "iss"]},
            )
        except jwt.ExpiredSignatureError as e:
            raise PermissionError("Access token has expired") from e
        except jwt.InvalidTokenError as e:
            raise PermissionError(f"Invalid access token: {e}") from e

        return UserId.from_str(payload["sub"])

    def hash_refresh_token(self, token: str) -> str:
        salt = _bcrypt.gensalt()
        return _bcrypt.hashpw(token[:72].encode(), salt).decode()

    def verify_refresh_token_hash(self, token: str, hashed: str) -> bool:
        return _bcrypt.checkpw(token[:72].encode(), hashed.encode())
