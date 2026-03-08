"""Google OAuth2 provider implementation."""

from __future__ import annotations

import httpx
import jwt

from src.domain.ports.auth_provider import AuthProvider, OAuthUserInfo
from src.domain.value_objects.oauth_provider import OAuthProvider

GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_CERTS_URL = "https://www.googleapis.com/oauth2/v3/certs"
GOOGLE_ISSUER = "https://accounts.google.com"


class GoogleOAuthProvider(AuthProvider):
    def __init__(self, client_id: str, client_secret: str) -> None:
        self._client_id = client_id
        self._client_secret = client_secret

    async def exchange_code(self, code: str, redirect_uri: str) -> OAuthUserInfo:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                GOOGLE_TOKEN_URL,
                data={
                    "code": code,
                    "client_id": self._client_id,
                    "client_secret": self._client_secret,
                    "redirect_uri": redirect_uri,
                    "grant_type": "authorization_code",
                },
            )
            resp.raise_for_status()
            token_data = resp.json()

        id_token = token_data["id_token"]
        jwks_client = jwt.PyJWKClient(GOOGLE_CERTS_URL)
        signing_key = jwks_client.get_signing_key_from_jwt(id_token)

        claims = jwt.decode(
            id_token,
            signing_key.key,
            algorithms=["RS256"],
            audience=self._client_id,
            issuer=GOOGLE_ISSUER,
        )

        return OAuthUserInfo(
            email=claims["email"],
            name=claims.get("name"),
            provider=OAuthProvider.GOOGLE,
            subject=claims["sub"],
        )
