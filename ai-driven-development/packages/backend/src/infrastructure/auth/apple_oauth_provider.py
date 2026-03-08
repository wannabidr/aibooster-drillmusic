"""Apple Sign-In provider implementation."""

from __future__ import annotations

import jwt

from src.domain.ports.auth_provider import AuthProvider, OAuthUserInfo
from src.domain.value_objects.oauth_provider import OAuthProvider

APPLE_JWKS_URL = "https://appleid.apple.com/auth/keys"
APPLE_ISSUER = "https://appleid.apple.com"


class AppleOAuthProvider(AuthProvider):
    def __init__(self, client_id: str) -> None:
        self._client_id = client_id

    async def exchange_code(self, code: str, redirect_uri: str) -> OAuthUserInfo:
        """Validate Apple identity_token (passed as 'code' from desktop app).

        The desktop app uses ASAuthorizationAppleIDProvider natively and sends
        the identity_token JWT directly. We validate it against Apple's public keys.
        """
        jwks_client = jwt.PyJWKClient(APPLE_JWKS_URL)
        signing_key = jwks_client.get_signing_key_from_jwt(code)

        claims = jwt.decode(
            code,
            signing_key.key,
            algorithms=["RS256"],
            audience=self._client_id,
            issuer=APPLE_ISSUER,
        )

        return OAuthUserInfo(
            email=claims.get("email", ""),
            name=None,
            provider=OAuthProvider.APPLE,
            subject=claims["sub"],
        )
