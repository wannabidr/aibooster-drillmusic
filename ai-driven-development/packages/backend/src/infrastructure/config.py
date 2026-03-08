"""Application settings via pydantic-settings."""

from __future__ import annotations

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_prefix": "AIDJ_"}

    database_url: str
    jwt_private_key_path: str
    jwt_public_key_path: str

    google_client_id: str = ""
    google_client_secret: str = ""
    apple_client_id: str = ""

    cors_origins: str = "tauri://localhost"

    stripe_api_key: str = ""
    stripe_webhook_secret: str = ""
    stripe_pro_monthly_price_id: str = ""
    stripe_pro_annual_price_id: str = ""

    def load_jwt_private_key(self) -> str:
        with open(self.jwt_private_key_path) as f:
            return f.read()

    def load_jwt_public_key(self) -> str:
        with open(self.jwt_public_key_path) as f:
            return f.read()
