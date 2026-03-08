"""Tests for JWTTokenService."""

from __future__ import annotations

import time

import jwt
import pytest
from src.domain.value_objects.user_id import UserId
from src.infrastructure.auth.jwt_token_service import JWTTokenService


@pytest.fixture
def token_service(rsa_keys):
    private_key, public_key = rsa_keys
    return JWTTokenService(private_key, public_key)


def test_create_token_pair(token_service):
    uid = UserId.generate()
    pair = token_service.create_token_pair(uid)
    assert pair.access_token
    assert pair.refresh_token
    assert pair.expires_in == 900
    assert len(pair.refresh_token) == 128  # 64 bytes hex


def test_verify_valid_access_token(token_service):
    uid = UserId.generate()
    pair = token_service.create_token_pair(uid)
    result = token_service.verify_access_token(pair.access_token)
    assert result == uid


def test_verify_expired_token_raises(rsa_keys):
    private_key, public_key = rsa_keys
    payload = {
        "sub": str(UserId.generate()),
        "iat": time.time() - 3600,
        "exp": time.time() - 1800,
        "iss": "ai-dj-assist",
    }
    expired_token = jwt.encode(payload, private_key, algorithm="RS256")
    service = JWTTokenService(private_key, public_key)
    with pytest.raises(PermissionError, match="expired"):
        service.verify_access_token(expired_token)


def test_verify_tampered_token_raises(token_service, rsa_keys):
    uid = UserId.generate()
    pair = token_service.create_token_pair(uid)
    tampered = pair.access_token[:-5] + "XXXXX"
    with pytest.raises(PermissionError, match="Invalid"):
        token_service.verify_access_token(tampered)


def test_hash_and_verify_refresh_token(token_service):
    token = "my-refresh-token"
    hashed = token_service.hash_refresh_token(token)
    assert token_service.verify_refresh_token_hash(token, hashed) is True
    assert token_service.verify_refresh_token_hash("wrong", hashed) is False
