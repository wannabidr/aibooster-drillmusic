"""Tests for rate limiter config."""

from __future__ import annotations

from src.interface.middleware.rate_limiter import (
    GENERAL_LIMIT,
    LOGIN_LIMIT,
    REFRESH_LIMIT,
    limiter,
)


def test_rate_limit_values():
    assert LOGIN_LIMIT == "5/minute"
    assert REFRESH_LIMIT == "10/minute"
    assert GENERAL_LIMIT == "100/minute"


def test_limiter_exists():
    assert limiter is not None
