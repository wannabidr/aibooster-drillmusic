"""Tests for Session entity."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from tests.conftest import make_session


def test_session_not_expired():
    session = make_session(expires_at=datetime.now(UTC) + timedelta(days=30))
    assert session.is_expired(datetime.now(UTC)) is False


def test_session_expired():
    session = make_session(expires_at=datetime.now(UTC) - timedelta(hours=1))
    assert session.is_expired(datetime.now(UTC)) is True


def test_revoke_session():
    session = make_session(revoked=False)
    revoked = session.revoke()
    assert revoked.revoked is True
    assert session.revoked is False  # original unchanged
