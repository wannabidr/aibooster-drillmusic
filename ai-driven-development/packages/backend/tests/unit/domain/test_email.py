"""Tests for Email value object."""

from __future__ import annotations

import pytest
from src.domain.value_objects.email import Email


def test_valid_email():
    email = Email("user@example.com")
    assert email.value == "user@example.com"


def test_email_normalizes_to_lowercase():
    email = Email("User@EXAMPLE.com")
    assert email.value == "user@example.com"


def test_email_rejects_empty():
    with pytest.raises(ValueError, match="1-255"):
        Email("")


def test_email_rejects_no_at():
    with pytest.raises(ValueError, match="Invalid email"):
        Email("notanemail")


def test_email_rejects_too_long():
    long = "a" * 250 + "@b.com"
    with pytest.raises(ValueError, match="1-255"):
        Email(long)
