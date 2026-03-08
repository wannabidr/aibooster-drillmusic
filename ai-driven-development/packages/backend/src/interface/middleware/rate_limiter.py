"""Rate limiting middleware using slowapi."""

from __future__ import annotations

from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

LOGIN_LIMIT = "5/minute"
REFRESH_LIMIT = "10/minute"
GENERAL_LIMIT = "100/minute"
