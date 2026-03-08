"""Beatport Link streaming adapter.

Implements StreamingProvider using the Beatport API.
Beatport is the primary electronic music store/streaming service for DJs,
providing BPM, key, and genre metadata that is highly relevant for DJ workflows.

Uses HTTP client directly as there is no official Python SDK.
"""

from __future__ import annotations

import logging
from typing import Any
from urllib.parse import quote_plus, urlencode

from src.domain.ports.streaming_provider import (
    StreamingProvider,
    StreamingSearchResult,
    StreamingTrackMetadata,
)

logger = logging.getLogger(__name__)

_BEATPORT_API_BASE = "https://api.beatport.com/v4"
_BEATPORT_AUTH_URL = "https://api.beatport.com/v4/auth/o/token/"


class BeatportAdapter(StreamingProvider):
    """Beatport Link API adapter.

    Provides track search, metadata retrieval, and DJ-specific audio features
    (BPM, key, genre) from Beatport's catalog. Beatport is the gold standard
    for electronic music metadata.

    Requires Beatport Link API credentials (client_id, client_secret).
    """

    def __init__(self) -> None:
        self._access_token: str | None = None
        self._authenticated = False
        self._http: Any = None

    def provider_name(self) -> str:
        return "beatport"

    def authenticate(self, credentials: dict[str, str]) -> bool:
        try:
            import urllib.request
            import json

            client_id = credentials.get("client_id", "")
            client_secret = credentials.get("client_secret", "")

            if not client_id or not client_secret:
                logger.error("Beatport credentials missing client_id or client_secret")
                return False

            # OAuth2 client credentials flow
            data = urlencode({
                "grant_type": "client_credentials",
                "client_id": client_id,
                "client_secret": client_secret,
            }).encode("utf-8")

            req = urllib.request.Request(
                _BEATPORT_AUTH_URL,
                data=data,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                method="POST",
            )

            with urllib.request.urlopen(req, timeout=15) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                self._access_token = body.get("access_token")

            if self._access_token:
                self._authenticated = True
                logger.info("Beatport authentication successful")
                return True

            logger.warning("Beatport auth response missing access_token")
            return False
        except Exception:
            logger.exception("Beatport authentication failed")
            self._authenticated = False
            return False

    def is_authenticated(self) -> bool:
        return self._authenticated and self._access_token is not None

    def search_tracks(
        self,
        query: str,
        *,
        offset: int = 0,
        limit: int = 25,
    ) -> StreamingSearchResult:
        self._require_auth()
        limit = min(limit, 50)

        try:
            params = urlencode({
                "q": query,
                "page": (offset // limit) + 1,
                "per_page": limit,
                "type": "tracks",
            })
            url = f"{_BEATPORT_API_BASE}/catalog/search/?{params}"
            data = self._api_get(url)

            if data is None:
                return StreamingSearchResult()

            tracks_data = data.get("tracks", {})
            items = tracks_data.get("results", [])
            total = tracks_data.get("count", 0)

            tracks = [self._parse_track(item) for item in items]

            return StreamingSearchResult(
                tracks=tracks,
                total=total,
                offset=offset,
                limit=limit,
                has_more=(offset + limit) < total,
            )
        except Exception:
            logger.exception("Beatport search failed for query: %s", query)
            return StreamingSearchResult()

    def get_track_metadata(self, provider_track_id: str) -> StreamingTrackMetadata | None:
        self._require_auth()
        try:
            url = f"{_BEATPORT_API_BASE}/catalog/tracks/{provider_track_id}/"
            data = self._api_get(url)
            if data is None:
                return None
            return self._parse_track(data)
        except Exception:
            logger.exception("Failed to fetch Beatport track: %s", provider_track_id)
            return None

    def get_preview_url(self, provider_track_id: str) -> str | None:
        self._require_auth()
        try:
            url = f"{_BEATPORT_API_BASE}/catalog/tracks/{provider_track_id}/"
            data = self._api_get(url)
            if data is None:
                return None
            return data.get("sample_url") or data.get("preview", {}).get("uri")
        except Exception:
            logger.exception("Failed to get Beatport preview for: %s", provider_track_id)
            return None

    def get_audio_features(self, provider_track_id: str) -> dict[str, float] | None:
        self._require_auth()
        try:
            url = f"{_BEATPORT_API_BASE}/catalog/tracks/{provider_track_id}/"
            data = self._api_get(url)
            if data is None:
                return None

            result: dict[str, float] = {}
            bpm = data.get("bpm")
            if bpm is not None:
                result["bpm"] = float(bpm)

            # Beatport key is highly accurate for electronic music
            key_data = data.get("key")
            if key_data and isinstance(key_data, dict):
                # Key notation stored in name field
                result["_has_key"] = 1.0

            return result if result else None
        except Exception:
            logger.exception("Failed to get Beatport features for: %s", provider_track_id)
            return None

    def get_key_notation(self, provider_track_id: str) -> str | None:
        """Get the musical key as standard notation from Beatport."""
        self._require_auth()
        try:
            url = f"{_BEATPORT_API_BASE}/catalog/tracks/{provider_track_id}/"
            data = self._api_get(url)
            if data is None:
                return None
            key_data = data.get("key")
            if key_data and isinstance(key_data, dict):
                return key_data.get("camelot_number") or key_data.get("name")
            return None
        except Exception:
            logger.exception("Failed to get key for: %s", provider_track_id)
            return None

    def _require_auth(self) -> None:
        if not self.is_authenticated():
            raise RuntimeError(
                "Beatport adapter is not authenticated. Call authenticate() first."
            )

    def _api_get(self, url: str) -> dict[str, Any] | None:
        """Make an authenticated GET request to the Beatport API."""
        import json
        import urllib.request

        req = urllib.request.Request(
            url,
            headers={
                "Authorization": f"Bearer {self._access_token}",
                "Accept": "application/json",
            },
            method="GET",
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception:
            logger.exception("Beatport API request failed: %s", url)
            return None

    @staticmethod
    def _parse_track(item: dict[str, Any]) -> StreamingTrackMetadata:
        artists = item.get("artists", [])
        artist_name = artists[0].get("name", "") if artists else ""

        release = item.get("release", {})
        album_name = release.get("name", "") if release else ""

        artwork = item.get("image", {})
        artwork_url = artwork.get("uri") if artwork else None

        key_data = item.get("key")
        key_str = None
        if key_data and isinstance(key_data, dict):
            key_str = key_data.get("name")

        genre_data = item.get("genre")
        genre = genre_data.get("name", "") if genre_data and isinstance(genre_data, dict) else ""

        return StreamingTrackMetadata(
            provider="beatport",
            provider_track_id=str(item.get("id", "")),
            title=item.get("name", ""),
            artist=artist_name,
            album=album_name,
            duration_ms=int(item.get("length_ms", 0)),
            bpm=item.get("bpm"),
            key=key_str,
            genre=genre,
            preview_url=item.get("sample_url"),
            artwork_url=artwork_url,
            isrc=item.get("isrc"),
            release_date=item.get("publish_date") or item.get("new_release_date"),
        )
