"""Spotify streaming adapter.

Implements StreamingProvider using the Spotify Web API.
Uses spotipy for OAuth and API calls.
"""

from __future__ import annotations

import logging
from typing import Any

from src.domain.ports.streaming_provider import (
    StreamingProvider,
    StreamingSearchResult,
    StreamingTrackMetadata,
)

logger = logging.getLogger(__name__)

# Spotify key notation mapping (pitch_class, mode) -> standard notation
_SPOTIFY_KEY_MAP: dict[tuple[int, int], str] = {
    (0, 1): "C",
    (0, 0): "Cm",
    (1, 1): "Db",
    (1, 0): "Dbm",
    (2, 1): "D",
    (2, 0): "Dm",
    (3, 1): "Eb",
    (3, 0): "Ebm",
    (4, 1): "E",
    (4, 0): "Em",
    (5, 1): "F",
    (5, 0): "Fm",
    (6, 1): "F#",
    (6, 0): "F#m",
    (7, 1): "G",
    (7, 0): "Gm",
    (8, 1): "Ab",
    (8, 0): "Abm",
    (9, 1): "A",
    (9, 0): "Am",
    (10, 1): "Bb",
    (10, 0): "Bbm",
    (11, 1): "B",
    (11, 0): "Bm",
}


class SpotifyAdapter(StreamingProvider):
    """Spotify Web API adapter.

    Provides track search, metadata retrieval, and audio feature extraction
    via the Spotify Web API. Requires a Spotify Developer application with
    client_id and client_secret.
    """

    def __init__(self) -> None:
        self._client: Any = None
        self._authenticated = False

    def provider_name(self) -> str:
        return "spotify"

    def authenticate(self, credentials: dict[str, str]) -> bool:
        try:
            import spotipy
            from spotipy.oauth2 import SpotifyClientCredentials

            client_id = credentials.get("client_id", "")
            client_secret = credentials.get("client_secret", "")

            if not client_id or not client_secret:
                logger.error("Spotify credentials missing client_id or client_secret")
                return False

            auth_manager = SpotifyClientCredentials(
                client_id=client_id,
                client_secret=client_secret,
            )
            self._client = spotipy.Spotify(auth_manager=auth_manager)
            # Verify auth by making a lightweight API call
            self._client.categories(limit=1)
            self._authenticated = True
            logger.info("Spotify authentication successful")
            return True
        except Exception:
            logger.exception("Spotify authentication failed")
            self._authenticated = False
            return False

    def is_authenticated(self) -> bool:
        return self._authenticated and self._client is not None

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
            results = self._client.search(q=query, type="track", limit=limit, offset=offset)
            tracks_data = results.get("tracks", {})
            items = tracks_data.get("items", [])
            total = tracks_data.get("total", 0)

            tracks = [self._parse_track(item) for item in items]

            return StreamingSearchResult(
                tracks=tracks,
                total=total,
                offset=offset,
                limit=limit,
                has_more=(offset + limit) < total,
            )
        except Exception:
            logger.exception("Spotify search failed for query: %s", query)
            return StreamingSearchResult()

    def get_track_metadata(self, provider_track_id: str) -> StreamingTrackMetadata | None:
        self._require_auth()
        try:
            track = self._client.track(provider_track_id)
            if track is None:
                return None
            return self._parse_track(track)
        except Exception:
            logger.exception("Failed to fetch Spotify track: %s", provider_track_id)
            return None

    def get_preview_url(self, provider_track_id: str) -> str | None:
        self._require_auth()
        try:
            track = self._client.track(provider_track_id)
            return track.get("preview_url") if track else None
        except Exception:
            logger.exception("Failed to get preview URL for: %s", provider_track_id)
            return None

    def get_audio_features(self, provider_track_id: str) -> dict[str, float] | None:
        self._require_auth()
        try:
            features = self._client.audio_features([provider_track_id])
            if not features or features[0] is None:
                return None

            feat = features[0]
            result: dict[str, float] = {}

            if feat.get("tempo") is not None:
                result["bpm"] = round(float(feat["tempo"]), 1)
            if feat.get("energy") is not None:
                result["energy"] = round(float(feat["energy"]) * 100, 1)
            if feat.get("danceability") is not None:
                result["danceability"] = round(float(feat["danceability"]) * 100, 1)
            if feat.get("valence") is not None:
                result["valence"] = round(float(feat["valence"]) * 100, 1)
            if feat.get("loudness") is not None:
                result["loudness"] = round(float(feat["loudness"]), 2)

            key_num = feat.get("key")
            mode_num = feat.get("mode")
            if key_num is not None and mode_num is not None and key_num >= 0:
                key_str = _SPOTIFY_KEY_MAP.get((int(key_num), int(mode_num)))
                if key_str:
                    result["key"] = 0.0  # placeholder; actual key stored separately
                    result["_key_notation"] = 0.0  # see _key_notation property

            return result if result else None
        except Exception:
            logger.exception("Failed to get audio features for: %s", provider_track_id)
            return None

    def get_key_notation(self, provider_track_id: str) -> str | None:
        """Get the musical key as standard notation (e.g. 'Am', 'C')."""
        self._require_auth()
        try:
            features = self._client.audio_features([provider_track_id])
            if not features or features[0] is None:
                return None
            feat = features[0]
            key_num = feat.get("key")
            mode_num = feat.get("mode")
            if key_num is not None and mode_num is not None and key_num >= 0:
                return _SPOTIFY_KEY_MAP.get((int(key_num), int(mode_num)))
            return None
        except Exception:
            logger.exception("Failed to get key notation for: %s", provider_track_id)
            return None

    def _require_auth(self) -> None:
        if not self.is_authenticated():
            raise RuntimeError("Spotify adapter is not authenticated. Call authenticate() first.")

    @staticmethod
    def _parse_track(item: dict[str, Any]) -> StreamingTrackMetadata:
        artists = item.get("artists", [])
        artist_name = artists[0].get("name", "") if artists else ""
        album = item.get("album", {})
        album_name = album.get("name", "")
        images = album.get("images", [])
        artwork = images[0].get("url", "") if images else None
        release_date = album.get("release_date")

        external_ids = item.get("external_ids", {})
        isrc = external_ids.get("isrc")

        return StreamingTrackMetadata(
            provider="spotify",
            provider_track_id=item.get("id", ""),
            title=item.get("name", ""),
            artist=artist_name,
            album=album_name,
            duration_ms=item.get("duration_ms", 0),
            preview_url=item.get("preview_url"),
            artwork_url=artwork,
            isrc=isrc,
            release_date=release_date,
        )
