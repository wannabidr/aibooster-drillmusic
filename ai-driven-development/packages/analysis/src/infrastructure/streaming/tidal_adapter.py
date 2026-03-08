"""Tidal streaming adapter.

Implements StreamingProvider using the Tidal API via tidalapi.
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


class TidalAdapter(StreamingProvider):
    """Tidal API adapter.

    Provides track search and metadata retrieval via the Tidal API.
    Requires OAuth2 credentials (client_id and access_token).
    """

    def __init__(self) -> None:
        self._session: Any = None
        self._authenticated = False

    def provider_name(self) -> str:
        return "tidal"

    def authenticate(self, credentials: dict[str, str]) -> bool:
        try:
            import tidalapi

            self._session = tidalapi.Session()

            access_token = credentials.get("access_token")
            refresh_token = credentials.get("refresh_token")
            token_type = credentials.get("token_type", "Bearer")

            if access_token:
                # Use existing tokens
                self._session.load_oauth_session(
                    token_type=token_type,
                    access_token=access_token,
                    refresh_token=refresh_token or "",
                )
            else:
                logger.error("Tidal credentials require at least access_token")
                return False

            self._authenticated = self._session.check_login()
            if self._authenticated:
                logger.info("Tidal authentication successful")
            else:
                logger.warning("Tidal session login check failed")
            return self._authenticated
        except Exception:
            logger.exception("Tidal authentication failed")
            self._authenticated = False
            return False

    def is_authenticated(self) -> bool:
        return self._authenticated and self._session is not None

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
            results = self._session.search(query, models=[self._get_track_model()], limit=limit)
            tracks_data = results.get("tracks", []) if isinstance(results, dict) else []

            tracks = []
            for i, item in enumerate(tracks_data):
                if i < offset:
                    continue
                if len(tracks) >= limit:
                    break
                meta = self._parse_track(item)
                if meta:
                    tracks.append(meta)

            total = len(tracks_data)
            return StreamingSearchResult(
                tracks=tracks,
                total=total,
                offset=offset,
                limit=limit,
                has_more=(offset + limit) < total,
            )
        except Exception:
            logger.exception("Tidal search failed for query: %s", query)
            return StreamingSearchResult()

    def get_track_metadata(self, provider_track_id: str) -> StreamingTrackMetadata | None:
        self._require_auth()
        try:
            track = self._session.track(int(provider_track_id))
            if track is None:
                return None
            return self._parse_track(track)
        except Exception:
            logger.exception("Failed to fetch Tidal track: %s", provider_track_id)
            return None

    def get_preview_url(self, provider_track_id: str) -> str | None:
        self._require_auth()
        try:
            track = self._session.track(int(provider_track_id))
            if track is None:
                return None
            # tidalapi provides stream URL via get_url method
            stream_url = track.get_url()
            return stream_url
        except Exception:
            logger.exception("Failed to get Tidal preview URL for: %s", provider_track_id)
            return None

    def get_audio_features(self, provider_track_id: str) -> dict[str, float] | None:
        # Tidal does not expose audio features API like Spotify
        # Return None; analysis must be done locally via audio preview
        return None

    def _require_auth(self) -> None:
        if not self.is_authenticated():
            raise RuntimeError("Tidal adapter is not authenticated. Call authenticate() first.")

    def _get_track_model(self) -> Any:
        import tidalapi

        return tidalapi.media.Track

    @staticmethod
    def _parse_track(track: Any) -> StreamingTrackMetadata | None:
        try:
            artist_name = ""
            if hasattr(track, "artist") and track.artist:
                artist_name = track.artist.name if hasattr(track.artist, "name") else str(track.artist)
            elif hasattr(track, "artists") and track.artists:
                artist_name = track.artists[0].name if hasattr(track.artists[0], "name") else ""

            album_name = ""
            artwork_url = None
            if hasattr(track, "album") and track.album:
                album_name = track.album.name if hasattr(track.album, "name") else ""
                if hasattr(track.album, "image"):
                    try:
                        artwork_url = track.album.image(640)
                    except Exception:
                        pass

            duration_ms = 0
            if hasattr(track, "duration"):
                duration_ms = int(track.duration) * 1000

            isrc = getattr(track, "isrc", None)

            return StreamingTrackMetadata(
                provider="tidal",
                provider_track_id=str(track.id) if hasattr(track, "id") else "",
                title=track.name if hasattr(track, "name") else "",
                artist=artist_name,
                album=album_name,
                duration_ms=duration_ms,
                artwork_url=artwork_url,
                isrc=isrc,
            )
        except Exception:
            logger.exception("Failed to parse Tidal track")
            return None
