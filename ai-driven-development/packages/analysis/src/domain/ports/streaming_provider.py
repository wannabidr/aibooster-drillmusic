"""Streaming provider port (abstract interface).

Defines the contract for streaming service adapters (Beatport, Tidal, Spotify).
Domain layer -- ZERO external dependencies.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass(frozen=True)
class StreamingTrackMetadata:
    """Metadata for a track available on a streaming service."""

    provider: str
    provider_track_id: str
    title: str
    artist: str
    album: str = ""
    duration_ms: int = 0
    bpm: float | None = None
    key: str | None = None
    genre: str = ""
    preview_url: str | None = None
    artwork_url: str | None = None
    isrc: str | None = None
    release_date: str | None = None


@dataclass(frozen=True)
class StreamingSearchResult:
    """Paginated search result from a streaming provider."""

    tracks: list[StreamingTrackMetadata] = field(default_factory=list)
    total: int = 0
    offset: int = 0
    limit: int = 25
    has_more: bool = False


class StreamingProvider(ABC):
    """Abstract interface for streaming service integration.

    Each streaming service (Beatport Link, Tidal, Spotify) implements this port.
    Infrastructure adapters provide the concrete implementations.
    """

    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider identifier (e.g. 'beatport', 'tidal', 'spotify')."""
        ...

    @abstractmethod
    def authenticate(self, credentials: dict[str, str]) -> bool:
        """Authenticate with the streaming service.

        Args:
            credentials: Provider-specific auth credentials (client_id, client_secret,
                         access_token, refresh_token, etc.)

        Returns:
            True if authentication succeeded.
        """
        ...

    @abstractmethod
    def is_authenticated(self) -> bool:
        """Check if the current session is authenticated."""
        ...

    @abstractmethod
    def search_tracks(
        self,
        query: str,
        *,
        offset: int = 0,
        limit: int = 25,
    ) -> StreamingSearchResult:
        """Search for tracks on the streaming service.

        Args:
            query: Free-text search query.
            offset: Pagination offset.
            limit: Maximum results to return (max 50).

        Returns:
            Paginated search result with track metadata.
        """
        ...

    @abstractmethod
    def get_track_metadata(self, provider_track_id: str) -> StreamingTrackMetadata | None:
        """Fetch full metadata for a single track by its provider-specific ID.

        Returns:
            Track metadata, or None if not found.
        """
        ...

    @abstractmethod
    def get_preview_url(self, provider_track_id: str) -> str | None:
        """Get a streamable preview URL for a track (typically 30-90 seconds).

        Returns:
            Direct URL to the audio preview, or None if unavailable.
        """
        ...

    @abstractmethod
    def get_audio_features(self, provider_track_id: str) -> dict[str, float] | None:
        """Get provider-reported audio features (BPM, key, energy, etc.).

        Not all providers support this. Returns None if unavailable.

        Returns:
            Dict with keys like 'bpm', 'energy', 'danceability', 'valence', etc.
        """
        ...
