"""SearchStreaming use case -- unified search across streaming providers."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.domain.ports.streaming_provider import (
    StreamingProvider,
    StreamingSearchResult,
    StreamingTrackMetadata,
)


@dataclass(frozen=True)
class UnifiedSearchResult:
    """Combined search results from multiple streaming providers and local library."""

    results: list[StreamingTrackMetadata] = field(default_factory=list)
    total_by_provider: dict[str, int] = field(default_factory=dict)
    query: str = ""


class SearchStreaming:
    """Search across one or more streaming providers.

    Supports querying individual providers or all at once,
    with results merged and deduplicated by ISRC when possible.
    """

    def __init__(self, providers: dict[str, StreamingProvider]) -> None:
        self._providers = providers

    def execute(
        self,
        query: str,
        *,
        provider_names: list[str] | None = None,
        offset: int = 0,
        limit: int = 25,
    ) -> UnifiedSearchResult:
        """Search for tracks across streaming providers.

        Args:
            query: Search query string.
            provider_names: List of provider names to search. If None, search all.
            offset: Pagination offset.
            limit: Max results per provider.

        Returns:
            Unified search result with tracks from all queried providers.
        """
        if not query.strip():
            return UnifiedSearchResult(query=query)

        targets = self._resolve_providers(provider_names)
        all_tracks: list[StreamingTrackMetadata] = []
        totals: dict[str, int] = {}

        for name, provider in targets.items():
            if not provider.is_authenticated():
                continue
            result: StreamingSearchResult = provider.search_tracks(
                query, offset=offset, limit=limit
            )
            all_tracks.extend(result.tracks)
            totals[name] = result.total

        # Deduplicate by ISRC across providers (prefer Beatport > Tidal > Spotify)
        deduped = self._deduplicate(all_tracks)

        return UnifiedSearchResult(
            results=deduped,
            total_by_provider=totals,
            query=query,
        )

    def _resolve_providers(
        self, names: list[str] | None
    ) -> dict[str, StreamingProvider]:
        if names is None:
            return dict(self._providers)
        return {n: self._providers[n] for n in names if n in self._providers}

    @staticmethod
    def _deduplicate(
        tracks: list[StreamingTrackMetadata],
    ) -> list[StreamingTrackMetadata]:
        """Remove duplicate tracks across providers using ISRC matching.

        Priority order: beatport > tidal > spotify (Beatport has best DJ metadata).
        """
        _PROVIDER_PRIORITY = {"beatport": 0, "tidal": 1, "spotify": 2}

        seen_isrcs: dict[str, StreamingTrackMetadata] = {}
        no_isrc: list[StreamingTrackMetadata] = []

        for track in tracks:
            if not track.isrc:
                no_isrc.append(track)
                continue

            existing = seen_isrcs.get(track.isrc)
            if existing is None:
                seen_isrcs[track.isrc] = track
            else:
                # Keep the one with higher priority (lower number)
                existing_pri = _PROVIDER_PRIORITY.get(existing.provider, 99)
                new_pri = _PROVIDER_PRIORITY.get(track.provider, 99)
                if new_pri < existing_pri:
                    seen_isrcs[track.isrc] = track

        return list(seen_isrcs.values()) + no_isrc
