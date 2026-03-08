"""AnalyzeStreamingTrack use case -- on-demand analysis of streaming tracks.

Downloads a preview clip from the streaming provider and runs partial
audio analysis (BPM, key, energy) on it. Also merges provider-reported
metadata (Beatport BPM/key is considered authoritative for electronic music).
"""

from __future__ import annotations

import logging
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from src.domain.ports.audio_analyzer import AudioAnalyzer
from src.domain.ports.streaming_provider import StreamingProvider, StreamingTrackMetadata

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StreamingAnalysisResult:
    """Result of analyzing a streaming track's preview."""

    provider: str
    provider_track_id: str
    title: str
    artist: str
    bpm: float | None = None
    key: str | None = None
    energy: float | None = None
    genre: str = ""
    bpm_source: str = ""  # "provider" or "analysis"
    key_source: str = ""  # "provider" or "analysis"
    preview_analyzed: bool = False


class AnalyzeStreamingTrack:
    """Analyze a streaming track using preview audio and provider metadata.

    Strategy:
    1. Fetch provider metadata (may include BPM, key from Beatport/Spotify).
    2. Download preview audio clip.
    3. Run local audio analysis on preview.
    4. Merge results: prefer provider BPM/key for Beatport, local analysis otherwise.
    """

    def __init__(
        self,
        providers: dict[str, StreamingProvider],
        analyzer: AudioAnalyzer,
    ) -> None:
        self._providers = providers
        self._analyzer = analyzer

    def execute(
        self,
        provider_name: str,
        provider_track_id: str,
    ) -> StreamingAnalysisResult:
        """Analyze a streaming track.

        Args:
            provider_name: Which streaming provider ('beatport', 'tidal', 'spotify').
            provider_track_id: Provider-specific track ID.

        Returns:
            Analysis result with BPM, key, energy, and source attribution.

        Raises:
            ValueError: If provider is unknown or not authenticated.
        """
        provider = self._providers.get(provider_name)
        if provider is None:
            raise ValueError(f"Unknown streaming provider: {provider_name}")
        if not provider.is_authenticated():
            raise ValueError(f"Provider '{provider_name}' is not authenticated")

        # Step 1: Get provider metadata
        metadata = provider.get_track_metadata(provider_track_id)
        if metadata is None:
            raise ValueError(
                f"Track not found: {provider_name}/{provider_track_id}"
            )

        # Step 2: Get provider audio features
        provider_features = provider.get_audio_features(provider_track_id)

        # Step 3: Try to analyze preview audio
        preview_result = self._analyze_preview(provider, provider_track_id)

        # Step 4: Merge results
        return self._merge_results(metadata, provider_features, preview_result)

    def _analyze_preview(
        self,
        provider: StreamingProvider,
        provider_track_id: str,
    ) -> _PreviewAnalysis | None:
        """Download and analyze the preview audio clip."""
        preview_url = provider.get_preview_url(provider_track_id)
        if not preview_url:
            logger.info("No preview URL available for %s", provider_track_id)
            return None

        try:
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                tmp_path = tmp.name
                urllib.request.urlretrieve(preview_url, tmp_path)

            # Create a minimal AudioTrack for the analyzer
            import uuid

            from src.domain.entities.audio_track import AudioTrack

            temp_track = AudioTrack(
                id=uuid.uuid4(),
                file_path=tmp_path,
                file_hash=f"streaming-preview-{provider_track_id}",
                title="preview",
            )

            result = self._analyzer.analyze(temp_track)

            return _PreviewAnalysis(
                bpm=result.bpm.value,
                key=f"{result.key.root}{'m' if result.key.mode == 'minor' else ''}",
                energy=result.energy.overall,
            )
        except Exception:
            logger.exception("Preview analysis failed for %s", provider_track_id)
            return None
        finally:
            # Clean up temp file
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                pass

    @staticmethod
    def _merge_results(
        metadata: StreamingTrackMetadata,
        provider_features: dict[str, float] | None,
        preview: _PreviewAnalysis | None,
    ) -> StreamingAnalysisResult:
        """Merge provider metadata with local analysis.

        Priority:
        - Beatport BPM/key: authoritative for electronic music
        - Spotify audio features: good BPM, reasonable key
        - Local preview analysis: fallback
        """
        bpm: float | None = None
        bpm_source = ""
        key: str | None = None
        key_source = ""
        energy: float | None = None

        # Provider metadata (Beatport includes BPM/key directly)
        if metadata.bpm is not None:
            bpm = metadata.bpm
            bpm_source = "provider"
        if metadata.key is not None:
            key = metadata.key
            key_source = "provider"

        # Provider audio features (Spotify)
        if provider_features:
            if bpm is None and "bpm" in provider_features:
                bpm = provider_features["bpm"]
                bpm_source = "provider"
            if energy is None and "energy" in provider_features:
                energy = provider_features["energy"]

        # Local preview analysis (fallback)
        if preview:
            if bpm is None:
                bpm = preview.bpm
                bpm_source = "analysis"
            if key is None:
                key = preview.key
                key_source = "analysis"
            if energy is None:
                energy = preview.energy

        return StreamingAnalysisResult(
            provider=metadata.provider,
            provider_track_id=metadata.provider_track_id,
            title=metadata.title,
            artist=metadata.artist,
            bpm=bpm,
            key=key,
            energy=energy,
            genre=metadata.genre,
            bpm_source=bpm_source,
            key_source=key_source,
            preview_analyzed=preview is not None,
        )


@dataclass(frozen=True)
class _PreviewAnalysis:
    """Internal result from analyzing a preview audio clip."""

    bpm: float
    key: str
    energy: float
