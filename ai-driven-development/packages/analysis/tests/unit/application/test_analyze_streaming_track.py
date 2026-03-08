"""Unit tests for AnalyzeStreamingTrack use case."""

import uuid
from unittest.mock import MagicMock, patch

import pytest

from src.application.use_cases.analyze_streaming_track import AnalyzeStreamingTrack
from src.domain.entities.analysis_result import AnalysisResult
from src.domain.ports.streaming_provider import StreamingTrackMetadata
from src.domain.value_objects.bpm_value import BPMValue
from src.domain.value_objects.energy_profile import EnergyProfile
from src.domain.value_objects.key_signature import KeySignature


def _make_metadata(
    provider: str = "beatport",
    track_id: str = "123",
    bpm: float | None = None,
    key: str | None = None,
    genre: str = "",
) -> StreamingTrackMetadata:
    return StreamingTrackMetadata(
        provider=provider,
        provider_track_id=track_id,
        title="Test Track",
        artist="Test Artist",
        bpm=bpm,
        key=key,
        genre=genre,
        preview_url="https://example.com/preview.mp3",
    )


def _make_analysis_result() -> AnalysisResult:
    return AnalysisResult(
        id=uuid.uuid4(),
        track_id=uuid.uuid4(),
        bpm=BPMValue(126.0),
        key=KeySignature("Cm"),
        energy=EnergyProfile(overall=72.0),
    )


def _make_provider(
    name: str = "beatport",
    authenticated: bool = True,
    metadata: StreamingTrackMetadata | None = None,
    features: dict[str, float] | None = None,
    preview_url: str | None = "https://example.com/preview.mp3",
) -> MagicMock:
    provider = MagicMock()
    provider.provider_name.return_value = name
    provider.is_authenticated.return_value = authenticated
    provider.get_track_metadata.return_value = metadata
    provider.get_audio_features.return_value = features
    provider.get_preview_url.return_value = preview_url
    return provider


class TestAnalyzeStreamingTrack:
    def test_unknown_provider_raises(self):
        use_case = AnalyzeStreamingTrack(providers={}, analyzer=MagicMock())
        with pytest.raises(ValueError, match="Unknown streaming provider"):
            use_case.execute("nonexistent", "123")

    def test_unauthenticated_provider_raises(self):
        provider = _make_provider(authenticated=False)
        use_case = AnalyzeStreamingTrack(
            providers={"beatport": provider}, analyzer=MagicMock()
        )
        with pytest.raises(ValueError, match="not authenticated"):
            use_case.execute("beatport", "123")

    def test_track_not_found_raises(self):
        provider = _make_provider(metadata=None)
        use_case = AnalyzeStreamingTrack(
            providers={"beatport": provider}, analyzer=MagicMock()
        )
        with pytest.raises(ValueError, match="Track not found"):
            use_case.execute("beatport", "999")

    def test_beatport_bpm_and_key_from_provider(self):
        metadata = _make_metadata(bpm=128.0, key="Am", genre="Tech House")
        provider = _make_provider(metadata=metadata, preview_url=None)
        analyzer = MagicMock()

        use_case = AnalyzeStreamingTrack(
            providers={"beatport": provider}, analyzer=analyzer
        )
        result = use_case.execute("beatport", "123")

        assert result.bpm == 128.0
        assert result.bpm_source == "provider"
        assert result.key == "Am"
        assert result.key_source == "provider"
        assert result.genre == "Tech House"
        assert result.preview_analyzed is False
        analyzer.analyze.assert_not_called()

    def test_spotify_features_used(self):
        metadata = _make_metadata(provider="spotify")
        features = {"bpm": 130.0, "energy": 85.0, "danceability": 70.0}
        provider = _make_provider(
            name="spotify", metadata=metadata, features=features, preview_url=None
        )
        analyzer = MagicMock()

        use_case = AnalyzeStreamingTrack(
            providers={"spotify": provider}, analyzer=analyzer
        )
        result = use_case.execute("spotify", "123")

        assert result.bpm == 130.0
        assert result.bpm_source == "provider"
        assert result.energy == 85.0

    @patch("src.application.use_cases.analyze_streaming_track.urllib.request.urlretrieve")
    def test_preview_analysis_fallback(self, mock_urlretrieve):
        metadata = _make_metadata(provider="tidal")
        provider = _make_provider(name="tidal", metadata=metadata)

        analysis = _make_analysis_result()
        analyzer = MagicMock()
        analyzer.analyze.return_value = analysis

        use_case = AnalyzeStreamingTrack(
            providers={"tidal": provider}, analyzer=analyzer
        )
        result = use_case.execute("tidal", "123")

        assert result.bpm == 126.0
        assert result.bpm_source == "analysis"
        assert result.key == "Cm"
        assert result.key_source == "analysis"
        assert result.energy == 72.0
        assert result.preview_analyzed is True

    @patch("src.application.use_cases.analyze_streaming_track.urllib.request.urlretrieve")
    def test_provider_metadata_takes_priority_over_analysis(self, mock_urlretrieve):
        metadata = _make_metadata(bpm=128.0, key="Am")
        provider = _make_provider(metadata=metadata)

        analysis = _make_analysis_result()  # bpm=126.0, key=Cm
        analyzer = MagicMock()
        analyzer.analyze.return_value = analysis

        use_case = AnalyzeStreamingTrack(
            providers={"beatport": provider}, analyzer=analyzer
        )
        result = use_case.execute("beatport", "123")

        # Provider BPM/key should win over analysis
        assert result.bpm == 128.0
        assert result.bpm_source == "provider"
        assert result.key == "Am"
        assert result.key_source == "provider"
        # But energy comes from analysis
        assert result.energy == 72.0
        assert result.preview_analyzed is True

    def test_no_preview_url_skips_analysis(self):
        metadata = _make_metadata(provider="tidal")
        provider = _make_provider(name="tidal", metadata=metadata, preview_url=None)
        analyzer = MagicMock()

        use_case = AnalyzeStreamingTrack(
            providers={"tidal": provider}, analyzer=analyzer
        )
        result = use_case.execute("tidal", "123")

        assert result.preview_analyzed is False
        assert result.bpm is None
        analyzer.analyze.assert_not_called()

    @patch("src.application.use_cases.analyze_streaming_track.urllib.request.urlretrieve")
    def test_preview_analysis_failure_graceful(self, mock_urlretrieve):
        metadata = _make_metadata(provider="spotify")
        provider = _make_provider(name="spotify", metadata=metadata)

        analyzer = MagicMock()
        analyzer.analyze.side_effect = RuntimeError("Audio decode failed")

        use_case = AnalyzeStreamingTrack(
            providers={"spotify": provider}, analyzer=analyzer
        )
        result = use_case.execute("spotify", "123")

        # Should not raise, just skip preview analysis
        assert result.preview_analyzed is False
        assert result.bpm is None

    def test_result_includes_track_info(self):
        metadata = StreamingTrackMetadata(
            provider="beatport",
            provider_track_id="456",
            title="Deep Vibes",
            artist="DJ Test",
            genre="Deep House",
        )
        provider = _make_provider(metadata=metadata, preview_url=None)
        analyzer = MagicMock()

        use_case = AnalyzeStreamingTrack(
            providers={"beatport": provider}, analyzer=analyzer
        )
        result = use_case.execute("beatport", "456")

        assert result.provider == "beatport"
        assert result.provider_track_id == "456"
        assert result.title == "Deep Vibes"
        assert result.artist == "DJ Test"
        assert result.genre == "Deep House"
