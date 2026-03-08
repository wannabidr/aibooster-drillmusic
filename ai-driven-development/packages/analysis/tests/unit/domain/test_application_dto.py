"""Unit tests for application DTOs."""


class TestAnalysisRequest:
    def test_create_request(self):
        from src.application.dto.analysis_request import AnalysisRequest

        req = AnalysisRequest(file_path="/music/track.mp3")
        assert req.file_path == "/music/track.mp3"
        assert req.force_reanalyze is False

    def test_create_request_with_force(self):
        from src.application.dto.analysis_request import AnalysisRequest

        req = AnalysisRequest(file_path="/music/track.mp3", force_reanalyze=True)
        assert req.force_reanalyze is True


class TestAnalysisResponse:
    def test_create_response(self):
        from src.application.dto.analysis_response import AnalysisResponse

        resp = AnalysisResponse(
            track_id="abc-123",
            file_path="/music/track.mp3",
            bpm=128.0,
            bpm_confidence=0.95,
            key="Am",
            key_camelot="8A",
            key_confidence=0.85,
            energy_overall=75.0,
            energy_trajectory="build",
            fingerprint="AQAA...",
            cached=False,
        )
        assert resp.bpm == 128.0
        assert resp.key_camelot == "8A"
        assert resp.cached is False
