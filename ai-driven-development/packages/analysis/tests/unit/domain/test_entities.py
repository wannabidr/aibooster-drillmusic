"""Unit tests for Python domain entities."""

import uuid


class TestAudioTrack:
    """Tests for AudioTrack entity."""

    def test_create_audio_track(self):
        from src.domain.entities.audio_track import AudioTrack

        track = AudioTrack(
            id=uuid.uuid4(),
            file_path="/music/track.mp3",
            file_hash="abc123",
        )
        assert track.file_path == "/music/track.mp3"
        assert track.file_hash == "abc123"
        assert track.analysis_status == "pending"

    def test_track_equality_by_id(self):
        from src.domain.entities.audio_track import AudioTrack

        track_id = uuid.uuid4()
        track1 = AudioTrack(id=track_id, file_path="/a.mp3", file_hash="aaa")
        track2 = AudioTrack(id=track_id, file_path="/b.mp3", file_hash="bbb")
        assert track1 == track2

    def test_track_inequality(self):
        from src.domain.entities.audio_track import AudioTrack

        track1 = AudioTrack(id=uuid.uuid4(), file_path="/a.mp3", file_hash="aaa")
        track2 = AudioTrack(id=uuid.uuid4(), file_path="/a.mp3", file_hash="aaa")
        assert track1 != track2

    def test_mark_as_analyzed(self):
        from src.domain.entities.audio_track import AudioTrack

        track = AudioTrack(id=uuid.uuid4(), file_path="/a.mp3", file_hash="aaa")
        updated = track.mark_as_analyzed()
        assert updated.analysis_status == "analyzed"
        assert track.analysis_status == "pending"  # original unchanged

    def test_mark_as_failed(self):
        from src.domain.entities.audio_track import AudioTrack

        track = AudioTrack(id=uuid.uuid4(), file_path="/a.mp3", file_hash="aaa")
        updated = track.mark_as_failed("timeout")
        assert updated.analysis_status == "failed"
        assert updated.failure_reason == "timeout"


class TestAnalysisResult:
    """Tests for AnalysisResult entity."""

    def test_create_analysis_result(self):
        from src.domain.entities.analysis_result import AnalysisResult
        from src.domain.value_objects.bpm_value import BPMValue
        from src.domain.value_objects.energy_profile import EnergyProfile
        from src.domain.value_objects.key_signature import KeySignature

        track_id = uuid.uuid4()
        result = AnalysisResult(
            id=uuid.uuid4(),
            track_id=track_id,
            bpm=BPMValue(128.0),
            key=KeySignature("Am"),
            energy=EnergyProfile(overall=75.0),
        )
        assert result.track_id == track_id
        assert result.bpm.value == 128.0
        assert result.key.root == "A"
        assert result.energy.overall == 75.0

    def test_result_with_fingerprint(self):
        from src.domain.entities.analysis_result import AnalysisResult
        from src.domain.value_objects.bpm_value import BPMValue
        from src.domain.value_objects.energy_profile import EnergyProfile
        from src.domain.value_objects.key_signature import KeySignature

        result = AnalysisResult(
            id=uuid.uuid4(),
            track_id=uuid.uuid4(),
            bpm=BPMValue(140.0),
            key=KeySignature("Cm"),
            energy=EnergyProfile(overall=85.0),
            fingerprint="AQAA...",
        )
        assert result.fingerprint == "AQAA..."

    def test_result_equality_by_id(self):
        from src.domain.entities.analysis_result import AnalysisResult
        from src.domain.value_objects.bpm_value import BPMValue
        from src.domain.value_objects.energy_profile import EnergyProfile
        from src.domain.value_objects.key_signature import KeySignature

        result_id = uuid.uuid4()
        r1 = AnalysisResult(
            id=result_id,
            track_id=uuid.uuid4(),
            bpm=BPMValue(128.0),
            key=KeySignature("Am"),
            energy=EnergyProfile(overall=75.0),
        )
        r2 = AnalysisResult(
            id=result_id,
            track_id=uuid.uuid4(),
            bpm=BPMValue(140.0),
            key=KeySignature("Cm"),
            energy=EnergyProfile(overall=85.0),
        )
        assert r1 == r2
