"""Integration tests for SQLite track repository."""

import uuid

import pytest
from src.domain.entities.analysis_result import AnalysisResult
from src.domain.entities.audio_track import AudioTrack
from src.domain.value_objects.bpm_value import BPMValue
from src.domain.value_objects.energy_profile import EnergyProfile
from src.domain.value_objects.key_signature import KeySignature


def _make_track(file_path: str = "/music/track.mp3", file_hash: str = "abc123") -> AudioTrack:
    return AudioTrack(id=uuid.uuid4(), file_path=file_path, file_hash=file_hash)


def _make_result(track: AudioTrack) -> AnalysisResult:
    return AnalysisResult(
        id=uuid.uuid4(),
        track_id=track.id,
        bpm=BPMValue(128.0),
        key=KeySignature("Am"),
        energy=EnergyProfile(overall=75.0),
        fingerprint="AQAA...",
    )


@pytest.fixture
def repo(tmp_path):
    from src.infrastructure.persistence.sqlite_track_repository import (
        SQLiteTrackRepository,
    )

    db_path = str(tmp_path / "test.db")
    return SQLiteTrackRepository(db_path)


class TestSQLiteTrackRepository:
    def test_save_and_find_by_id(self, repo):
        track = _make_track()
        repo.save(track)
        found = repo.find_by_id(track.id)
        assert found is not None
        assert found.id == track.id
        assert found.file_path == track.file_path

    def test_find_by_hash(self, repo):
        track = _make_track(file_hash="unique_hash_123")
        repo.save(track)
        found = repo.find_by_hash("unique_hash_123")
        assert found is not None
        assert found.file_hash == "unique_hash_123"

    def test_find_by_hash_miss(self, repo):
        found = repo.find_by_hash("nonexistent")
        assert found is None

    def test_find_all(self, repo):
        repo.save(_make_track(file_path="/a.mp3", file_hash="aaa"))
        repo.save(_make_track(file_path="/b.mp3", file_hash="bbb"))
        all_tracks = repo.find_all()
        assert len(all_tracks) == 2

    def test_delete(self, repo):
        track = _make_track()
        repo.save(track)
        repo.delete(track.id)
        found = repo.find_by_id(track.id)
        assert found is None

    def test_save_analysis_result(self, repo):
        track = _make_track()
        result = _make_result(track)
        repo.save(track)
        repo.save_analysis(result)
        found = repo.find_analysis_by_track_id(track.id)
        assert found is not None
        assert found.bpm.value == 128.0
        assert found.key.root == "A"
        assert found.key.mode == "minor"
        assert found.energy.overall == 75.0

    def test_update_analysis_result(self, repo):
        track = _make_track()
        result1 = _make_result(track)
        repo.save(track)
        repo.save_analysis(result1)

        result2 = AnalysisResult(
            id=result1.id,
            track_id=track.id,
            bpm=BPMValue(140.0),
            key=KeySignature("Cm"),
            energy=EnergyProfile(overall=85.0),
        )
        repo.save_analysis(result2)
        found = repo.find_analysis_by_track_id(track.id)
        assert found is not None
        assert found.bpm.value == 140.0
