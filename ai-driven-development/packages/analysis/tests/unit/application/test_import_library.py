"""Tests for ImportLibrary use case."""

import json
from unittest.mock import MagicMock

import pytest

from src.application.use_cases.import_library import ImportLibrary, ImportResult
from src.domain.ports.track_repository import TrackRepository
from src.infrastructure.parsers.parse_result import LibraryParseResult


@pytest.fixture
def mock_repo():
    repo = MagicMock(spec=TrackRepository)
    repo.find_by_hash.return_value = None  # No duplicates by default
    return repo


@pytest.fixture
def import_library(mock_repo):
    return ImportLibrary(mock_repo)


class TestImportLibrary:
    def test_imports_rekordbox_xml(self, import_library, mock_repo, tmp_path):
        # Create a minimal Rekordbox XML fixture
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<DJ_PLAYLISTS Version="1.0.0">
  <PRODUCT Name="rekordbox" Version="6.0"/>
  <COLLECTION Entries="1">
    <TRACK TrackID="1" Name="Test Track" Artist="Test Artist"
           TotalTime="300" AverageBpm="128.00" Tonality="Am"
           Location="file:///music/test.wav"/>
  </COLLECTION>
  <PLAYLISTS><NODE Type="0" Name="root" Count="0"/></PLAYLISTS>
</DJ_PLAYLISTS>"""
        xml_file = tmp_path / "library.xml"
        xml_file.write_text(xml_content)

        result = import_library.execute(str(xml_file))
        assert isinstance(result, ImportResult)
        assert result.source == "rekordbox"
        assert result.imported >= 1

    def test_detects_source_by_extension(self, import_library):
        assert import_library._detect_source("library.xml") == "rekordbox"
        assert import_library._detect_source("collection.nml") == "traktor"
        assert import_library._detect_source("tracks.csv") == "csv"
        assert import_library._detect_source("tracks.tsv") == "csv"
        assert import_library._detect_source("something.crate") == "serato"

    def test_skips_duplicates(self, import_library, mock_repo, tmp_path):
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<DJ_PLAYLISTS Version="1.0.0">
  <PRODUCT Name="rekordbox" Version="6.0"/>
  <COLLECTION Entries="1">
    <TRACK TrackID="1" Name="Dup" Artist="Artist"
           TotalTime="300" AverageBpm="128.00" Tonality="Am"
           Location="file:///music/dup.wav"/>
  </COLLECTION>
  <PLAYLISTS><NODE Type="0" Name="root" Count="0"/></PLAYLISTS>
</DJ_PLAYLISTS>"""
        xml_file = tmp_path / "library.xml"
        xml_file.write_text(xml_content)

        # Simulate existing track
        mock_repo.find_by_hash.return_value = MagicMock()

        result = import_library.execute(str(xml_file))
        assert result.skipped >= 1
        assert result.imported == 0

    def test_unsupported_source_raises(self, import_library):
        with pytest.raises(ValueError, match="Unsupported source"):
            import_library.execute("file.xyz", source="unknown")

    def test_explicit_source_overrides_detection(self, import_library, mock_repo, tmp_path):
        nml_content = """<?xml version="1.0" encoding="UTF-8"?>
<NML VERSION="19"><COLLECTION ENTRIES="0"></COLLECTION>
<PLAYLISTS><NODE TYPE="FOLDER" NAME="$ROOT"><SUBNODES COUNT="0"/></NODE></PLAYLISTS></NML>"""
        nml_file = tmp_path / "collection.nml"
        nml_file.write_text(nml_content)

        result = import_library.execute(str(nml_file), source="traktor")
        assert result.source == "traktor"

    def test_imports_mixcloud_json(self, import_library, mock_repo, tmp_path):
        data = {
            "name": "Test Mix",
            "sections": [
                {
                    "track": {"name": "Track A", "artist": {"name": "Artist A"}},
                    "start_time": 0,
                    "section_type": "track",
                }
            ],
        }
        json_file = tmp_path / "mix.json"
        json_file.write_text(json.dumps(data))

        result = import_library.execute(str(json_file), source="mixcloud")
        assert result.source == "mixcloud"
        assert result.imported == 1
        assert result.playlists == 1

    def test_imports_soundcloud_json(self, import_library, mock_repo, tmp_path):
        data = {
            "title": "SC Mix",
            "tracks": [
                {
                    "title": "Track B",
                    "user": {"full_name": "Artist B"},
                    "duration": 300000,
                }
            ],
        }
        json_file = tmp_path / "sc.json"
        json_file.write_text(json.dumps(data))

        result = import_library.execute(str(json_file), source="soundcloud")
        assert result.source == "soundcloud"
        assert result.imported == 1

    def test_imports_csv(self, import_library, mock_repo, tmp_path):
        csv_file = tmp_path / "tracks.csv"
        csv_file.write_text("title,artist,file_path\nTrack C,Artist C,/music/c.mp3\n")

        result = import_library.execute(str(csv_file))
        assert result.source == "csv"
        assert result.imported == 1

    def test_json_auto_detects_mixcloud(self, import_library, tmp_path):
        data = {"name": "Mix", "sections": []}
        json_file = tmp_path / "mix.json"
        json_file.write_text(json.dumps(data))
        assert import_library._detect_json_source(str(json_file)) == "mixcloud"

    def test_json_auto_detects_soundcloud(self, import_library, tmp_path):
        data = {"title": "Mix", "tracks": []}
        json_file = tmp_path / "sc.json"
        json_file.write_text(json.dumps(data))
        assert import_library._detect_json_source(str(json_file)) == "soundcloud"
