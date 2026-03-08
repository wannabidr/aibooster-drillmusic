"""Tests for ChromaprintFingerprinter (mocked fpcalc)."""

import hashlib
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from src.infrastructure.fingerprint.chromaprint_fingerprinter import ChromaprintFingerprinter


@pytest.fixture
def fingerprinter():
    return ChromaprintFingerprinter()


class TestChromaprintFingerprinter:
    @patch("src.infrastructure.fingerprint.chromaprint_fingerprinter.subprocess.run")
    def test_generate_fingerprint(self, mock_run, fingerprinter):
        mock_run.return_value = MagicMock(
            stdout="DURATION=120\nFINGERPRINT=AQAAEklKaUmSRCk\n",
            returncode=0,
        )

        result = fingerprinter.generate_fingerprint("/path/to/track.wav")
        assert result == "AQAAEklKaUmSRCk"
        mock_run.assert_called_once()

    @patch("src.infrastructure.fingerprint.chromaprint_fingerprinter.subprocess.run")
    def test_fpcalc_not_found_raises(self, mock_run, fingerprinter):
        mock_run.side_effect = FileNotFoundError()

        with pytest.raises(RuntimeError, match="fpcalc not found"):
            fingerprinter.generate_fingerprint("/path/to/track.wav")

    @patch("src.infrastructure.fingerprint.chromaprint_fingerprinter.subprocess.run")
    def test_fpcalc_no_fingerprint_raises(self, mock_run, fingerprinter):
        mock_run.return_value = MagicMock(stdout="DURATION=120\n", returncode=0)

        with pytest.raises(RuntimeError, match="no fingerprint"):
            fingerprinter.generate_fingerprint("/path/to/track.wav")

    def test_compute_file_hash(self, fingerprinter):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(b"fake audio data for hashing")
            f.flush()

            result = fingerprinter.compute_file_hash(f.name)

            expected = hashlib.sha256(b"fake audio data for hashing").hexdigest()
            assert result == expected

    def test_same_file_same_hash(self, fingerprinter):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(b"identical content")
            f.flush()

            hash1 = fingerprinter.compute_file_hash(f.name)
            hash2 = fingerprinter.compute_file_hash(f.name)
            assert hash1 == hash2
