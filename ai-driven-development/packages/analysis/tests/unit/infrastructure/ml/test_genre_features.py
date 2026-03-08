"""Unit tests for genre feature extraction."""

import numpy as np
import pytest

from src.infrastructure.ml.genre_feature_extractor import extract_genre_features


class TestGenreFeatureExtraction:
    """Tests for audio feature extraction pipeline."""

    @pytest.fixture
    def sine_audio(self):
        """1 second of 440Hz sine wave at 44100Hz."""
        sr = 44100
        t = np.linspace(0, 1.0, sr, endpoint=False)
        return np.sin(2 * np.pi * 440 * t).astype(np.float32), sr

    @pytest.fixture
    def noise_audio(self):
        """1 second of white noise."""
        rng = np.random.default_rng(42)
        sr = 44100
        return rng.standard_normal(sr).astype(np.float32) * 0.5, sr

    @pytest.fixture
    def silence_audio(self):
        """1 second of silence."""
        sr = 44100
        return np.zeros(sr, dtype=np.float32), sr

    def test_output_shape(self, sine_audio):
        audio, sr = sine_audio
        features = extract_genre_features(audio, sr)
        assert features.shape == (42,)

    def test_output_dtype(self, sine_audio):
        audio, sr = sine_audio
        features = extract_genre_features(audio, sr)
        assert features.dtype == np.float32

    def test_no_nans(self, sine_audio):
        audio, sr = sine_audio
        features = extract_genre_features(audio, sr)
        assert not np.any(np.isnan(features))

    def test_no_infs(self, sine_audio):
        audio, sr = sine_audio
        features = extract_genre_features(audio, sr)
        assert not np.any(np.isinf(features))

    def test_different_signals_produce_different_features(self, sine_audio, noise_audio):
        sine, sr1 = sine_audio
        noise, sr2 = noise_audio
        f_sine = extract_genre_features(sine, sr1)
        f_noise = extract_genre_features(noise, sr2)
        assert not np.allclose(f_sine, f_noise)

    def test_mfcc_components(self, sine_audio):
        """First 20 features are MFCCs."""
        audio, sr = sine_audio
        features = extract_genre_features(audio, sr)
        mfccs = features[:20]
        # MFCCs should have nonzero values for tonal signal
        assert np.any(mfccs != 0.0)

    def test_spectral_centroid(self, sine_audio):
        """Feature index 20 is spectral centroid."""
        audio, sr = sine_audio
        features = extract_genre_features(audio, sr)
        centroid = features[20]
        # 440Hz sine should have centroid near 440Hz
        assert 200 < centroid < 1000

    def test_spectral_rolloff(self, sine_audio):
        """Feature index 21 is spectral rolloff."""
        audio, sr = sine_audio
        features = extract_genre_features(audio, sr)
        rolloff = features[21]
        # Rolloff should be positive
        assert rolloff > 0

    def test_zero_crossing_rate(self, sine_audio, noise_audio):
        """Feature index 23 is ZCR. Noise has higher ZCR than sine."""
        sine, sr = sine_audio
        noise, sr2 = noise_audio
        zcr_sine = extract_genre_features(sine, sr)[23]
        zcr_noise = extract_genre_features(noise, sr2)[23]
        assert zcr_noise > zcr_sine

    def test_rms_energy(self, sine_audio, silence_audio):
        """Feature index 24 is RMS energy."""
        sine, sr1 = sine_audio
        silence, sr2 = silence_audio
        rms_sine = extract_genre_features(sine, sr1)[24]
        rms_silence = extract_genre_features(silence, sr2)[24]
        assert rms_sine > rms_silence

    def test_tempo_histogram(self, sine_audio):
        """Features 25-32 are tempo histogram bins."""
        audio, sr = sine_audio
        features = extract_genre_features(audio, sr)
        tempo_hist = features[25:33]
        assert tempo_hist.shape == (8,)
        # Histogram should be non-negative
        assert np.all(tempo_hist >= 0)

    def test_onset_stats(self, sine_audio):
        """Features 33-36 are onset strength stats."""
        audio, sr = sine_audio
        features = extract_genre_features(audio, sr)
        onset = features[33:37]
        assert onset.shape == (4,)

    def test_spectral_contrast(self, sine_audio):
        """Features 37-41 are spectral contrast bands."""
        audio, sr = sine_audio
        features = extract_genre_features(audio, sr)
        contrast = features[37:42]
        assert contrast.shape == (5,)

    def test_very_short_audio(self):
        """Audio shorter than a frame should return zeros."""
        audio = np.zeros(100, dtype=np.float32)
        features = extract_genre_features(audio, 44100)
        assert features.shape == (42,)
        assert np.all(features == 0.0)

    def test_deterministic(self, sine_audio):
        """Same input should produce same output."""
        audio, sr = sine_audio
        f1 = extract_genre_features(audio, sr)
        f2 = extract_genre_features(audio, sr)
        np.testing.assert_array_equal(f1, f2)
