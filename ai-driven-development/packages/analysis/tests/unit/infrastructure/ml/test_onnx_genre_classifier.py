"""Unit tests for OnnxGenreClassifier adapter."""

import numpy as np
import pytest

from src.infrastructure.ml.onnx_genre_classifier import OnnxGenreClassifier


class TestOnnxGenreClassifier:
    @pytest.fixture
    def classifier(self):
        return OnnxGenreClassifier(model_path=None)

    @pytest.fixture
    def sine_audio(self):
        sr = 44100
        t = np.linspace(0, 1.0, sr, endpoint=False)
        return np.sin(2 * np.pi * 440 * t).astype(np.float32), sr

    def test_extract_features_shape(self, classifier, sine_audio):
        audio, sr = sine_audio
        features = classifier.extract_features(audio, sr)
        assert features.shape == (42,)

    def test_classify_returns_64_dim(self, classifier, sine_audio):
        audio, sr = sine_audio
        features = classifier.extract_features(audio, sr)
        embedding = classifier.classify(features)
        assert len(embedding) == 64

    def test_classify_returns_list_of_floats(self, classifier, sine_audio):
        audio, sr = sine_audio
        features = classifier.extract_features(audio, sr)
        embedding = classifier.classify(features)
        assert isinstance(embedding, list)
        assert all(isinstance(x, float) for x in embedding)

    def test_embedding_is_normalized(self, classifier, sine_audio):
        audio, sr = sine_audio
        features = classifier.extract_features(audio, sr)
        embedding = classifier.classify(features)
        norm = np.linalg.norm(embedding)
        assert abs(norm - 1.0) < 0.01

    def test_deterministic_fallback(self, sine_audio):
        """Fallback projection is deterministic across instances."""
        audio, sr = sine_audio
        c1 = OnnxGenreClassifier(model_path=None)
        c2 = OnnxGenreClassifier(model_path=None)
        f = c1.extract_features(audio, sr)
        e1 = c1.classify(f)
        e2 = c2.classify(f)
        np.testing.assert_array_almost_equal(e1, e2)

    def test_different_audio_different_embeddings(self, classifier):
        sr = 44100
        t = np.linspace(0, 1.0, sr, endpoint=False)
        sine440 = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        sine880 = np.sin(2 * np.pi * 880 * t).astype(np.float32)

        f1 = classifier.extract_features(sine440, sr)
        f2 = classifier.extract_features(sine880, sr)
        e1 = classifier.classify(f1)
        e2 = classifier.classify(f2)
        assert not np.allclose(e1, e2, atol=0.01)

    def test_nonexistent_model_path_uses_fallback(self):
        classifier = OnnxGenreClassifier(model_path="/nonexistent/model.onnx")
        assert classifier._session is None
        assert classifier._projection is not None

    def test_embedding_dim_constant(self):
        assert OnnxGenreClassifier.EMBEDDING_DIM == 64

    def test_feature_dim_constant(self):
        assert OnnxGenreClassifier.FEATURE_DIM == 42
