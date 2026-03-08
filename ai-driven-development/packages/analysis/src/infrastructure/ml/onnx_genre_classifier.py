"""ONNX-based genre classifier adapter."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.domain.ports.genre_classifier import GenreClassifier
from src.infrastructure.ml.genre_feature_extractor import extract_genre_features


class OnnxGenreClassifier(GenreClassifier):
    """Genre classifier that uses an ONNX model for inference.

    If no model file is available, falls back to a random projection
    that produces deterministic 64-dim embeddings from features.
    This allows the pipeline to work end-to-end before the ML engineer
    provides the trained model.
    """

    EMBEDDING_DIM = 64
    FEATURE_DIM = 42

    def __init__(self, model_path: str | None = None, sample_rate: int = 44100) -> None:
        self._sample_rate = sample_rate
        self._session = None
        self._projection: np.ndarray | None = None

        if model_path and Path(model_path).exists():
            try:
                import onnxruntime as ort

                self._session = ort.InferenceSession(model_path)
            except ImportError:
                self._init_fallback_projection()
        else:
            self._init_fallback_projection()

    def _init_fallback_projection(self) -> None:
        rng = np.random.default_rng(seed=42)
        self._projection = rng.standard_normal((self.FEATURE_DIM, self.EMBEDDING_DIM)).astype(
            np.float32
        )

    def extract_features(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        return extract_genre_features(audio, sample_rate)

    def classify(self, features: np.ndarray) -> list[float]:
        if self._session is not None:
            input_name = self._session.get_inputs()[0].name
            input_data = features.reshape(1, -1).astype(np.float32)
            outputs = self._session.run(None, {input_name: input_data})
            embedding = outputs[0].flatten()
        else:
            embedding = self._fallback_classify(features)

        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding.tolist()

    def _fallback_classify(self, features: np.ndarray) -> np.ndarray:
        """Deterministic projection when no ONNX model is available."""
        assert self._projection is not None
        feat = features.flatten().astype(np.float32)
        if len(feat) < self.FEATURE_DIM:
            feat = np.pad(feat, (0, self.FEATURE_DIM - len(feat)))
        elif len(feat) > self.FEATURE_DIM:
            feat = feat[: self.FEATURE_DIM]
        return feat @ self._projection
