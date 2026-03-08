"""ONNX export and inference for recommendation model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from .model import TrackPairMLP


def export_to_onnx(model: TrackPairMLP, path: Path, input_dim: int = 10) -> None:
    """Export trained PyTorch model to ONNX format."""
    model.eval()
    dummy_input = torch.randn(1, input_dim)
    torch.onnx.export(
        model,
        dummy_input,
        str(path),
        input_names=["features"],
        output_names=["score"],
        dynamic_axes={"features": {0: "batch"}, "score": {0: "batch"}},
        opset_version=17,
    )


class ONNXRecommendationInference:
    """ONNX Runtime wrapper for recommendation model inference."""

    def __init__(self, model_path: Path):
        self._session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict compatibility scores for a batch of track pair features.

        Args:
            features: (N, 10) float32 array of concatenated track pair features.

        Returns:
            (N,) float32 array of compatibility scores in [0, 1].
        """
        if features.dtype != np.float32:
            features = features.astype(np.float32)
        outputs = self._session.run(None, {"features": features})
        return outputs[0].squeeze(axis=1)
