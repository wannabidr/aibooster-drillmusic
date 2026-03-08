"""Tests for ML recommendation model training, ONNX export, and inference."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.infrastructure.ml.dataset import generate_synthetic_dataset, split_dataset
from src.infrastructure.ml.model import TrackPairMLP, train_model
from src.infrastructure.ml.onnx_export import export_to_onnx, ONNXRecommendationInference


class TestSyntheticDataset:
    def test_generate_dataset_shape(self):
        features, targets = generate_synthetic_dataset(n_samples=500)
        assert features.shape == (500, 10)  # 5 features per track * 2 tracks
        assert targets.shape == (500,)

    def test_targets_in_valid_range(self):
        _, targets = generate_synthetic_dataset(n_samples=200)
        assert np.all(targets >= 0.0)
        assert np.all(targets <= 1.0)

    def test_features_normalized(self):
        features, _ = generate_synthetic_dataset(n_samples=200)
        # BPM features should be normalized 0-1
        assert np.all(features >= -0.1)
        assert np.all(features <= 1.1)

    def test_split_dataset_proportions(self):
        features, targets = generate_synthetic_dataset(n_samples=1000)
        train, val, test = split_dataset(features, targets)
        train_f, train_t = train
        val_f, val_t = val
        test_f, test_t = test
        total = len(train_f) + len(val_f) + len(test_f)
        assert total == 1000
        assert len(train_f) == 700  # 70%
        assert len(val_f) == 150  # 15%
        assert len(test_f) == 150  # 15%

    def test_compatible_pairs_score_high(self):
        """Tracks with same BPM/key/energy should have high compatibility."""
        features, targets = generate_synthetic_dataset(n_samples=2000, seed=42)
        # Find pairs where features are very similar (small diff)
        bpm_diff = np.abs(features[:, 0] - features[:, 5])
        key_diff = np.abs(features[:, 1] - features[:, 6])
        similar_mask = (bpm_diff < 0.05) & (key_diff < 0.05)
        if np.any(similar_mask):
            avg_score = targets[similar_mask].mean()
            assert avg_score > 0.5


class TestTrackPairMLP:
    def test_model_output_shape(self):
        import torch

        model = TrackPairMLP(input_dim=10, hidden_dim=32)
        x = torch.randn(16, 10)
        out = model(x)
        assert out.shape == (16, 1)

    def test_model_output_range(self):
        """Output should be in [0, 1] due to sigmoid."""
        import torch

        model = TrackPairMLP(input_dim=10, hidden_dim=32)
        x = torch.randn(100, 10)
        out = model(x)
        assert torch.all(out >= 0.0)
        assert torch.all(out <= 1.0)

    def test_train_model_reduces_loss(self):
        features, targets = generate_synthetic_dataset(n_samples=500, seed=42)
        train_data, val_data, _ = split_dataset(features, targets)
        model, history = train_model(
            train_data, val_data, input_dim=10, hidden_dim=32, epochs=20, lr=0.01
        )
        assert history["train_loss"][-1] < history["train_loss"][0]

    def test_model_accuracy_above_70_percent(self):
        """Quality gate: accuracy on test set must exceed 70%."""
        features, targets = generate_synthetic_dataset(n_samples=5000, seed=42)
        train_data, val_data, test_data = split_dataset(features, targets)
        model, _ = train_model(
            train_data, val_data, input_dim=10, hidden_dim=64, epochs=50, lr=0.005
        )
        import torch

        model.eval()
        test_f, test_t = test_data
        x = torch.tensor(test_f, dtype=torch.float32)
        y = torch.tensor(test_t, dtype=torch.float32)
        with torch.no_grad():
            preds = model(x).squeeze()
        # Accuracy: predictions within 0.2 of target
        accuracy = ((preds - y).abs() < 0.2).float().mean().item()
        assert accuracy > 0.70, f"Model accuracy {accuracy:.2%} below 70% threshold"


class TestONNXExport:
    def test_export_creates_file(self):
        import torch

        model = TrackPairMLP(input_dim=10, hidden_dim=32)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.onnx"
            export_to_onnx(model, path, input_dim=10)
            assert path.exists()
            assert path.stat().st_size > 0

    def test_onnx_inference_produces_scores(self):
        import torch

        model = TrackPairMLP(input_dim=10, hidden_dim=32)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.onnx"
            export_to_onnx(model, path, input_dim=10)
            inference = ONNXRecommendationInference(path)
            features = np.random.rand(50, 10).astype(np.float32)
            scores = inference.predict(features)
            assert scores.shape == (50,)
            assert np.all(scores >= 0.0)
            assert np.all(scores <= 1.0)

    def test_onnx_inference_matches_pytorch(self):
        """ONNX output should closely match PyTorch output."""
        import torch

        model = TrackPairMLP(input_dim=10, hidden_dim=32)
        model.eval()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.onnx"
            export_to_onnx(model, path, input_dim=10)
            inference = ONNXRecommendationInference(path)

            test_input = np.random.rand(20, 10).astype(np.float32)

            # PyTorch prediction
            with torch.no_grad():
                pt_out = model(torch.tensor(test_input)).squeeze().numpy()

            # ONNX prediction
            onnx_out = inference.predict(test_input)

            np.testing.assert_allclose(pt_out, onnx_out, atol=1e-5)

    def test_onnx_inference_performance(self):
        """Quality gate: inference < 50ms for 1000 candidates."""
        import time
        import torch

        model = TrackPairMLP(input_dim=10, hidden_dim=64)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.onnx"
            export_to_onnx(model, path, input_dim=10)
            inference = ONNXRecommendationInference(path)

            features = np.random.rand(1000, 10).astype(np.float32)
            # Warmup
            inference.predict(features)

            start = time.perf_counter()
            scores = inference.predict(features)
            elapsed_ms = (time.perf_counter() - start) * 1000

            assert scores.shape == (1000,)
            assert elapsed_ms < 50, f"Inference took {elapsed_ms:.1f}ms, exceeds 50ms limit"
