"""Script to train the recommendation model and export to ONNX."""

from pathlib import Path

from .dataset import generate_synthetic_dataset, split_dataset
from .model import train_model
from .onnx_export import export_to_onnx

MODEL_DIR = Path(__file__).parent
MODEL_PATH = MODEL_DIR / "recommendation_model.onnx"


def main() -> None:
    print("Generating synthetic dataset (10000 samples)...")
    features, targets = generate_synthetic_dataset(n_samples=10000, seed=42)

    print("Splitting into train/val/test...")
    train_data, val_data, test_data = split_dataset(features, targets)

    print("Training model...")
    model, history = train_model(
        train_data, val_data, input_dim=10, hidden_dim=64, epochs=100, lr=0.005
    )
    print(f"  Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"  Final val loss:   {history['val_loss'][-1]:.4f}")

    # Evaluate on test set
    import torch

    model.eval()
    test_f, test_t = test_data
    x = torch.tensor(test_f, dtype=torch.float32)
    y = torch.tensor(test_t, dtype=torch.float32)
    with torch.no_grad():
        preds = model(x).squeeze()
    accuracy = ((preds - y).abs() < 0.2).float().mean().item()
    print(f"  Test accuracy (within 0.2): {accuracy:.2%}")

    print(f"Exporting to ONNX: {MODEL_PATH}")
    export_to_onnx(model, MODEL_PATH, input_dim=10)
    print(f"  Model size: {MODEL_PATH.stat().st_size / 1024:.1f} KB")
    print("Done!")


if __name__ == "__main__":
    main()
