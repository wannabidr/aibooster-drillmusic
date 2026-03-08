"""PyTorch MLP model for track pair compatibility scoring."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class TrackPairMLP(nn.Module):
    """Lightweight MLP for scoring track pair compatibility.

    Input: concatenated normalized features of two tracks (default 10 dims).
    Output: compatibility score in [0, 1].
    """

    def __init__(self, input_dim: int = 10, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_model(
    train_data: tuple[np.ndarray, np.ndarray],
    val_data: tuple[np.ndarray, np.ndarray],
    input_dim: int = 10,
    hidden_dim: int = 64,
    epochs: int = 50,
    lr: float = 0.005,
    batch_size: int = 64,
) -> tuple[TrackPairMLP, dict[str, list[float]]]:
    """Train the MLP model and return model + training history."""
    model = TrackPairMLP(input_dim=input_dim, hidden_dim=hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_x = torch.tensor(train_data[0], dtype=torch.float32)
    train_y = torch.tensor(train_data[1], dtype=torch.float32).unsqueeze(1)
    val_x = torch.tensor(val_data[0], dtype=torch.float32)
    val_y = torch.tensor(val_data[1], dtype=torch.float32).unsqueeze(1)

    train_loader = DataLoader(
        TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True
    )

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    for _epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch_x)
        history["train_loss"].append(epoch_loss / len(train_x))

        model.eval()
        with torch.no_grad():
            val_pred = model(val_x)
            val_loss = criterion(val_pred, val_y).item()
        history["val_loss"].append(val_loss)

    return model, history
