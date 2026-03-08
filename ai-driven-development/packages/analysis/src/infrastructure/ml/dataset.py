"""Synthetic dataset generation for track pair compatibility scoring."""

from __future__ import annotations

import numpy as np


def _camelot_distance(key_a: int, key_b: int) -> int:
    """Circular distance on Camelot wheel (1-12)."""
    diff = abs(key_a - key_b)
    return min(diff, 12 - diff)


def _compute_compatibility(
    bpm_a: float,
    key_num_a: int,
    key_mode_a: int,
    energy_a: float,
    genre_a: float,
    bpm_b: float,
    key_num_b: int,
    key_mode_b: int,
    energy_b: float,
    genre_b: float,
) -> float:
    """Rule-based compatibility score mirroring the TypeScript scorer."""
    # BPM compatibility (normalized 0-1)
    bpm_diff = abs(bpm_a - bpm_b)
    if bpm_diff < 0.02:
        bpm_score = 1.0
    elif bpm_diff < 0.05:
        bpm_score = 0.85
    elif bpm_diff < 0.1:
        bpm_score = 0.7
    else:
        bpm_score = max(0.0, 1.0 - bpm_diff * 3)

    # Key compatibility
    key_dist = _camelot_distance(key_num_a, key_num_b)
    same_mode = key_mode_a == key_mode_b
    if key_dist == 0 and same_mode:
        key_score = 1.0
    elif key_dist == 0:
        key_score = 0.85
    elif key_dist == 1 and same_mode:
        key_score = 0.9
    elif key_dist == 1:
        key_score = 0.75
    elif key_dist == 2 and same_mode:
        key_score = 0.6
    else:
        key_score = max(0.0, 1.0 - key_dist * 0.15)

    # Energy compatibility
    energy_diff = abs(energy_a - energy_b)
    energy_score = max(0.0, 1.0 - energy_diff * 2)

    # Genre compatibility
    genre_diff = abs(genre_a - genre_b)
    genre_score = max(0.0, 1.0 - genre_diff * 1.5)

    # Weighted combination matching TypeScript weights
    score = (
        bpm_score * 0.2
        + key_score * 0.25
        + energy_score * 0.25
        + genre_score * 0.15
        + 0.5 * 0.15  # history neutral
    )
    return float(np.clip(score, 0.0, 1.0))


def generate_synthetic_dataset(
    n_samples: int = 5000, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic track pair features and compatibility targets.

    Features per track (5): bpm_norm, key_num_norm, key_mode, energy_norm, genre_embed
    Total features: 10 (5 per track in pair)

    Returns:
        (features, targets) where features is (n_samples, 10) and targets is (n_samples,)
    """
    rng = np.random.default_rng(seed)

    features = np.zeros((n_samples, 10), dtype=np.float32)
    targets = np.zeros(n_samples, dtype=np.float32)

    for i in range(n_samples):
        bpm_a = rng.uniform(0.0, 1.0)
        key_num_a = rng.integers(1, 13)
        key_mode_a = rng.integers(0, 2)
        energy_a = rng.uniform(0.0, 1.0)
        genre_a = rng.uniform(0.0, 1.0)

        bpm_b = rng.uniform(0.0, 1.0)
        key_num_b = rng.integers(1, 13)
        key_mode_b = rng.integers(0, 2)
        energy_b = rng.uniform(0.0, 1.0)
        genre_b = rng.uniform(0.0, 1.0)

        features[i] = [
            bpm_a, key_num_a / 12.0, float(key_mode_a), energy_a, genre_a,
            bpm_b, key_num_b / 12.0, float(key_mode_b), energy_b, genre_b,
        ]

        score = _compute_compatibility(
            bpm_a, key_num_a, key_mode_a, energy_a, genre_a,
            bpm_b, key_num_b, key_mode_b, energy_b, genre_b,
        )
        # Add small noise so model learns to generalize
        noise = rng.normal(0, 0.02)
        targets[i] = float(np.clip(score + noise, 0.0, 1.0))

    return features, targets


def split_dataset(
    features: np.ndarray, targets: np.ndarray, train_ratio: float = 0.7, val_ratio: float = 0.15
) -> tuple[
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
]:
    """Split dataset into train/val/test sets."""
    n = len(features)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    return (
        (features[:train_end], targets[:train_end]),
        (features[train_end:val_end], targets[train_end:val_end]),
        (features[val_end:], targets[val_end:]),
    )
