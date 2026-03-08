"""Genre classifier port (abstract interface)."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class GenreClassifier(ABC):
    @abstractmethod
    def extract_features(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Extract genre-relevant audio features (MFCCs, spectral, rhythm).

        Returns a 1-D feature vector suitable for model input.
        """
        ...

    @abstractmethod
    def classify(self, features: np.ndarray) -> list[float]:
        """Run inference on extracted features, return 64-dim genre embedding."""
        ...
