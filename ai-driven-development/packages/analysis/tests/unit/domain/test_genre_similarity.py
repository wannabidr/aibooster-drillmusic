"""Unit tests for cosine similarity scoring between genre embeddings."""

import math

import pytest

from src.domain.services.genre_similarity import cosine_similarity, genre_score


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = [1.0, 2.0, 3.0]
        assert abs(cosine_similarity(v, v) - 1.0) < 1e-6

    def test_opposite_vectors(self):
        a = [1.0, 0.0, 0.0]
        b = [-1.0, 0.0, 0.0]
        assert abs(cosine_similarity(a, b) - (-1.0)) < 1e-6

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert abs(cosine_similarity(a, b)) < 1e-6

    def test_similar_vectors(self):
        a = [1.0, 2.0, 3.0]
        b = [1.1, 2.1, 3.1]
        sim = cosine_similarity(a, b)
        assert sim > 0.99

    def test_empty_vectors(self):
        assert cosine_similarity([], []) == 0.0

    def test_zero_vectors(self):
        assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0

    def test_different_lengths_returns_zero(self):
        assert cosine_similarity([1.0, 2.0], [1.0]) == 0.0

    def test_64_dim_vectors(self):
        """Test with actual embedding dimensionality."""
        a = [float(i) / 64 for i in range(64)]
        b = [float(i + 1) / 64 for i in range(64)]
        sim = cosine_similarity(a, b)
        assert -1.0 <= sim <= 1.0


class TestGenreScore:
    def test_identical_embeddings_max_score(self):
        v = [0.1] * 64
        score = genre_score(v, v)
        assert abs(score - 1.0) < 1e-6

    def test_opposite_embeddings_min_score(self):
        a = [1.0, 0.0, 0.0]
        b = [-1.0, 0.0, 0.0]
        score = genre_score(a, b)
        assert abs(score - 0.0) < 1e-6

    def test_orthogonal_embeddings_mid_score(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        score = genre_score(a, b)
        assert abs(score - 0.5) < 1e-6

    def test_score_range(self):
        """Genre score should always be in [0, 1]."""
        import random

        random.seed(42)
        for _ in range(100):
            a = [random.gauss(0, 1) for _ in range(64)]
            b = [random.gauss(0, 1) for _ in range(64)]
            s = genre_score(a, b)
            assert 0.0 <= s <= 1.0

    def test_similar_genres_high_score(self):
        techno_a = [0.8, 0.9, 0.1, 0.2] + [0.5] * 60
        techno_b = [0.85, 0.88, 0.12, 0.22] + [0.5] * 60
        jazz = [-0.5, -0.3, 0.9, 0.8] + [-0.5] * 60

        score_same = genre_score(techno_a, techno_b)
        score_diff = genre_score(techno_a, jazz)
        assert score_same > score_diff

    def test_symmetry(self):
        a = [1.0, 2.0, 3.0]
        b = [4.0, 5.0, 6.0]
        assert abs(genre_score(a, b) - genre_score(b, a)) < 1e-10
