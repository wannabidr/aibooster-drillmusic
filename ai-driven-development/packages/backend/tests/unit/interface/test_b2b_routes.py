"""Tests for B2B API routes."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.application.dto.trend_dto import (
    AnonymizedTrackDTO,
    BpmBucketDTO,
    BpmDistributionResponseDTO,
    GenreTrendItemDTO,
    GenreTrendResponseDTO,
    TrendingTransitionDTO,
)
from src.application.use_cases.query_trends import QueryTrends
from src.domain.entities.api_client import ApiClient
from src.domain.value_objects.api_tier import ApiTier
from src.interface.middleware.api_key_middleware import get_api_client
from src.interface.routes.b2b_routes import _get_query_trends, router


def _make_client(tier: ApiTier = ApiTier.ENTERPRISE) -> ApiClient:
    return ApiClient(
        client_id=uuid.uuid4(),
        organization="Test Label",
        api_key_hash="hash",
        tier=tier,
        is_active=True,
        created_at=datetime.now(UTC),
    )


@pytest.fixture
def mock_query_trends() -> AsyncMock:
    return AsyncMock(spec=QueryTrends)


@pytest.fixture
def app(mock_query_trends: AsyncMock) -> FastAPI:
    app = FastAPI()
    client = _make_client(ApiTier.ENTERPRISE)
    app.dependency_overrides[get_api_client] = lambda: client
    app.dependency_overrides[_get_query_trends] = lambda: mock_query_trends
    app.include_router(router)
    return app


@pytest.fixture
def test_client(app: FastAPI) -> TestClient:
    return TestClient(app)


class TestGenreTrendsEndpoint:
    def test_returns_genre_trends(
        self, test_client: TestClient, mock_query_trends: AsyncMock
    ) -> None:
        mock_query_trends.genre_trends.return_value = GenreTrendResponseDTO(
            region="global",
            days=30,
            trends=[
                GenreTrendItemDTO(genre="techno", play_count=1000, change_pct=15.0),
            ],
        )

        resp = test_client.get("/api/v1/b2b/trends/genre?days=30")

        assert resp.status_code == 200
        data = resp.json()
        assert data["region"] == "global"
        assert len(data["trends"]) == 1
        assert data["trends"][0]["genre"] == "techno"

    def test_region_filter(
        self, test_client: TestClient, mock_query_trends: AsyncMock
    ) -> None:
        mock_query_trends.genre_trends.return_value = GenreTrendResponseDTO(
            region="eu", days=7, trends=[]
        )

        resp = test_client.get("/api/v1/b2b/trends/genre?region=eu&days=7")
        assert resp.status_code == 200

    def test_invalid_days(self, test_client: TestClient) -> None:
        resp = test_client.get("/api/v1/b2b/trends/genre?days=0")
        assert resp.status_code == 422

    def test_days_over_max(self, test_client: TestClient) -> None:
        resp = test_client.get("/api/v1/b2b/trends/genre?days=500")
        assert resp.status_code == 422


class TestBpmDistributionEndpoint:
    def test_returns_bpm_data(
        self, test_client: TestClient, mock_query_trends: AsyncMock
    ) -> None:
        mock_query_trends.bpm_distribution.return_value = BpmDistributionResponseDTO(
            genre="house",
            days=30,
            buckets=[BpmBucketDTO(bpm_min=120.0, bpm_max=125.0, count=200)],
            mean_bpm=122.5,
            median_bpm=122.0,
        )

        resp = test_client.get("/api/v1/b2b/trends/bpm?genre=house&days=30")

        assert resp.status_code == 200
        data = resp.json()
        assert data["mean_bpm"] == 122.5
        assert len(data["buckets"]) == 1


class TestTopTracksEndpoint:
    def test_returns_tracks(
        self, test_client: TestClient, mock_query_trends: AsyncMock
    ) -> None:
        mock_query_trends.top_tracks.return_value = [
            AnonymizedTrackDTO(
                fingerprint="fp_abc", genre="techno", bpm=138.0, key="Am", play_count=200
            ),
        ]

        resp = test_client.get("/api/v1/b2b/trends/tracks?genre=techno&days=30&limit=10")

        assert resp.status_code == 200
        data = resp.json()
        assert len(data["tracks"]) == 1
        assert data["tracks"][0]["fingerprint"] == "fp_abc"

    def test_basic_tier_rejected(
        self, app: FastAPI, mock_query_trends: AsyncMock
    ) -> None:
        basic_client = _make_client(ApiTier.BASIC)
        app.dependency_overrides[get_api_client] = lambda: basic_client
        client = TestClient(app)

        resp = client.get("/api/v1/b2b/trends/tracks?days=30")
        assert resp.status_code == 403
        assert "pro" in resp.json()["detail"].lower()

    def test_pro_tier_allowed(
        self, app: FastAPI, mock_query_trends: AsyncMock
    ) -> None:
        pro_client = _make_client(ApiTier.PRO)
        app.dependency_overrides[get_api_client] = lambda: pro_client
        mock_query_trends.top_tracks.return_value = []
        client = TestClient(app)

        resp = client.get("/api/v1/b2b/trends/tracks?days=30")
        assert resp.status_code == 200


class TestTrendingTransitionsEndpoint:
    def test_returns_transitions(
        self, test_client: TestClient, mock_query_trends: AsyncMock
    ) -> None:
        mock_query_trends.trending_transitions.return_value = [
            TrendingTransitionDTO(
                track_a_fingerprint="fp_a",
                track_b_fingerprint="fp_b",
                frequency=42,
                genre="house",
            ),
        ]

        resp = test_client.get("/api/v1/b2b/trends/transitions?days=30&limit=20")

        assert resp.status_code == 200
        data = resp.json()
        assert len(data["transitions"]) == 1
        assert data["transitions"][0]["frequency"] == 42

    def test_basic_tier_rejected(
        self, app: FastAPI, mock_query_trends: AsyncMock
    ) -> None:
        basic_client = _make_client(ApiTier.BASIC)
        app.dependency_overrides[get_api_client] = lambda: basic_client
        client = TestClient(app)

        resp = client.get("/api/v1/b2b/trends/transitions?days=30")
        assert resp.status_code == 403

    def test_pro_tier_rejected(
        self, app: FastAPI, mock_query_trends: AsyncMock
    ) -> None:
        pro_client = _make_client(ApiTier.PRO)
        app.dependency_overrides[get_api_client] = lambda: pro_client
        client = TestClient(app)

        resp = client.get("/api/v1/b2b/trends/transitions?days=30")
        assert resp.status_code == 403
        assert "enterprise" in resp.json()["detail"].lower()

    def test_enterprise_tier_allowed(
        self, test_client: TestClient, mock_query_trends: AsyncMock
    ) -> None:
        mock_query_trends.trending_transitions.return_value = []

        resp = test_client.get("/api/v1/b2b/trends/transitions?days=30")
        assert resp.status_code == 200


class TestApiKeyAuth:
    def test_missing_api_key_returns_401(self, mock_query_trends: AsyncMock) -> None:
        app = FastAPI()
        app.dependency_overrides[_get_query_trends] = lambda: mock_query_trends
        # Don't override get_api_client -- use the real one which needs X-Api-Key
        app.include_router(router)
        client = TestClient(app)

        resp = client.get("/api/v1/b2b/trends/genre?days=30")
        assert resp.status_code in (401, 501)
