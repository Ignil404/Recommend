"""
Tests for the FastAPI recommender service.
"""

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.models.collaborative import ItemItemKNNRecommender
from src.service.app import (
    create_app,
    ensure_model,
    load_model,
    save_model,
)
from tests.test_collaborative import make_clustered_ratings


@pytest.fixture
def service_data():
    """Clustered ratings plus a matching movie metadata table."""
    ratings, held_out = make_clustered_ratings()
    movie_ids = sorted(ratings["movie_id"].unique())
    movies = pd.DataFrame(
        {
            "movie_id": movie_ids,
            "title": [f"Movie {movie_id}" for movie_id in movie_ids],
            "genres": ["Drama"] * len(movie_ids),
        }
    )
    return ratings, held_out, movies


@pytest.fixture
def client(service_data):
    """TestClient around an app with a pre-fitted KNN model."""
    ratings, _, movies = service_data
    model = ItemItemKNNRecommender(n_neighbors=5)
    model.fit(ratings)
    app = create_app(model=model, ratings=ratings, movies=movies)
    with TestClient(app) as test_client:
        yield test_client


class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_reports_model_and_sizes(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "ok"
        assert payload["model"] == "ItemItemKNNRecommender"
        assert payload["n_users"] == 40
        assert payload["n_items"] == 12


class TestRecommendationsEndpoint:
    """Tests for GET /recommendations/{user_id}."""

    def test_known_user_gets_ranked_items(self, client, service_data):
        _, held_out, _ = service_data
        # Each synthetic user has only one unseen item, so ask for one
        response = client.get("/recommendations/1", params={"k": 1})
        assert response.status_code == 200
        payload = response.json()
        assert payload["user_id"] == 1
        assert payload["known_user"] is True
        assert len(payload["items"]) == 1
        # The synthetic design requires the held-out item to rank first
        assert payload["items"][0]["movie_id"] == held_out[1]

    def test_titles_are_attached(self, client):
        response = client.get("/recommendations/2", params={"k": 2})
        items = response.json()["items"]
        for item in items:
            assert item["title"] == f"Movie {item['movie_id']}"

    def test_seen_items_are_excluded(self, client, service_data):
        ratings, _, _ = service_data
        seen = set(ratings[ratings["user_id"] == 3]["movie_id"].tolist())
        response = client.get("/recommendations/3", params={"k": 5})
        recommended = {item["movie_id"] for item in response.json()["items"]}
        assert not recommended & seen

    def test_cold_start_user_is_flagged(self, client):
        response = client.get("/recommendations/9999", params={"k": 3})
        assert response.status_code == 200
        payload = response.json()
        assert payload["known_user"] is False
        assert len(payload["items"]) == 3

    def test_k_is_validated(self, client):
        assert client.get("/recommendations/1", params={"k": 0}).status_code == 422
        assert (
            client.get("/recommendations/1", params={"k": 101}).status_code == 422
        )

    def test_unknown_user_id_is_rejected(self, client):
        assert client.get("/recommendations/abc").status_code == 422


class TestStartupFitting:
    """Tests for the default startup path (artifact load / model fitting)."""

    def test_model_is_fitted_on_startup(self, service_data, tmp_path):
        ratings, _, movies = service_data
        artifact = tmp_path / "model.joblib"
        app = create_app(ratings=ratings, movies=movies, path=artifact)
        with TestClient(app) as test_client:
            payload = test_client.get("/health").json()
            assert payload["model"] == "ItemItemKNNRecommender"
            assert payload["n_users"] == 40
        # First startup fits the model and persists the artifact
        assert artifact.exists()

    def test_startup_loads_existing_artifact(self, service_data, tmp_path):
        ratings, _, movies = service_data
        artifact = tmp_path / "model.joblib"
        # Pre-built artifact from other ratings must be preferred over `ratings`
        other, _ = make_clustered_ratings(n_users=10)
        ensure_model(ratings=other, path=artifact)

        app = create_app(ratings=ratings, movies=movies, path=artifact)
        with TestClient(app) as test_client:
            payload = test_client.get("/health").json()
            assert payload["n_users"] == 10

    def test_missing_ratings_fails_without_artifact(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("DATA_DIR", str(tmp_path / "no_such_dir"))
        app = create_app(path=tmp_path / "absent.joblib", ratings=None)
        with pytest.raises(FileNotFoundError):
            with TestClient(app):
                pass


class TestModelPersistence:
    """Tests for save_model / load_model / ensure_model."""

    def test_roundtrip_preserves_recommendations(self, service_data, tmp_path):
        ratings, held_out, _ = service_data
        model = ItemItemKNNRecommender(n_neighbors=5)
        model.fit(ratings)

        artifact = tmp_path / "knn.joblib"
        save_model(model, artifact)
        restored = load_model(artifact)

        assert type(restored) is ItemItemKNNRecommender
        for user_id in (1, 17, 40):
            assert restored.recommend(user_id, k=1) == model.recommend(
                user_id, k=1
            )
            assert restored.recommend(user_id, k=1) == [held_out[user_id]]

    def test_ensure_model_creates_missing_artifact(
        self, service_data, tmp_path
    ):
        ratings, _, _ = service_data
        artifact = tmp_path / "nested" / "model.joblib"
        ensure_model(ratings=ratings, path=artifact)
        assert artifact.exists()

    def test_ensure_model_reuses_existing_artifact(
        self, service_data, tmp_path
    ):
        ratings, _, _ = service_data
        artifact = tmp_path / "model.joblib"
        first = ensure_model(ratings=ratings, path=artifact)
        # A second call must load, not fit: pass empty ratings that would
        # crash a refit
        second = ensure_model(ratings=ratings.iloc[0:0], path=artifact)
        assert type(second) is type(first)
        assert second.recommend(1, k=1) == first.recommend(1, k=1)
