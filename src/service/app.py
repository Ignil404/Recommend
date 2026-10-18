"""FastAPI serving layer for the recommender models."""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query

from src.models.baseline import BaseRecommender
from src.models.collaborative import ItemItemKNNRecommender

DEFAULT_MODEL_K = 50

logger = logging.getLogger(__name__)


def data_dir() -> str:
    """Return the ratings directory (DATA_DIR env var or default)."""
    return os.environ.get("DATA_DIR", "data/ml-1m")


def model_path() -> Path:
    """Return the model artifact path (MODEL_PATH env or default)."""
    return Path(os.environ.get("MODEL_PATH", "artifacts/recommender.joblib"))


def load_ratings(data_dir_path: str | None = None) -> pd.DataFrame:
    """Load the MovieLens ratings file.

    Args:
        data_dir_path: Directory containing ratings.dat. Defaults to DATA_DIR.

    Returns:
        DataFrame with columns [user_id, movie_id, rating, timestamp]
    """
    directory = data_dir_path or data_dir()
    return pd.read_csv(
        f"{directory}/ratings.dat",
        sep="::",
        names=["user_id", "movie_id", "rating", "timestamp"],
        engine="python",
        encoding="latin-1",
    )


def load_movies(data_dir_path: str | None = None) -> pd.DataFrame:
    """Load the MovieLens movies file.

    Args:
        data_dir_path: Directory containing movies.dat. Defaults to DATA_DIR.

    Returns:
        DataFrame with columns [movie_id, title, genres]
    """
    directory = data_dir_path or data_dir()
    return pd.read_csv(
        f"{directory}/movies.dat",
        sep="::",
        names=["movie_id", "title", "genres"],
        engine="python",
        encoding="latin-1",
    )


def save_model(model: BaseRecommender, path: Path | str) -> Path:
    """Persist a fitted model to disk.

    Args:
        model: Fitted recommender
        path: Destination file path (parent directories are created)

    Returns:
        The path written
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, destination)
    logger.info("Model saved to %s", destination)
    return destination


def load_model(path: Path | str) -> BaseRecommender:
    """Load a persisted model from disk.

    Args:
        path: Artifact file path

    Returns:
        The fitted recommender
    """
    model = joblib.load(path)
    logger.info("Model loaded from %s", path)
    return model


def ensure_model(
    ratings: pd.DataFrame | None = None,
    path: Path | str | None = None,
) -> BaseRecommender:
    """Load the model artifact, fitting and saving it when missing.

    Args:
        ratings: Ratings used when the artifact must be created. Defaults to
            the DATA_DIR ratings file.
        path: Artifact path. Defaults to the MODEL_PATH env var.

    Returns:
        Fitted (or loaded) recommender
    """
    artifact = Path(path) if path is not None else model_path()
    if artifact.exists():
        return load_model(artifact)
    train = ratings if ratings is not None else load_ratings()
    model = ItemItemKNNRecommender(n_neighbors=DEFAULT_MODEL_K)
    model.fit(train)
    logger.info("Model fitted on %d ratings", len(train))
    save_model(model, artifact)
    return model


def create_app(
    model: BaseRecommender | None = None,
    ratings: pd.DataFrame | None = None,
    movies: pd.DataFrame | None = None,
    path: Path | str | None = None,
) -> FastAPI:
    """Build the FastAPI application around a recommender model.

    Args:
        model: Pre-fitted model. When None, the artifact at `path` is loaded,
            or fitted from ratings and saved there when the artifact is
            missing.
        ratings: Ratings used to fit the startup model. Defaults to MovieLens.
        movies: Movie metadata used to attach titles. Defaults to MovieLens.
        path: Model artifact path. Defaults to the MODEL_PATH env var.

    Returns:
        FastAPI application exposing /health and /recommendations
    """
    state: dict[str, object] = {"model": model, "movies": movies}

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if state["model"] is None:
            state["movies"] = (
                movies if movies is not None else load_movies()
            )
            state["model"] = ensure_model(ratings=ratings, path=path)
        yield
        state.clear()

    app = FastAPI(
        title="Recommender API",
        description="Item-item KNN recommender over MovieLens 1M",
        version="1.0.0",
        lifespan=lifespan,
    )

    def _require_model() -> BaseRecommender:
        fitted = state["model"]
        if fitted is None:
            raise HTTPException(status_code=503, detail="Model not fitted")
        return fitted

    @app.get("/health")
    def health() -> dict[str, object]:
        """Report service status and model size."""
        fitted = _require_model()
        user_map = getattr(fitted, "user_id_to_idx_", {})
        item_map = getattr(fitted, "item_id_to_idx_", {})
        return {
            "status": "ok",
            "model": type(fitted).__name__,
            "n_users": len(user_map),
            "n_items": len(item_map),
        }

    @app.get("/recommendations/{user_id}")
    def recommendations(
        user_id: int,
        k: int = Query(default=10, ge=1, le=100),
    ) -> dict[str, object]:
        """Return the top-K unseen items for a user.

        Users without training interactions are served from the popularity
        fallback and flagged with known_user=false.
        """
        fitted = _require_model()
        known_user = user_id in getattr(fitted, "user_id_to_idx_", {})
        item_ids = fitted.recommend(user_id, k=k)

        movie_table = state["movies"]
        titles: dict[int, str] = {}
        if isinstance(movie_table, pd.DataFrame) and not movie_table.empty:
            titles = dict(zip(movie_table["movie_id"], movie_table["title"]))

        items = [
            {"movie_id": item_id, "title": titles.get(item_id)}
            for item_id in item_ids
        ]
        return {
            "user_id": user_id,
            "known_user": known_user,
            "k": k,
            "items": items,
        }

    return app


def main() -> None:
    """Run the API server with uvicorn (HOST/PORT env vars)."""
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(create_app(), host=host, port=port)


if __name__ == "__main__":
    main()
