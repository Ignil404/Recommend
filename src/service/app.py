"""FastAPI serving layer for the recommender models."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, Query

from src.models.baseline import BaseRecommender
from src.models.collaborative import ItemItemKNNRecommender

DATA_DIR = "data/ml-1m"
DEFAULT_MODEL_K = 50

logger = logging.getLogger(__name__)


def load_ratings(data_dir: str = DATA_DIR) -> pd.DataFrame:
    """Load the MovieLens ratings file.

    Args:
        data_dir: Directory containing ratings.dat

    Returns:
        DataFrame with columns [user_id, movie_id, rating, timestamp]
    """
    return pd.read_csv(
        f"{data_dir}/ratings.dat",
        sep="::",
        names=["user_id", "movie_id", "rating", "timestamp"],
        engine="python",
        encoding="latin-1",
    )


def load_movies(data_dir: str = DATA_DIR) -> pd.DataFrame:
    """Load the MovieLens movies file.

    Args:
        data_dir: Directory containing movies.dat

    Returns:
        DataFrame with columns [movie_id, title, genres]
    """
    return pd.read_csv(
        f"{data_dir}/movies.dat",
        sep="::",
        names=["movie_id", "title", "genres"],
        engine="python",
        encoding="latin-1",
    )


def create_app(
    model: BaseRecommender | None = None,
    ratings: pd.DataFrame | None = None,
    movies: pd.DataFrame | None = None,
) -> FastAPI:
    """Build the FastAPI application around a recommender model.

    Args:
        model: Pre-fitted model. When None, an item-item KNN model is fitted
            on the full ratings file during application startup.
        ratings: Ratings used to fit the startup model. Defaults to MovieLens.
        movies: Movie metadata used to attach titles. Defaults to MovieLens.

    Returns:
        FastAPI application exposing /health and /recommendations
    """
    state: dict[str, object] = {"model": model, "movies": movies}

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if state["model"] is None:
            train = ratings if ratings is not None else load_ratings()
            state["movies"] = (
                movies if movies is not None else load_movies()
            )
            fitted = ItemItemKNNRecommender(n_neighbors=DEFAULT_MODEL_K)
            fitted.fit(train)
            state["model"] = fitted
            logger.info("Model fitted on %d ratings", len(train))
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
    """Run the API server with uvicorn."""
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    uvicorn.run(create_app(), host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
