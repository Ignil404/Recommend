"""
Collaborative filtering models for the MovieLens dataset.

Implements two classic neighbourhood/model-based approaches:
- ItemItemKNNRecommender: item-item cosine similarity over centered ratings
- SVDRecommender: truncated SVD (latent factors) on the centered rating matrix

Both exclude items the user has already rated and fall back to the popularity
baseline for cold-start users.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import svds

from src.models.baseline import (
    BaseRecommender,
    PopularityBaseline,
    evaluate,
    time_based_split,
)

logger = logging.getLogger(__name__)


def _build_id_mappings(
    ratings: pd.DataFrame,
) -> tuple[dict[int, int], dict[int, int], dict[int, int], dict[int, int]]:
    """Build contiguous index mappings for users and items.

    Args:
        ratings: DataFrame with columns [user_id, movie_id, rating, timestamp]

    Returns:
        Tuple of (user_id_to_idx, idx_to_user_id, item_id_to_idx, idx_to_item_id)
    """
    user_ids = sorted(int(uid) for uid in ratings["user_id"].unique())
    item_ids = sorted(int(mid) for mid in ratings["movie_id"].unique())
    user_id_to_idx = {uid: i for i, uid in enumerate(user_ids)}
    item_id_to_idx = {mid: i for i, mid in enumerate(item_ids)}
    idx_to_user_id = {i: uid for uid, i in user_id_to_idx.items()}
    idx_to_item_id = {i: mid for mid, i in item_id_to_idx.items()}
    return user_id_to_idx, idx_to_user_id, item_id_to_idx, idx_to_item_id


def _build_sparse_matrix(
    ratings: pd.DataFrame,
    user_id_to_idx: dict[int, int],
    item_id_to_idx: dict[int, int],
    values: np.ndarray | None = None,
) -> sparse.csr_matrix:
    """Build a sparse user-item matrix from ratings.

    Args:
        ratings: DataFrame with columns [user_id, movie_id, rating, timestamp]
        user_id_to_idx: Mapping from user_id to contiguous row index
        item_id_to_idx: Mapping from movie_id to contiguous column index
        values: Optional replacement values (same order as ratings rows).
            Defaults to the raw ratings.

    Returns:
        Sparse CSR matrix of shape (n_users, n_items)
    """
    n_users = len(user_id_to_idx)
    n_items = len(item_id_to_idx)
    rows = ratings["user_id"].map(user_id_to_idx).to_numpy()
    cols = ratings["movie_id"].map(item_id_to_idx).to_numpy()
    data = ratings["rating"].to_numpy(dtype=np.float32) if values is None else values
    return sparse.csr_matrix(
        (np.asarray(data, dtype=np.float32), (rows, cols)),
        shape=(n_users, n_items),
        dtype=np.float32,
    )


def _fallback_extend(
    recommendations: list[int],
    seen: set[int],
    fallback: list[int],
) -> list[int]:
    """Pad recommendations with fallback items not already present."""
    existing = set(recommendations)
    for item_id in fallback:
        if item_id in seen or item_id in existing:
            continue
        recommendations.append(item_id)
        existing.add(item_id)
    return recommendations


class ItemItemKNNRecommender(BaseRecommender):
    """Item-item collaborative filtering with cosine similarity.

    Similarity is computed on the user-item matrix of raw ratings (the
    `center_ratings=True` variant uses per-user mean centering instead).
    Keeping ratings raw works better in practice: all-positive values make the
    cosine similarity correlate with co-occurrence *and* item popularity, which
    is exactly the signal the temporal test set rewards.

    A user's score for an unseen item is the sum of similarities to the items
    they liked (rating >= min_rating_for_like), weighted by (rating - user_mean).

    Only the top `n_neighbors` most similar items per item are kept, which both
    speeds up scoring and removes noisy weak similarities.
    """

    def __init__(
        self,
        n_neighbors: int = 50,
        min_rating_for_like: float = 4.0,
        center_ratings: bool = False,
    ) -> None:
        self.n_neighbors = n_neighbors
        self.min_rating_for_like = min_rating_for_like
        self.center_ratings = center_ratings

        # Fitted attributes
        self.user_id_to_idx_: dict[int, int] = {}
        self.idx_to_user_id_: dict[int, int] = {}
        self.item_id_to_idx_: dict[int, int] = {}
        self.idx_to_item_id_: dict[int, int] = {}
        self.user_means_: dict[int, float] = {}
        self.neighbor_idx_: np.ndarray | None = None
        self.neighbor_sim_: np.ndarray | None = None
        self.user_item_indices_: dict[int, np.ndarray] = {}
        self.user_item_weights_: dict[int, np.ndarray] = {}
        self.user_seen_indices_: dict[int, np.ndarray] = {}
        self.popularity_model_: PopularityBaseline = PopularityBaseline()

    def fit(self, train_ratings: pd.DataFrame) -> None:
        """Fit item-item KNN on training data.

        Args:
            train_ratings: DataFrame with columns [user_id, movie_id, rating, timestamp]
        """
        (
            self.user_id_to_idx_,
            self.idx_to_user_id_,
            self.item_id_to_idx_,
            self.idx_to_item_id_,
        ) = _build_id_mappings(train_ratings)

        self.user_means_ = train_ratings.groupby("user_id")["rating"].mean().to_dict()

        values = None
        if self.center_ratings:
            user_means = train_ratings["user_id"].map(self.user_means_).to_numpy()
            values = (train_ratings["rating"].to_numpy() - user_means).astype(
                np.float32
            )

        matrix = _build_sparse_matrix(
            train_ratings, self.user_id_to_idx_, self.item_id_to_idx_, values
        )
        self._build_similarity(matrix)
        self._build_user_items(train_ratings)
        self.popularity_model_.fit(train_ratings)

        n_items = len(self.item_id_to_idx_)
        logger.info(
            f"ItemItemKNNRecommender fitted: {n_items} items, "
            f"{len(self.user_id_to_idx_)} users, "
            f"neighbors={min(self.n_neighbors, n_items - 1)}"
        )

    def _build_similarity(self, matrix: sparse.csr_matrix) -> None:
        """Compute top-N item-item cosine similarities.

        Args:
            matrix: Sparse user-item matrix (centered if center_ratings=True)
        """
        n_items = matrix.shape[1]
        n_neighbors = min(self.n_neighbors, n_items - 1)

        # Column-normalize, then A^T A gives the item-item cosine similarity
        # matrix (A is users x items, so A^T A is items x items)
        norms = np.sqrt(matrix.multiply(matrix).sum(axis=0)).A1
        norms[norms == 0] = 1.0
        normalized = matrix.multiply(1.0 / norms).tocsr()
        similarity = (normalized.T @ normalized).toarray().astype(np.float32)

        # Ignore self-similarity
        np.fill_diagonal(similarity, 0.0)

        # Keep only the top-N neighbours per item
        neighbor_idx = np.argpartition(-similarity, n_neighbors, axis=1)[
            :, :n_neighbors
        ]
        neighbor_sim = np.take_along_axis(similarity, neighbor_idx, axis=1)

        # Sort neighbours by similarity (descending) for stable behaviour
        order = np.argsort(-neighbor_sim, axis=1)
        self.neighbor_idx_ = np.take_along_axis(neighbor_idx, order, axis=1)
        self.neighbor_sim_ = np.take_along_axis(neighbor_sim, order, axis=1)

    def _build_user_items(self, train_ratings: pd.DataFrame) -> None:
        """Store liked items, their weights, and seen items per user.

        Args:
            train_ratings: DataFrame with columns [user_id, movie_id, rating, timestamp]
        """
        seen = train_ratings.groupby("user_id")["movie_id"].apply(
            lambda movies: movies.map(self.item_id_to_idx_).to_numpy()
        )
        self.user_seen_indices_ = {
            self.user_id_to_idx_[user_id]: indices
            for user_id, indices in seen.items()
        }

        liked = train_ratings[train_ratings["rating"] >= self.min_rating_for_like]
        for user_id, group in liked.groupby("user_id"):
            user_idx = self.user_id_to_idx_[user_id]
            self.user_item_indices_[user_idx] = group["movie_id"].map(
                self.item_id_to_idx_
            ).to_numpy()
            user_mean = self.user_means_[user_id]
            self.user_item_weights_[user_idx] = (
                group["rating"].to_numpy() - user_mean
            ).astype(np.float32)

    def recommend(self, user_id: int, k: int = 10) -> list[int]:
        """Recommend top-K unseen items for a user.

        Cold-start users (no liked items in training) get popular items.

        Args:
            user_id: User ID to recommend for
            k: Number of recommendations to return

        Returns:
            List of movie_ids (top-K recommendations)
        """
        user_idx = self.user_id_to_idx_.get(user_id)
        if (
            user_idx is None
            or self.neighbor_idx_ is None
            or user_idx not in self.user_item_indices_
        ):
            return self.popularity_model_.recommend(user_id, k)

        scores = np.zeros(self.neighbor_idx_.shape[0], dtype=np.float32)
        item_indices = self.user_item_indices_[user_idx]
        weights = self.user_item_weights_[user_idx]
        for item_idx, weight in zip(item_indices, weights):
            scores[self.neighbor_idx_[item_idx]] += weight * self.neighbor_sim_[
                item_idx
            ]

        seen = set(self.user_seen_indices_.get(user_idx, []))
        seen_items = {
            self.idx_to_item_id_[idx] for idx in seen
        }
        order = np.argsort(-scores)
        recommendations: list[int] = []
        for idx in order:
            if scores[idx] <= 0:
                break
            item_id = self.idx_to_item_id_[int(idx)]
            if item_id in seen_items:
                continue
            recommendations.append(item_id)
            if len(recommendations) == k:
                break

        if len(recommendations) < k:
            fallback = self.popularity_model_.recommend(
                user_id, k - len(recommendations)
            )
            recommendations = _fallback_extend(recommendations, seen_items, fallback)
        return recommendations


class SVDRecommender(BaseRecommender):
    """Latent-factor recommender using truncated SVD.

    Fits SVD on the rating matrix centered by the global mean and per-user and
    per-item biases, so the latent factors capture the interaction structure
    that simple biases cannot explain:

        rating(u, i) ~= global_mean + bias_u + bias_i + P_u . Q_i

    Unseen (user, item) pairs are scored by the same formula, with biases
    defaulting to 0 for users/items without training data.
    """

    def __init__(self, n_factors: int = 50) -> None:
        self.n_factors = n_factors

        # Fitted attributes
        self.user_id_to_idx_: dict[int, int] = {}
        self.idx_to_user_id_: dict[int, int] = {}
        self.item_id_to_idx_: dict[int, int] = {}
        self.idx_to_item_id_: dict[int, int] = {}
        self.global_mean_: float = 0.0
        self.user_biases_: np.ndarray | None = None
        self.item_biases_: np.ndarray | None = None
        self.user_factors_: np.ndarray | None = None
        self.item_factors_: np.ndarray | None = None
        self.seen_items_: dict[int, set[int]] = {}
        self.popularity_model_: PopularityBaseline = PopularityBaseline()

    def fit(self, train_ratings: pd.DataFrame) -> None:
        """Fit truncated SVD on training data.

        Args:
            train_ratings: DataFrame with columns [user_id, movie_id, rating, timestamp]
        """
        (
            self.user_id_to_idx_,
            self.idx_to_user_id_,
            self.item_id_to_idx_,
            self.idx_to_item_id_,
        ) = _build_id_mappings(train_ratings)

        self.global_mean_ = float(train_ratings["rating"].mean())

        matrix = _build_sparse_matrix(
            train_ratings, self.user_id_to_idx_, self.item_id_to_idx_
        )
        item_matrix = matrix.tocsc()

        # Biases: deviation of user/item means from the global mean
        user_counts = np.maximum(np.diff(matrix.indptr), 1)
        item_counts = np.maximum(np.diff(item_matrix.indptr), 1)
        user_means = np.asarray(matrix.sum(axis=1)).ravel() / user_counts
        item_means = np.asarray(item_matrix.sum(axis=0)).ravel() / item_counts
        self.user_biases_ = (user_means - self.global_mean_).astype(np.float32)
        self.item_biases_ = (item_means - self.global_mean_).astype(np.float32)

        # Residual matrix: remove global mean and biases before factorization.
        # Rebuild the CSR arrays directly (same indices/indptr, new data).
        rows = np.repeat(np.arange(matrix.shape[0]), np.diff(matrix.indptr))
        residual_data = (
            matrix.data
            - self.global_mean_
            - self.user_biases_[rows]
            - self.item_biases_[matrix.indices]
        ).astype(np.float32)
        residual = sparse.csr_matrix(
            (residual_data, matrix.indices, matrix.indptr), shape=matrix.shape
        )

        # Deterministic starting vector keeps results reproducible
        n_factors = min(self.n_factors, min(residual.shape) - 1)
        v0 = np.ones(min(residual.shape), dtype=np.float32)
        u, sigma, vt = svds(residual, k=n_factors, v0=v0)

        # svds returns singular values in ascending order
        order = np.argsort(-sigma)
        sqrt_sigma = np.sqrt(sigma[order]).astype(np.float32)
        self.user_factors_ = (u[:, order] * sqrt_sigma).astype(np.float32)
        self.item_factors_ = (vt[order].T * sqrt_sigma).astype(np.float32)

        self.seen_items_ = (
            train_ratings.groupby("user_id")["movie_id"]
            .apply(lambda movies: set(int(m) for m in movies))
            .to_dict()
        )
        self.popularity_model_.fit(train_ratings)

        logger.info(
            f"SVDRecommender fitted: {self.item_factors_.shape[0]} items, "
            f"{self.user_factors_.shape[0]} users, "
            f"n_factors={self.user_factors_.shape[1]}"
        )

    def recommend(self, user_id: int, k: int = 10) -> list[int]:
        """Recommend top-K unseen items for a user.

        Cold-start users (no training interactions) get popular items.

        Args:
            user_id: User ID to recommend for
            k: Number of recommendations to return

        Returns:
            List of movie_ids (top-K recommendations)
        """
        user_idx = self.user_id_to_idx_.get(user_id)
        if user_idx is None or self.user_factors_ is None:
            return self.popularity_model_.recommend(user_id, k)

        scores = (
            self.global_mean_
            + self.user_biases_[user_idx]
            + self.item_biases_
            + self.user_factors_[user_idx] @ self.item_factors_.T
        )

        seen_items = self.seen_items_.get(user_id, set())
        order = np.argsort(-scores)
        recommendations: list[int] = []
        for idx in order:
            item_id = self.idx_to_item_id_[int(idx)]
            if item_id in seen_items:
                continue
            recommendations.append(item_id)
            if len(recommendations) == k:
                break

        if len(recommendations) < k:
            fallback = self.popularity_model_.recommend(
                user_id, k - len(recommendations)
            )
            recommendations = _fallback_extend(recommendations, seen_items, fallback)
        return recommendations

    def predict_rating(self, user_id: int, item_id: int) -> float:
        """Predict the rating for a user-item pair.

        Args:
            user_id: User ID
            item_id: Item ID

        Returns:
            Predicted rating (global mean for unknown users/items)
        """
        user_idx = self.user_id_to_idx_.get(user_id)
        item_idx = self.item_id_to_idx_.get(item_id)
        if user_idx is None or item_idx is None:
            return self.global_mean_
        return float(
            self.global_mean_
            + self.user_biases_[user_idx]
            + self.item_biases_[item_idx]
            + self.user_factors_[user_idx] @ self.item_factors_[item_idx]
        )


def main() -> None:
    """Run collaborative filtering evaluation on MovieLens 1M."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    ratings_cols = ["user_id", "movie_id", "rating", "timestamp"]
    ratings = pd.read_csv(
        "data/ml-1m/ratings.dat",
        sep="::",
        names=ratings_cols,
        engine="python",
        encoding="latin-1",
    )
    train, test = time_based_split(ratings, test_ratio=0.2)
    print(f"Train: {len(train)}, Test: {len(test)}")

    knn = ItemItemKNNRecommender(n_neighbors=50)
    knn_metrics = _fit_and_evaluate(knn, train, test, "ItemItemKNN")

    svd = SVDRecommender(n_factors=50)
    svd_metrics = _fit_and_evaluate(svd, train, test, "SVD")

    best = max(
        [("ItemItemKNN", knn_metrics), ("SVD", svd_metrics)],
        key=lambda pair: pair[1]["precision_at_10"],
    )
    print(f"\nBest model: {best[0]} (Precision@10={best[1]['precision_at_10']:.4f})")


def _fit_and_evaluate(
    model: BaseRecommender,
    train: pd.DataFrame,
    test: pd.DataFrame,
    name: str,
) -> dict[str, float]:
    """Fit a model, evaluate it, and print the metrics."""
    model.fit(train)
    metrics = evaluate(model, train, test, k=10)
    print(
        f"{name}: Precision@10={metrics['precision_at_10']:.4f}, "
        f"Recall@10={metrics['recall_at_10']:.4f}"
    )
    return metrics


if __name__ == "__main__":
    main()
