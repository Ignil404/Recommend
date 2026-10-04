"""
Baseline recommender models for MovieLens dataset.

Implements popularity-based and mean-rating baselines with time-based evaluation.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BaseRecommender(ABC):
    """Abstract base class for recommender models."""

    @abstractmethod
    def fit(self, train_ratings: pd.DataFrame) -> None:
        """Fit the model on training data.

        Args:
            train_ratings: DataFrame with columns [user_id, movie_id, rating, timestamp]
        """

    @abstractmethod
    def recommend(self, user_id: int, k: int = 10) -> list[int]:
        """Recommend top-K items for a user.

        Args:
            user_id: User ID to recommend for
            k: Number of recommendations to return

        Returns:
            List of movie_ids (top-K recommendations)
        """


class PopularityBaseline(BaseRecommender):
    """Popularity-based baseline recommender.

    Recommends globally most popular items (by rating count) to all users.
    """

    def __init__(self) -> None:
        self.popular_items_: list[int] = []
        self.item_popularity_: dict[int, int] = {}

    def fit(self, train_ratings: pd.DataFrame) -> None:
        """Fit popularity model on training data.

        Args:
            train_ratings: DataFrame with columns [user_id, movie_id, rating, timestamp]
        """
        # Count ratings per item
        self.item_popularity_ = train_ratings.groupby("movie_id").size().to_dict()
        # Sort by popularity (descending)
        self.popular_items_ = sorted(
            self.item_popularity_.keys(),
            key=lambda x: self.item_popularity_[x],
            reverse=True,
        )
        logger.info(f"PopularityBaseline fitted on {len(train_ratings)} ratings, {len(self.popular_items_)} items")

    def recommend(self, user_id: int, k: int = 10) -> list[int]:
        """Return top-K popular items (same for all users).

        Args:
            user_id: User ID (unused, kept for interface compatibility)
            k: Number of recommendations

        Returns:
            List of top-K movie_ids by global popularity
        """
        return self.popular_items_[:k]


class MeanRatingBaseline(BaseRecommender):
    """Mean-rating baseline recommender with fallback strategies.

    Scores items by their mean rating in training data.
    Falls back to global mean for cold items/users.
    """

    def __init__(self, min_ratings: int = 5) -> None:
        self.min_ratings = min_ratings
        self.item_means_: dict[int, float] = {}
        self.user_means_: dict[int, float] = {}
        self.global_mean_: float = 0.0
        self.popular_items_: list[int] = []

    def fit(self, train_ratings: pd.DataFrame) -> None:
        """Fit mean-rating model on training data.

        Args:
            train_ratings: DataFrame with columns [user_id, movie_id, rating, timestamp]
        """
        # Global mean
        self.global_mean_ = float(train_ratings["rating"].mean())

        # Item means (only for items with sufficient ratings)
        item_stats = train_ratings.groupby("movie_id")["rating"].agg(["mean", "count"])
        self.item_means_ = item_stats[item_stats["count"] >= self.min_ratings]["mean"].to_dict()

        # User means
        self.user_means_ = train_ratings.groupby("user_id")["rating"].mean().to_dict()

        # Popular items as fallback for cold-start recommendations
        item_counts = train_ratings.groupby("movie_id").size()
        self.popular_items_ = item_counts.sort_values(ascending=False).index.tolist()

        logger.info(
            f"MeanRatingBaseline fitted: global_mean={self.global_mean_:.3f}, "
            f"{len(self.item_means_)} items with >= {self.min_ratings} ratings, "
            f"{len(self.user_means_)} users"
        )

    def recommend(self, user_id: int, k: int = 10) -> list[int]:
        """Recommend top-K items by predicted rating for a user.

        For cold-start users, falls back to popular items.
        For items with insufficient ratings, uses global mean.

        Args:
            user_id: User ID to recommend for
            k: Number of recommendations

        Returns:
            List of top-K movie_ids by predicted rating
        """
        # Score all items
        scores = {}
        for item_id in self.popular_items_:
            if item_id in self.item_means_:
                scores[item_id] = self.item_means_[item_id]
            else:
                scores[item_id] = self.global_mean_

        # Sort by predicted rating (descending)
        sorted_items = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        return sorted_items[:k]

    def predict_rating(self, user_id: int, item_id: int) -> float:
        """Predict rating for a user-item pair.

        Falls back to global mean if either user or item is cold-start
        (insufficient ratings in training).

        Args:
            user_id: User ID
            item_id: Item ID

        Returns:
            Predicted rating
        """
        if item_id in self.item_means_ and user_id in self.user_means_:
            return self.item_means_[item_id]
        return self.global_mean_


def time_based_split(
    ratings: pd.DataFrame,
    test_ratio: float = 0.2,
    timestamp_col: str = "timestamp",
    user_col: str = "user_id",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split ratings chronologically per user.

    For each user, the last `test_ratio` fraction of interactions by timestamp
    goes to test set, the rest to train set.

    Args:
        ratings: DataFrame with columns [user_id, movie_id, rating, timestamp]
        test_ratio: Fraction of interactions per user to use as test (default 0.2)
        timestamp_col: Name of timestamp column
        user_col: Name of user ID column

    Returns:
        Tuple of (train_df, test_df)
    """
    # Sort by user then timestamp
    ratings_sorted = ratings.sort_values([user_col, timestamp_col])

    # Compute split point per user
    user_counts = ratings_sorted.groupby(user_col).size()
    test_counts = (user_counts * test_ratio).astype(int).clip(lower=1)

    # Build train/test masks
    train_indices = []
    test_indices = []

    for user_id, group in ratings_sorted.groupby(user_col):
        n_test = test_counts.get(user_id, 1)
        n_train = len(group) - n_test

        if n_train > 0:
            train_indices.extend(group.iloc[:n_train].index.tolist())
        test_indices.extend(group.iloc[n_train:].index.tolist())

    train_df = ratings_sorted.loc[train_indices].reset_index(drop=True)
    test_df = ratings_sorted.loc[test_indices].reset_index(drop=True)

    logger.info(f"Time-based split: train={len(train_df)}, test={len(test_df)}")

    # Verify no data leakage
    verify_no_leakage(train_df, test_df, user_col, timestamp_col)

    return train_df, test_df


def verify_no_leakage(
    train: pd.DataFrame,
    test: pd.DataFrame,
    user_col: str = "user_id",
    timestamp_col: str = "timestamp",
) -> None:
    """Verify no test timestamp precedes any train timestamp for the same user.

    Args:
        train: Training DataFrame
        test: Test DataFrame
        user_col: Name of user ID column
        timestamp_col: Name of timestamp column

    Raises:
        ValueError: If data leakage detected
    """
    for user_id in test[user_col].unique():
        train_user = train[train[user_col] == user_id]
        test_user = test[test[user_col] == user_id]

        if len(train_user) == 0:
            continue

        max_train_ts = train_user[timestamp_col].max()
        min_test_ts = test_user[timestamp_col].min()

        if min_test_ts < max_train_ts:
            raise ValueError(
                f"Data leakage detected for user {user_id}: "
                f"test timestamp {min_test_ts} < train timestamp {max_train_ts}"
            )

    logger.debug("No data leakage verified")


def precision_at_k(
    recommendations: list[int],
    relevant_items: set[int],
    k: int = 10,
) -> float:
    """Compute Precision@K.

    Args:
        recommendations: List of recommended item IDs
        relevant_items: Set of relevant item IDs for the user
        k: Cutoff (default 10)

    Returns:
        Precision@K score
    """
    if not recommendations or k == 0:
        return 0.0
    rec_k = set(recommendations[:k])
    return len(rec_k & relevant_items) / k


def recall_at_k(
    recommendations: list[int],
    relevant_items: set[int],
    k: int = 10,
) -> float:
    """Compute Recall@K.

    Args:
        recommendations: List of recommended item IDs
        relevant_items: Set of relevant item IDs for the user
        k: Cutoff (default 10)

    Returns:
        Recall@K score
    """
    if not relevant_items or k == 0:
        return 0.0
    rec_k = set(recommendations[:k])
    return len(rec_k & relevant_items) / len(relevant_items)


def evaluate(
    model: BaseRecommender,
    train_ratings: pd.DataFrame,
    test_ratings: pd.DataFrame,
    k: int = 10,
    user_col: str = "user_id",
    item_col: str = "movie_id",
) -> dict[str, float]:
    """Evaluate model using Precision@K and Recall@K with macro-averaging.

    Args:
        model: Fitted recommender model
        train_ratings: Training data (used for fitting, passed for context)
        test_ratings: Test data with held-out interactions
        k: Cutoff for metrics (default 10)
        user_col: Name of user ID column
        item_col: Name of item ID column

    Returns:
        Dictionary with 'precision_at_k' and 'recall_at_k' (macro-averaged)
    """
    # Build test relevance sets per user
    test_relevant = test_ratings.groupby(user_col)[item_col].apply(set).to_dict()

    # Only evaluate users present in test set
    test_users = list(test_relevant.keys())

    precisions = []
    recalls = []

    for user_id in test_users:
        relevant = test_relevant[user_id]
        if not relevant:
            continue

        recommendations = model.recommend(user_id, k=k)
        precisions.append(precision_at_k(recommendations, relevant, k))
        recalls.append(recall_at_k(recommendations, relevant, k))

    macro_precision = np.mean(precisions) if precisions else 0.0
    macro_recall = np.mean(recalls) if recalls else 0.0

    logger.info(f"Evaluation (k={k}): Precision@{k}={macro_precision:.4f}, Recall@{k}={macro_recall:.4f}")
    logger.info(f"Evaluated on {len(test_users)} users")

    return {
        f"precision_at_{k}": macro_precision,
        f"recall_at_{k}": macro_recall,
        "n_users_evaluated": len(test_users),
    }


def main() -> None:
    """Main function to run baseline evaluation on MovieLens 1M."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Load data
    data_dir = "data/ml-1m"
    ratings_cols = ["user_id", "movie_id", "rating", "timestamp"]
    ratings = pd.read_csv(
        f"{data_dir}/ratings.dat",
        sep="::",
        names=ratings_cols,
        engine="python",
        encoding="latin-1",
    )

    print(f"Loaded {len(ratings)} ratings from {ratings['user_id'].nunique()} users and {ratings['movie_id'].nunique()} movies")

    # Time-based split
    train, test = time_based_split(ratings, test_ratio=0.2)
    print(f"Train: {len(train)}, Test: {len(test)}")

    # Popularity Baseline
    print("\n--- Popularity Baseline ---")
    pop_model = PopularityBaseline()
    pop_model.fit(train)
    pop_metrics = evaluate(pop_model, train, test, k=10)
    print(f"Precision@10: {pop_metrics['precision_at_10']:.4f}")
    print(f"Recall@10:    {pop_metrics['recall_at_10']:.4f}")

    # Mean Rating Baseline
    print("\n--- Mean Rating Baseline ---")
    mean_model = MeanRatingBaseline(min_ratings=5)
    mean_model.fit(train)
    mean_metrics = evaluate(mean_model, train, test, k=10)
    print(f"Precision@10: {mean_metrics['precision_at_10']:.4f}")
    print(f"Recall@10:    {mean_metrics['recall_at_10']:.4f}")


if __name__ == "__main__":
    main()