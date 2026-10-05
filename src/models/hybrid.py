"""
Hybrid recommender combining content-based cosine similarity with popularity.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.models.baseline import BaseRecommender, PopularityBaseline
from src.models.content_based import ContentBasedRecommender

logger = logging.getLogger(__name__)


class HybridRecommender(BaseRecommender):
    """Hybrid recommender combining content-based cosine similarity with popularity.

    Score = alpha * cosine_similarity + (1 - alpha) * normalized_popularity

    - Popularity is min-max normalized, computed on train only
    - User profile weights = rating - user_mean (for liked items: rating >= 4)
    - Reuses ContentBased item vectors and PopularityBaseline
    - Excludes seen items from recommendations
    """

    def __init__(
        self,
        alpha: float = 0.5,
        min_rating_for_like: float = 4.0,
        tfidf_max_features: int = 100,
        tfidf_ngram_range: tuple[int, int] = (1, 2),
    ) -> None:
        self.alpha = alpha
        self.min_rating_for_like = min_rating_for_like

        # Content-based component (for item vectors and user profiles)
        self.content_model = ContentBasedRecommender(
            min_rating_for_like=min_rating_for_like,
            tfidf_max_features=tfidf_max_features,
            tfidf_ngram_range=tfidf_ngram_range,
        )

        # Popularity component (for min-max normalized popularity)
        self.popularity_model = PopularityBaseline()

        # Fitted attributes
        self.item_popularity_: dict[int, int] = {}
        self.popularity_min_: float = 0.0
        self.popularity_max_: float = 0.0
        self.user_means_: dict[int, float] = {}
        self.train_ratings_: pd.DataFrame | None = None

    def fit(self, train_ratings: pd.DataFrame, data_dir: str = "data/ml-1m") -> None:
        """Fit hybrid model on training data.

        Args:
            train_ratings: DataFrame with columns [user_id, movie_id, rating, timestamp]
            data_dir: Directory containing movies.dat, users.dat, ratings.dat
        """
        self.train_ratings_ = train_ratings.copy()

        # Fit content-based model (builds item vectors and user profiles with raw weights)
        self.content_model.fit(train_ratings, data_dir=data_dir)

        # Store original content-based user profiles for alpha=1.0
        self._original_user_profiles_ = self.content_model.user_profiles_.copy()

        # Fit popularity model
        self.popularity_model.fit(train_ratings)

        # Store popularity counts for min-max normalization
        self.item_popularity_ = self.popularity_model.item_popularity_.copy()
        if self.item_popularity_:
            self.popularity_min_ = min(self.item_popularity_.values())
            self.popularity_max_ = max(self.item_popularity_.values())

        # Compute user means for centering weights
        self.user_means_ = train_ratings.groupby("user_id")["rating"].mean().to_dict()

        # Rebuild user profiles with centered weights (rating - user_mean) for hybrid
        self._build_centered_user_profiles()

        logger.info(
            f"HybridRecommender fitted: alpha={self.alpha}, "
            f"{len(self.content_model.item_id_to_idx_)} items, "
            f"{len(self.content_model.user_profiles_)} user profiles"
        )

    def _build_centered_user_profiles(self) -> None:
        """Build user profiles with weights = rating - user_mean for liked items."""
        assert self.train_ratings_ is not None
        assert self.content_model.item_vectors_ is not None

        # Filter to liked items (rating >= threshold)
        liked = self.train_ratings_[
            self.train_ratings_["rating"] >= self.min_rating_for_like
        ]

        if liked.empty:
            logger.warning("No liked items found in training data")
            return

        # Build profile per user with centered weights
        for user_id, group in liked.groupby("user_id"):
            user_mean = self.user_means_.get(user_id, 0.0)
            item_ids = group["movie_id"].tolist()
            ratings = group["rating"].tolist()

            # Get vectors for items that exist in our item set
            valid_items = []
            valid_weights = []
            for item_id, rating in zip(item_ids, ratings):
                if item_id in self.content_model.item_id_to_idx_:
                    valid_items.append(self.content_model.item_id_to_idx_[item_id])
                    valid_weights.append(rating - user_mean)  # Centered weight

            if not valid_items:
                continue

            # Weighted mean of item vectors
            item_vecs = self.content_model.item_vectors_[valid_items]
            weights = np.array(valid_weights, dtype=float)

            # Handle case where all weights are zero or negative
            if np.all(weights <= 0):
                # Fall back to uniform weights
                weights = np.ones_like(weights) / len(weights)
            else:
                # Use absolute values for weighting, preserving sign for direction
                weights = np.abs(weights)
                weights = weights / weights.sum()

            user_profile = np.average(item_vecs, axis=0, weights=weights)
            self.content_model.user_profiles_[user_id] = user_profile

        logger.debug(f"Built centered profiles for {len(self.content_model.user_profiles_)} users")

    def _normalize_popularity(self, item_id: int) -> float:
        """Min-max normalize popularity for an item."""
        if item_id not in self.item_popularity_:
            return 0.0
        if self.popularity_max_ == self.popularity_min_:
            return 1.0
        return (self.item_popularity_[item_id] - self.popularity_min_) / (
            self.popularity_max_ - self.popularity_min_
        )

    def recommend(self, user_id: int, k: int = 10) -> list[int]:
        """Recommend top-K unseen items by hybrid score.

        Args:
            user_id: User ID to recommend for
            k: Number of recommendations

        Returns:
            List of top-K movie_ids by hybrid score
        """
        # Special case: alpha=0 -> pure popularity
        if self.alpha == 0.0:
            return self.popularity_model.recommend(user_id, k)

        # Special case: alpha=1.0 -> pure cosine with original content-based profiles
        if self.alpha == 1.0:
            # Temporarily restore original profiles
            original_profiles = self.content_model.user_profiles_
            self.content_model.user_profiles_ = self._original_user_profiles_
            try:
                return self.content_model.recommend(user_id, k)
            finally:
                self.content_model.user_profiles_ = original_profiles

        # Cold-start: no content profile -> fallback to popularity
        if (
            user_id not in self.content_model.user_profiles_
            or self.content_model.similarity_matrix_ is None
        ):
            logger.debug(f"Cold-start user {user_id}, falling back to popularity")
            return self.popularity_model.recommend(user_id, k)

        # Get user profile (centered weights for hybrid)
        user_profile = self.content_model.user_profiles_[user_id].reshape(1, -1)

        # Compute cosine similarity to all items
        cosine_sims = cosine_similarity(user_profile, self.content_model.item_vectors_).flatten()

        # Get items user has already seen in training
        seen_items: set[int] = set()
        if self.train_ratings_ is not None:
            seen_items = set(
                self.train_ratings_[self.train_ratings_["user_id"] == user_id]["movie_id"].tolist()
            )

        # Compute hybrid scores for all items
        hybrid_scores = {}
        for idx, item_id in self.content_model.idx_to_item_id_.items():
            if item_id in seen_items:
                continue
            cosine_score = cosine_sims[idx]
            popularity_score = self._normalize_popularity(item_id)
            hybrid_scores[item_id] = self.alpha * cosine_score + (1 - self.alpha) * popularity_score

        # Sort by hybrid score (descending)
        sorted_items = sorted(hybrid_scores.keys(), key=lambda x: hybrid_scores[x], reverse=True)

        recommendations = sorted_items[:k]

        # If not enough items, pad with popularity fallback
        if len(recommendations) < k:
            fallback = self.popularity_model.recommend(user_id, k - len(recommendations))
            for item_id in fallback:
                if item_id not in seen_items and item_id not in recommendations:
                    recommendations.append(item_id)
                    if len(recommendations) >= k:
                        break

        return recommendations[:k]


def tune_alpha_on_val(
    train_ratings: pd.DataFrame,
    val_ratings: pd.DataFrame,
    alphas: list[float] | None = None,
    k: int = 10,
    data_dir: str = "data/ml-1m",
) -> tuple[float, dict]:
    """Tune alpha on validation set by Precision@10.

    Args:
        train_ratings: Training data
        val_ratings: Validation data
        alphas: List of alpha values to try (default: 0.0 to 1.0 step 0.1)
        k: Cutoff for metrics
        data_dir: Directory containing movies.dat

    Returns:
        Tuple of (best_alpha, results_dict)
    """
    if alphas is None:
        alphas = [round(i * 0.1, 1) for i in range(11)]  # 0.0, 0.1, ..., 1.0

    from src.models.baseline import evaluate

    results = {}
    best_alpha = 0.0
    best_precision = -1.0

    for alpha in alphas:
        model = HybridRecommender(alpha=alpha)
        model.fit(train_ratings, data_dir=data_dir)
        metrics = evaluate(model, train_ratings, val_ratings, k=k)
        results[alpha] = {
            "precision_at_10": metrics["precision_at_10"],
            "recall_at_10": metrics["recall_at_10"],
        }
        if metrics["precision_at_10"] > best_precision:
            best_precision = metrics["precision_at_10"]
            best_alpha = alpha

    return best_alpha, results


def main() -> None:
    """Main function to run hybrid tuning and evaluation on MovieLens 1M."""
    import sys
    sys.path.insert(0, ".")
    from src.models.baseline import time_based_train_val_test_split

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

    # Time-based train/val/test split
    train, val, test = time_based_train_val_test_split(
        ratings, val_ratio=0.1, test_ratio=0.2
    )
    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

    # Tune alpha on validation set
    print("\n--- Tuning alpha on validation set ---")
    alphas = [round(i * 0.1, 1) for i in range(11)]
    best_alpha, results = tune_alpha_on_val(train, val, alphas=alphas, k=10, data_dir=data_dir)

    print("\nValidation curve (alpha, P@10, R@10):")
    for alpha in alphas:
        r = results[alpha]
        marker = " *" if alpha == best_alpha else ""
        print(f"  alpha={alpha:.1f}: P@10={r['precision_at_10']:.4f}, R@10={r['recall_at_10']:.4f}{marker}")

    # Evaluate best alpha on test set
    print(f"\n--- Best alpha = {best_alpha} on test set ---")
    best_model = HybridRecommender(alpha=best_alpha)
    best_model.fit(train, data_dir=data_dir)
    from src.models.baseline import evaluate
    test_metrics = evaluate(best_model, train, test, k=10)
    print(f"Precision@10: {test_metrics['precision_at_10']:.4f}")
    print(f"Recall@10:    {test_metrics['recall_at_10']:.4f}")


if __name__ == "__main__":
    main()