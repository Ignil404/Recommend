"""
Tests for hybrid recommender model.
"""

import pandas as pd
import pytest

from src.models.baseline import (
    PopularityBaseline,
    time_based_train_val_test_split,
)
from src.models.content_based import ContentBasedRecommender
from src.models.hybrid import HybridRecommender, tune_alpha_on_val


class TestHybridRecommender:
    """Tests for HybridRecommender."""

    @pytest.fixture
    def train_val_test(self):
        """Create train/val/test split for testing."""
        ratings_cols = ["user_id", "movie_id", "rating", "timestamp"]
        ratings = pd.read_csv(
            "data/ml-1m/ratings.dat",
            sep="::",
            names=ratings_cols,
            engine="python",
            encoding="latin-1",
        )
        train, val, test = time_based_train_val_test_split(
            ratings, val_ratio=0.1, test_ratio=0.2
        )
        return train, val, test

    def test_alpha_zero_equals_popularity(self, train_val_test):
        """Test that alpha=0 gives same ranking as PopularityBaseline."""
        train, _, _ = train_val_test

        hybrid = HybridRecommender(alpha=0.0)
        hybrid.fit(train)

        pop = PopularityBaseline()
        pop.fit(train)

        # Test for a few users
        for user_id in [1, 2, 3, 100, 500]:
            hybrid_recs = hybrid.recommend(user_id, k=20)
            pop_recs = pop.recommend(user_id, k=20)
            assert hybrid_recs == pop_recs, f"Mismatch for user {user_id}"

    def test_alpha_one_equals_pure_cosine(self, train_val_test):
        """Test that alpha=1 gives same ranking as pure cosine (ContentBased)."""
        train, _, _ = train_val_test

        hybrid = HybridRecommender(alpha=1.0)
        hybrid.fit(train)

        content = ContentBasedRecommender(min_rating_for_like=4.0)
        content.fit(train)

        # Test for a few users with profiles
        test_users = list(content.user_profiles_.keys())[:5]
        for user_id in test_users:
            hybrid_recs = hybrid.recommend(user_id, k=20)
            content_recs = content.recommend(user_id, k=20)
            assert hybrid_recs == content_recs, f"Mismatch for user {user_id}"

    def test_seen_items_excluded(self, train_val_test):
        """Test that seen items are excluded from recommendations."""
        train, _, _ = train_val_test

        hybrid = HybridRecommender(alpha=0.5)
        hybrid.fit(train)

        for user_id in [1, 2, 3, 100]:
            seen_items = set(
                train[train["user_id"] == user_id]["movie_id"].tolist()
            )
            recs = hybrid.recommend(user_id, k=20)
            # None of the recommended items should be in seen_items
            assert not any(item in seen_items for item in recs), (
                f"Seen item in recommendations for user {user_id}"
            )

    def test_val_test_no_time_overlap(self, train_val_test):
        """Test that val and test do not overlap in time for any user."""
        train, val, test = train_val_test

        for user_id in test["user_id"].unique():
            val_user = val[val["user_id"] == user_id]
            test_user = test[test["user_id"] == user_id]

            if len(val_user) == 0 or len(test_user) == 0:
                continue

            max_val_ts = val_user["timestamp"].max()
            min_test_ts = test_user["timestamp"].min()

            # Allow equality for boundary timestamps
            assert min_test_ts >= max_val_ts, (
                f"Time overlap for user {user_id}: "
                f"val max={max_val_ts}, test min={min_test_ts}"
            )

        # Also check train vs val
        for user_id in val["user_id"].unique():
            train_user = train[train["user_id"] == user_id]
            val_user = val[val["user_id"] == user_id]

            if len(train_user) == 0 or len(val_user) == 0:
                continue

            max_train_ts = train_user["timestamp"].max()
            min_val_ts = val_user["timestamp"].min()

            assert min_val_ts >= max_train_ts, (
                f"Time overlap train/val for user {user_id}"
            )

    def test_cold_start_fallback(self, train_val_test):
        """Test that cold-start users fall back to popularity."""
        train, _, _ = train_val_test

        hybrid = HybridRecommender(alpha=0.5)
        hybrid.fit(train)

        pop = PopularityBaseline()
        pop.fit(train)

        # Create a user not in training data
        cold_user = 99999
        hybrid_recs = hybrid.recommend(cold_user, k=10)
        pop_recs = pop.recommend(cold_user, k=10)

        assert hybrid_recs == pop_recs

    def test_alpha_tuning_returns_best(self, train_val_test):
        """Test that alpha tuning returns a valid best alpha."""
        train, val, _ = train_val_test

        best_alpha, results = tune_alpha_on_val(
            train, val, alphas=[0.0, 0.5, 1.0], k=10
        )

        assert best_alpha in [0.0, 0.5, 1.0]
        assert len(results) == 3
        for alpha in [0.0, 0.5, 1.0]:
            assert "precision_at_10" in results[alpha]
            assert "recall_at_10" in results[alpha]
            assert 0 <= results[alpha]["precision_at_10"] <= 1
            assert 0 <= results[alpha]["recall_at_10"] <= 1


class TestTrainValTestSplit:
    """Tests for time_based_train_val_test_split function."""

    def test_split_ratios(self):
        """Test that split ratios are approximately correct."""
        ratings_cols = ["user_id", "movie_id", "rating", "timestamp"]
        ratings = pd.read_csv(
            "data/ml-1m/ratings.dat",
            sep="::",
            names=ratings_cols,
            engine="python",
            encoding="latin-1",
        )

        train, val, test = time_based_train_val_test_split(
            ratings, val_ratio=0.1, test_ratio=0.2
        )

        total = len(ratings)
        # Allow some flexibility due to integer rounding
        assert 0.65 <= len(train) / total <= 0.75
        assert 0.05 <= len(val) / total <= 0.15
        assert 0.15 <= len(test) / total <= 0.25

    def test_no_data_leakage(self):
        """Test no data leakage across splits."""
        ratings_cols = ["user_id", "movie_id", "rating", "timestamp"]
        ratings = pd.read_csv(
            "data/ml-1m/ratings.dat",
            sep="::",
            names=ratings_cols,
            engine="python",
            encoding="latin-1",
        )

        train, val, test = time_based_train_val_test_split(
            ratings, val_ratio=0.1, test_ratio=0.2
        )

        # Train < Val < Test in time for each user
        for user_id in ratings["user_id"].unique():
            train_user = train[train["user_id"] == user_id]
            val_user = val[val["user_id"] == user_id]
            test_user = test[test["user_id"] == user_id]

            if len(train_user) > 0 and len(val_user) > 0:
                assert train_user["timestamp"].max() <= val_user["timestamp"].min()
            if len(val_user) > 0 and len(test_user) > 0:
                assert val_user["timestamp"].max() <= test_user["timestamp"].min()
            if len(train_user) > 0 and len(test_user) > 0:
                assert train_user["timestamp"].max() <= test_user["timestamp"].min()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])