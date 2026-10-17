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
from src.models.hybrid import (
    CFHybridRecommender,
    HybridRecommender,
    tune_alpha_on_val,
    tune_cf_weights_on_val,
)


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

class TestCFHybridRecommender:
    """Tests for the three-way CF + content + popularity hybrid (synthetic)."""

    @pytest.fixture
    def cluster_data(self):
        """Clustered synthetic ratings plus the held-out item per user."""
        from tests.test_collaborative import make_clustered_ratings

        return make_clustered_ratings()

    @pytest.fixture
    def train(self, cluster_data):
        """Training ratings."""
        return cluster_data[0]

    @pytest.fixture
    def model(self, train):
        """Fitted CF hybrid with balanced weights."""
        model = CFHybridRecommender(
            cf_weight=0.7, content_weight=0.2, n_neighbors=5
        )
        model.fit(train)
        return model

    def test_invalid_weights_rejected(self):
        """Weights must be non-negative and sum to at most 1."""
        with pytest.raises(ValueError):
            CFHybridRecommender(cf_weight=0.8, content_weight=0.3)
        with pytest.raises(ValueError):
            CFHybridRecommender(cf_weight=-0.1, content_weight=0.0)

    def test_ranks_held_out_item_first(self, model, cluster_data):
        """The unseen same-cluster item must rank first for every weight mix."""
        _, held_out = cluster_data
        for cf_weight, content_weight in [(1.0, 0.0), (0.7, 0.2), (0.5, 0.5)]:
            tuned = CFHybridRecommender(
                cf_weight=cf_weight,
                content_weight=content_weight,
                n_neighbors=5,
            )
            tuned.fit(cluster_data[0])
            assert tuned.recommend(1, k=1) == [held_out[1]]

    def test_seen_items_excluded(self, model, train):
        """Rated items must never be recommended."""
        for user_id in [1, 11]:
            seen = set(train[train["user_id"] == user_id]["movie_id"].tolist())
            recs = model.recommend(user_id, k=3)
            assert not any(item in seen for item in recs)

    def test_cold_start_falls_back_to_popularity(self, model, train):
        """Users without training data get popular items."""
        popularity = PopularityBaseline()
        popularity.fit(train)
        assert model.recommend(999, k=5) == popularity.recommend(999, k=5)

    def test_recommendations_are_python_ints(self, model):
        """Recommendation lists must contain plain ints (serializable)."""
        for item_id in model.recommend(1, k=3):
            assert type(item_id) is int

    def test_components_are_normalized(self, model):
        """Each component is min-max normalized to [0, 1]."""
        cf_scores, content_scores, popularity_scores = model._component_scores(1)
        for scores in (cf_scores, content_scores, popularity_scores):
            assert float(scores.min()) >= 0.0
            assert float(scores.max()) <= 1.0

    def test_tuning_returns_grid_member(self, train):
        """Weight tuning picks a candidate from the grid and scores it."""
        val = train.sample(frac=0.2, random_state=1)
        best, results = tune_cf_weights_on_val(
            train,
            val,
            cf_weights=(0.6, 1.0),
            content_weights=(0.0, 0.3),
            n_val_users=20,
        )
        assert best in {(0.6, 0.0), (0.6, 0.3), (1.0, 0.0)}
        assert set(results) == {(0.6, 0.0), (0.6, 0.3), (1.0, 0.0)}
        assert results[best] == max(results.values())
