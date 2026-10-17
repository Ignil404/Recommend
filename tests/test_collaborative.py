"""
Tests for collaborative filtering models.
"""

import numpy as np
import pandas as pd
import pytest

from src.models.baseline import (
    PopularityBaseline,
    RandomRecommender,
    evaluate,
    time_based_split,
)
from src.models.collaborative import (
    ALSRecommender,
    ItemItemKNNRecommender,
    SVDRecommender,
)

CLUSTER_A = list(range(1, 7))
CLUSTER_B = list(range(7, 13))


def make_clustered_ratings(n_users: int = 40):
    """Create synthetic ratings with two opposing user/item clusters.

    Users 1..n/2 (cluster A) rate every cluster A item 5.0 except one random
    item that is held out; they rate every cluster B item 2.0. Users n/2+1..n
    do the opposite. The held-out item is the only unseen item per user, so a
    working collaborative model must rank it first.

    Args:
        n_users: Even number of users to generate

    Returns:
        Tuple of (ratings DataFrame, mapping of user_id to held-out movie_id)
    """
    rng = np.random.default_rng(42)
    rows = []
    held_out: dict[int, int] = {}
    for user_id in range(1, n_users + 1):
        own = CLUSTER_A if user_id <= n_users // 2 else CLUSTER_B
        other = CLUSTER_B if user_id <= n_users // 2 else CLUSTER_A
        dropped = int(rng.choice(own))
        held_out[user_id] = dropped
        for movie_id in own:
            if movie_id == dropped:
                continue
            rows.append(
                {
                    "user_id": user_id,
                    "movie_id": movie_id,
                    "rating": 5.0,
                    "timestamp": movie_id,
                }
            )
        for movie_id in other:
            rows.append(
                {
                    "user_id": user_id,
                    "movie_id": movie_id,
                    "rating": 2.0,
                    "timestamp": 100 + movie_id,
                }
            )
    return pd.DataFrame(rows), held_out


@pytest.fixture
def cluster_data():
    """Clustered training data plus held-out item per user."""
    return make_clustered_ratings()


class TestItemItemKNNRecommender:
    """Tests for ItemItemKNNRecommender."""

    @pytest.fixture
    def train(self, cluster_data):
        """Training ratings."""
        return cluster_data[0]

    @pytest.fixture
    def model(self, train):
        """Fitted item-item KNN model."""
        model = ItemItemKNNRecommender(n_neighbors=5)
        model.fit(train)
        return model

    def test_ranks_held_out_item_first(self, model, cluster_data):
        """The unseen same-cluster item must be ranked first."""
        _, held_out = cluster_data
        for user_id in [1, 7, 20, 40]:
            assert model.recommend(user_id, k=1) == [held_out[user_id]]

    def test_seen_items_excluded(self, model, train):
        """Rated items must never be recommended."""
        for user_id in [1, 5, 11, 40]:
            seen = set(train[train["user_id"] == user_id]["movie_id"].tolist())
            recs = model.recommend(user_id, k=3)
            assert not any(item in seen for item in recs)

    def test_cold_start_user_falls_back_to_popularity(self, model, train):
        """Users without training data get popular items."""
        popularity = PopularityBaseline()
        popularity.fit(train)
        assert model.recommend(999, k=5) == popularity.recommend(999, k=5)

    def test_recommendations_are_python_ints(self, model):
        """Recommendation lists must contain plain ints (serializable)."""
        for item_id in model.recommend(1, k=3):
            assert type(item_id) is int

    def test_returns_requested_number_of_items(self, model):
        """Model returns exactly k items when enough unseen items exist."""
        assert len(model.recommend(1, k=1)) == 1

    def test_deterministic(self, train):
        """Fitting twice on the same data gives identical recommendations."""
        first = ItemItemKNNRecommender(n_neighbors=5)
        second = ItemItemKNNRecommender(n_neighbors=5)
        first.fit(train)
        second.fit(train)
        for user_id in [1, 11]:
            assert first.recommend(user_id, k=3) == second.recommend(user_id, k=3)

    def test_centered_variant_still_separates_clusters(self, train):
        """Centered-rating variant also finds the held-out cluster item."""
        model = ItemItemKNNRecommender(n_neighbors=5, center_ratings=True)
        model.fit(train)
        assert model.recommend(1, k=1) == [_held_out(train, 1)]

    def test_similarity_shape_is_item_by_item(self, model, train):
        """The similarity table must be indexed by items, not users."""
        n_items = train["movie_id"].nunique()
        assert model.neighbor_idx_.shape[0] == n_items

    def test_neighbors_exclude_self(self, model):
        """An item must never be its own neighbour."""
        for item_idx in range(model.neighbor_idx_.shape[0]):
            assert item_idx not in model.neighbor_idx_[item_idx]

    def test_higher_min_rating_fewer_profiles(self, train):
        """Raising the like threshold can only shrink the profile set."""
        strict = ItemItemKNNRecommender(min_rating_for_like=5.5)
        strict.fit(train)
        assert strict.user_item_indices_ == {}


class TestSVDRecommender:
    """Tests for SVDRecommender."""

    @pytest.fixture
    def train(self, cluster_data):
        """Training ratings."""
        return cluster_data[0]

    @pytest.fixture
    def model(self, cluster_data):
        """Fitted SVD model."""
        model = SVDRecommender(n_factors=2)
        model.fit(cluster_data[0])
        return model

    def test_ranks_held_out_item_first(self, model, cluster_data):
        """The unseen same-cluster item must be ranked first."""
        _, held_out = cluster_data
        for user_id in [1, 7, 20, 40]:
            assert model.recommend(user_id, k=1) == [held_out[user_id]]

    def test_seen_items_excluded(self, model, train):
        """Rated items must never be recommended."""
        for user_id in [1, 11]:
            seen = set(train[train["user_id"] == user_id]["movie_id"].tolist())
            recs = model.recommend(user_id, k=3)
            assert not any(item in seen for item in recs)

    def test_cold_start_user_falls_back_to_popularity(self, model, train):
        """Users without training data get popular items."""
        popularity = PopularityBaseline()
        popularity.fit(train)
        assert model.recommend(999, k=5) == popularity.recommend(999, k=5)

    def test_predict_rating_known_pair(self, model):
        """Predictions for known pairs are finite and in rating scale."""
        for user_id, movie_id in [(1, 1), (20, 6), (1, 12)]:
            prediction = model.predict_rating(user_id, movie_id)
            assert np.isfinite(prediction)
            assert 0.5 <= prediction <= 5.5

    def test_predict_rating_unknown_user(self, model):
        """Unknown users get the global mean."""
        assert model.predict_rating(999, 1) == pytest.approx(model.global_mean_)

    def test_recommendations_are_python_ints(self, model):
        """Recommendation lists must contain plain ints (serializable)."""
        for item_id in model.recommend(1, k=3):
            assert type(item_id) is int

    def test_deterministic(self, train):
        """The fixed svds start vector makes fitting reproducible."""
        first = SVDRecommender(n_factors=2)
        second = SVDRecommender(n_factors=2)
        first.fit(train)
        second.fit(train)
        assert first.recommend(1, k=3) == second.recommend(1, k=3)


def _held_out(train: pd.DataFrame, user_id: int) -> int:
    """Return the item from the user's own cluster that the user never rated."""
    cluster = CLUSTER_A if user_id <= 20 else CLUSTER_B
    rated = set(train[train["user_id"] == user_id]["movie_id"].tolist())
    missing = [movie_id for movie_id in cluster if movie_id not in rated]
    return missing[0]


class TestCollaborativeOnMovieLens:
    """Smoke tests on the real MovieLens 1M dataset."""

    @pytest.fixture(scope="class")
    def train_test(self):
        """Real MovieLens train/test split."""
        ratings_cols = ["user_id", "movie_id", "rating", "timestamp"]
        ratings = pd.read_csv(
            "data/ml-1m/ratings.dat",
            sep="::",
            names=ratings_cols,
            engine="python",
            encoding="latin-1",
        )
        train, test = time_based_split(ratings, test_ratio=0.2)
        return train, test

    @pytest.mark.slow
    def test_item_knn_beats_random(self, train_test):
        """Item-item KNN must beat the random baseline on real data."""
        train, test = train_test

        sample_users = test["user_id"].unique()[:200]
        sample_test = test[test["user_id"].isin(sample_users)]

        knn = ItemItemKNNRecommender(n_neighbors=50)
        knn.fit(train)
        knn_metrics = evaluate(knn, train, sample_test, k=10)

        random_model = RandomRecommender(seed=42)
        random_model.fit(train)
        random_metrics = evaluate(random_model, train, sample_test, k=10)

        assert knn_metrics["precision_at_10"] > random_metrics["precision_at_10"]

    @pytest.mark.slow
    def test_svd_produces_valid_recommendations(self, train_test):
        """SVD recommendations are unique, valid, and exclude seen items."""
        train, _ = train_test

        svd = SVDRecommender(n_factors=20)
        svd.fit(train)

        for user_id in [1, 100, 1000]:
            recs = svd.recommend(user_id, k=10)
            assert len(recs) == 10
            assert len(set(recs)) == 10
            seen = train[train["user_id"] == user_id]["movie_id"].tolist()
            assert not any(item in seen for item in recs)


class TestALSRecommender:
    """Tests for the implicit-feedback ALS recommender."""

    @pytest.fixture
    def cluster_data(self):
        """Clustered training data plus held-out item per user."""
        return make_clustered_ratings()

    @pytest.fixture
    def train(self, cluster_data):
        """Training ratings."""
        return cluster_data[0]

    @pytest.fixture
    def model(self, train):
        """Fitted ALS model."""
        model = ALSRecommender(n_factors=2, iterations=10)
        model.fit(train)
        return model

    def test_ranks_held_out_item_first(self, model, cluster_data):
        """The unseen same-cluster item must be ranked first."""
        _, held_out = cluster_data
        for user_id in [1, 7, 20, 40]:
            assert model.recommend(user_id, k=1) == [held_out[user_id]]

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

    def test_deterministic(self, train):
        """The fixed random seed makes factor initialization reproducible."""
        first = ALSRecommender(n_factors=2, iterations=5)
        second = ALSRecommender(n_factors=2, iterations=5)
        first.fit(train)
        second.fit(train)
        assert first.recommend(1, k=3) == second.recommend(1, k=3)

    def test_preferences_are_binary(self, train):
        """Liked items carry preference 1.0, confidence exceeds 1.0."""
        model = ALSRecommender(n_factors=2, iterations=2)
        model.fit(train)
        user_idx = model.user_id_to_idx_[1]
        partners, values = model.user_items_[user_idx]
        assert set(np.unique(values[:, 1])) <= {0.0, 1.0}
        assert float(values[:, 0].min()) > 1.0
        assert partners.size > 0
