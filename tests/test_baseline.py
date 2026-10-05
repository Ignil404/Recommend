"""
Tests for baseline recommender models.
"""

import pandas as pd
import pytest

from src.models.baseline import (
    MeanRatingBaseline,
    PopularityBaseline,
    evaluate,
    precision_at_k,
    recall_at_k,
    time_based_split,
    verify_no_leakage,
)


class TestPrecisionAtK:
    """Tests for precision_at_k function."""

    def test_perfect_precision(self):
        """All recommendations are relevant."""
        recs = [1, 2, 3]
        relevant = {1, 2, 3}
        assert precision_at_k(recs, relevant, k=3) == 1.0

    def test_zero_precision(self):
        """No recommendations are relevant."""
        recs = [4, 5, 6]
        relevant = {1, 2, 3}
        assert precision_at_k(recs, relevant, k=3) == 0.0

    def test_partial_precision(self):
        """Some recommendations are relevant."""
        recs = [1, 2, 4]
        relevant = {1, 2, 3}
        assert precision_at_k(recs, relevant, k=3) == 2 / 3

    def test_k_larger_than_recs(self):
        """k larger than number of recommendations."""
        recs = [1, 2]
        relevant = {1, 2, 3}
        assert precision_at_k(recs, relevant, k=5) == 2 / 5

    def test_empty_recommendations(self):
        """Empty recommendations returns 0."""
        assert precision_at_k([], {1, 2}, k=10) == 0.0

    def test_k_zero(self):
        """k=0 returns 0."""
        assert precision_at_k([1, 2], {1, 2}, k=0) == 0.0


class TestRecallAtK:
    """Tests for recall_at_k function."""

    def test_perfect_recall(self):
        """All relevant items in recommendations."""
        recs = [1, 2, 3, 4, 5]
        relevant = {1, 2, 3}
        assert recall_at_k(recs, relevant, k=5) == 1.0

    def test_zero_recall(self):
        """No relevant items in recommendations."""
        recs = [4, 5, 6]
        relevant = {1, 2, 3}
        assert recall_at_k(recs, relevant, k=3) == 0.0

    def test_partial_recall(self):
        """Some relevant items in recommendations."""
        recs = [1, 2, 4, 5, 6]
        relevant = {1, 2, 3}
        assert recall_at_k(recs, relevant, k=5) == 2 / 3

    def test_empty_relevant(self):
        """Empty relevant set returns 0."""
        assert recall_at_k([1, 2], set(), k=10) == 0.0


class TestTimeBasedSplit:
    """Tests for time_based_split function."""

    def test_split_preserves_order(self):
        """Test timestamps in test are after train for each user."""
        # Create synthetic data: 2 users, 10 interactions each
        data = []
        for user in [1, 2]:
            for i in range(10):
                data.append({
                    "user_id": user,
                    "movie_id": i + 1,
                    "rating": 4.0,
                    "timestamp": 1000 + user * 1000 + i * 10,  # Increasing timestamps
                })
        ratings = pd.DataFrame(data)

        train, test = time_based_split(ratings, test_ratio=0.2)

        # Verify no leakage
        verify_no_leakage(train, test)

    def test_split_ratios(self):
        """Test approximate split ratios per user."""
        # 5 users, 100 interactions each
        data = []
        for user in range(1, 6):
            for i in range(100):
                data.append({
                    "user_id": user,
                    "movie_id": i + 1,
                    "rating": 4.0,
                    "timestamp": 1000 + i,
                })
        ratings = pd.DataFrame(data)

        train, test = time_based_split(ratings, test_ratio=0.2)

        # Each user should have ~80 train, ~20 test
        for user in range(1, 6):
            train_user = train[train["user_id"] == user]
            test_user = test[test["user_id"] == user]
            assert len(train_user) >= 75  # Allow some flexibility
            assert len(test_user) >= 15

    def test_no_data_leakage(self):
        """Verify verify_no_leakage catches leakage."""
        # Create data where test has earlier timestamp than train
        train = pd.DataFrame({
            "user_id": [1],
            "movie_id": [1],
            "rating": [4.0],
            "timestamp": [2000],
        })
        test = pd.DataFrame({
            "user_id": [1],
            "movie_id": [2],
            "rating": [4.0],
            "timestamp": [1000],  # Earlier than train!
        })

        with pytest.raises(ValueError, match="Data leakage detected"):
            verify_no_leakage(train, test)

    def test_cold_start_user(self):
        """Test user with only test interactions (no train)."""
        data = []
        # User 1: 5 interactions
        for i in range(5):
            data.append({"user_id": 1, "movie_id": i, "rating": 4.0, "timestamp": 1000 + i})
        # User 2: only 1 interaction - should go to train (minimum 1 train item)
        data.append({"user_id": 2, "movie_id": 10, "rating": 4.0, "timestamp": 2000})

        ratings = pd.DataFrame(data)
        train, test = time_based_split(ratings, test_ratio=0.2)

        # User 2 has 1 interaction, goes to train (minimum 1 train item enforced)
        assert len(train[train["user_id"] == 2]) == 1
        assert len(test[test["user_id"] == 2]) == 0


class TestPopularityBaseline:
    """Tests for PopularityBaseline."""

    def test_fit_and_recommend(self):
        """Test basic fit and recommend."""
        train = pd.DataFrame({
            "user_id": [1, 1, 2, 2, 3],
            "movie_id": [1, 2, 1, 3, 1],  # Item 1: 3 ratings, Item 2: 1, Item 3: 1
            "rating": [4, 5, 3, 4, 5],
            "timestamp": [1000, 1001, 1002, 1003, 1004],
        })

        model = PopularityBaseline()
        model.fit(train)

        # Item 1 should be most popular
        recs = model.recommend(user_id=1, k=3)
        assert recs[0] == 1  # Most popular
        assert len(recs) == 3

    def test_same_recommendations_for_all_users(self):
        """Popularity baseline gives same recs to all users."""
        train = pd.DataFrame({
            "user_id": [1, 1, 2, 2],
            "movie_id": [1, 2, 1, 3],
            "rating": [4, 5, 3, 4],
            "timestamp": [1000, 1001, 1002, 1003],
        })

        model = PopularityBaseline()
        model.fit(train)

        recs_user1 = model.recommend(user_id=1, k=2)
        recs_user2 = model.recommend(user_id=2, k=2)
        assert recs_user1 == recs_user2


class TestMeanRatingBaseline:
    """Tests for MeanRatingBaseline."""

    def test_fit_and_recommend(self):
        """Test basic fit and recommend."""
        train = pd.DataFrame({
            "user_id": [1, 1, 2, 2, 3],
            "movie_id": [1, 2, 1, 3, 1],
            "rating": [4, 5, 3, 4, 5],
            "timestamp": [1000, 1001, 1002, 1003, 1004],
        })

        model = MeanRatingBaseline(min_ratings=1)  # Low threshold for test
        model.fit(train)

        recs = model.recommend(user_id=1, k=2)
        assert len(recs) == 2
        # Item 2 has highest mean rating (5.0)
        assert recs[0] == 2

    def test_fallback_to_global_mean(self):
        """Test fallback to global mean for cold items."""
        train = pd.DataFrame({
            "user_id": [1, 1, 2],
            "movie_id": [1, 2, 1],  # Item 1: 2 ratings, Item 2: 1 rating
            "rating": [4, 5, 3],
            "timestamp": [1000, 1001, 1002],
        })

        model = MeanRatingBaseline(min_ratings=2)  # Item 2 has only 1 rating
        model.fit(train)

        # Item 2 should fall back to global mean
        assert model.predict_rating(1, 2) == model.global_mean_
        # Item 1 has enough ratings
        assert model.predict_rating(1, 1) == 3.5  # Mean of 4 and 3

    def test_cold_start_user(self):
        """Test cold-start user gets global mean."""
        train = pd.DataFrame({
            "user_id": [1, 1],
            "movie_id": [1, 2],
            "rating": [4, 5],
            "timestamp": [1000, 1001],
        })

        model = MeanRatingBaseline(min_ratings=1)
        model.fit(train)

        # User 999 not in training
        assert model.predict_rating(999, 1) == model.global_mean_


class TestEvaluate:
    """Tests for evaluate function."""

    def test_evaluate_popularity(self):
        """Test evaluate with PopularityBaseline."""
        # Create data where we know the answer
        train = pd.DataFrame({
            "user_id": [1, 1, 2, 2, 3, 3],
            "movie_id": [1, 2, 1, 3, 1, 4],  # Item 1 most popular (3 ratings)
            "rating": [4, 5, 3, 4, 5, 4],
            "timestamp": [1000, 1001, 1002, 1003, 1004, 1005],
        })
        test = pd.DataFrame({
            "user_id": [1, 2, 3],
            "movie_id": [1, 1, 1],  # All users have item 1 in test
            "rating": [5, 4, 5],
            "timestamp": [2000, 2001, 2002],
        })

        model = PopularityBaseline()
        model.fit(train)
        metrics = evaluate(model, train, test, k=1)

        # All users should get item 1 as top rec, which is relevant
        assert metrics["precision_at_1"] == 1.0
        assert metrics["recall_at_1"] == 1.0

    def test_evaluate_mean_rating(self):
        """Test evaluate with MeanRatingBaseline."""
        train = pd.DataFrame({
            "user_id": [1, 1, 2, 2],
            "movie_id": [1, 2, 1, 3],
            "rating": [4, 5, 3, 4],
            "timestamp": [1000, 1001, 1002, 1003],
        })
        test = pd.DataFrame({
            "user_id": [1, 2],
            "movie_id": [2, 3],  # User 1 liked item 2, user 2 liked item 3
            "rating": [5, 4],
            "timestamp": [2000, 2001],
        })

        model = MeanRatingBaseline(min_ratings=1)
        model.fit(train)
        metrics = evaluate(model, train, test, k=1)

        # Should compute metrics without error
        assert "precision_at_1" in metrics
        assert "recall_at_1" in metrics
        assert 0 <= metrics["precision_at_1"] <= 1
        assert 0 <= metrics["recall_at_1"] <= 1


class TestIntegration:
    """Integration tests with real MovieLens data."""

    @pytest.mark.slow
    def test_full_pipeline(self):
        """Test full pipeline on MovieLens 1M (if available)."""
        try:
            ratings_cols = ["user_id", "movie_id", "rating", "timestamp"]
            ratings = pd.read_csv(
                "data/ml-1m/ratings.dat",
                sep="::",
                names=ratings_cols,
                engine="python",
                encoding="latin-1",
            )
        except FileNotFoundError:
            pytest.skip("MovieLens data not available")

        train, test = time_based_split(ratings, test_ratio=0.2)

        # Popularity baseline
        pop_model = PopularityBaseline()
        pop_model.fit(train)
        pop_metrics = evaluate(pop_model, train, test, k=10)

        assert 0 <= pop_metrics["precision_at_10"] <= 1
        assert 0 <= pop_metrics["recall_at_10"] <= 1
        assert pop_metrics["n_users_evaluated"] > 0

        # Mean rating baseline
        mean_model = MeanRatingBaseline(min_ratings=5)
        mean_model.fit(train)
        mean_metrics = evaluate(mean_model, train, test, k=10)

        assert 0 <= mean_metrics["precision_at_10"] <= 1
        assert 0 <= mean_metrics["recall_at_10"] <= 1