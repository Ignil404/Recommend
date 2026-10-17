"""
Content-based recommender model for MovieLens dataset.

Uses item features (genres, release decade) and TF-IDF over genre tokens
to build item profiles, then recommends items similar to user's profile.
"""

from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer

from src.models.baseline import BaseRecommender, PopularityBaseline

logger = logging.getLogger(__name__)


def extract_year_from_title(title: str) -> int | None:
    """Extract release year from movie title.

    MovieLens titles typically end with "(YYYY)".

    Args:
        title: Movie title string

    Returns:
        Year as integer, or None if not found
    """
    match = re.search(r"\((\d{4})\)$", title.strip())
    if match:
        return int(match.group(1))
    return None


def decade_from_year(year: int | None) -> str | None:
    """Convert year to decade string for one-hot encoding.

    Args:
        year: Release year

    Returns:
        Decade string like "1990s", or None if year is None
    """
    if year is None:
        return None
    return f"{(year // 10) * 10}s"


def parse_genres(genres_str: str) -> list[str]:
    """Parse pipe-separated genre string into list.

    Args:
        genres_str: Genre string like "Action|Adventure|Sci-Fi"

    Returns:
        List of genre strings
    """
    if not genres_str or genres_str == "(no genres listed)":
        return []
    return [g.strip() for g in genres_str.split("|")]


class ContentBasedRecommender(BaseRecommender):
    """Content-based recommender using item features and user profiles.

    Item features combine:
    - Genre one-hot encoding (MultiLabelBinarizer)
    - Decade one-hot encoding (from title year)
    - TF-IDF over genre tokens (captures genre co-occurrence patterns)

    User profile = rating-weighted mean of liked item vectors (rating >= 4).
    Recommendations = top-K unseen items by cosine similarity to user profile.
    Cold-start fallback = PopularityBaseline.
    """

    def __init__(
        self,
        min_rating_for_like: float = 4.0,
        tfidf_max_features: int = 100,
        tfidf_ngram_range: tuple[int, int] = (1, 2),
    ) -> None:
        self.min_rating_for_like = min_rating_for_like
        self.tfidf_max_features = tfidf_max_features
        self.tfidf_ngram_range = tfidf_ngram_range

        # Fitted components
        self.mlb_genres_: MultiLabelBinarizer | None = None
        self.mlb_decades_: MultiLabelBinarizer | None = None
        self.tfidf_vectorizer_: TfidfVectorizer | None = None
        self.item_vectors_: np.ndarray | None = None
        self.item_id_to_idx_: dict[int, int] = {}
        self.idx_to_item_id_: dict[int, int] = {}
        self.similarity_matrix_: np.ndarray | None = None
        self.user_profiles_: dict[int, np.ndarray] = {}
        self.popularity_fallback_: PopularityBaseline = PopularityBaseline()
        self.train_ratings_: pd.DataFrame | None = None
        self.movies_df_: pd.DataFrame | None = None

    def fit(self, train_ratings: pd.DataFrame, data_dir: str = "data/ml-1m") -> None:
        """Fit content-based model on training data.

        Args:
            train_ratings: DataFrame with columns [user_id, movie_id, rating, timestamp]
            data_dir: Directory containing movies.dat, users.dat, ratings.dat
        """
        self.train_ratings_ = train_ratings.copy()

        # Load movie metadata
        self.movies_df_ = pd.read_csv(
            f"{data_dir}/movies.dat",
            sep="::",
            names=["movie_id", "title", "genres"],
            engine="python",
            encoding="latin-1",
        )

        # Build item features
        self._build_item_features()

        # Build user profiles from training data
        self._build_user_profiles()

        # Fit popularity fallback for cold-start
        self.popularity_fallback_.fit(train_ratings)

        vec_dim = self.item_vectors_.shape[1] if self.item_vectors_ is not None else 0
        logger.info(
            f"ContentBasedRecommender fitted: {len(self.item_id_to_idx_)} items, "
            f"{len(self.user_profiles_)} user profiles, item vector dim={vec_dim}"
        )

    def _build_item_features(self) -> None:
        """Build combined item feature vectors from genres, decade, and TF-IDF."""
        movies = self.movies_df_.copy()

        # Parse genres
        movies["genre_list"] = movies["genres"].apply(parse_genres)

        # Extract decade from title
        movies["year"] = movies["title"].apply(extract_year_from_title)
        movies["decade"] = movies["year"].apply(decade_from_year)
        movies["decade_list"] = movies["decade"].apply(lambda d: [d] if d else [])

        # Genre one-hot encoding
        self.mlb_genres_ = MultiLabelBinarizer()
        genre_features = self.mlb_genres_.fit_transform(movies["genre_list"])

        # Decade one-hot encoding
        self.mlb_decades_ = MultiLabelBinarizer()
        decade_features = self.mlb_decades_.fit_transform(movies["decade_list"])

        # TF-IDF over genre tokens (joined as text)
        movies["genre_text"] = movies["genre_list"].apply(lambda g: " ".join(g))
        self.tfidf_vectorizer_ = TfidfVectorizer(
            max_features=self.tfidf_max_features,
            ngram_range=self.tfidf_ngram_range,
            lowercase=True,
        )
        tfidf_features = (
            self.tfidf_vectorizer_.fit_transform(movies["genre_text"]).toarray()
        )

        # Combine all features
        combined_features = np.hstack([genre_features, decade_features, tfidf_features])

        # Store item mappings (only for items present in training ratings)
        train_items = set(self.train_ratings_["movie_id"].unique())
        movies = movies[movies["movie_id"].isin(train_items)].reset_index(drop=True)

        self.item_id_to_idx_ = {row["movie_id"]: i for i, row in movies.iterrows()}
        self.idx_to_item_id_ = {i: row["movie_id"] for i, row in movies.iterrows()}

        # Subset features to training items only
        train_indices = [
            self.item_id_to_idx_[mid] for mid in sorted(self.item_id_to_idx_.keys())
        ]
        self.item_vectors_ = combined_features[train_indices]

        # Compute item-item cosine similarity matrix
        self.similarity_matrix_ = cosine_similarity(self.item_vectors_)

        logger.debug(
            f"Item features: genres={genre_features.shape[1]}, "
            f"decades={decade_features.shape[1]}, tfidf={tfidf_features.shape[1]}"
        )

    def _build_user_profiles(self) -> None:
        """Build user profiles as rating-weighted mean of liked item vectors.

        Liked items = items rated >= min_rating_for_like in training data.
        """
        assert self.train_ratings_ is not None
        assert self.item_vectors_ is not None

        # Filter to liked items (rating >= threshold)
        liked_mask = self.train_ratings_["rating"] >= self.min_rating_for_like
        liked = self.train_ratings_[liked_mask]

        if liked.empty:
            logger.warning("No liked items found in training data")
            return

        # Build profile per user
        for user_id, group in liked.groupby("user_id"):
            item_ids = group["movie_id"].tolist()
            ratings = group["rating"].tolist()

            # Get vectors for items that exist in our item set
            valid_items = []
            valid_weights = []
            for item_id, rating in zip(item_ids, ratings):
                if item_id in self.item_id_to_idx_:
                    valid_items.append(self.item_id_to_idx_[item_id])
                    valid_weights.append(rating)

            if not valid_items:
                continue

            # Weighted mean of item vectors
            item_vecs = self.item_vectors_[valid_items]
            weights = np.array(valid_weights, dtype=float)
            weights = weights / weights.sum()  # Normalize

            user_profile = np.average(item_vecs, axis=0, weights=weights)
            self.user_profiles_[user_id] = user_profile

        logger.debug(f"Built profiles for {len(self.user_profiles_)} users")

    def recommend(self, user_id: int, k: int = 10) -> list[int]:
        """Recommend top-K unseen items by similarity to user profile.

        Args:
            user_id: User ID to recommend for
            k: Number of recommendations

        Returns:
            List of top-K movie_ids by cosine similarity to user profile
        """
        # Cold-start: no profile -> fallback to popularity
        if user_id not in self.user_profiles_ or self.similarity_matrix_ is None:
            logger.debug(f"Cold-start user {user_id}, falling back to popularity")
            return self.popularity_fallback_.recommend(user_id, k)

        # Get user profile
        user_profile = self.user_profiles_[user_id].reshape(1, -1)

        # Compute similarity to all items
        item_sims = cosine_similarity(user_profile, self.item_vectors_).flatten()

        # Get items user has already seen in training
        seen_items: set[int] = set()
        if self.train_ratings_ is not None:
            user_ratings = self.train_ratings_
            user_movies = user_ratings[user_ratings["user_id"] == user_id]
            seen_items = set(user_movies["movie_id"].tolist())

        # Sort items by similarity, excluding seen
        item_indices = np.argsort(item_sims)[::-1]  # Descending

        recommendations = []
        for idx in item_indices:
            item_id = self.idx_to_item_id_[idx]
            if item_id not in seen_items:
                recommendations.append(item_id)
                if len(recommendations) >= k:
                    break

        # If not enough unseen items, pad with popularity fallback
        if len(recommendations) < k:
            n_needed = k - len(recommendations)
            fallback = self.popularity_fallback_.recommend(user_id, n_needed)
            for item_id in fallback:
                if item_id not in seen_items and item_id not in recommendations:
                    recommendations.append(item_id)
                    if len(recommendations) >= k:
                        break

        return recommendations[:k]

    def get_item_vector(self, item_id: int) -> np.ndarray | None:
        """Get feature vector for an item.

        Args:
            item_id: Movie ID

        Returns:
            Item feature vector, or None if item not in training set
        """
        if item_id not in self.item_id_to_idx_:
            return None
        return self.item_vectors_[self.item_id_to_idx_[item_id]]

    def get_similar_items(self, item_id: int, k: int = 10) -> list[tuple[int, float]]:
        """Get top-K most similar items to a given item.

        Args:
            item_id: Movie ID
            k: Number of similar items to return

        Returns:
            List of (item_id, similarity) tuples
        """
        if item_id not in self.item_id_to_idx_ or self.similarity_matrix_ is None:
            return []

        idx = self.item_id_to_idx_[item_id]
        sims = self.similarity_matrix_[idx]
        similar_indices = np.argsort(sims)[::-1][1:k+1]  # Exclude self

        return [(self.idx_to_item_id_[i], float(sims[i])) for i in similar_indices]


def main() -> None:
    """Main function to run content-based evaluation on MovieLens 1M."""
    import sys
    sys.path.insert(0, ".")
    from src.models.baseline import evaluate, time_based_split

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

    print(
        f"Loaded {len(ratings)} ratings from {ratings['user_id'].nunique()} users "
        f"and {ratings['movie_id'].nunique()} movies"
    )

    # Time-based split
    train, test = time_based_split(ratings, test_ratio=0.2)
    print(f"Train: {len(train)}, Test: {len(test)}")

    # Content-Based
    print("\n--- Content-Based ---")
    cb_model = ContentBasedRecommender(min_rating_for_like=4.0)
    cb_model.fit(train)
    cb_metrics = evaluate(cb_model, train, test, k=10)
    print(f"Precision@10: {cb_metrics['precision_at_10']:.4f}")
    print(f"Recall@10:    {cb_metrics['recall_at_10']:.4f}")


if __name__ == "__main__":
    main()