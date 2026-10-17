"""Benchmark all recommender models on MovieLens 1M (temporal 80/20 split).

Usage (from the repository root):
    python -m scripts.benchmark
"""

import logging
import time

import pandas as pd

from src.models.baseline import (
    MeanRatingBaseline,
    PopularityBaseline,
    RandomRecommender,
    evaluate,
    time_based_split,
)
from src.models.collaborative import ItemItemKNNRecommender, SVDRecommender
from src.models.content_based import ContentBasedRecommender
from src.models.hybrid import CFHybridRecommender, HybridRecommender

DATA_DIR = "data/ml-1m"
RATINGS_COLS = ["user_id", "movie_id", "rating", "timestamp"]
K = 10
BEST_ALPHA = 0.2
# Tuned on the validation split by tune_cf_weights_on_val
BEST_CF = (0.7, 0.1)


def load_ratings() -> pd.DataFrame:
    """Load raw MovieLens 1M ratings."""
    return pd.read_csv(
        f"{DATA_DIR}/ratings.dat",
        sep="::",
        names=RATINGS_COLS,
        engine="python",
        encoding="latin-1",
    )


def run(name: str, model, train: pd.DataFrame, test: pd.DataFrame) -> dict[str, float]:
    """Fit a model and evaluate it, printing and returning the metrics."""
    start = time.time()
    model.fit(train)
    metrics = evaluate(model, train, test, k=K)
    elapsed = time.time() - start
    print(
        f"{name:15s} P@{K}={metrics['precision_at_10']:.4f} "
        f"R@{K}={metrics['recall_at_10']:.4f} ({elapsed:.1f}s)"
    )
    return metrics


def main() -> None:
    """Run the full benchmark."""
    logging.basicConfig(level=logging.WARNING)

    ratings = load_ratings()
    train, test = time_based_split(ratings, test_ratio=0.2)
    print(
        f"train={len(train)} test={len(test)} "
        f"users={train['user_id'].nunique()}\n"
    )

    run("Random", RandomRecommender(seed=42), train, test)
    run("Popularity", PopularityBaseline(), train, test)
    run("MeanRating", MeanRatingBaseline(min_ratings=5), train, test)
    run("ItemItemKNN", ItemItemKNNRecommender(n_neighbors=50), train, test)
    run("SVD", SVDRecommender(n_factors=50), train, test)
    run("ContentBased", ContentBasedRecommender(min_rating_for_like=4.0), train, test)
    run(f"Hybrid(a={BEST_ALPHA})", HybridRecommender(alpha=BEST_ALPHA), train, test)
    run(
        f"CFHybrid(cf={BEST_CF[0]},c={BEST_CF[1]})",
        CFHybridRecommender(cf_weight=BEST_CF[0], content_weight=BEST_CF[1]),
        train,
        test,
    )


if __name__ == "__main__":
    main()
