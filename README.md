# recomendation

MovieLens 1M recommender system — from EDA to hybrid and collaborative
filtering models, with temporal evaluation and an OpenSpec-driven spec workflow.

## Project structure

```
data/ml-1m/               # MovieLens 1M (downloaded, git-ignored)
notebooks/
  01_eda.ipynb            # Exploratory data analysis
  01_eda_executed.ipynb   # Executed EDA
  02_models.ipynb         # Comparison of all 7 models + alpha tuning
openspec/
  specs/                  # Specs: data-ingestion, eda, models/*
  changes/                # Active changes (archive/ holds finished ones)
scripts/
  download_movielens.py   # Download + SHA256-verified dataset fetch
  benchmark.py            # Reproduce the model comparison table
src/models/
  baseline.py             # Random, Popularity, MeanRating + splits/metrics
  content_based.py        # TF-IDF item vectors + user profiles
  hybrid.py               # score = α·cosine + (1−α)·normalized popularity
  collaborative.py        # item-item KNN + truncated SVD latent factors
tests/                    # pytest suite
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Download the dataset (SHA256-verified):

```bash
python scripts/download_movielens.py
```

## Usage

```bash
# Evaluate baselines (Random / Popularity / MeanRating)
python -m src.models.baseline

# Content-based recommender
python -m src.models.content_based

# Hybrid recommender with alpha tuning on the validation set
python -m src.models.hybrid

# Collaborative filtering (item-item KNN + truncated SVD)
python -m src.models.collaborative

# Full model comparison
jupyter nbconvert --to notebook --execute notebooks/02_models.ipynb

# Reproduce the results table below
python -m scripts.benchmark
```

## Evaluation protocol

- **Temporal split**: per user, the most recent interactions are held out as test
  (`time_based_split`), with an optional train/val/test variant for tuning
  (`time_based_train_val_test_split`).
- **No leakage**: `verify_no_leakage()` asserts no test timestamp precedes a train
  timestamp for the same user.
- **Metrics**: macro-averaged Precision@10 and Recall@10 over test users
  (`evaluate()`).

## Results (MovieLens 1M, temporal 80/20 split, k=10)

797 758 train / 202 451 test ratings, 6 040 users.

| Model | Precision@10 | Recall@10 | Fit+eval time |
|---|---|---|---|
| Random | 0.0090 | 0.0028 | ~1 s |
| Popularity | 0.0433 | 0.0227 | <1 s |
| MeanRating | 0.0174 | 0.0075 | ~4 s |
| SVD (50 factors) | 0.0031 | 0.0003 | ~2 s |
| ContentBased | 0.0087 | 0.0024 | ~10 s |
| Hybrid (α=0.2) | 0.1029 | 0.0378 | ~20 s |
| ItemItemKNN (50 neighbours) | **0.1334** | **0.0608** | ~13 s |
| CFHybrid (0.7 CF / 0.1 content / 0.2 pop) | 0.1297 | 0.0578 | ~19 s |

Item-item KNN uses raw ratings (no per-user centering); `n_neighbors=50` is the
best of a sweep over {5, 10, 20, 50, 100, 200, 400} (0.1233 → 0.1335, flat).
CFHybrid weights come from `tune_cf_weights_on_val` on the validation split.

Takeaways:
- **Item-item KNN is the best model**: +29% Precision@10 and +61% Recall@10 over
  the content+popularity hybrid. Co-rating behaviour is a much stronger signal
  than item metadata on MovieLens 1M.
- **Three-way blending does not help**: adding content and popularity weights on
  top of CF (0.1297) is slightly worse than pure CF (0.1333). Co-rating
  similarity already carries the popularity signal, and the content signal is
  too weak to add anything. The validation-tuned weights (0.7/0.1) also did not
  transfer to the test split.
- **Centering ratings hurts** (0.1080 vs 0.1334): raw positive values make the
  cosine similarity carry the popularity signal that the temporal test set
  rewards.
- **SVD is the worst model despite the best fit** (train RMSE 0.78): ranking by
  predicted rating surfaces obscure items with 1-2 training ratings instead of
  broadly liked ones. Optimizing RMSE does not optimize top-N ranking —
  implicit-feedback models (ALS on binary interactions) are the fix.
- Content-based alone is worse than the popularity baseline; its signal only
  helps when blended with popularity (α=0.2).

## Quality checks

```bash
pytest -q                 # full suite (~7 min; includes slow tests)
pytest -q -m "not slow"   # fast subset
ruff check .              # lint
```

## Roadmap

- [x] Stage 1 — data ingestion (MovieLens 1M, verified download) + spec
- [x] Stage 2 — EDA + spec
- [x] Stage 3 — baselines (Popularity / MeanRating / Random) + temporal split + metrics
- [x] Stage 4 — content-based and hybrid models with α tuning
- [x] Stage 5 — collaborative filtering (item-item KNN, truncated SVD) + specs for
      all models
- [ ] Stage 6 — richer metrics (NDCG@K, MAP, coverage/diversity), ranking-aware
      matrix factorization (implicit-feedback ALS)
- [ ] Stage 7 — serving (FastAPI) and CI
