# recomendation

MovieLens 1M recommender system — from EDA to hybrid and collaborative
filtering models, with temporal evaluation and an OpenSpec-driven spec workflow.

## Project structure

```
data/ml-1m/               # MovieLens 1M (downloaded, git-ignored)
notebooks/
  01_eda.ipynb            # Exploratory data analysis
  01_eda_executed.ipynb   # Executed EDA
  02_models.ipynb         # Comparison of all 9 models + alpha tuning
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
  collaborative.py        # item-item KNN, truncated SVD, implicit ALS
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

# Collaborative filtering (item-item KNN + SVD + implicit ALS)
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
- **Metrics**: macro-averaged Precision@10, Recall@10, NDCG@10 and MAP@10 over
  test users, plus catalogue coverage and average novelty in bits
  (`evaluate()`).

## Results (MovieLens 1M, temporal 80/20 split, k=10)

797 758 train / 202 451 test ratings, 6 040 users.

| Model | P@10 | R@10 | NDCG@10 | Coverage | Novelty |
|---|---|---|---|---|---|
| Random | 0.0090 | 0.0028 | 0.0094 | 1.000 | 13.44 |
| Popularity | 0.0433 | 0.0227 | 0.0453 | 0.003 | 8.30 |
| MeanRating | 0.0174 | 0.0075 | 0.0140 | 0.003 | 11.94 |
| SVD (50 factors) | 0.0031 | 0.0003 | 0.0051 | 0.036 | 19.22 |
| ALS (implicit, 50 factors) | 0.0036 | 0.0007 | 0.0038 | 0.105 | 13.86 |
| ContentBased | 0.0087 | 0.0024 | 0.0089 | 0.511 | 13.68 |
| Hybrid (α=0.2) | 0.1029 | 0.0378 | 0.1119 | 0.031 | 8.46 |
| CFHybrid (0.7 CF / 0.1 content / 0.2 pop) | 0.1297 | 0.0578 | 0.1425 | 0.150 | 8.80 |
| ItemItemKNN (50 neighbours) | **0.1334** | **0.0608** | **0.1463** | 0.184 | 8.95 |

Item-item KNN uses raw ratings (no per-user centering); `n_neighbors=50` is the
best of a sweep over {5, 10, 20, 50, 100, 200, 400} (0.1233 → 0.1335, flat).
CFHybrid weights come from `tune_cf_weights_on_val` on the validation split.
ALS is a sweep over factors {50, 100}, alpha {20, 40, 100} and reg
{0.01, 0.1, 1.0} — best P@10 = 0.0042 (factors=100).

Takeaways:
- **Item-item KNN is the best model on every accuracy metric**: +29%
  Precision@10 and +21% NDCG@10 over the content+popularity hybrid. Co-rating
  behaviour is a much stronger signal than item metadata on MovieLens 1M.
- **The metrics separate the failure modes**:
  - Popularity recommends ~11 distinct items to everyone (coverage 0.003) — it
    works only because the temporal test set skews popular.
  - SVD is the most novel (19.2 bits) and the least accurate despite the best
    fit (train RMSE 0.78): RMSE-optimized factors surface items with 1-2
    ratings.
  - ContentBased has the best coverage (0.511) of any useful model, but its
    precision stays below the popularity baseline.
  - ItemItemKNN sits in the useful middle: 0.184 coverage at the top accuracy.
- **Implicit ALS does not fix the latent-factor problem here** (0.0036 vs SVD's
  0.0031): the preference objective helps, but the items excluded as "seen"
  are exactly the ones the model fits best, so the top of the unseen ranking is
  still niche items. On this temporal split, beating item-item CF requires
  co-occurrence signal, not a better factorization objective.
- **Three-way blending does not help**: adding content and popularity weights
  on top of CF (0.1297) is slightly worse than pure CF (0.1333), and the
  validation-tuned weights (0.7/0.1) did not transfer to the test split.
- **Centering ratings hurts** item-item CF (0.1080 vs 0.1334): raw positive
  values make the cosine similarity carry the popularity signal that the
  temporal test set rewards.

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
- [x] Stage 6 — richer metrics (NDCG@K, MAP, coverage, novelty) + implicit-feedback
      ALS (documented negative result: item-item KNN still wins)
- [ ] Stage 7 — serving (FastAPI) and CI
