# recomendation

MovieLens 1M recommender system — from EDA to hybrid (content-based + popularity) models,
with temporal evaluation and an OpenSpec-driven spec workflow.

## Project structure

```
data/ml-1m/               # MovieLens 1M (downloaded, git-ignored)
notebooks/
  01_eda.ipynb            # Exploratory data analysis
  01_eda_executed.ipynb   # Executed EDA
  02_models.ipynb         # Comparison of all 5 models + alpha tuning
openspec/
  specs/                  # Specs: data-ingestion, eda, models
  changes/                # Active changes (archive/ holds finished ones)
scripts/
  download_movielens.py   # Download + SHA256-verified dataset fetch
  benchmark.py            # Reproduce the model comparison table
src/models/
  baseline.py             # Random, Popularity, MeanRating + splits/metrics
  content_based.py        # TF-IDF item vectors + user profiles
  hybrid.py               # score = α·cosine + (1−α)·normalized popularity
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
| Random | 0.0090 | 0.0028 | ~2 s |
| Popularity | 0.0433 | 0.0227 | <1 s |
| MeanRating | 0.0174 | 0.0075 | ~13 s |
| ContentBased | 0.0087 | 0.0024 | ~30 s |
| Hybrid (α=0.2) | **0.1029** | **0.0378** | ~75 s |

Best α = 0.2, tuned on a separate validation split by Precision@10.

Takeaways:
- Content-based alone is worse than the popularity baseline — sparse TF-IDF
  features over genres/title/year are not enough signal.
- Combining it with popularity (α=0.2) more than doubles Popularity's P@10;
  the two signals are complementary.
- Absolute numbers are low because only liked items (rating ≥ 4) in the held-out
  tail are relevant — expected for a 10-item cutoff on 3 700+ movies.

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
- [ ] Stage 5 — collaborative filtering (item-item KNN, matrix factorization)
- [ ] Stage 6 — richer metrics (NDCG@K, MAP, coverage/diversity) + unified benchmark
- [ ] Stage 7 — serving (FastAPI) and CI
