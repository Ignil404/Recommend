# Specification: models/collaborative-filtering

## Purpose

Implements collaborative filtering models that learn from user-item interaction
patterns rather than item metadata: item-item KNN over centered ratings and
truncated SVD latent factors. Both reuse the shared evaluation protocol
(temporal split, Precision@10 / Recall@10).

## Requirements

### Requirement: Shared indexing and cold-start fallback
The system SHALL map raw user and item IDs to contiguous indices, and SHALL
fall back to popularity-based recommendations for users with no training
interactions.

#### Scenario: Cold-start user gets popular items
- **WHEN** `recommend()` is called for a user ID absent from training data
- **THEN** the same top-K items as the popularity baseline are returned

#### Scenario: Plain integer IDs returned
- **WHEN** recommendations are returned by any collaborative model
- **THEN** every item ID is a Python `int` (JSON-serializable), not a NumPy scalar

### Requirement: Item-item KNN model
The system SHALL compute item-item cosine similarity on the user-item matrix of
raw ratings (with an optional per-user mean centered variant), keeping only the
top-N neighbours per item.

#### Scenario: Score by similarity to liked items
- **WHEN** scoring an unseen item for a user
- **THEN** the score is the sum over the user's liked items of
  similarity(item, liked) * (rating - user_mean)

#### Scenario: Seen items are excluded
- **WHEN** recommendations are generated for a user
- **THEN** no item the user rated in training appears in the output

#### Scenario: Deterministic fit
- **WHEN** the model is fitted twice on the same training data
- **THEN** the resulting recommendations are identical

### Requirement: SVD latent-factor model
The system SHALL factorize the rating matrix residual (global mean and per-user,
per-item biases removed) into `n_factors` latent factors with a deterministic
start vector.

#### Scenario: Rating prediction formula
- **WHEN** a rating is predicted for a known user-item pair
- **THEN** the prediction equals global_mean + user_bias + item_bias + P_u · Q_i

#### Scenario: Unknown pair falls back to global mean
- **WHEN** a rating is predicted for an unknown user or item
- **THEN** the global mean rating is returned

#### Scenario: Reproducible factorization
- **WHEN** the model is fitted twice on the same training data
- **THEN** the latent factors (and therefore the recommendations) match

### Requirement: Evaluation on the shared protocol
The system SHALL be evaluated with the shared benchmark script on the temporal
80/20 split at k=10, alongside all other models.

#### Scenario: Benchmark covers collaborative models
- **WHEN** `python -m scripts.benchmark` runs
- **THEN** item-item KNN and SVD results are reported with the other models
