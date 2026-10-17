# Specification: evaluation/ranking-metrics

## Purpose

Evaluates recommenders with ranking-aware and diversity metrics, so that models
can be compared not only by hit rate but by where hits land in the list and how
broad the recommendations are.

## Requirements

### Requirement: NDCG@K
The system SHALL compute Normalized Discounted Cumulative Gain@K with binary
relevance, discounting hits by `1 / log2(rank + 1)` and normalizing by the ideal
ranking of the same relevant set.

#### Scenario: Perfect ranking
- **WHEN** all relevant items appear first in the recommendation list
- **THEN** NDCG@K equals 1.0

#### Scenario: Rank discount
- **WHEN** the same hit moves to a lower rank
- **THEN** NDCG@K decreases

### Requirement: MAP@K
The system SHALL compute Average Precision@K, averaging precision at each hit
position and normalizing by `min(|relevant|, K)`.

#### Scenario: Hand-computed case
- **WHEN** recommendations are [1, 2, 3], relevant items are {1, 3} and K = 3
- **THEN** AP@K equals `(1/1 + 2/3) / 2`

### Requirement: Coverage and novelty
The system SHALL report catalogue coverage (share of distinct training items
that appear in any user's top-K) and average novelty, defined as the mean
self-information `-log2(count / total)` of the recommended items.

#### Scenario: Popularity has low coverage and novelty
- **WHEN** the popularity baseline is evaluated
- **THEN** coverage equals the share of the catalogue in its top-K list and
  novelty is lower than for a model recommending rare items

#### Scenario: Obscure items are more novel
- **WHEN** two items have different training rating counts
- **THEN** the less rated item has higher novelty

### Requirement: Extended evaluate() contract
The system SHALL extend `evaluate()` to return, alongside precision and recall,
`ndcg_at_k`, `map_at_k`, `coverage`, and `novelty`, without breaking existing
callers that read the precision/recall keys.

#### Scenario: Backwards-compatible keys
- **WHEN** any existing model is evaluated
- **THEN** `precision_at_k` and `recall_at_k` are still present and unchanged
