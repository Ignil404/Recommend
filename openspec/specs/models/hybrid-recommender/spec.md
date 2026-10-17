# Specification: models/hybrid-recommender

## Purpose

Combines the content-based cosine signal with the popularity signal in a single
tunable score, so that weak-but-personal content features are regularized by
global popularity.

## Requirements

### Requirement: Tunable blended score
The system SHALL score items as
`alpha * cosine_similarity + (1 - alpha) * minmax_normalized_popularity`, with
popularity computed on training data only.

#### Scenario: Alpha equals zero
- **WHEN** alpha = 0.0
- **THEN** the ranking matches the popularity baseline exactly

#### Scenario: Alpha equals one
- **WHEN** alpha = 1.0
- **THEN** the ranking matches the pure content-based model exactly

#### Scenario: Popularity normalization
- **WHEN** popularity is normalized
- **THEN** min-max normalization maps the least to 0.0 and the most popular
  item to 1.0

### Requirement: Centered user profiles
The system SHALL build user profiles from liked items (rating >= threshold)
weighted by (rating - user mean), so strict raters do not dominate.

#### Scenario: Profile weights are centered
- **WHEN** a user rates an item above their own mean
- **THEN** that item contributes a positive weight to the profile

### Requirement: Temporal validation tuning
The system SHALL tune alpha on a validation split with a chronological
train/val/test partition per user, and report test metrics for the best alpha.

#### Scenario: No temporal leakage during tuning
- **WHEN** alpha is tuned on the validation split
- **THEN** no validation interaction precedes a training interaction for the
  same user

### Requirement: Cold-start fallback
The system SHALL fall back to popularity for users with no liked training items.

#### Scenario: Unknown user
- **WHEN** recommendations are requested for a user without training history
- **THEN** popularity recommendations are returned
