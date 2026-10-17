# Specification: models/content-based-recommender

## Purpose

Recommends items by matching item metadata (genres, release decade, TF-IDF over
genre tokens) against a rating-weighted user profile.

## Requirements

### Requirement: Item feature construction
The system SHALL build item vectors from genre one-hot encoding, decade
one-hot encoding, and TF-IDF over genre tokens, restricted to items present in
training data.

#### Scenario: Features come only from training items
- **WHEN** the model is fitted on training ratings
- **THEN** item vectors exist only for items that appear in training

#### Scenario: Missing metadata is tolerated
- **WHEN** a movie has no parseable year or `(no genres listed)`
- **THEN** the item is still represented by its remaining features

### Requirement: User profiles from liked items
The system SHALL build a user profile as the rating-weighted mean of vectors for
items the user liked (rating >= `min_rating_for_like`).

#### Scenario: Disliked items do not enter the profile
- **WHEN** a user rated an item below the like threshold
- **THEN** that item's vector is not part of the profile

### Requirement: Cosine recommendations
The system SHALL recommend the top-K unseen items by cosine similarity to the
user profile, falling back to popularity for cold-start users.

#### Scenario: Seen items are excluded
- **WHEN** recommendations are generated
- **THEN** items the user rated in training are never returned

#### Scenario: Cold-start user gets popular items
- **WHEN** a user has no liked items in training
- **THEN** popularity recommendations are returned
