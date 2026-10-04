# Specification: models/baseline-recommender

## Purpose

Implements baseline recommender models (popularity-based and mean-rating) with time-based train/test split and evaluates them using Precision@10 and Recall@10 metrics.

## Requirements

### Requirement: Time-based train/test split
The system SHALL split the dataset chronologically per user, using the most recent interactions as test set.

#### Scenario: Per-user temporal split
- **WHEN** the split function runs
- **THEN** for each user, the last 20% of interactions by timestamp are held out as test, the rest as train

#### Scenario: No data leakage
- **WHEN** evaluating on test set
- **THEN** no test interaction timestamp is earlier than any train interaction for the same user

### Requirement: Popularity baseline model
The system SHALL recommend the globally most popular items (by rating count) to all users.

#### Scenario: Top-K popular items returned
- **WHEN** recommending for any user
- **THEN** the same top-K items by global popularity are returned regardless of user

### Requirement: Mean rating baseline model
The system SHALL fall back to global mean rating for items with insufficient interactions, and user mean for cold-start users.

#### Scenario: Item mean fallback
- **WHEN** an item has fewer than 5 ratings in training
- **THEN** the global mean rating is used for scoring

#### Scenario: User mean fallback
- **WHEN** a user has no training interactions
- **THEN** the global mean rating is used for all items

### Requirement: Precision@10 evaluation
The system SHALL compute Precision@10 on the time-based test set.

#### Scenario: Precision@10 computed
- **WHEN** evaluation runs
- **THEN** Precision@10 = (relevant items in top-10) / 10, averaged across all test users

### Requirement: Recall@10 evaluation
The system SHALL compute Recall@10 on the time-based test set.

#### Scenario: Recall@10 computed
- **WHEN** evaluation runs
- **THEN** Recall@10 = (relevant items in top-10) / (total relevant items for user), averaged across all test users