# Specification: eda/movielens-analysis

## Purpose

Performs exploratory data analysis on the MovieLens 1M dataset to understand rating distributions, sparsity patterns, user activity levels, and item popularity long-tail characteristics.

## Requirements

### Requirement: Rating distribution analysis
The system SHALL compute and visualize the distribution of ratings (1-5) across the dataset.

#### Scenario: Rating histogram generated
- **WHEN** the EDA notebook runs
- **THEN** a histogram showing count of each rating value (1 through 5) is produced with summary statistics (mean, median, std)

### Requirement: Sparsity analysis
The system SHALL calculate and report the sparsity of the user-item interaction matrix.

#### Scenario: Sparsity metrics reported
- **WHEN** the EDA notebook runs
- **THEN** the sparsity percentage (1 - (num_ratings / (num_users * num_items))) is computed and displayed

### Requirement: User activity analysis
The system SHALL analyze the distribution of ratings per user.

#### Scenario: User activity distribution visualized
- **WHEN** the EDA notebook runs
- **THEN** a histogram/log-log plot of ratings-per-user is produced with percentiles (50th, 90th, 99th)

### Requirement: Item popularity long-tail analysis
The system SHALL analyze the distribution of ratings per item (movie) to characterize the long tail.

#### Scenario: Item popularity distribution visualized
- **WHEN** the EDA notebook runs
- **THEN** a log-log plot of item popularity (ratings per movie) is produced showing the Pareto/Zipf distribution