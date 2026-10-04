# Specification: data-ingestion/movielens-1m

## Purpose

Downloads and stores the MovieLens 1M dataset locally for offline recommender system development.

## Requirements

### Requirement: Dataset download
The system SHALL download the MovieLens 1M dataset from the official GroupLens repository when executed.

#### Scenario: Successful download
- **WHEN** the download script runs
- **THEN** the dataset files (ratings.dat, users.dat, movies.dat) are saved to `data/ml-1m/`

#### Scenario: Resume partial download
- **WHEN** the download is interrupted and re-run
- **THEN** the script resumes or verifies existing files without re-downloading

### Requirement: Data integrity verification
The system SHALL verify downloaded file sizes and checksums against known values.

#### Scenario: Checksum validation passes
- **WHEN** files exist locally with correct checksums
- **THEN** the script reports success without re-downloading

#### Scenario: Checksum validation fails
- **WHEN** files are corrupted or incomplete
- **THEN** the script re-downloads the affected files