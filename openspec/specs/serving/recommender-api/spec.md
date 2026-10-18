# Specification: serving/recommender-api

## Purpose

Exposes the best available recommender model over HTTP so recommendations can
be consumed by any client, and keeps the serving contract testable.

## Requirements

### Requirement: Startup model fitting
The system SHALL fit the item-item KNN model on the full ratings file during
application startup (lifespan), unless a pre-fitted model is injected into
`create_app()`.

#### Scenario: Default startup
- **WHEN** the application starts without an injected model
- **THEN** an `ItemItemKNNRecommender` is fitted on the training ratings
  before requests are served

#### Scenario: Injected model
- **WHEN** `create_app()` receives a pre-fitted model
- **THEN** startup performs no fitting and serves the injected model

### Requirement: Health endpoint
The system SHALL provide `GET /health` reporting the service status, the
concrete model class and the number of users and items in the fitted model.

#### Scenario: Healthy service
- **WHEN** a client calls `/health` after startup
- **THEN** the response contains `status: "ok"`, the model class name, and
  `n_users` / `n_items` counts

### Requirement: Recommendation endpoint
The system SHALL provide `GET /recommendations/{user_id}` returning the top-K
unseen items with their titles, where K is a validated query parameter in
[1, 100] (default 10).

#### Scenario: Known user
- **WHEN** a client requests recommendations for a user present in training
- **THEN** the response contains `known_user: true` and K items, each with
  `movie_id` and `title`

#### Scenario: Cold-start user
- **WHEN** a client requests recommendations for an unknown user
- **THEN** the response contains `known_user: false` and popularity-fallback
  items instead of an error

#### Scenario: Invalid input
- **WHEN** a client requests `k=0`, `k=101`, or a non-integer user id
- **THEN** the service responds with HTTP 422 and does not crash

### Requirement: Model persistence
The system SHALL persist the fitted model as a joblib artifact and load it on
startup, so restarts do not repeat the fit. The artifact path comes from the
MODEL_PATH environment variable (default `artifacts/recommender.joblib`).

#### Scenario: Cold start without artifact
- **WHEN** the service starts and the artifact does not exist
- **THEN** the model is fitted on the ratings file and saved to MODEL_PATH

#### Scenario: Warm start
- **WHEN** the service starts and the artifact exists
- **THEN** the model is loaded from disk and no fitting happens

#### Scenario: Prebuilt artifact
- **WHEN** `python -m src.service.build_model` runs
- **THEN** the artifact is created or refreshed before the server starts

### Requirement: Environment configuration
The system SHALL read its configuration from environment variables: DATA_DIR
(ratings directory), MODEL_PATH (artifact), HOST and PORT (bind address).

#### Scenario: Container binding
- **WHEN** the service runs with HOST=0.0.0.0 and PORT=8000
- **THEN** uvicorn listens on all interfaces on the given port

### Requirement: Containerized deployment
The system SHALL provide a Dockerfile that installs the package, downloads
the SHA256-verified dataset, prebuilds the model artifact and starts the API
with a /health-based container healthcheck. CI SHALL build the image and
smoke-test the running container.

#### Scenario: Image build
- **WHEN** `docker build -t recommender .` runs on a clean checkout
- **THEN** the image contains the installed package, the dataset and the
  model artifact, and the container serves recommendations without fitting

### Requirement: Automated verification
The system SHALL cover the API with pytest tests using an injected model on
synthetic data, and CI SHALL run linting, the test suite, and an HTTP
smoke test of the real server.

#### Scenario: CI pipeline
- **WHEN** a commit is pushed or a pull request is opened
- **THEN** GitHub Actions installs the package, runs `ruff check`, downloads
  the dataset, runs `pytest`, and curls `/health` and
  `/recommendations/{user_id}` against a live server
