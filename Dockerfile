FROM python:3.11-slim

WORKDIR /app

# Dependencies first for layer caching (package install also makes
# `import src.*` available inside the image)
COPY pyproject.toml README.md ./
COPY src ./src
COPY scripts ./scripts
RUN pip install --no-cache-dir .

# Dataset (SHA256-verified, skips when present) and prebuilt model artifact,
# so the container starts without the ~15 s fit
RUN python scripts/download_movielens.py \
    && python -m src.service.build_model

ENV HOST=0.0.0.0 \
    PORT=8000 \
    MODEL_PATH=artifacts/recommender.joblib

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request, sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).status == 200 else 1)"

CMD ["python", "-m", "src.service.app"]
