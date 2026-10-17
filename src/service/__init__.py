"""Serving layer: FastAPI application around the recommender models."""

from src.service.app import create_app

__all__ = ["create_app"]
