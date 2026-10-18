"""Build (or refresh) the model artifact used by the serving layer.

Run before starting the server to move the ~15 s fit out of startup:

    python -m src.service.build_model
"""

from __future__ import annotations

import logging

from src.service.app import ensure_model


def main() -> None:
    """Fit the model on the ratings file and persist it."""
    logging.basicConfig(level=logging.INFO)
    model = ensure_model()
    n_users = len(getattr(model, "user_id_to_idx_", {}))
    n_items = len(getattr(model, "item_id_to_idx_", {}))
    print(f"Artifact ready: {type(model).__name__} ({n_users} users, {n_items} items)")


if __name__ == "__main__":
    main()
