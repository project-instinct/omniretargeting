"""Fallback locations of the licensed SMPL-X model files."""
from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def smplx_model_search_paths(explicit=None):
    """Explicit directory first, then ``$OMNIRETARGETING_SMPLX_MODELS`` and the repository defaults."""
    paths = [explicit, os.environ.get("OMNIRETARGETING_SMPLX_MODELS"),
             str(_REPO_ROOT / "assets/body_models"), "data/body_models/smplx"]
    return [str(path) for path in paths if path]
