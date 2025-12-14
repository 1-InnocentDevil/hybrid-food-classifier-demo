# src/paths.py
from __future__ import annotations

import os
from pathlib import Path

def get_project_root() -> Path:
    """Return repository root.

    Priority:
    1) HYBRID_ROOT environment variable (absolute or relative path)
    2) Repo root inferred from this file location (src/paths.py -> repo root)
    """
    env = os.getenv("HYBRID_ROOT", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return Path(__file__).resolve().parents[1]

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
