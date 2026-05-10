"""Optional Git metadata for export manifests (avoid duplicating subprocess logic)."""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional


def repo_root_from_here() -> Path:
    """``src/export/git_meta.py`` -> training repo root (parent of ``src``)."""
    return Path(__file__).resolve().parents[2]


def git_sha_training_repo(*, cwd: Optional[Path] = None) -> Optional[str]:
    root = cwd if cwd is not None else repo_root_from_here()
    try:
        cp = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if cp.returncode == 0 and cp.stdout:
            return cp.stdout.strip()
    except OSError:
        pass
    return None
