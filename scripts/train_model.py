#!/usr/bin/env python3
"""
Train the full MMA model (ELO, features, multinomial regression) and write ``model.pkl``.

Implements the same command as ``python main.py train``; logic lives in ``src/cli/train.py``.

By default Tier-1 regression training excludes fights on/after the configured holdout
(default 2023-01-01); use ``--no-holdout`` only to train on all post-era rows (e.g. shipping refit).

Run from the repository root (recommended)::

    python scripts/train_model.py
    python scripts/train_model.py --data-dir ./data --model-path ./out/model.pkl
    python scripts/train_model.py --no-holdout
    python scripts/train_model.py --holdout-start 2022-06-01
    python scripts/train_model.py --elo-cache ./data/elo_model_cache.pkl
    python scripts/train_model.py --full-rebuild --no-scrape
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cli.train import build_train_parser, cmd_train  # noqa: E402


def main() -> None:
    args = build_train_parser().parse_args()
    cmd_train(args)


if __name__ == "__main__":
    main()
