#!/usr/bin/env python3
"""
Weekly pipeline: reload data, rebuild ELO, refresh or refit regression state, export deploy JSON.

**refresh (steps 1–5)** — Emit the five ``export_artifacts`` JSON files after reloading CSVs,
``build_elo()``, and ``train_regression(fit_model=False)`` so ``elo_states``, ``style_axes``,
``fighter_profiles`` include **elo_trajectories** when ``build_elo(..., record_trajectories=True)``
(this is enabled by default; use ``--no-record-elo-trajectories`` to skip).

**retrain (steps 1–6)** — Same data + ELO path, then full ``train_regression()`` (new **W**,
bootstrap, artifact audit), saves the pickle, then exports all five JSONs. Step **6** is the
multinomial refit; upcoming cards stay ``export_upcoming_events.py``.

**Hyperparameters (both subcommands):** Uses the ``Config`` already stored in the loaded pickle
(Huber ``delta``, ``l2_lambda``, L-BFGS limits, bootstrap count/seed, ELO fields, holdout dates,
etc.). There is **no** walk-forward or random search here — that is only for initial validation /
selection (e.g. ``python -m src.cli.run_phase3_tuning`` and related docs). ``retrain`` refits
**coefficients** (and optional bootstrap draws for CIs) under that **fixed** config when you add
data and want a new **W**, not a new hyperparameter sweep.

Usage (repo root)::

    python scripts/weekly_update.py refresh
    python scripts/weekly_update.py retrain

Defaults: ``model.pkl`` under ``<repo>/data/``, data dir ``<repo>/data``, JSON out ``<repo>/JSON_exports``.
Override with ``--model-path``, ``--data-dir``, ``--out-dir`` as needed.

Flags match ``export_artifacts.py`` for ``--as-of-date``, ``--copy-to-mma-ai``, etc.
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import export_artifacts as export_artifacts_mod  # noqa: E402
from src.pipeline import MMAPredictor  # noqa: E402


def _copy_to_mma_ai(out_dir: Path, mma_ai_dir: Path | None) -> None:
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    import copy_exports_to_mma_ai as _cex

    dest = Path(mma_ai_dir).resolve() if mma_ai_dir else _cex.default_mma_ai_artifacts_dir()
    copied = _cex.copy_json_from_dir(out_dir, dest)
    print(f"Copied {len(copied)} JSON file(s) -> {dest}", flush=True)


def cmd_refresh(args: argparse.Namespace) -> int:
    model_path = Path(args.model_path).resolve()
    data_dir = Path(args.data_dir).resolve()
    out_dir = Path(args.out_dir).resolve()

    print(f"[weekly_update refresh] load pickle {model_path}", flush=True)
    pred = MMAPredictor.load(model_path)
    print(f"[weekly_update refresh] load_data {data_dir}", flush=True)
    pred.load_data(data_dir)
    print("[weekly_update refresh] build_elo ...", flush=True)
    pred.build_elo(
        elo_progress_every=args.elo_progress_every,
        record_trajectories=args.record_elo_trajectories,
    )
    print("[weekly_update refresh] train_regression(fit_model=False) ...", flush=True)
    pred.train_regression(fit_model=False, matrix_progress_every=args.matrix_progress_every)

    as_of: Optional[date] = None
    if args.as_of_date:
        as_of = date.fromisoformat(args.as_of_date)

    export_artifacts_mod.export_all(pred, out_dir, as_of=as_of)
    print(f"[weekly_update refresh] Wrote 5 JSON files under {out_dir}", flush=True)

    if args.save_model:
        pred.save(model_path)
        print(f"[weekly_update refresh] Saved pickle {model_path}", flush=True)

    if args.copy_to_mma_ai:
        _copy_to_mma_ai(out_dir, args.mma_ai_artifacts_dir)
    return 0


def cmd_retrain(args: argparse.Namespace) -> int:
    model_path = Path(args.model_path).resolve()
    data_dir = Path(args.data_dir).resolve()
    out_dir = Path(args.out_dir).resolve()

    print(f"[weekly_update retrain] load pickle (config + warm state) {model_path}", flush=True)
    pred = MMAPredictor.load(model_path)
    print(f"[weekly_update retrain] load_data {data_dir}", flush=True)
    pred.load_data(data_dir)
    print("[weekly_update retrain] build_elo ...", flush=True)
    pred.build_elo(
        elo_progress_every=args.elo_progress_every,
        record_trajectories=args.record_elo_trajectories,
    )
    print("[weekly_update retrain] train_regression() full fit ...", flush=True)
    pred.train_regression(matrix_progress_every=args.matrix_progress_every)

    pred.save(model_path)
    print(f"[weekly_update retrain] Saved pickle {model_path}", flush=True)

    as_of: Optional[date] = None
    if args.as_of_date:
        as_of = date.fromisoformat(args.as_of_date)

    export_artifacts_mod.export_all(pred, out_dir, as_of=as_of)
    print(f"[weekly_update retrain] Wrote 5 JSON files under {out_dir}", flush=True)

    if args.copy_to_mma_ai:
        _copy_to_mma_ai(out_dir, args.mma_ai_artifacts_dir)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Weekly update: refresh ELO + matrix + JSON, or full retrain + export.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--model-path",
        type=Path,
        default=ROOT / "data" / "model.pkl",
        help="Pickle path (default: <repo>/data/model.pkl)",
    )
    common.add_argument("--data-dir", type=Path, default=ROOT / "data")
    common.add_argument("--out-dir", type=Path, default=ROOT / "JSON_exports")
    common.add_argument("--as-of-date", type=str, default=None, help="YYYY-MM-DD for ELO/style export")
    common.add_argument("--elo-progress-every", type=int, default=2000)
    common.add_argument("--matrix-progress-every", type=int, default=500)
    common.add_argument("--copy-to-mma-ai", action="store_true")
    common.add_argument("--mma-ai-artifacts-dir", type=Path, default=None)
    common.add_argument(
        "--record-elo-trajectories",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Record per-fight ELO points for fighter_profiles elo_trajectories (default: true)",
    )

    sp_r = sub.add_parser("refresh", parents=[common], help="Rebuild ELO + training matrix; keep W from pickle (steps 1–5).")
    sp_r.add_argument(
        "--save-model",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write pickle after refresh so it stores fresh ELO (default: true)",
    )

    sub.add_parser("retrain", parents=[common], help="Full train_regression + save pickle + export (step 6 + 1–5).")

    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "refresh":
        return cmd_refresh(args)
    if args.command == "retrain":
        return cmd_retrain(args)
    raise SystemExit(f"unknown command {args.command!r}")


if __name__ == "__main__":
    raise SystemExit(main())
