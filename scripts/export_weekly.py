#!/usr/bin/env python3
"""
Weekly deploy export: refresh ELO/style snapshots and JSON without refitting *W*, or full retrain.

**refresh (steps 1–5)** — Emit the five ``export_artifacts`` JSON files after reloading CSVs,
``build_elo()``, and ``train_regression(fit_model=False)`` so ``elo_states``, ``style_axes``,
``fighter_profiles``, ``reference_distributions``, and ``model_weights`` (unchained **W** from the
pickle) match the latest data. Optionally updates the pickle so local ``predict`` sees fresh ELO.

**retrain (steps 1–6)** — Same data + ELO path, then full ``train_regression()`` (new **W**,
bootstrap, artifact audit), saves the pickle, then exports all five JSONs. Step **6** is the
multinomial refit; upcoming cards stay ``export_upcoming_events.py``.

Usage (repo root)::

    python scripts/export_weekly.py refresh --model-path data/model.pkl --data-dir data --out-dir JSON_exports
    python scripts/export_weekly.py retrain --model-path data/model.pkl --data-dir data --out-dir JSON_exports

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

    print(f"[export_weekly refresh] load pickle {model_path}", flush=True)
    pred = MMAPredictor.load(model_path)
    print(f"[export_weekly refresh] load_data {data_dir}", flush=True)
    pred.load_data(data_dir)
    print("[export_weekly refresh] build_elo ...", flush=True)
    pred.build_elo(elo_progress_every=args.elo_progress_every)
    print("[export_weekly refresh] train_regression(fit_model=False) ...", flush=True)
    pred.train_regression(fit_model=False, matrix_progress_every=args.matrix_progress_every)

    as_of: Optional[date] = None
    if args.as_of_date:
        as_of = date.fromisoformat(args.as_of_date)

    export_artifacts_mod.export_all(pred, out_dir, as_of=as_of)
    print(f"[export_weekly refresh] Wrote 5 JSON files under {out_dir}", flush=True)

    if args.save_model:
        pred.save(model_path)
        print(f"[export_weekly refresh] Saved pickle {model_path}", flush=True)

    if args.copy_to_mma_ai:
        _copy_to_mma_ai(out_dir, args.mma_ai_artifacts_dir)
    return 0


def cmd_retrain(args: argparse.Namespace) -> int:
    model_path = Path(args.model_path).resolve()
    data_dir = Path(args.data_dir).resolve()
    out_dir = Path(args.out_dir).resolve()

    print(f"[export_weekly retrain] load pickle (config + warm state) {model_path}", flush=True)
    pred = MMAPredictor.load(model_path)
    print(f"[export_weekly retrain] load_data {data_dir}", flush=True)
    pred.load_data(data_dir)
    print("[export_weekly retrain] build_elo ...", flush=True)
    pred.build_elo(elo_progress_every=args.elo_progress_every)
    print("[export_weekly retrain] train_regression() full fit ...", flush=True)
    pred.train_regression(matrix_progress_every=args.matrix_progress_every)

    pred.save(model_path)
    print(f"[export_weekly retrain] Saved pickle {model_path}", flush=True)

    as_of: Optional[date] = None
    if args.as_of_date:
        as_of = date.fromisoformat(args.as_of_date)

    export_artifacts_mod.export_all(pred, out_dir, as_of=as_of)
    print(f"[export_weekly retrain] Wrote 5 JSON files under {out_dir}", flush=True)

    if args.copy_to_mma_ai:
        _copy_to_mma_ai(out_dir, args.mma_ai_artifacts_dir)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Weekly export: refresh JSON without refit, or retrain then export.")
    sub = p.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model-path", type=Path, required=True)
    common.add_argument("--data-dir", type=Path, default=ROOT / "data")
    common.add_argument("--out-dir", type=Path, default=ROOT / "JSON_exports")
    common.add_argument("--as-of-date", type=str, default=None, help="YYYY-MM-DD for ELO/style export")
    common.add_argument("--elo-progress-every", type=int, default=2000)
    common.add_argument("--matrix-progress-every", type=int, default=500)
    common.add_argument("--copy-to-mma-ai", action="store_true")
    common.add_argument("--mma-ai-artifacts-dir", type=Path, default=None)

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
