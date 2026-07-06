#!/usr/bin/env python3
"""
Weekly pipeline: reload data, rebuild ELO, refresh or refit regression state, export deploy JSON.

**refresh (steps 1–5)** — Emit the five ``export_artifacts`` JSON files after reloading CSVs,
``build_elo()``, and ``train_regression(fit_model=False)`` so ``elo_states``, ``style_axes``,
``fighter_profiles`` include **elo_trajectories** when ``build_elo(..., record_trajectories=True)``
(this is enabled by default; use ``--no-record-elo-trajectories`` to skip).

**retrain (steps 1–6)** — Same data + ELO path, then full ``train_regression()`` (new **W**,
bootstrap, artifact audit), saves the pickle, then exports all five JSONs. Step **6** is the
multinomial refit.

**Upcoming events** — After the five JSONs, both subcommands also write ``upcoming_events.json``
via ``export_upcoming_events``'s ``build_upcoming_events_doc``, sourced from whichever of ESPN's
``fightcenter`` scrape or UFCStats' scrape produced fresh data this run — **ESPN preferred**
(reliable in CI; see ``docs/ufc-com-upcoming-scrape-plan.md`` §0), UFCStats as fallback when ESPN's
attempt fails but UFCStats' doesn't. If neither produced fresh data this run (both blocked/failed,
or ``--no-scrape``), the export is skipped rather than re-shipping a stale ``*_cards.json`` carried
over from a prior run. With ``--no-scrape`` (CI's split-step flow), this script never scrapes
itself, so the CI workflow gates a separate ``export_upcoming_events.py`` step on
``ci_try_refresh_data``'s ``espn_upcoming_scraped``/``ufcstats_upcoming_scraped`` step outputs
instead.

**Hyperparameters (both subcommands):** Uses the ``Config`` already stored in the loaded pickle
(Huber ``delta``, ``l2_lambda``, L-BFGS limits, bootstrap count/seed, ELO fields, holdout dates,
etc.). There is **no** walk-forward or random search here — that is only for initial validation /
selection (e.g. ``python -m src.cli.run_phase3_tuning`` and related docs). ``retrain`` refits
**coefficients** (and optional bootstrap draws for CIs) under that **fixed** config when you add
data and want a new **W**, not a new hyperparameter sweep.

Usage (repo root)::

    python scripts/weekly_update.py refresh
    python scripts/weekly_update.py retrain

By default each run calls ``refresh_data()`` first (ESPN incremental fights, ESPN profiles,
ESPN ID audit, then UFCStats gap-fill / upcoming when reachable). Use ``--no-scrape`` only when
CSVs are already current (e.g. CI after ``ci_try_refresh_data``). Local smoke:
``refresh --smoke-test`` (caps ESPN scan; skips rookie eventlog fetches).

GitHub Actions: ``ci_try_refresh_data`` then ``weekly_update … --no-scrape``.

Defaults: ``model.pkl`` under ``<repo>/data/``, data dir ``<repo>/data``, JSON out ``<repo>/JSON_exports``.
Override with ``--model-path``, ``--data-dir``, ``--out-dir`` as needed.

Flags match ``export_artifacts.py`` for ``--as-of-date``, ``--copy-to-mma-ai``, etc.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import export_artifacts as export_artifacts_mod  # noqa: E402
from src.data.espn_upcoming import DEFAULT_ESPN_UPCOMING_CARDS_JSON  # noqa: E402
from src.data.refresh import DataRefreshError, refresh_data  # noqa: E402
from src.data.ufcstats_upcoming import DEFAULT_UPCOMING_CARDS_JSON  # noqa: E402
from src.export.upcoming_events_doc import build_upcoming_events_doc  # noqa: E402
from src.pipeline import MMAPredictor  # noqa: E402


def _maybe_refresh_csvs(data_dir: Path, args: argparse.Namespace, label: str) -> Tuple[bool, bool]:
    """Returns ``(ufcstats_upcoming_scraped, espn_upcoming_scraped)`` for export gating."""
    if args.no_scrape:
        print(f"[weekly_update {label}] skip refresh_data (--no-scrape)", flush=True)
        return False, False
    print(f"[weekly_update {label}] refresh_data (ESPN + audit) ...", flush=True)
    try:
        result = refresh_data(
            data_dir,
            run_audit=not args.no_audit,
            fail_on_audit_reject=not args.allow_audit_failures,
            require_fight_updates=args.require_fight_updates,
            fetch_rookie_audit=not args.skip_rookie_audit,
            espn_max_events=args.espn_max_events,
            espn_max_competitions=args.espn_max_competitions,
            espn_verbose=not args.quiet_espn,
        )
    except DataRefreshError as e:
        print(f"[weekly_update {label}] refresh_data failed: {e}", flush=True)
        raise SystemExit(1) from e
    debut_n = 0
    state_path = data_dir / "espn_ingest_state.json"
    if state_path.is_file():
        with open(state_path, encoding="utf-8") as f:
            debut_n = len((json.load(f).get("last_run") or {}).get("new_fighters") or [])
    print(
        f"[weekly_update {label}] refresh_data ok: {result.fights_updated} fight row update(s), "
        f"{debut_n} new espn_* fighter id(s), audit={'pass' if result.audit_passed else 'fail'} "
        f"(see [espn debut] lines above)",
        flush=True,
    )
    return result.upcoming_cards_scraped, result.espn_upcoming_cards_scraped


def _maybe_export_upcoming_events(
    data_dir: Path,
    out_dir: Path,
    *,
    ufcstats_scraped: bool,
    espn_scraped: bool,
    label: str,
) -> None:
    """Prefer ESPN's upcoming-cards file (reliable in CI); fall back to UFCStats' if only
    that one is fresh this run. Skip entirely rather than re-ship a stale file when neither is.
    """
    if espn_scraped:
        cards_path, source = data_dir / DEFAULT_ESPN_UPCOMING_CARDS_JSON, "ESPN"
    elif ufcstats_scraped:
        cards_path, source = data_dir / DEFAULT_UPCOMING_CARDS_JSON, "UFCStats"
    else:
        print(
            f"[weekly_update {label}] skip upcoming_events export "
            "(no fresh upcoming-cards scrape this run)",
            flush=True,
        )
        return
    if not cards_path.is_file():
        print(f"[weekly_update {label}] skip upcoming_events export ({cards_path} missing)", flush=True)
        return
    cards = json.loads(cards_path.read_text(encoding="utf-8"))
    doc = build_upcoming_events_doc(cards)
    out_path = out_dir / "upcoming_events.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    print(f"[weekly_update {label}] Wrote {out_path} (source: {source})", flush=True)


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

    ufcstats_scraped, espn_scraped = _maybe_refresh_csvs(data_dir, args, "refresh")
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
    _maybe_export_upcoming_events(
        data_dir, out_dir, ufcstats_scraped=ufcstats_scraped, espn_scraped=espn_scraped, label="refresh"
    )

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

    ufcstats_scraped, espn_scraped = _maybe_refresh_csvs(data_dir, args, "retrain")
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
    _maybe_export_upcoming_events(
        data_dir, out_dir, ufcstats_scraped=ufcstats_scraped, espn_scraped=espn_scraped, label="retrain"
    )

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
    common.add_argument(
        "--no-scrape",
        action="store_true",
        help="Skip refresh_data(); use existing CSVs under --data-dir (CI passes this when scrape runs separately).",
    )
    common.add_argument(
        "--no-audit",
        action="store_true",
        help="Skip ESPN duplicate/rookie audit after ingest.",
    )
    common.add_argument(
        "--allow-audit-failures",
        action="store_true",
        help="Log audit rejects but do not exit (debug only).",
    )
    common.add_argument(
        "--skip-rookie-audit",
        action="store_true",
        help="Skip ESPN eventlog rookie checks (offline / faster).",
    )
    common.add_argument(
        "--require-fight-updates",
        action="store_true",
        help="Fail if ESPN changes 0 fights (CI strict mode; use for local CI-parity smoke).",
    )
    common.add_argument(
        "--espn-max-events",
        type=int,
        default=None,
        help="Limit ESPN incremental to N events (local smoke test).",
    )
    common.add_argument(
        "--espn-max-competitions",
        type=int,
        default=None,
        help="Stop after N bout updates in one incremental run (local smoke test).",
    )
    common.add_argument(
        "--quiet-espn",
        action="store_true",
        help="Less per-event ESPN ingest logging.",
    )
    common.add_argument(
        "--smoke-test",
        action="store_true",
        help="Shorthand: --espn-max-events 3 --espn-max-competitions 8 --skip-rookie-audit "
        "(still runs audit on any new espn_* ids from that sample).",
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


def _apply_smoke_test_defaults(args: argparse.Namespace) -> None:
    if not args.smoke_test:
        return
    if args.espn_max_events is None:
        args.espn_max_events = 3
    if args.espn_max_competitions is None:
        args.espn_max_competitions = 8
    args.skip_rookie_audit = True


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    _apply_smoke_test_defaults(args)
    if args.command == "refresh":
        return cmd_refresh(args)
    if args.command == "retrain":
        return cmd_retrain(args)
    raise SystemExit(f"unknown command {args.command!r}")


if __name__ == "__main__":
    raise SystemExit(main())
