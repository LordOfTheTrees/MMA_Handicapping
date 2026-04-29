#!/usr/bin/env python3
"""
Phase-3 tuning harness: walk-forward (selection block) + pristine test (2023–2025),
CSV/JSON output, and simple matplotlib figures. See ``docs/hyperparameter-tuning.md``.

* **Baseline** (default): one fixed ``Config()`` for every selection year; pristine uses
  the same default config.
* **Selection search** (``--selection-search``): ``--n-trials`` (default 50) random
  trials per calendar year, warm-starting each year from the previous year’s winner;
  pristine 2023–2025 uses the **frozen 2022 winner** config, not a fresh ``Config()``.
* **Debug** one year: ``--search-outer-year Y --n-trials N`` (cannot combine with
  ``--selection-search``).

From repo root::

    python scripts/run_phase3_tuning.py --data-dir ./data --out-dir ./data/phase3_eval

ELO is cached under ``<out-dir>/elo_walkforward_cache.pkl`` by default to speed folds.
"""
from __future__ import annotations

import argparse
import copy
import csv
import dataclasses
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import Config  # noqa: E402
from src.eval.fight_scoring import Tier1SliceScore  # noqa: E402
from src.eval.tuning_harness import (  # noqa: E402
    first_valid_outer_eval_year,
    make_trial_progress_bar,
    run_pristine_years,
    run_random_search_for_outer_year,
    run_selection_campaign_with_search,
    run_selection_walkforward_baseline,
)
from src.eval.tuning_plots import (  # noqa: E402
    plot_combined_log_loss_trajectory,
    plot_pristine_yoy_bars,
)

# Planning defaults (``docs/hyperparameter-tuning.md``)
PRISTINE_YEARS = (2023, 2024, 2025)
SELECTION_END_YEAR = 2022


def _slice_to_dict(name: str, y: int, s: Tier1SliceScore) -> Dict[str, Any]:
    wcs = {
        k: {
            "n": v.n,
            "mean_log_loss": v.mean_log_loss,
            "mean_brier": v.mean_brier,
            "accuracy": v.accuracy,
            "macro_f1": v.macro_f1,
            "wl_f1": v.wl_f1,
            "finish_f1": v.finish_f1,
        }
        for k, v in s.by_weight_class.items()
    }
    return {
        "segment": name,
        "year": y,
        "n": s.n,
        "mean_log_loss": s.mean_log_loss,
        "mean_brier": s.mean_brier,
        "accuracy": s.accuracy,
        "macro_f1": s.macro_f1,
        "wl_f1": s.wl_f1,
        "finish_f1": s.finish_f1,
        "by_weight_class": wcs,
    }


def _write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = ["segment", "year", "n", "mean_log_loss", "mean_brier", "accuracy", "macro_f1", "wl_f1", "finish_f1"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})


def _write_json(payload: Dict[str, Any], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Phase-3 walk-forward + pristine eval (see docs/hyperparameter-tuning.md).",
    )
    p.add_argument("--data-dir", type=Path, default=Path("data"), help="CSV data directory")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/phase3_eval"),
        help="Output directory (CSV, JSON, figures, ELO cache)",
    )
    p.add_argument(
        "--no-walkforward",
        action="store_true",
        help="Skip selection block (2007–2022 baseline); only run pristine 2023–2025",
    )
    p.add_argument(
        "--no-pristine",
        action="store_true",
        help="Only run selection walkforward; no pristine test years",
    )
    p.add_argument(
        "--selection-start",
        type=int,
        default=None,
        help="First eval year in selection block (default: first valid year: master_start+2, e.g. 2007 when master is 2005)",
    )
    p.add_argument(
        "--selection-end",
        type=int,
        default=SELECTION_END_YEAR,
        help="Last eval year in selection block (default 2022)",
    )
    p.add_argument(
        "--no-elo-cache",
        action="store_true",
        help="Rebuild ELO every fold (slow); default writes <out>/elo_walkforward_cache.pkl",
    )
    p.add_argument(
        "--selection-search",
        action="store_true",
        help="Run random search (see --n-trials) per selection year, warm-start chain; "
        "pristine uses frozen end-year winner, not default Config()",
    )
    p.add_argument(
        "--search-outer-year",
        type=int,
        default=None,
        help="If set with --n-trials, run one random-search outer year (very slow; "
        "incompatible with --selection-search).",
    )
    p.add_argument(
        "--n-trials",
        type=int,
        default=0,
        help="Search trials per outer year. With --selection-search, defaults to 50 when 0. "
        "For --search-outer-year, must be > 0 to run the single-year block.",
    )
    p.add_argument(
        "--inner-last-k",
        type=int,
        default=3,
        help="Inner walk-forward: mean log-loss over last K years before outer year (search only)",
    )
    p.add_argument(
        "--inner-full",
        action="store_true",
        help="Search only: use full inner walk-forward (slower) instead of last-K inner years",
    )
    p.add_argument("--seed", type=int, default=42, help="RNG seed for random search")
    p.add_argument(
        "--no-search-progress",
        action="store_true",
        help="Disable tqdm trial progress/ETA for --selection-search and --search-outer-year",
    )
    args = p.parse_args()

    if args.no_walkforward and args.no_pristine:
        print("Cannot use --no-walkforward and --no-pristine together.")
        sys.exit(1)
    if args.selection_search and args.search_outer_year is not None:
        print("Cannot use --selection-search and --search-outer-year together.", file=sys.stderr)
        sys.exit(1)

    data_dir = Path(args.data_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    elo_cache: Optional[Path] = None if args.no_elo_cache else (out_dir / "elo_walkforward_cache.pkl").resolve()

    base = Config()
    m = base.master_start_year
    sel_start = args.selection_start if args.selection_start is not None else first_valid_outer_eval_year(m)
    sel_end = args.selection_end
    if sel_end < sel_start:
        print("selection-end must be >= selection-start", file=sys.stderr)
        sys.exit(1)

    n_trials = int(args.n_trials)
    if args.selection_search and n_trials == 0:
        n_trials = 50
    if args.selection_search and n_trials < 1:
        print("With --selection-search, n_trials must be >= 1 (use default 50 by passing 0).", file=sys.stderr)
        sys.exit(1)

    all_rows: List[Dict[str, Any]] = []
    selection_ll: List[Tuple[int, float]] = []
    pristine_ll: List[Tuple[int, float]] = []
    used_selection_search = False
    frozen: Config = copy.deepcopy(base)
    report_extra: Dict[str, Any] = {}

    show_search_progress = not args.no_search_progress

    print(f"Data: {data_dir}  |  Out: {out_dir}")
    if elo_cache:
        print(f"ELO cache: {elo_cache}  (use --no-elo-cache to disable)\n")
    else:
        print("ELO: full rebuild every fold  (--no-elo-cache)\n")

    t0 = datetime.now()
    if not args.no_walkforward:
        if args.selection_search:
            used_selection_search = True
            print(
                f"Selection block: random search  n_trials={n_trials}  years {sel_start}–{sel_end}  "
                f"inner_last_k={args.inner_last_k}  inner_full={args.inner_full}  seed={args.seed} ...",
            )
            rng = np.random.default_rng(args.seed)
            campaign, fwd_slices, frozen = run_selection_campaign_with_search(
                base,
                data_dir,
                sel_start,
                sel_end,
                n_trials,
                rng,
                inner_last_k=args.inner_last_k,
                use_full_inner=args.inner_full,
                skip_bootstrap=True,
                elo_cache_path=elo_cache,
                show_progress=show_search_progress,
            )
            for y, s in fwd_slices:
                all_rows.append(_slice_to_dict("selection", y, s))
                selection_ll.append((y, s.mean_log_loss))
            report_extra["selection_campaign"] = {
                "n_trials": n_trials,
                "inner_last_k": args.inner_last_k,
                "use_full_inner": args.inner_full,
                "seed": args.seed,
                "years": campaign,
            }
            report_extra["frozen_winner_config"] = dataclasses.asdict(frozen)
            print(
                f"  Done: {len(fwd_slices)} years, frozen end-year {sel_end} winner, "
                f"wall time {datetime.now() - t0}\n"
            )
        else:
            print(f"Selection block: baseline walk-forward (single Config)  years {sel_start}–{sel_end} ...")
            sel = run_selection_walkforward_baseline(
                base, data_dir, sel_start, sel_end, skip_bootstrap=True, elo_cache_path=elo_cache
            )
            for y, s in sel:
                all_rows.append(_slice_to_dict("selection", y, s))
                selection_ll.append((y, s.mean_log_loss))
            print(f"  Done: {len(sel)} years, wall time {datetime.now() - t0}\n")

    t0 = datetime.now()
    if not args.no_pristine:
        pcfg = "frozen {0} search winner".format(sel_end) if used_selection_search else "default Config()"
        print(f"Pristine test: {pcfg}, years {list(PRISTINE_YEARS)} ...")
        pris = run_pristine_years(
            frozen, data_dir, PRISTINE_YEARS, skip_bootstrap=True, elo_cache_path=elo_cache
        )
        for y, s in pris:
            all_rows.append(_slice_to_dict("pristine", y, s))
            pristine_ll.append((y, s.mean_log_loss))
        print(f"  Done: {len(pris)} years, wall time {datetime.now() - t0}\n")

    # CSV
    csv_path = out_dir / "phase3_metrics.csv"
    _write_csv(all_rows, csv_path)
    print(f"Wrote {csv_path}")

    # JSON (full + weight class slices)
    report: Dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "selection_mode": "random_search_per_year" if used_selection_search else "baseline_walkforward",
        "pristine_uses_frozen_winner": used_selection_search and not args.no_pristine and not args.no_walkforward,
        "data_dir": str(data_dir),
        "selection_years": [sel_start, sel_end] if not args.no_walkforward else None,
        "pristine_years": list(PRISTINE_YEARS) if not args.no_pristine else None,
        "rows": all_rows,
    }
    report.update(report_extra)
    if args.search_outer_year is not None and int(args.n_trials) > 0:
        t0 = datetime.now()
        n_one = int(args.n_trials)
        print(
            f"Random search (single year): outer_year={args.search_outer_year}  "
            f"n_trials={n_one}  inner_last_k={args.inner_last_k}  inner_full={args.inner_full} ...",
        )
        if show_search_progress:
            print(
                f"  [selection] 1 calendar year × {n_one} random-walk trials = {n_one} bar steps; "
                "each step = one trial (inner + forward fits). ETA refines after a few trials.\n",
                flush=True,
            )
        pbar = make_trial_progress_bar(
            n_one,
            desc=f"Y{args.search_outer_year}",
            disabled=not show_search_progress,
        )
        rng = np.random.default_rng(args.seed)
        try:
            _wcfg, binner, fs, tri = run_random_search_for_outer_year(
                base,
                data_dir,
                args.search_outer_year,
                n_one,
                rng,
                inner_last_k=args.inner_last_k,
                use_full_inner=args.inner_full,
                warm_start=None,
                skip_bootstrap=True,
                elo_cache_path=elo_cache,
                pbar=pbar,
            )
        finally:
            if pbar is not None:
                pbar.close()
        report["random_search"] = {
            "outer_year": args.search_outer_year,
            "n_trials": n_one,
            "inner_last_k": args.inner_last_k,
            "use_full_inner": args.inner_full,
            "best_inner_mean_log_loss": binner,
            "forward_n": fs.n,
            "forward_mean_log_loss": fs.mean_log_loss,
            "trial_rows": [
                {"trial": a, "inner_mean_ll": b, "forward_mean_ll": c} for a, b, c in tri
            ],
            "wall_time_sec": (datetime.now() - t0).total_seconds(),
        }
        print(
            f"  Best inner mean log-loss: {binner:.4f}  |  forward {args.search_outer_year}: "
            f"n={fs.n}  ll={fs.mean_log_loss:.4f}  (wall {report['random_search']['wall_time_sec']:.0f}s)\n",
        )

    json_path = out_dir / "phase3_report.json"
    _write_json(report, json_path)
    print(f"Wrote {json_path}")

    # Plots
    pristine_bars_title = (
        f"Pristine test set (2023–2025): Tier-1 mean metrics (fighter A, frozen {sel_end} search winner)"
        if (used_selection_search and not args.no_walkforward)
        else "Pristine test set (2023–2025): Tier-1 mean metrics (fighter A, default Config)"
    )
    if not args.no_pristine and len(pristine_ll) == len(PRISTINE_YEARS):
        pr = sorted([r for r in all_rows if r["segment"] == "pristine"], key=lambda r: r["year"])
        plot_pristine_yoy_bars(
            [r["year"] for r in pr],
            [r["mean_log_loss"] for r in pr],
            [r["mean_brier"] for r in pr],
            [r["macro_f1"] for r in pr],
            out_dir / "pristine_test_yoy.png",
            title=pristine_bars_title,
        )
        print(f"Wrote {out_dir / 'pristine_test_yoy.png'}")

    comb_path = out_dir / "log_loss_selection_and_pristine.png"
    sel_x: Optional[List[int]] = [a for a, _ in selection_ll] if selection_ll else None
    sel_y: Optional[List[float]] = [b for _, b in selection_ll] if selection_ll else None
    pr_x = [a for a, _ in pristine_ll] if pristine_ll else []
    pr_y = [b for _, b in pristine_ll] if pristine_ll else []
    sel_leg = (
        f"Selection (N={n_trials} trials/yr, warm-start, last {args.inner_last_k} inner years)"
        if used_selection_search
        else "Selection (single baseline Config per year)"
    )
    if args.inner_full and used_selection_search:
        sel_leg = f"Selection (N={n_trials} trials/yr, warm-start, full inner walk-forward)"
    pr_leg = (
        f"Pristine test (frozen {sel_end} search winner)" if (used_selection_search and not args.no_walkforward) else
        "Pristine test (2023–2025, same frozen default Config)"
    )
    if selection_ll or pristine_ll:
        plot_combined_log_loss_trajectory(
            sel_x, sel_y, pr_x, pr_y, comb_path,
            selection_legend=sel_leg,
            pristine_legend=pr_leg,
        )
        print(f"Wrote {comb_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
