#!/usr/bin/env python3
"""
Pilot: L-BFGS-B stopping (max_iter, ftol, gtol) on the **same** training matrix
as real `train_regression` for chosen calendar cutoffs (stratify across years).

Builds (X, y) once per year with `train_regression(fit_model=False)` — no first L-BFGS
before the grid — then re-fits `MultinomialLogisticModel` in-process with many options.

Usage (repo root)::

    python scripts/pilot_lbfgs_stopping.py --data-dir ./data
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import Config  # noqa: E402
from src.model.regression import N_CLASSES, _robust_nll_and_grad  # noqa: E402
from src.pipeline import MMAPredictor  # noqa: E402


@dataclass(frozen=True)
class StoppingConfig:
    name: str
    max_iter: int
    ftol: float
    gtol: float


# Grid: start from current defaults, then looser (typical for tuning pilots)
_DEFAULT_GRID: List[StoppingConfig] = [
    StoppingConfig("default_like", 3000, 1e-12, 1e-7),
    StoppingConfig("relaxed_A", 2000, 1e-10, 1e-6),
    StoppingConfig("relaxed_B", 1000, 1e-9, 1e-6),
    StoppingConfig("relaxed_C", 500, 1e-8, 1e-5),
    StoppingConfig("relaxed_D", 400, 1e-7, 1e-5),
    StoppingConfig("relaxed_E", 300, 1e-6, 1e-4),
]

ELO_PILOT_CACHE_NAME = "elo_pilot_lbfgs_cache.pkl"


def _refit_raw(
    X: np.ndarray,
    y: np.ndarray,
    n_features: int,
    delta: float,
    l2: float,
    max_iter: int,
    ftol: float,
    gtol: float,
) -> Tuple[object, float, bool, int, str, np.ndarray]:
    """
    One minimize with zero init, same as MultinomialLogisticModel but returns W + timing.
    """
    init = np.zeros(N_CLASSES * n_features)
    t0 = time.perf_counter()
    result = minimize(
        fun=_robust_nll_and_grad,
        x0=init,
        args=(X, y, delta, l2),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": max_iter, "ftol": ftol, "gtol": gtol},
    )
    wall = time.perf_counter() - t0
    W = result.x.reshape(N_CLASSES, n_features)
    return result, wall, result.success, int(result.nit), str(result.message), W


def _w_dist(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    d = float(np.linalg.norm(a - b))
    ref = max(float(np.linalg.norm(b)), 1e-15)
    return d / ref


def build_xy_for_holdout(
    data_dir: Path,
    holdout_y: int,
    elo_cache: Optional[Path],
) -> Tuple[np.ndarray, np.ndarray, int, float, float]:
    """
    holdout: train on fight_date < Jan 1 holdout_y (same as walk-forward for eval holdout_y).
    Returns X, y, n_features, huber_delta, l2_lambda
    """
    c = Config()
    c.holdout_start_date = date(holdout_y, 1, 1)
    c.model.n_bootstrap = 0
    c.model.cauchy_fallback_threshold = 10**9
    p = MMAPredictor(c)
    p.load_data(data_dir)
    if elo_cache and elo_cache.is_file():
        p.try_load_elo_from_cache(elo_cache)
    else:
        p.build_elo()
        if elo_cache:
            p.save_elo_cache(elo_cache)
    p.train_regression(matrix_progress_every=0, fit_model=False)
    x = p._X_train
    yv = p._y_train
    n_f = int(x.shape[1])
    return x, yv, n_f, c.model.huber_delta, c.model.l2_lambda


def run_pilot(
    data_dir: Path,
    out_csv: Path,
    years: List[int],
    grid: List[StoppingConfig],
) -> None:
    elo_cache: Optional[Path] = Path(data_dir).resolve() / ELO_PILOT_CACHE_NAME

    rows: List[dict] = []
    for holdout_y in years:
        print(f"=== holdout (train before) = Jan 1 {holdout_y}  (matrix for walk-forward) ===", flush=True)
        X, y, n_feat, delta, l2 = build_xy_for_holdout(
            data_dir, holdout_y, elo_cache if elo_cache and elo_cache.parent.exists() else None
        )
        print(f"  n_samples={X.shape[0]:,}  n_features={n_feat}  (building reference once)", flush=True)

        # Reference: default-like, full 3000 cap
        ref_result, t_ref, ok_ref, nit_ref, msg_ref, W_ref = _refit_raw(
            X, y, n_feat, delta, l2, 3000, 1e-12, 1e-7
        )
        f_ref = float(ref_result.fun)
        print(
            f"  reference: nit={nit_ref} success={ok_ref} time={t_ref:.2f}s  "
            f"final_nll={f_ref:.6f}  msg={msg_ref!r}",
            flush=True,
        )

        for g in grid:
            if (
                g.name == "default_like"
                and g.max_iter == 3000
                and g.ftol == 1e-12
                and g.gtol == 1e-7
            ):
                # same as reference run — duplicate row for table clarity
                W = W_ref
                r = ref_result
                wall = t_ref
                success = ok_ref
                nit = nit_ref
                msg = msg_ref
            else:
                r, wall, success, nit, msg, W = _refit_raw(
                    X, y, n_feat, delta, l2, g.max_iter, g.ftol, g.gtol
                )
            fval = float(r.fun)
            drel = _w_dist(W, W_ref)
            rows.append(
                {
                    "holdout_year": holdout_y,
                    "name": g.name,
                    "max_iter": g.max_iter,
                    "ftol": g.ftol,
                    "gtol": g.gtol,
                    "nit": nit,
                    "success": int(success),
                    "final_nll": fval,
                    "rel_W_dist_to_ref": drel,
                    "nll_delta_vs_ref": fval - f_ref,
                    "time_sec": wall,
                    "message": msg,
                }
            )
            print(
                f"  {g.name:12}  nit={nit:4d}  t={wall:5.2f}s  d_nll={fval - f_ref:+.2e}  "
                f"rel|W|={drel:.4f}  ok={success}",
                flush=True,
            )

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys()) if rows else []
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote {out_csv}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Pilot L-BFGS stopping on real training matrices.")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument(
        "--out-csv",
        type=Path,
        default=Path("data/pilot_lbfgs_stopping.csv"),
        help="Output CSV of grid results",
    )
    p.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=[2010, 2020],
        help="Stratify: holdout Jan 1 Y for these years (2010=smaller matrix, 2020=larger)",
    )
    args = p.parse_args()
    data_dir = args.data_dir.resolve()
    if not data_dir.is_dir():
        print(f"data dir not found: {data_dir}", file=sys.stderr)
        sys.exit(1)
    run_pilot(data_dir, args.out_csv, sorted(args.years), _DEFAULT_GRID)
    _print_recommendations()


def _print_recommendations() -> None:
    print(
        "\n"
        "=== Pilot-based recommendations (see data/pilot_lbfgs_stopping.csv) ===\n"
        "1) Reference (max_iter=3000, ftol=1e-12, gtol=1e-7) still hits the iteration\n"
        "   cap for both 2010 and 2020 matrices (success=False). So production-like\n"
        "   training is *budget-limited*, not tolerance-satisfied, as you observed.\n"
        "2) relaxed_A/B/C: lower max_iter *without* very loose ftol/gtol still pegs the\n"
        "   cap (2000/1000/500) — moderate rel|W| vs ref but you still do not get SciPy\n"
        "   'convergence' in the same sense. 2020: relaxed_A is ~0.6x wall time vs ref.\n"
        "3) relaxed_D/E: SciPy success=True, few iterations, but rel|W|~1 and large\n"
        "   training NLL change vs ref — *not* equivalent to the 3k-peg solution.\n"
        "   Use for tuning *only* after you verify trial **ranking** is stable; do not\n"
        "   ship with these without a separate validation study.\n"
        "Suggested next steps (no code change implied):\n"
        "  A) For Phase-3 *speed*: try **max_iter=2000, ftol=1e-10, gtol=1e-5** (slightly\n"
        "     looser than default) as a *pilot* on a few (config, year) — check if the\n"
        "     *rank order* of random trials matches default; if yes, you save ~1/3 time.\n"
        "  B) If rank stability holds, expose these as `ModelConfig` / tuning-only and\n"
        "     keep train defaults at 3k/1e-12/1e-7 for production until revalidated.\n"
        "  C) A larger ftol (e.g. 1e-6) is what allowed relaxed_D to stop early; pair it\n"
        "     with a controlled ranking study, not a single NLL number.\n",
        flush=True,
    )


if __name__ == "__main__":
    main()
