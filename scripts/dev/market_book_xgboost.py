#!/usr/bin/env python3
"""Walk-forward +EV book with an XGBoost head on the same 12 features and maps.

Optional extra: ``pip install -r requirements-benchmark.txt`` (not in ``requirements.txt``).

Same join, stake maps, and plots as ``python -m src.eval.market_book``. Does **not**
fit the multinomial; skips L-BFGS. Writes a **separate** out-dir (default
``data/market_eval_xgb``). Never writes ``JSON_exports/``. Does not splice onto
the logit series.

XGBoost params match ``scripts/dev/rebuild_efficacy_table.py`` (unweighted
``multi:softprob``). Sample weights default off so the comparison is the same
head as the published efficacy table, not a recency-weighted variant.

Run from repo root::

    python scripts/dev/market_book_xgboost.py \\
        --data-dir ./data \\
        --out-dir ./data/market_eval_xgb \\
        --elo-cache ./data/market_eval/elo_walkforward_cache.pkl \\
        --logit-json ./data/market_eval/market_book.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import Config  # noqa: E402
from src.data.loader import load_fighter_profiles, load_ufcstats_fights  # noqa: E402
from src.eval.fight_scoring import filter_tier1_fights_in_calendar_year  # noqa: E402
from src.eval.market_book import (  # noqa: E402
    MAIN_CARD_SIZE,
    PRIMARY_ODDS_SOURCE,
    BookAccum,
    _DEFAULT_JUREK,
    _DEFAULT_MDABBERT,
    _json_default,
    align_class_proba,
    assert_out_dir_allowed,
    assign_slice_tags,
    book_year,
    default_sidecar,
    join_posted_lines,
    load_mdabbert_rows,
    predict_p6_from_included,
)
from src.eval.tuning_harness import (  # noqa: E402
    fit_predictor_for_train_before,
    train_before_for_eval_year,
)

# Same as scripts/dev/rebuild_efficacy_table.py
XGB_PARAMS = dict(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    seed=42,
)


def _require_xgboost():
    try:
        import xgboost as xgb
    except ImportError as exc:
        raise SystemExit(
            "xgboost is not installed. This script is optional:\n"
            "  pip install -r requirements-benchmark.txt"
        ) from exc
    return xgb


def _print_book_line(prefix: str, year_rep: Dict[str, Any]) -> None:
    tw = year_rep["two_way"]
    fav = year_rep["two_way_favorite"]
    mh = year_rep["method"]
    print(
        f"  {prefix} two_way n_priced={tw['n_priced']} n+={tw['n_plus_ev']}  "
        f"two_way_fav n_priced={fav['n_priced']} n+={fav['n_plus_ev']}  "
        f"method n_priced={mh['n_priced']} n+={mh['n_plus_ev']}",
        flush=True,
    )


def _pooled_line(label: str, node: Dict[str, Any]) -> str:
    n_plus = int(node.get("n_plus_ev") or 0)
    cov = node.get("coverage")
    mean_p = node.get("mean_model_p")
    q = node.get("mean_posted_implied")
    hit = node.get("hit_rate")
    profit = (node.get("flat_1u") or {}).get("realized_profit_units")
    g = (node.get("quarter_kelly") or {}).get("realized_log_growth")

    def _f(x: Any, nd: int = 2) -> str:
        try:
            v = float(x)
        except (TypeError, ValueError):
            return "n/a"
        if v != v:
            return "n/a"
        return f"{v:.{nd}f}"

    return (
        f"{label:22} n+={n_plus:5d}  cov={_f(cov, 2)}  "
        f"P={_f(mean_p, 3)}  q={_f(q, 3)}  hit={_f(hit, 3)}  "
        f"1u={_f(profit, 1)}  1/4K g={_f(g, 2)}"
    )


def _fit_xgb(xgb, X_train, y_train, *, sample_weight=None):
    """Native booster so we do not need scikit-learn (not in requirements.txt).

    Params match ``rebuild_efficacy_table.py`` / ``XGBClassifier``: 300 trees,
    depth 6, eta 0.05, subsample/colsample 0.9, hist, seed 42. ``num_class=6``
    keeps columns in encode_outcome order 0..5 even if a fold misses a label.
    """
    dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weight)
    params = {
        "objective": "multi:softprob",
        "num_class": 6,
        "max_depth": XGB_PARAMS["max_depth"],
        "eta": XGB_PARAMS["learning_rate"],
        "subsample": XGB_PARAMS["subsample"],
        "colsample_bytree": XGB_PARAMS["colsample_bytree"],
        "tree_method": "hist",
        "eval_metric": "mlogloss",
        "seed": XGB_PARAMS["seed"],
        "nthread": 0,
    }
    return xgb.train(params, dtrain, num_boost_round=int(XGB_PARAMS["n_estimators"]))


def _predict_proba(xgb, bst, X):
    raw = bst.predict(xgb.DMatrix(X))
    raw = raw.reshape(-1, 6)
    return align_class_proba(raw, range(6), n_classes=6)


def run_xgb_market_book(
    data_dir: Path,
    out_dir: Path,
    *,
    start_year: int = 2013,
    end_year: int = 2025,
    jurek_path: Optional[Path] = None,
    mdabbert_path: Optional[Path] = None,
    elo_cache: Optional[Path] = None,
    logit_json: Optional[Path] = None,
    sample_weight: str = "none",
) -> Dict[str, Any]:
    xgb = _require_xgboost()
    data_dir = Path(data_dir).resolve()
    out_dir = assert_out_dir_allowed(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jurek_path = Path(jurek_path).resolve() if jurek_path else default_sidecar(data_dir, _DEFAULT_JUREK)
    mdabbert_path = (
        Path(mdabbert_path).resolve() if mdabbert_path else default_sidecar(data_dir, _DEFAULT_MDABBERT)
    )
    if not jurek_path.is_file() and not mdabbert_path.is_file():
        raise FileNotFoundError(
            f"Need at least one sidecar CSV. Missing:\n  {jurek_path}\n  {mdabbert_path}"
        )
    default_elo = data_dir / "market_eval" / "elo_walkforward_cache.pkl"
    elo_cache = Path(elo_cache).resolve() if elo_cache else (
        default_elo if default_elo.is_file() else (out_dir / "elo_walkforward_cache.pkl")
    )

    fights_csv = data_dir / "ufcstats_fights.csv"
    profiles_csv = data_dir / "fighter_profiles.csv"
    fights = load_ufcstats_fights(fights_csv)
    profiles = load_fighter_profiles(profiles_csv)
    posted, join_stats = join_posted_lines(fights, profiles, jurek_path, mdabbert_path)
    md_rows = load_mdabbert_rows(mdabbert_path) if mdabbert_path.is_file() else []
    tags = assign_slice_tags(fights, profiles, md_rows)
    print(
        f"[market_book_xgb] join  jurek={join_stats['n_jurek']}  "
        f"mdabbert_fill={join_stats['n_mdabbert_fill']}  "
        f"two_way={join_stats['n_two_way']}  method={join_stats['n_method']}",
        flush=True,
    )
    print(
        f"[market_book_xgb] XGB unweighted={sample_weight == 'none'}  "
        f"params={XGB_PARAMS}  elo_cache={elo_cache}",
        flush=True,
    )

    cfg = Config()
    years_out: Dict[str, Any] = {}
    pooled = BookAccum()
    for y in range(int(start_year), int(end_year) + 1):
        print(
            f"[market_book_xgb] fit year {y} (train_before={train_before_for_eval_year(y)}; "
            "skip L-BFGS) ...",
            flush=True,
        )
        pred = fit_predictor_for_train_before(
            cfg,
            data_dir,
            train_before_for_eval_year(y),
            skip_bootstrap=True,
            elo_cache_path=elo_cache,
            fit_regression=False,
        )
        X_train = pred._X_train
        y_train = pred._y_train
        if X_train is None or y_train is None or len(y_train) == 0:
            print(f"  skip {y}: empty train matrix", flush=True)
            continue
        w_train = pred._train_weights if sample_weight == "recency" else None
        bst = _fit_xgb(xgb, X_train, y_train, sample_weight=w_train)
        year_fights = filter_tier1_fights_in_calendar_year(
            pred.fights, pred.config.master_start_year, y
        )
        if not year_fights:
            print(f"  skip {y}: no tier-1 eval fights", flush=True)
            continue
        X_eval, _, _, included = pred.build_xyw_for_fights(
            year_fights, matrix_progress_every=0, progress_prefix=f"  [eval {y}]"
        )
        probs = _predict_proba(xgb, bst, X_eval)
        year_acc = book_year(
            included,
            posted,
            predict_p6_from_included(included, probs),
            tags,
        )
        pooled.merge(year_acc)
        years_out[str(y)] = year_acc.as_report()
        _print_book_line(f"jurek {y}", years_out[str(y)])

    report: Dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "start_year": int(start_year),
        "end_year": int(end_year),
        "head": "xgboost",
        "config": (
            "Config() features + PIT ELO; XGBClassifier multi:softprob "
            f"(unweighted={sample_weight == 'none'}); skip L-BFGS"
        ),
        "xgb_params": dict(XGB_PARAMS),
        "sample_weight": sample_weight,
        "stake_rules": ["full_kelly", "half_kelly", "quarter_kelly", "flat_1u"],
        "stake_maps": {
            "two_way": "isolated max-edge (baseline)",
            "method": "isolated max-edge (baseline)",
            "two_way_simul": "simultaneous Kelly on listed mutex contracts (backing only)",
            "method_simul": "simultaneous Kelly on listed mutex contracts (backing only)",
            "two_way_favorite": (
                "model-preferred moneyline only, e>0 on that side; "
                "drop underround boards (q_A+q_B<1); no method max-edge"
            ),
        },
        "slice_rules": {
            "card": (
                "mdabbert billed order (main-first); "
                f"main_card=first {MAIN_CARD_SIZE}; "
                f"prelim_main_event=index {MAIN_CARD_SIZE}; "
                "generic_prelims=later; title=mdabbert title_bout or weight_class_raw; "
                "overlapping labels; not crossed with gender/weight"
            ),
            "gender": "men/women from WeightClass; catch/unknown=other",
            "weight_class": "FightRecord.weight_class; one-way only",
        },
        "join": join_stats,
        "odds_tapes": {
            "primary": PRIMARY_ODDS_SOURCE,
            "fill": "mdabbert",
            "rule": (
                "One source per fight; jurek fight_id wins even if method columns are empty. "
                "YoY, slices, and simul figures use the jurek tape only. "
                "mdabbert fill is a separate rollup and is never spliced onto those series."
            ),
        },
        "sidecars": {"jurek": str(jurek_path), "mdabbert": str(mdabbert_path)},
        "years": years_out,
        "slices_pooled": pooled.as_report(),
    }
    json_path = out_dir / "market_book.json"
    json_path.write_text(json.dumps(report, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(f"[market_book_xgb] wrote {json_path}", flush=True)

    from src.eval.tuning_plots import (
        plot_market_book_favorite_compare,
        plot_market_book_fill_tape,
        plot_market_book_head_compare,
        plot_market_book_slices,
        plot_market_book_simul_compare,
        plot_market_book_yoy,
    )

    xgb_title = "XGBoost head (same 12 features): jurek tape only"
    plot_market_book_yoy(report, out_dir / "market_book_yoy.png", title=xgb_title)
    plot_market_book_slices(report, out_dir / "market_book_slices.png")
    plot_market_book_simul_compare(report, out_dir / "market_book_simul.png")
    plot_market_book_favorite_compare(
        report,
        out_dir / "market_book_favorite.png",
        title="XGBoost two-way: max-edge vs model-favorite only",
    )
    plot_market_book_fill_tape(report, out_dir / "market_book_mdabbert_fill.png")
    print(f"[market_book_xgb] wrote figures under {out_dir}", flush=True)

    pooled_rep = report["slices_pooled"]
    print("\n[market_book_xgb] pooled jurek 2013–end:", flush=True)
    for key, lab in (
        ("two_way", "XGB max-edge"),
        ("two_way_favorite", "XGB favorite-only"),
        ("method", "XGB method"),
    ):
        print("  " + _pooled_line(lab, pooled_rep[key]), flush=True)

    if logit_json is not None and Path(logit_json).is_file():
        logit = json.loads(Path(logit_json).read_text(encoding="utf-8"))
        cmp_path = out_dir / "market_book_logit_vs_xgb.png"
        plot_market_book_head_compare(logit, report, cmp_path)
        print(f"[market_book_xgb] wrote {cmp_path}", flush=True)
        lp = logit.get("slices_pooled") or {}
        same_window = (
            int(logit.get("start_year") or -1) == int(start_year)
            and int(logit.get("end_year") or -1) == int(end_year)
        )
        if same_window:
            print("\n[market_book_xgb] logit (same window, from --logit-json):", flush=True)
            for key, lab in (
                ("two_way", "logit max-edge"),
                ("two_way_favorite", "logit favorite-only"),
                ("method", "logit method"),
            ):
                if key in lp:
                    print("  " + _pooled_line(lab, lp[key]), flush=True)
        else:
            print(
                f"[market_book_xgb] logit JSON window "
                f"{logit.get('start_year')}-{logit.get('end_year')} "
                f"!= this run {start_year}-{end_year}; overlay figure uses intersecting years only",
                flush=True,
            )
    else:
        print("[market_book_xgb] no --logit-json; skip overlay figure", flush=True)
    return report


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(
        description=(
            "Walk-forward +EV book with XGBoost on the same features and maps as "
            "src.eval.market_book. Optional extra; does not change production train."
        )
    )
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data") / "market_eval_xgb",
        help="JSON+PNG output (must not be JSON_exports/)",
    )
    p.add_argument("--start-year", type=int, default=2013)
    p.add_argument("--end-year", type=int, default=2025)
    p.add_argument("--jurek", type=Path, default=None)
    p.add_argument("--mdabbert", type=Path, default=None)
    p.add_argument(
        "--elo-cache",
        type=Path,
        default=None,
        help="PIT ELO cache (default: data/market_eval/elo_walkforward_cache.pkl if present)",
    )
    p.add_argument(
        "--logit-json",
        type=Path,
        default=Path("data") / "market_eval" / "market_book.json",
        help="Logit book JSON for overlay figure (skip if missing)",
    )
    p.add_argument(
        "--sample-weight",
        choices=("none", "recency"),
        default="none",
        help="none = same as efficacy-table XGB (default); recency = matrix 1/(1+days/365)",
    )
    args = p.parse_args(list(argv) if argv is not None else None)
    run_xgb_market_book(
        args.data_dir,
        args.out_dir,
        start_year=args.start_year,
        end_year=args.end_year,
        jurek_path=args.jurek,
        mdabbert_path=args.mdabbert,
        elo_cache=args.elo_cache,
        logit_json=args.logit_json,
        sample_weight=args.sample_weight,
    )


if __name__ == "__main__":
    main()
