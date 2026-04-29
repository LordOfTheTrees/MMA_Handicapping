"""
Walk-forward and tuning evaluation harness.

Builds on point-in-time ELO + regression: for each (train_before, eval_year) pair,
trains the multinomial on Tier-1 rows with ``fight_date < train_before`` and scores
all decisive Tier-1 fights in ``eval_year`` (calendar), fighter A perspective.

See ``docs/hyperparameter-tuning.md`` for the selection / pristine / ship split.
"""
from __future__ import annotations

import copy
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np

from ..config import Config
from ..pipeline import MMAPredictor
from .fight_scoring import (
    Tier1SliceScore,
    filter_tier1_fights_in_calendar_year,
    score_tier1_fight_slice,
)

if TYPE_CHECKING:
    pass


def make_trial_progress_bar(
    total: int,
    *,
    desc: str,
    disabled: bool,
) -> Any:
    """``tqdm`` instance for *total* trial steps, or ``None`` if *disabled* or ``tqdm`` not installed."""
    if disabled or total <= 0:
        return None
    try:
        from tqdm import tqdm
    except ImportError:
        return None
    return tqdm(
        total=total,
        desc=desc,
        unit="trial",
        mininterval=0.5,
        dynamic_ncols=True,
    )


def train_before_for_eval_year(eval_year: int) -> date:
    """All regression training rows use ``fight_date <`` this date (first day of *eval_year*)."""
    return date(eval_year, 1, 1)


def default_inner_eval_years(master_start_year: int, outer_y: int) -> List[int]:
    """
    Calendar years used for the **inner** walk-forward to rank random-search trials
    for a given *outer* year ``outer_y``.

    The first year we can eval with non-empty post-era training is *master_start_year* + 1
    (train on ``fight_date <`` Jan 1 of that year+1, which includes a full *master* year).
    The inner set is ``[master+1, outer_y)`` — all forward eval years **strictly before** *outer_y*.
    If empty, caller must skip outer year (too early) or use a different protocol.
    """
    lo = master_start_year + 1
    return list(range(lo, outer_y))


def first_valid_outer_eval_year(master_start_year: int) -> int:
    """
    Smallest *outer* year for which ``default_inner_eval_years`` is non-empty:
    at least one inner eval year (requires ``outer_y >= master + 2`` with default 2005 -> 2007).
    """
    return master_start_year + 2


def fit_predictor_for_train_before(
    config: Config,
    data_dir: Path,
    train_before: date,
    *,
    skip_bootstrap: bool = False,
    elo_cache_path: Optional[Path] = None,
) -> MMAPredictor:
    """
    Full pipeline: load data, PIT ELO on all fights, fit regression on Tier-1 rows
    with ``fight_date < train_before`` (and ``master_start_year`` filter).

    ``skip_bootstrap``: if True, set ``n_bootstrap = 0`` on a copy to speed retrain loops
    (CIs are not used for point log-loss in tuning).
    ``elo_cache_path``: if set, reuse / save a point-in-time ELO cache (same fight count
    and ``ELOConfig``) to skip full ``build_elo`` across many walk-forward folds.
    """
    c = copy.deepcopy(config)
    c.holdout_start_date = train_before
    if skip_bootstrap:
        c.model.n_bootstrap = 0
        # train_regression runs bootstrap when ESS >= cauchy_threshold; raise threshold so
        # tuning loops skip expensive resamples (point log-loss / forward eval only).
        c.model.cauchy_fallback_threshold = 10**9
    p = MMAPredictor(c)
    p.load_data(data_dir)
    cache = Path(elo_cache_path).resolve() if elo_cache_path else None
    if cache and p.try_load_elo_from_cache(cache):
        pass
    else:
        p.build_elo()
        if cache:
            p.save_elo_cache(cache)
    p.train_regression()
    return p


def forward_log_loss_for_eval_year(
    config: Config,
    data_dir: Path,
    eval_year: int,
    *,
    skip_bootstrap: bool = True,
    elo_cache_path: Optional[Path] = None,
) -> Tuple[Tier1SliceScore, MMAPredictor]:
    """
    One expanding walk-forward step: train on all Tier-1 data strictly before *eval_year*,
    score all decisive Tier-1 fights in calendar *eval_year*.
    """
    p = fit_predictor_for_train_before(
        config,
        data_dir,
        train_before_for_eval_year(eval_year),
        skip_bootstrap=skip_bootstrap,
        elo_cache_path=elo_cache_path,
    )
    m = p.config.master_start_year
    fights = filter_tier1_fights_in_calendar_year(p.fights, m, eval_year)
    s = score_tier1_fight_slice(p, fights)
    return s, p


def inner_mean_log_loss(
    config: Config,
    data_dir: Path,
    outer_y: int,
    inner_eval_years: Optional[Sequence[int]] = None,
    *,
    skip_bootstrap: bool = True,
    elo_cache_path: Optional[Path] = None,
) -> Tuple[float, List[Tuple[int, float]]]:
    """
    Inner objective for ranking the 50 random-search trials at outer year *outer_y*:
    mean of per-calendar-year log-loss over *inner_eval_years* (default: all in
    ``[master+1, outer_y)``). Year *outer_y* is **never** included.

    Returns:
        (mean_log_loss, [(year, log_loss), ...] per inner year, nan years skipped in mean)
    """
    m = config.master_start_year
    years: List[int]
    if inner_eval_years is not None:
        years = [int(y) for y in inner_eval_years]
    else:
        years = default_inner_eval_years(m, outer_y)
    if not years:
        return float("nan"), []

    per: List[Tuple[int, float]] = []
    lls: List[float] = []
    for y in years:
        s, _ = forward_log_loss_for_eval_year(
            config,
            data_dir,
            y,
            skip_bootstrap=skip_bootstrap,
            elo_cache_path=elo_cache_path,
        )
        if s.n == 0:
            continue
        per.append((y, s.mean_log_loss))
        lls.append(s.mean_log_loss)
    if not lls:
        return float("nan"), per
    return float(np.mean(lls)), per


def inner_mean_log_loss_last_k_years(
    config: Config,
    data_dir: Path,
    outer_y: int,
    k: int = 3,
    *,
    skip_bootstrap: bool = True,
    elo_cache_path: Optional[Path] = None,
) -> Tuple[float, List[Tuple[int, float]]]:
    """
    Cheaper inner score: only the last *k* inner calendar years before *outer_y*.
    """
    full = default_inner_eval_years(config.master_start_year, outer_y)
    if not full:
        return float("nan"), []
    sub = full[-k:] if len(full) > k else full
    return inner_mean_log_loss(
        config,
        data_dir,
        outer_y,
        inner_eval_years=sub,
        skip_bootstrap=skip_bootstrap,
        elo_cache_path=elo_cache_path,
    )


# --- Pristine block (single frozen config, no search) ---


def run_pristine_years(
    config: Config,
    data_dir: Path,
    years: Sequence[int] = (2023, 2024, 2025),
    *,
    skip_bootstrap: bool = True,
    elo_cache_path: Optional[Path] = None,
) -> List[Tuple[int, Tier1SliceScore]]:
    """
    Expanding walk-forward for a **frozen** config: for each y in *years* (in order),
    train with cutoff Jan 1 y, then score calendar year y. Each step uses a fresh fit
    (state accumulates in training data, not in weights — correct for PIT check).
    """
    out: List[Tuple[int, Tier1SliceScore]] = []
    for y in years:
        s, _ = forward_log_loss_for_eval_year(
            config, data_dir, y, skip_bootstrap=skip_bootstrap, elo_cache_path=elo_cache_path
        )
        out.append((y, s))
    return out


def run_selection_walkforward_baseline(
    config: Config,
    data_dir: Path,
    start_year: int,
    end_year: int,
    *,
    skip_bootstrap: bool = True,
    elo_cache_path: Optional[Path] = None,
) -> List[Tuple[int, Tier1SliceScore]]:
    """
    For each calendar year in ``[start_year, end_year]`` (inclusive), one forward
    step with a **fixed** *config* (no random search). Use for the selection-regime
    diagnostic curve. See ``first_valid_outer_eval_year`` for a valid *start_year*.
    """
    out: List[Tuple[int, Tier1SliceScore]] = []
    for y in range(start_year, end_year + 1):
        s, _ = forward_log_loss_for_eval_year(
            config, data_dir, y, skip_bootstrap=skip_bootstrap, elo_cache_path=elo_cache_path
        )
        out.append((y, s))
    return out


def run_random_search_for_outer_year(
    base_config: Config,
    data_dir: Path,
    outer_y: int,
    n_trials: int,
    rng: np.random.Generator,
    *,
    inner_last_k: int = 3,
    use_full_inner: bool = False,
    warm_start: Optional[Config] = None,
    skip_bootstrap: bool = True,
    elo_cache_path: Optional[Path] = None,
    pbar: Any = None,
) -> Tuple[Config, float, Tier1SliceScore, List[Tuple[int, float, float]]]:
    """
    For one outer year *outer_y*, run *n_trials* trials: trial 0 = *warm_start* or
    *base_config*; trials 1..*n*−1 = ``sample_random_config`` draws. Rank by inner
    mean log-loss (``inner_mean_log_loss`` over all inner years, or
    ``inner_mean_log_loss_last_k_years`` when *use_full_inner* is False), then the
    winner's **forward** score on calendar *outer_y* (from the same fit used to rank
    that trial, not re-fit).

    Returns:
        (winner_config, best_inner_mean_log_loss, forward_tier1_slice, trial log rows)
    """
    from .tuning_priors import sample_random_config

    if n_trials < 1:
        raise ValueError("n_trials must be at least 1")
    best_inner = float("inf")
    best_cfg: Optional[Config] = None
    best_fwd: Optional[Tier1SliceScore] = None
    rows: List[Tuple[int, float, float]] = []
    for t in range(n_trials):
        if t == 0:
            c = copy.deepcopy(warm_start) if warm_start is not None else copy.deepcopy(base_config)
        else:
            c = sample_random_config(rng, base_config)
        if use_full_inner:
            inner_mean, _ = inner_mean_log_loss(
                c,
                data_dir,
                outer_y,
                None,
                skip_bootstrap=skip_bootstrap,
                elo_cache_path=elo_cache_path,
            )
        else:
            inner_mean, _ = inner_mean_log_loss_last_k_years(
                c,
                data_dir,
                outer_y,
                k=inner_last_k,
                skip_bootstrap=skip_bootstrap,
                elo_cache_path=elo_cache_path,
            )
        ill = float(inner_mean) if inner_mean == inner_mean else float("nan")
        fs, _ = forward_log_loss_for_eval_year(
            c, data_dir, outer_y, skip_bootstrap=skip_bootstrap, elo_cache_path=elo_cache_path
        )
        rows.append((t, ill, fs.mean_log_loss))
        if ill == ill and ill < best_inner:
            best_inner = ill
            best_cfg = c
            best_fwd = fs
        if pbar is not None:
            pbar.set_postfix_str(f"calendar_y={outer_y}  walk {t + 1}/{n_trials}", refresh=True)
            pbar.update(1)
    if best_cfg is None or best_fwd is None:
        best_cfg = copy.deepcopy(base_config)
        best_inner = float("nan")
        best_fwd, _ = forward_log_loss_for_eval_year(
            best_cfg, data_dir, outer_y, skip_bootstrap=skip_bootstrap, elo_cache_path=elo_cache_path
        )
    return best_cfg, best_inner, best_fwd, rows


def run_selection_campaign_with_search(
    base_config: Config,
    data_dir: Path,
    outer_start: int,
    outer_end: int,
    n_trials: int,
    rng: np.random.Generator,
    *,
    inner_last_k: int = 3,
    use_full_inner: bool = False,
    skip_bootstrap: bool = True,
    elo_cache_path: Optional[Path] = None,
    show_progress: bool = True,
) -> Tuple[List[Dict[str, Any]], List[Tuple[int, Tier1SliceScore]], Config]:
    """
    Phase-3 **selection** block: for each calendar year ``outer_y`` in
    ``[outer_start, outer_end]`` (inclusive), run *n_trials* random search trials
    (trial 0 warm-started from the **previous** year's winning ``Config``; first year
    uses *base_config*), rank by inner mean log-loss, record forward mean log-loss on
    *outer_y*. Carries the winner to the next year.

    The returned ``Config`` is the *outer_end* year's winner (use as **frozen** config
    for the 2023–2025 pristine block when *outer_end* is 2022).

    Each dict in the list: ``outer_year``, ``best_inner_mean_log_loss``,
    ``n_trials``, ``use_full_inner``, ``inner_last_k``,
    ``forward_mean_log_loss``, ``forward_n_fights``,
    ``trial_rows`` (``trial, inner_mean_ll, forward_mean_ll`` per trial).

    Also returns ``[(outer_year, forward_slice), ...]`` for full Tier-1 reporting (CSV, plots).
    """
    from datetime import datetime

    n_years = outer_end - outer_start + 1
    total_steps = n_years * n_trials
    pbar: Any = make_trial_progress_bar(
        total_steps,
        desc="Selection search",
        disabled=not show_progress,
    )
    if show_progress:
        print(
            f"  [selection] {n_years} calendar years × {n_trials} random-walk trials "
            f"= {total_steps} trial steps (one bar step per trial; each trial = inner + forward fits).",
            flush=True,
        )
        print(
            "  [selection] Bar shows [elapsed<remaining, rate]; remaining time is estimated and stabilizes after a few trials.",
            flush=True,
        )
        if show_progress and pbar is None and total_steps > 0:
            try:
                import tqdm  # noqa: F401
            except ImportError:
                print(
                    "  [selection] Install `tqdm` (see requirements.txt) for a live bar, speed, and ETA: pip install tqdm",
                    flush=True,
                )

    out: List[Dict[str, Any]] = []
    forward_by_year: List[Tuple[int, Tier1SliceScore]] = []
    prev_winner: Optional[Config] = None
    last_winner = copy.deepcopy(base_config)

    try:
        for iy, outer_y in enumerate(range(outer_start, outer_end + 1)):
            if pbar is not None:
                pbar.set_description(f"Y{outer_y} ({iy + 1}/{n_years})")
            elif show_progress:
                print(
                    f"  [selection] year {iy + 1}/{n_years}  calendar {outer_y}  —  {n_trials} random-walk trials …",
                    flush=True,
                )
            t0 = datetime.now()
            wcfg, binner, fwd, tri = run_random_search_for_outer_year(
                base_config,
                data_dir,
                outer_y,
                n_trials,
                rng,
                inner_last_k=inner_last_k,
                use_full_inner=use_full_inner,
                warm_start=prev_winner,
                skip_bootstrap=skip_bootstrap,
                elo_cache_path=elo_cache_path,
                pbar=pbar,
            )
            prev_winner = wcfg
            last_winner = wcfg
            forward_by_year.append((outer_y, fwd))
            wall = (datetime.now() - t0).total_seconds()
            if pbar is None and show_progress:
                print(
                    f"  [selection]   done calendar {outer_y} in {wall:.0f}s  "
                    f"forward log-loss (winner) = {fwd.mean_log_loss:.4f}",
                    flush=True,
                )
            out.append(
                {
                    "outer_year": outer_y,
                    "best_inner_mean_log_loss": binner,
                    "n_trials": n_trials,
                    "inner_last_k": inner_last_k if not use_full_inner else None,
                    "use_full_inner": use_full_inner,
                    "forward_mean_log_loss": fwd.mean_log_loss,
                    "forward_n_fights": fwd.n,
                    "forward_brier": fwd.mean_brier,
                    "forward_macro_f1": fwd.macro_f1,
                    "forward_wl_f1": fwd.wl_f1,
                    "forward_finish_f1": fwd.finish_f1,
                    "wall_time_sec": wall,
                    "trial_rows": [
                        {"trial": a, "inner_mean_ll": b, "forward_mean_ll": c} for a, b, c in tri
                    ],
                }
            )
    finally:
        if pbar is not None:
            pbar.close()
    return out, forward_by_year, last_winner


# --- Optional smoke / CLI ---


def _smoke(
    data_dir: Path,
    *,
    eval_year: int = 2020,
) -> None:
    """One forward fold, default config — sanity that data and paths work."""
    c = Config()
    s, _ = forward_log_loss_for_eval_year(c, data_dir, eval_year, skip_bootstrap=True)
    print(f"forward_mean_log_loss year={eval_year} n={s.n} ll={s.mean_log_loss:.4f} f1={s.macro_f1:.4f}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Tuning harness smoke: one forward year eval")
    p.add_argument("--data-dir", type=Path, default=Path("data"), help="CSV data directory")
    p.add_argument("--eval-year", type=int, default=2020, help="Calendar year to score")
    args = p.parse_args()
    _smoke(args.data_dir, eval_year=args.eval_year)
