#!/usr/bin/env python3
"""
Audit whether lex-smaller fighter_id (always fighter_a) correlates with age/cohort.

Hypothesis under "no accidental correlation": conditional on a matchup, who is older
should not systematically track lex order - i.e. age_lex_minus_B should center at 0.

Plots + simple regression (easy to read) plus binomial / Wilcoxon / Spearman checks.

Usage::

    python scripts/audit_lex_id_age_cohort.py --data-dir ./data
    python scripts/audit_lex_id_age_cohort.py --data-dir ./data \\
        --out-dir ./data/figures/lex_id_age_audit
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as scipy_stats

from src.data.loader import load_fighter_profiles, load_ufcstats_fights
from src.data.schema import DataTier


_FIGHT_CSV_CANDIDATES = ("ufcstats_fights.csv", "tier1_ufcstats.csv")


def _load_tier1_fights(data_dir: Path):
    for name in _FIGHT_CSV_CANDIDATES:
        p = data_dir / name
        if p.exists():
            return load_ufcstats_fights(p)
    raise FileNotFoundError(
        f"No {list(_FIGHT_CSV_CANDIDATES)} under {data_dir}"
    )


def _age_days_at(dob: date, fight_date: date) -> int:
    return (fight_date - dob).days


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Scatter + regression + robust tests: lex fighter_a vs age cohort.",
    )
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "data" / "figures" / "lex_id_age_audit",
    )
    args = ap.parse_args()

    data_dir = args.data_dir
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fights = _load_tier1_fights(data_dir)
    prof_path = data_dir / "fighter_profiles.csv"
    if not prof_path.exists():
        print(f"Missing {prof_path}", file=sys.stderr)
        return 1
    profiles = load_fighter_profiles(prof_path)

    tier1 = [f for f in fights if f.tier == DataTier.TIER_1]

    rows_age: list[tuple[date, float, date, date]] = []
    # (fight_date, age_diff_days, dob_a, dob_b)
    # age_diff_days = age_A - age_B on fight day (positive => fighter_a older).

    for f in tier1:
        a_id, b_id = f.fighter_a_id, f.fighter_b_id
        pa = profiles.get(a_id)
        pb = profiles.get(b_id)
        if pa is None or pb is None:
            continue
        if pa.date_of_birth is None or pb.date_of_birth is None:
            continue
        dob_a, dob_b = pa.date_of_birth, pb.date_of_birth
        aa = _age_days_at(dob_a, f.fight_date)
        ab = _age_days_at(dob_b, f.fight_date)
        age_diff_days = float(aa - ab)
        rows_age.append((f.fight_date, age_diff_days, dob_a, dob_b))

    if len(rows_age) < 50:
        print(f"Too few rows with DOB on both corners: {len(rows_age)}", file=sys.stderr)
        return 1

    fdates = np.array([t[0] for t in rows_age], dtype="datetime64[D]")
    age_diff = np.array([t[1] for t in rows_age], dtype=float)
    dob_a_arr = np.array([t[2] for t in rows_age], dtype="datetime64[D]")
    dob_b_arr = np.array([t[3] for t in rows_age], dtype="datetime64[D]")

    # --- Summary stats ---
    n = len(age_diff)
    k_a_older = int(np.sum(age_diff > 0))
    k_b_older = int(np.sum(age_diff < 0))
    k_tie_age = int(np.sum(age_diff == 0))

    mean_diff = float(np.mean(age_diff))
    median_diff = float(np.median(age_diff))

    # Binomial: under null, P(fighter_a older) = 0.5 (excluding exact ties)
    n_binom = k_a_older + k_b_older
    k_binom = k_a_older
    bt = scipy_stats.binomtest(k_binom, n_binom, 0.5) if n_binom > 0 else None

    # One-sample tests vs 0
    tt = scipy_stats.ttest_1samp(age_diff, 0.0, nan_policy="omit")
    wilcox = scipy_stats.wilcoxon(age_diff, zero_method="wilcox", alternative="two-sided")

    # Fight-date numeric for regression (days since epoch float)
    f_ord = np.array([d.astype("datetime64[D]").astype(float) for d in fdates])
    lr_time = scipy_stats.linregress(f_ord, age_diff)

    fight_year = np.array([date.fromisoformat(str(d)).year for d in fdates.astype(str)])
    lr_year = scipy_stats.linregress(fight_year.astype(float), age_diff)

    spear_fd_age = scipy_stats.spearmanr(f_ord, age_diff)
    spear_y_age = scipy_stats.spearmanr(fight_year, age_diff)

    # Bootstrap CI for mean age_diff (fight rows resampled i.i.d.)
    rng = np.random.default_rng(42)
    boots = []
    for _ in range(4000):
        samp = rng.choice(age_diff, size=n, replace=True)
        boots.append(float(np.mean(samp)))
    boots.sort()
    ci_lo = boots[int(0.025 * len(boots))]
    ci_hi = boots[int(0.975 * len(boots))]

    report_lines = [
        "",
        "=== Lex-ID vs age (fighter_a = lex-smaller id) ===",
        f"Tier-1 fights with DOB for both corners: {n:,}",
        f"fighter_a older (age_diff>0): {k_a_older:,}  |  fighter_b older: {k_b_older:,}  |  same age: {k_tie_age:,}",
        "",
        "If lex order were unrelated to age, we'd expect ~50% 'fighter_a older' "
        f"(excluding ties: {n_binom:,} fights).",
    ]
    if bt is not None:
        report_lines.append(
            f"Binomial test P(fighter_a older) vs 0.5: "
            f"k={k_binom}/{n_binom}, p={bt.pvalue:.4g}"
        )
    report_lines.extend(
        [
            f"Mean age_diff (days, A minus B at fight): {mean_diff:.2f}  "
            f"[bootstrap 95% CI {ci_lo:.2f}, {ci_hi:.2f}]",
            f"Median age_diff (days): {median_diff:.2f}",
            f"One-sample t-test mean!=0: statistic={tt.statistic:.4g}, p={tt.pvalue:.4g}",
            f"Wilcoxon signed-rank vs 0: statistic={wilcox.statistic:.4g}, p={wilcox.pvalue:.4g}",
            "",
            "Regression: age_diff_days ~ fight_date (ordinal days since epoch)",
            f"  slope (days per day): {lr_time.slope:.6g}, intercept: {lr_time.intercept:.4g}",
            f"  R^2={lr_time.rvalue ** 2:.6g}, p(slope)={lr_time.pvalue:.4g}",
            "",
            "Regression: age_diff_days ~ calendar year",
            f"  slope (days per year): {lr_year.slope:.6g}, intercept: {lr_year.intercept:.4g}",
            f"  R^2={lr_year.rvalue ** 2:.6g}, p(slope)={lr_year.pvalue:.4g}",
            "",
            f"Spearman rho(fight_date_ord, age_diff): {spear_fd_age.statistic:.4f}, p={spear_fd_age.pvalue:.4g}",
            f"Spearman rho(fight_year, age_diff): {spear_y_age.statistic:.4f}, p={spear_y_age.pvalue:.4g}",
        ]
    )

    # Fighter-level: lex rank vs debut year (first Tier-1 fight)
    first_seen: dict[str, date] = {}
    for f in tier1:
        for fid in (f.fighter_a_id, f.fighter_b_id):
            d0 = first_seen.get(fid)
            if d0 is None or f.fight_date < d0:
                first_seen[fid] = f.fight_date

    all_ids = sorted(first_seen.keys())
    rank_fracs = []
    debut_years = []
    for i, fid in enumerate(all_ids):
        rank_fracs.append(i / max(len(all_ids) - 1, 1))
        debut_years.append(first_seen[fid].year)

    lr_rank_year = scipy_stats.linregress(rank_fracs, debut_years)
    spear_rank_year = scipy_stats.spearmanr(rank_fracs, debut_years)

    report_lines.extend(
        [
            "",
            "=== Fighter-level: lex sort rank vs first Tier-1 fight year ===",
            f"Unique fighters with at least one Tier-1 fight: {len(all_ids):,}",
            "Regression: debut_year ~ lex_rank_fraction (0=min id, 1=max id)",
            f"  slope (years per rank unit): {lr_rank_year.slope:.4f}, intercept: {lr_rank_year.intercept:.4f}",
            f"  R^2={lr_rank_year.rvalue ** 2:.6g}, p(slope)={lr_rank_year.pvalue:.4g}",
            f"Spearman rho(rank_frac, debut_year): {spear_rank_year.statistic:.4f}, p={spear_rank_year.pvalue:.4g}",
        ]
    )

    report_text = "\n".join(report_lines)
    print(report_text, flush=True)
    (out_dir / "lex_id_age_audit_report.txt").write_text(report_text + "\n", encoding="utf-8")

    # --- Plot 1: fight date vs age_diff ---
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    x_dt = [date.fromisoformat(str(d)) for d in fdates.astype(str)]
    ax1.scatter(x_dt, age_diff, s=6, alpha=0.35, c="C0")
    ax1.axhline(0.0, color="k", lw=0.8, linestyle="--")
    x_num = mdates.date2num(x_dt)
    if len(x_num) >= 2:
        y_hat = lr_time.intercept + lr_time.slope * f_ord
        ax1.plot(x_dt, y_hat, color="C3", lw=2, label=f"OLS slope={lr_time.slope:.3e} d/d")
    ax1.set_xlabel("Fight date")
    ax1.set_ylabel("Age difference (days)\n(positive => fighter_a older)")
    ax1.set_title("Lex-smaller ID (fighter_a) age vs opponent vs fight date")
    ax1.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax1.xaxis.get_major_locator()))
    ax1.legend(loc="upper right")
    fig1.tight_layout()
    fig1.savefig(out_dir / "scatter_age_diff_vs_fight_date.png", dpi=150)
    plt.close(fig1)

    # --- Plot 2: DOB_a vs DOB_b ---
    fig2, ax2 = plt.subplots(figsize=(7, 7))
    xa = mdates.date2num([date.fromisoformat(str(d)) for d in dob_a_arr.astype(str)])
    xb = mdates.date2num([date.fromisoformat(str(d)) for d in dob_b_arr.astype(str)])
    ax2.scatter(xa, xb, s=5, alpha=0.25, c="C1")
    lim = [min(xa.min(), xb.min()), max(xa.max(), xb.max())]
    ax2.plot(lim, lim, "k--", lw=1, label="DOB equality")
    ax2.set_xlabel("fighter_a date of birth (lex-smaller id)")
    ax2.set_ylabel("fighter_b date of birth")
    ax2.set_title("DOB scatter (symmetry around diagonal if no lex-cohort tilt)")
    ax2.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax2.xaxis.get_major_locator()))
    ax2.yaxis.set_major_formatter(mdates.ConciseDateFormatter(ax2.yaxis.get_major_locator()))
    ax2.legend(loc="upper left")
    ax2.set_aspect("equal", adjustable="box")
    fig2.tight_layout()
    fig2.savefig(out_dir / "scatter_dob_a_vs_dob_b.png", dpi=150)
    plt.close(fig2)

    # --- Plot 3: fighter-level lex rank vs debut year ---
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    ax3.scatter(rank_fracs, debut_years, s=8, alpha=0.4, c="C2")
    rf = np.array(rank_fracs, dtype=float)
    yy = np.array(debut_years, dtype=float)
    if len(rf) >= 2:
        y_hat_r = lr_rank_year.intercept + lr_rank_year.slope * rf
        ax3.plot(rf, y_hat_r, color="C3", lw=2, label="OLS")
    ax3.set_xlabel("Lex rank fraction (sorted fighter_id, 0=min ... 1=max)")
    ax3.set_ylabel("Year of first Tier-1 fight in dataset")
    ax3.set_title("ID lex order vs career-start cohort (fighter-level)")
    ax3.legend(loc="upper left")
    fig3.tight_layout()
    fig3.savefig(out_dir / "scatter_lex_rank_vs_debut_year.png", dpi=150)
    plt.close(fig3)

    print(f"\nWrote PNGs and report under {out_dir.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
