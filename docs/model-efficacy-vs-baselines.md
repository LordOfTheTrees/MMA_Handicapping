# Model efficacy vs baselines (six-outcome UFC Tier‑1)

This note measures the **current multinomial logistic model** (ELO + style + matchup features + calibrated probabilities) against **uninformed random baselines** and an **ELO-only** probabilistic baseline, using the same **Tier‑1 decisive fights**, **fighter A perspective**, and **point-in-time ELO** as the rest of the pipeline.

**Primary metric:** mean **log-loss** (lower is better) — aligned with `eval-holdout`, `score_tier1_fight_slice`, and Phase 3 in [`hyperparameter-tuning.md`](hyperparameter-tuning.md).

**Secondary:** accuracy and macro F1 (higher is better for both); accuracy is easier to read but can hide miscalibration.

> **Correction (numbers recomputed).** Every figure previously published in this note was
> produced through a lookahead bug in `ELOModel.get_state`: historical queries returned each
> fighter's **terminal** ELO rather than their rating at fight time, so pre-fight feature rows
> carried the outcomes of fights that had not happened yet. Because `elo_differential` and the
> opponent-quality term in the style axes both read through that call, **every** feature row
> built by `build_xyw_for_fights` was affected — bespoke, XGBoost and the ELO-only baseline
> alike. The holdout cutoff did not help: the rows themselves contained future information.
>
> §3, §3.1 and §4 below are recomputed with point-in-time ELO. §5 has **not** been recomputed
> and is still leak-contaminated — see the warning there. Reproduce with
> `python scripts/dev/rebuild_efficacy_table.py --ab`, whose `--ab` column reruns the pre-fix
> code path and recovers the previously published values, confirming the provenance of the change.

---

## 1. Cohort and frozen configuration

| Item | Value |
|------|--------|
| Regression era floor | `master_start_year = 2005` |
| Training cutoff | Tier‑1 rows with `fight_date < 2023-01-01` feed the regression fit (when holdout is on) |
| Pristine **evaluation** cohort | **Calendar years 2023, 2024, 2025** only — decisive Tier‑1, fighter A (matches Phase 3 / `first_run_report.json`; not open-ended `≥ 2023`) |
| Frozen hyperparameters | **`frozen_winner_config`** in [`docs/first_run_report.json`](first_run_report.json) (selection ends 2022; pristine uses this config without re-search) |
| Report snapshot | `generated_utc`: **2026-04-24** (see JSON) — **superseded for §4**: those stored metrics predate the point-in-time ELO fix. §4 is recomputed from the current CSV; `first_run_report.json` itself has not been regenerated. |
| Cohort size | **1,529** decisive A-side Tier‑1 fights in 2023–2025. Note this is *not* the same slice as `eval-holdout`, which scores every row with `fight_date >= 2023-01-01` and pulls in 2026 cards (**1,742** on the current CSV). Compare §4 numbers only against the pristine cohort. |

Production defaults in [`src/config.py`](../src/config.py) match that frozen winner unless you override at train time.

---

## 2. Uninformed random baselines

These are **theoretical** references (same labels as the model: six mutually exclusive outcomes).

| Baseline | Definition | Mean log-loss | Multiclass accuracy “chance” |
|----------|------------|---------------|------------------------------|
| **Uniform six-way** | \(p_c = 1/6\) | \(\log 6 \approx\) **1.7918** | **16.67%** |
| **Fair coin (binary W/L)** | \(P(\text{A wins}) = 0.5\) | \(\log 2 \approx\) **0.6931** | **50%** (marginal W/L only) |

The CLI helper [`print_holdout_baseline_report`](../src/eval/holdout_metrics.py) prints the six-way and binary comparisons for any scored holdout slice.

---

## 3. ELO-only baseline (definition)

To isolate **strength rating** from **fight-style / interaction** learning, we use a **generative ELO-only** model:

1. **Point-in-time ELO** for A and B at `fight_date` (same Kalman ELO build as the full pipeline; frozen `logistic_divisor` from config).
2. **Binary win probability for A:** \(p_{\text{win}} = \text{expected\_score}(E_a, E_b)\) (standard logistic ELO formula — see [`expected_score`](../src/elo/elo.py)).
3. **Outcome method** is independent of the opponent given win/loss: use **global empirical** probabilities of each six-way label **conditional on A winning** vs **conditional on A losing**, estimated from **all Tier‑1 decisive fights with `fight_date < 2023-01-01`** (pre-holdout), fighter A perspective — pooled across divisions.

Then for each fight:

\[
\mathbb{P}(y \mid \text{A wins}) = p_{\text{win}} \cdot \hat{\pi}_{\text{win}}(y), \quad
\mathbb{P}(y \mid \text{A loses}) = (1 - p_{\text{win}}) \cdot \hat{\pi}_{\text{lose}}(y)
\]

with \(\hat{\pi}\) normalized within the win ({0,1,2}) or loss ({3,4,5}) classes.

**Marginal binary W/L** from this model uses \(p_{\text{win}}\) only (same collapse as `mean_wl_log_loss` in [`fight_scoring.py`](../src/eval/fight_scoring.py)).

This baseline is **weak on fine-grained method** by construction (static method priors), so the
full model should gain most on six-way log-loss.

An earlier version of this note also described it as **strong on W/L** ("ELO is built for that").
That is **not true** once ELO is queried point-in-time: the baseline scores **57.03%** W/L accuracy
with a marginal W/L log-loss of **0.6899** against a coin's **0.6931** — a gain of 0.003 nats, which
is close to nothing. The apparent W/L strength was the lookahead bug: the "ELO" being compared was
each fighter's end-of-dataset rating, which already encoded who won.

**Implementation:** [`scripts/dev/baseline_elo_only.py`](../scripts/dev/baseline_elo_only.py)
implements this specification (it was prose-only until the recomputation; see §6).

### 3.1 Empirical method priors (training side, pre‑2023)

Estimated on **6,366** decisive A-side Tier‑1 fights (`fight_date < 2023-01-01`, post‑2005 era filter), current `data/ufcstats_fights.csv`:

| Given | P(A wins by KO/TKO) | P(A wins by sub) | P(A wins by decision) |
|-------|---------------------|------------------|------------------------|
| **A wins** (n = 3,214) | 0.3286 | 0.1982 | 0.4732 |

| Given | P(A loses by decision) | P(A loses by KO/TKO) | P(A loses by sub) |
|-------|------------------------|---------------------|-------------------|
| **A loses** (n = 3,152) | 0.4753 | 0.3319 | 0.1929 |

(Recomputing these after a major scrape will shift them slightly.)

---

## 4. Headline comparison — pristine 2023–2025 (n = 1,529)

All three systems scored on one cohort from one ELO build by
[`scripts/dev/rebuild_efficacy_table.py`](../scripts/dev/rebuild_efficacy_table.py).

| System | Mean log-loss ↓ | Mean Brier ↓ | Accuracy ↑ | Macro F1 ↑ | W/L log-loss ↓ | W/L acc ↑ |
|--------|-----------------|--------------|------------|------------|----------------|-----------|
| **Uniform random (6-way)** | 1.7918 | — | 16.67% | ~0 | 0.6931 | 50.00% |
| **Full model (bespoke, frozen config)** | **1.6615** | **0.7899** | **31.59%** | 0.2054 | **0.6528** | **60.89%** |
| **XGBoost** (same tabular features) | 1.7108 | 0.8083 | 28.97% | **0.2201** | 0.6654 | 58.53% |
| **ELO-only (6-way, §3)** | 1.7090 | 0.8055 | 28.58% | 0.1258 | 0.6899 | 57.03% |

**Gains vs uniform:** bespoke log-loss is better than uniform by **0.130** nats/fight
(~**7.3%** relative reduction vs \(\log 6\)). The previously published figure was ~0.426 nats
(~23.8%); roughly two-thirds of that apparent gain was the ELO leak.

**Gains vs ELO-only:** bespoke is better by **0.048** nats/fight. The conclusion that the
regression layer improves calibration rather than only riding ELO **still holds** — bespoke
captures ~1.6× the gain over uniform that ELO-only does (0.130 vs 0.083 nats) — but the margin
is less than half the **~0.113** nats previously claimed.

**Bespoke vs XGBoost (same 1,529 fights, same feature rows):** bespoke remains **better on the
primary metric** — **1.6615 vs 1.7108**, **0.049** nats lower. This margin **grew** with the fix
(it was ~0.027 nats when both were leaking), so the case for the bespoke model over a
general-purpose GBDT is stronger now, not weaker. Brier and accuracy also favor bespoke; XGBoost
still posts **higher macro F1** (0.220 vs 0.205), the same tradeoff as before — different error
profile, not better probability quality for handicapping.

**ELO-only and XGBoost now land within 0.002 nats of each other** on the primary metric
(1.7090 vs 1.7108), and both sit close to uniform. Under the leak they looked well separated
(1.4808 vs 1.4115). No significance test has been run on that gap — it is reported as a point
estimate on a single cohort, and 0.002 nats should not be read as a meaningful ordering.

**W/L reality check:** the bespoke model's binary W/L accuracy is **60.89%**. Against a market
ceiling generally placed near 65%, that is a plausible if modest edge. The previously published
**~85.8%** (ELO-only, six-way argmax collapse) was not attainable performance — it was the model
reading end-of-dataset ratings. For W/L metrics on the open-ended holdout rather than this
pristine cohort, use **`eval-holdout`** (see [`Tier1SliceScore`](../src/eval/fight_scoring.py)),
remembering it scores a larger slice (n = 1,742).

### 4.1 What the leak was worth

Same data, same code, `get_state` toggled between the pre-fix and point-in-time implementations
(`rebuild_efficacy_table.py --ab`). The leaky column reproduces the previously published table,
which is what identifies the leak as the cause of the change.

| System | Leaky log-loss | Honest log-loss | Δ log-loss | Leaky W/L acc | Honest W/L acc | Δ W/L acc |
|--------|----------------|-----------------|-----------|---------------|----------------|-----------|
| Full model (bespoke) | 1.3646 | 1.6615 | −0.297 | 84.50% | 60.89% | −23.6 pts |
| XGBoost | 1.4115 | 1.7108 | −0.299 | 82.15% | 58.53% | −23.6 pts |
| ELO-only (§3) | 1.4808 | 1.7090 | −0.228 | 84.63% | 57.03% | −27.6 pts |

Reproduction against the previously published values: bespoke **1.3646** vs 1.3656 published,
ELO-only **1.4808** vs 1.4781. XGBoost recomputes to **1.4115** against 1.3930 published; that
row was generated on a later data snapshot (**2026-05-05**) than the bespoke row
(**2026-04-24**), which accounts for the residual gap.

**Feature weights shifted as much as the metrics.** Under the leak, `elo_differential` carried
**0.628** of total coefficient magnitude; honestly it carries **0.191**. Physical attributes rise
from 0.135 to **0.347** and become the largest family — `age_diff_days` 0.043 → **0.165**,
`reach_diff_cm` 0.040 → **0.102**. These features were not weak signals; they were crowded out by
a feature that already knew the outcome. Any prior conclusion about feature importance drawn from
the old fit should be treated as void
(see [`scripts/dev/ab_elo_leak_attribution.py`](../scripts/dev/ab_elo_leak_attribution.py)).

---

## 5. Performance over time (selection → pristine)

> **⚠ Not recomputed — these numbers are leak-contaminated.** Everything in §5 is read from
> [`first_run_report.json`](first_run_report.json), which was generated before the point-in-time
> ELO fix. Regenerating it means re-running the Phase‑3 harness (§6.2), not just re-scoring, so
> it is left as-is and flagged rather than silently updated. Expect the same direction of change
> as §4 — roughly +0.3 nats of log-loss and a large accuracy drop in every row — and treat the
> yearly *trajectory* as unverified. The §4 pristine pool below is superseded by the recomputed
> §4 table.

The committed JSON records **forward** multiclass metrics **year by year**:

- **Selection regime (2007–2022):** configuration was re-selected each year (random search / warm start per [`hyperparameter-tuning.md`](hyperparameter-tuning.md)); yearly forward scores are **diagnostic**, not a single static model curve.
- **Pristine (2023–2025):** **one** frozen 2022 winner — no further knob search.

### 5.1 Selection years (forward eval, n and scores)

| Year | n fights | Mean LL | Accuracy |
|------|----------|---------|----------|
| 2007 | 168 | 1.668 | 25.0% |
| 2008 | 201 | 1.662 | 23.9% |
| 2009 | 212 | 1.582 | 31.6% |
| 2010 | 249 | 1.639 | 24.9% |
| 2011 | 295 | 1.579 | 31.5% |
| 2012 | 331 | 1.548 | 30.2% |
| 2013 | 375 | 1.515 | 33.3% |
| 2014 | 493 | 1.513 | 33.5% |
| 2015 | 462 | 1.508 | 34.6% |
| 2016 | 483 | 1.538 | 34.6% |
| 2017 | 445 | 1.519 | 34.2% |
| 2018 | 468 | 1.550 | 36.1% |
| 2019 | 505 | 1.498 | 38.8% |
| 2020 | 442 | 1.493 | 35.1% |
| 2021 | 495 | 1.520 | 40.4% |
| 2022 | 505 | 1.452 | 34.9% |

**Selection block pooled (2007–2022):** n = **6,129**, mean log-loss ≈ **1.532**, accuracy ≈ **33.9%**, macro F1 ≈ **0.253** (from aggregating `first_run_report.json` rows).

### 5.2 Pristine years (frozen 2022 config)

| Year | n | Mean LL | Accuracy |
|------|---|---------|----------|
| 2023 | 503 | 1.408 | 37.8% |
| 2024 | 511 | 1.357 | 42.5% |
| 2025 | 515 | 1.333 | 43.5% |

Pooled pristine: §4. Trajectory is **flat to mildly improving** in log-loss across 2023–2025 on this snapshot — consistent with “no obvious post-freeze collapse,” though year-to-year variation is noisy.

---

## 6. Reproducing and refreshing

1. **Holdout report (full model artifact)** — Train with `holdout_start_date` set, then:  
   `python main.py --model-path ./data/model.pkl eval-holdout`  
   (prints random baselines from [`holdout_metrics.py`](../src/eval/holdout_metrics.py).)

2. **Phase 3 harness** — [`hyperparameter-tuning.md`](hyperparameter-tuning.md) §8:  
   `python -m src.cli.run_phase3_tuning --data-dir ./data --out-dir ./data/phase3_eval`  
   Regenerates CSV/JSON/plots; large runs use `--selection-search`.

3. **ELO-only baseline** — now scripted (it was prose-only, which is why its published numbers
   could not be re-derived):
   `python scripts/dev/baseline_elo_only.py --ab`
   Expect small drift if the fights CSV grows after the snapshot date.

4. **Whole §4 table in one run** — bespoke, ELO-only and XGBoost on one cohort and one ELO build:
   `python scripts/dev/rebuild_efficacy_table.py --ab`
   `--ab` adds the pre-fix leaky column used in §4.1. Bootstrap resamples are disabled (they feed
   prediction CIs via `compute_prediction_ci`, while scoring uses `predict_proba_point_only`), so
   metrics are identical to a full run and it finishes in ~1 minute per arm instead of ~50.

5. **Leak attribution / feature re-weighting** —
   `python scripts/dev/ab_elo_leak_attribution.py`
   Trains twice with only `get_state` swapped and prints the metric and coefficient-mass deltas.

---

## 7. XGBoost benchmark (same features and time split)

A **multiclass XGBoost** reference is implemented as a **standalone script** (optional dependency) so the core package stays lean.

| Item | Detail |
|------|--------|
| **Script** | [`scripts/dev/benchmark_xgboost_vs_holdout.py`](../scripts/dev/benchmark_xgboost_vs_holdout.py) |
| **Install** | `pip install xgboost` or `pip install -r requirements-benchmark.txt` |
| **Features** | [`MMAPredictor.build_xyw_for_fights`](../src/pipeline.py) — same construction as `train_regression` |
| **Split** | **Train:** `fight_date < holdout_start` (default `2023-01-01`). **Test (default):** only Tier‑1 fights in **`--eval-years`** (default `2023,2024,2025`) — same **pristine** cohort as [`first_run_report.json`](first_run_report.json), not every row with `fight_date ≥ holdout` (which would pull in **2026+** as your CSV grows). Use `--eval-mode expanding` for that behavior. |
| **Flags** | `--sample-weight recency` weights training rows (point L-BFGS is still unweighted); `--fit-logistic` fits [`MultinomialLogisticModel`](../src/model/regression.py) on identical `X_train` for a same-day head-to-head |
| **Metrics** | [`tier1_slice_score_from_probs`](../src/eval/fight_scoring.py) — same slice as `score_tier1_fight_slice` |

**Command:**

```bash
python   scripts/dev/benchmark_xgboost_vs_holdout.py   --data-dir ./data  --elo-cache ./data/elo_cache.pkl
```

**Expect:** on a typical laptop, **ELO build or cache load** dominates the first run; materializing **feature rows** (~6.5k train + ~1.5k test) is usually **several minutes** (style axes per row). **XGBoost fit** is typically on the order of **tens of seconds** at default `n_estimators=300`. Use `--matrix-progress-every 0` for quieter logs.

**Latest pristine result (see §4 table):** on the default cohort, **XGBoost did not beat** the
frozen bespoke model on mean log-loss (**1.7108 vs 1.6615**), and its shortfall is **wider** after
the point-in-time ELO fix than before it (0.049 vs ~0.027 nats). It remains a useful **nonlinear
sanity check** and future scoreboard if tunings change. Note that with honest ELO it lands within
0.002 nats of the far simpler ELO-only baseline — on this feature set the nonlinearity is buying
almost nothing.

**Cache note:** `--elo-cache` files written before the point-in-time fix are rejected
(`_elo_cache_v` bumped to 2), and pre-fix pickles raise on historical queries rather than silently
answering with terminal ELO. Delete any stale cache and let it rebuild; the ELO build itself is
seconds, not minutes.

Remaining roadmap: optional SHAP / calibration_bins if you promote GBDT; otherwise the linear model keeps `explain` and current SOTA on this slice.

---

## See also

- [`hyperparameter-tuning.md`](hyperparameter-tuning.md) — selection vs pristine protocol  
- [`validation-and-few-shot.md`](validation-and-few-shot.md) — leakage and split discipline  
- [`elo-modeling-status.md`](elo-modeling-status.md) — what ELO does and does not feed today  
- [`first_run_report.json`](first_run_report.json) — machine-readable yearly metrics + `frozen_winner_config`
