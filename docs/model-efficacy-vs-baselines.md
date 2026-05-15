# Model efficacy vs baselines (six-outcome UFC Tier‑1)

This note measures the **current multinomial logistic model** (ELO + style + matchup features + calibrated probabilities) against **uninformed random baselines** and an **ELO-only** probabilistic baseline, using the same **Tier‑1 decisive fights**, **fighter A perspective**, and **point-in-time ELO** as the rest of the pipeline.

**Primary metric:** mean **log-loss** (lower is better) — aligned with `eval-holdout`, `score_tier1_fight_slice`, and Phase 3 in [`hyperparameter-tuning.md`](hyperparameter-tuning.md).

**Secondary:** accuracy and macro F1 (higher is better for both); accuracy is easier to read but can hide miscalibration.

---

## 1. Cohort and frozen configuration

| Item | Value |
|------|--------|
| Regression era floor | `master_start_year = 2005` |
| Training cutoff | Tier‑1 rows with `fight_date < 2023-01-01` feed the regression fit (when holdout is on) |
| Pristine **evaluation** cohort | **Calendar years 2023, 2024, 2025** only — decisive Tier‑1, fighter A (matches Phase 3 / `first_run_report.json`; not open-ended `≥ 2023`) |
| Frozen hyperparameters | **`frozen_winner_config`** in [`docs/first_run_report.json`](first_run_report.json) (selection ends 2022; pristine uses this config without re-search) |
| Report snapshot | `generated_utc`: **2026-04-24** (see JSON); cohort *n* matches **1,529** decisive A-side Tier‑1 fights in 2023–2025 for both the saved report and the recomputed ELO baseline below |

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

This baseline is **strong on W/L** (ELO is built for that) but **weak on fine-grained method** (static method priors), so we expect the **full model to gain most on six-way log-loss**.

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

| System | Mean log-loss ↓ | Mean Brier ↓ | Accuracy ↑ | Macro F1 ↑ | Source / notes |
|--------|-----------------|--------------|------------|------------|----------------|
| **Uniform random (6-way)** | 1.7918 | — | 16.7% | ~0 | Theoretical |
| **Full model (bespoke, frozen config)** | **1.3656** | **0.7031** | **41.27%** | **0.2638** | [`first_run_report.json`](first_run_report.json) pristine pool; `generated_utc` **2026-04-24** |
| **XGBoost** (same tabular features, pristine eval) | 1.3930 | 0.7135 | 40.48% | 0.3305 | [`scripts/dev/benchmark_xgboost_vs_holdout.py`](../scripts/dev/benchmark_xgboost_vs_holdout.py) default `--eval-mode pristine`; run **2026-05-05** on current `data/` |
| **ELO-only (6-way, §3)** | 1.4781 | 0.7325 | 40.48% | 0.1790 | Recomputed same cohort + frozen ELO settings (see §3) |

**Bespoke vs XGBoost (same 1,529 fights, same feature rows):** the bespoke multinomial model is **better on the primary metric** — mean log-loss **1.366 vs 1.393** (**~0.027 nats** lower, ~**2.0%** relative improvement vs the XGB score). Brier and accuracy also **slightly favor** bespoke. XGBoost posts **higher macro F1** (0.33 vs 0.26) on this slice, i.e. different error tradeoffs, not better probability quality for handicapping.

**Gains vs uniform:** bespoke log-loss better than uniform by **~0.426** nats/fight (~**23.8%** relative reduction vs \(\log 6\)).

**Gains vs ELO-only:** bespoke log-loss better by **~0.113** nats/fight — the regression layer is **materially improving calibration** on the six-way outcome, not only riding ELO.

**XGBoost extras (same run):** marginal W/L log-loss **0.3730**; W/L accuracy from predicted class collapse **83.13%**; binary finish vs decision F1 **0.5750**.

**Binary W/L (ELO-only, marginal \(p_{\text{win}}\))** on the same 1,529 fights: mean log-loss **~0.457** vs coin baseline **0.693**; six-way **argmax** W/L accuracy ~**85.8%** (see §3). For bespoke W/L metrics vs holdout, use **`eval-holdout`** on a model trained with the same `holdout_start_date` (see [`Tier1SliceScore`](../src/eval/fight_scoring.py)).

---

## 5. Performance over time (selection → pristine)

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

3. **ELO-only baseline** — Not yet a first-class CLI; reproduce by scoring the pristine fight list with the construction in §3 (same loader, `build_elo` / cache, empirical priors from pre‑2023 rows). Expect small drift if the fights CSV grows after the snapshot date.

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

**Latest pristine result (see §4 table):** on the default cohort, **XGBoost did not beat** the frozen bespoke model on mean log-loss; it remains a useful **nonlinear sanity check** and future scoreboard if tunings change.

Remaining roadmap: optional SHAP / calibration_bins if you promote GBDT; otherwise the linear model keeps `explain` and current SOTA on this slice.

---

## See also

- [`hyperparameter-tuning.md`](hyperparameter-tuning.md) — selection vs pristine protocol  
- [`validation-and-few-shot.md`](validation-and-few-shot.md) — leakage and split discipline  
- [`elo-modeling-status.md`](elo-modeling-status.md) — what ELO does and does not feed today  
- [`first_run_report.json`](first_run_report.json) — machine-readable yearly metrics + `frozen_winner_config`
