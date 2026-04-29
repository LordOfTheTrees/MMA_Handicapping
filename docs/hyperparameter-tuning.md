# Hyperparameter Tuning: Architecture, Decisions, and Tradeoffs

This document records **non-code** design choices for Phase 3: how we search for `Config` values, how we avoid overfitting the evidence, and how that relates to a **time-ordered, non-IID** sport where fighters are all connected by the ELO network.

**Related code:** `src/config.py` (tunable fields), `src/cli/train.py` (train command shared by `main.py train` and `python -m src.cli.train`), `src/eval/tuning_harness.py` (walk-forward and scoring utilities).

---

## 1. Goals and metrics

- **Primary objective:** **mean log-loss** on held-out, decisive Tier-1 fights (fighter A’s perspective, same as `eval-holdout`). Log-loss rewards calibrated probabilities; that is what handicapping and EV use.
- **Secondary diagnostic:** **macro F1** over the six outcome classes. Useful to see if the model systematically picks the wrong class, but it ignores probability quality, so it is **not** an optimization target.
- **Stratified storage:** per **weight class** (and per year when applicable), we store the same metrics for later analysis. These slices are **not** used to pick `Config` — only to observe where the model hurts after the fact.

---

## 2. What “holdout” means with point-in-time ELO

We **trust** point-in-time (PIT) ELO and features: a prediction at fight time `t` only uses information from fights that occurred **strictly before** `t` for that weight class. Opponents’ histories are *supposed* to be correlated with yours; that is signal, not leakage.

What we **do** need to control is different:

- **Hyperparameter selection leakage:** comparing many `Config` values and picking the one that looks best on the same calendar span we later report. That **does** overfit, independent of PIT ELO, because the “winner” is an extremum over noisy folds.

- **ELO is always built on full history** in `train` before regression rows are built; each `get_state(..., fight_date)` is still PIT. So “train regression only on past fights” is implemented by **excluding** future rows from the multinomial fit, not by reprocessing ELO on a subset of fights (see harness: `holdout_start_date` on `Config`).

---

## 3. Time split (selection vs pristine)

| Block | Calendar | Role |
|-------|----------|------|
| **Selection** | 2005–2022 (outer years 2006–2022) | Re-tune each year, random search, warm-start, inner scoring — **only** on information strictly *before* each step’s evaluation target year (see below). |
| **Pristine** | 2023, 2024, 2025 | The **2022 winning `Config` only** (no re-tune). Check that forward mean log-loss does not **deteriorate** in a way that would reject shipping. |
| **Ship** | All data through last available date | **Refit** the frozen 2022 `Config` on **all** Tier-1 rows (no holdout) for the production artifact. Holdout is for validation, not for training the product model. |

Rationale: the **pristine** block was never used to *choose* among competing configs, so its trajectory is not subject to the same “winner’s curse” as a block where we ran 50 random trials per year.

---

## 4. Nested walk-forward with warm-started random search

**Outer loop (once per year Y in 2006…2022):**

1. Draw **50** trial `Config` values (pre-committed budget — **no** optional stopping, no plateau early-stop).
2. **Trial 1** = previous year’s **winning** `Config` (warm-start). **Trials 2–50** = i.i.d. random draws from **explicit, written-down** priors per knob. Random search adds **no** surrogate (no Gaussian process, no TPE), so we do not smuggle in smoothness or density assumptions; the only “prior” is the sampling distribution we declare.
3. **Rank** trials by an **inner** score: mean log-loss over a walk-forward on calendar years that are **strictly before Y** and within the training regime (see `tuning_harness` — default uses all such years, with options to use only the last *k* years for cost).
4. The **winner** of the 50 is year **Y**’s config; record **forward** mean log-loss on **calendar year Y** (fight-level, PIT) as that year’s dot on the **selection** diagnostic curve.
5. Carry the winner to year **Y+1** as warm-start (trial 1).

**Inner loop (per trial, when ranking for outer year Y):**  
The inner objective must **not** use the label distribution of year **Y** to pick among trials — otherwise the outer forward score for year Y is optimistically biased. The harness implements an inner walk-forward: for each eval year *v* &lt; *Y*, fit regression with all Tier-1 training rows with `fight_date` &lt; Jan 1 of *v* (i.e. train through end of *v*−1), then score all decisive Tier-1 fights in **calendar *v***. The inner score is the **mean** of per-year log-losses over the chosen inner years (e.g. `master_start_year` … *Y*−1, or a suffix such as the last 3 years only).

**Freeze:** the **2022** winner is the `Config` taken to the **pristine** block. No more search.

**Pristine block:** for 2023, 2024, 2025 only, fit with that frozen `Config` and record yearly mean log-loss. No 50-trial search here.

**Combined plot:** you get ~17 yearly points from the selection regime (config may change every year) **plus** 3 points from the same frozen 2022 config. Visually, distinguish the two segments so readers do not interpret the first 17 as a single static model’s trajectory.

---

## 5. Why random search (and not Bayesian optimization)

- **Random search** only assumes the **sampling distribution** you write for each parameter. It does **not** fit a second surrogate model to “log-loss vs hyperparameters,” so it stays aligned with a stack that is otherwise explicit (bootstrap, Cauchy fallbacks, etc.).
- **Bayesian optimization** (TPE, GP) assumes smoothness / structure of the black-box surface and can concentrate trials adaptively. That is powerful for speed, but it **reintroduces** priors you did not sign off on for the *selection* process itself.
- **OAT (one at a time)** is cheap but **cannot** find interactions between parameters (e.g. `k_base` and `logistic_divisor`).

---

## 6. Stopping rule and comparison budget

- Stopping is **not** “until plateau” (that is optional stopping and inflates the best observed score). We use a **fixed** budget of **50** trials per outer year, chosen **before** the first draw.
- The scarce resource is not wall time but **independent** selection signal: every extra trial on the same inner protocol spends some of the same “budget” to avoid winner’s curse. Parsimony and narrow priors matter more than clever search heuristics.

**Tiebreak:** if two trials are within a small log-loss **band** ε (campaign constant), prefer the **simpler** or **more regularized** `Config`.

---

## 7. Inherent limits (non-stationarity, ~20 years)

- There is **no** set of *IID* fight outcomes; the sport drifts, and the same athletes appear in multiple years.
- **~17 + 3** annual aggregates are **few** points for trend inference; we expect modest statistical power. Per-year log-loss is still precise in **N** (many fights per year), but **across** years, folds are correlated. Treat a “non-increasing” trajectory as **supporting** evidence, not a proof, unless failure is so clear you would not ship.

---

## 8. What the harness does (implementation)

- **`fit_predictor_for_train_before`:** load data, `build_elo` on full history, set `Config.holdout_start_date` so regression trains only on rows with `fight_date` **strictly before** the given cutoff (i.e. train through the prior calendar year when scoring year *v*). Optional **ELO cache** file reuses the same PIT ELO build across many folds when fight count and `ELOConfig` match.
- **`default_inner_eval_years`:** which calendar years are included in the **inner** mean (full range vs last *k* years before outer *Y*).
- **`inner_mean_log_loss` / `inner_mean_log_loss_last_k_years`:** inner score for one `Config` (for random-search trial ranking with last-*k* inner years for cost control).
- **`forward_log_loss_for_eval_year`:** one train+score for a **single** forward calendar year (used for the outer year’s point and for pristine years).
- **`run_selection_walkforward_baseline`:** one fixed `Config`, one forward point per year in a selection range (e.g. 2007–2022).
- **`run_pristine_years`:** fixed `Config`, default years **2023–2025** (pristine test block).
- **`run_random_search_for_outer_year`:** one outer year, *n* trials (warm start on trial 0 + `tuning_priors.sample_random_config` thereafter).
- **`run_selection_campaign_with_search`:** full selection block: Per calendar outer year, *n* trials (default 50) with previous year’s config as warm start; return per-year forward scores and the final-year winner `Config` for the pristine block.
- **`score_tier1_fight_slice`:** fight-level log-loss, Brier, accuracy, macro F1, and **per–weight-class** means for any fight list (aligned with PIT `predict`).

**End-to-end run (baseline + pristine + figures + CSV/JSON):** from the repo root,

```text
python -m src.cli.run_phase3_tuning --data-dir ./data --out-dir ./data/phase3_eval
```

Writes `phase3_metrics.csv`, `phase3_report.json`, `pristine_test_yoy.png` (2023–2025 bars), `log_loss_selection_and_pristine.png` (trajectory), and an ELO cache under `out-dir` to speed re-runs. Use `--no-walkforward` to only score the pristine years (faster). **Full protocol** (50 trials per selection year, warm-start chain, pristine uses frozen 2022 winner): add `--selection-search` (and optionally `--n-trials 50`, the default when this flag is set). For a **single-outer-year** debug run: `--search-outer-year 2020 --n-trials 50` (no `--selection-search`).

---

## 9. Pristine byproduct: case studies

Fights in 2023–2025 with the **highest** per-fight log-loss (or largest surprise vs implied confidence) are natural **article** case studies: where the model and reality disagree most, after we have already frozen `Config`.

---

## 10. Changelog of decision (short)

- **Apr 2026 — reference run on disk** — A full `python -m src.cli.run_phase3_tuning --selection-search` (default 50 trials/outer year, 2007–2022 → pristine 2023–2025) has been **executed**; outputs: `data/phase3_eval/phase3_report.json` (incl. `frozen_winner_config`), `phase3_metrics.csv`, plots, optional ELO cache. **Planned check:** a **faster** config pass (e.g. baseline walk-forward, 10–20 trials, or 2018–2022 only) to confirm **stability of rankings** before repeating 50/yr. **Economic** backtest (odds) remains **future** work.
- Random search, 50 trials, warm-start, fixed inner walk-forward, frozen 2022 `Config` for 2023–2025, full-data refit for ship.
- Log-loss primary, F1 secondary, weight-class slices stored, not used for selection.

For the Phase 3 checklist of individual knobs, see [`todo.md` §3.3](todo.md#33-phase-3-tuning-inventory-evaluate-on-holdout) and implementation fields in `src/config.py`.
