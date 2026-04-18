# Validation strategy, CV, and few-shot behavior

This note complements [`todo.md`](todo.md) Phase 2–3 and [`architecture.md`](architecture.md). It answers: **how to split data for honest scores**, and **how the current pipeline already mitigates thin per-fighter histories**.

---

## What you have today

- **Many fight rows**, but **fighters repeat** across rows. A random **IID K-fold on fights** leaks information: the same athlete appears in train and validation, so metrics look better than deployment.
- The **regression** is fit on **post-era** UFC fights only (`FeatureConfig.era_cutoff_year`, default **2013**). **ELO** and style axes use **full history** before each fight date (lookahead-free in training rows).
- **Training rows** currently use **all** post-era decisive outcomes (no holdout split in code yet). Phase 3 in [`todo.md`](todo.md) adds an explicit holdout.

---

## Recommended CV / test protocol (in order of preference)

### 1. Time-based holdout (primary)

Reserve a **recent calendar window** (e.g. fights from **2023-01-01** onward) for evaluation only:

- **Fit** the multinomial model on fights **strictly before** the cutoff (still building features with ELO/style **as of each fight date**).
- **Score** on the holdout: log-loss, Brier, calibration.

This matches **how you predict real cards** (past data only) and avoids optimistic leakage from future fights.

**Variant:** rolling **walk-forward** evaluation (expand training end date month by month; score the next chunk). More work, best picture of drift.

### 2. Fighter-grouped or event-grouped splits (when you need K-fold)

If you need multiple folds instead of one time cut:

- Assign **fighters** (or **events**) to folds so **no fighter appears in two folds** for validation *or* use **group K-fold** with `groups=event_id` or `groups=min(fighter_a, fighter_b)` (approximate).

Pure **event** grouping is weaker (same fighter can appear in train and val on different cards) but better than IID. **Fighter**-grouped CV is stricter and closer to “debut / few-shot” reality.

### 3. What to avoid

- **IID row shuffle** across all UFC fights.
- Validating on the same era you tuned **era_cutoff_year** on without a nested outer split (double-dipping).

---

## Few-shot / cold-start: what already helps

The codebase is already oriented toward **data-sparse fighters**:

| Mechanism | Role |
|-----------|------|
| **ELO + Kalman** | Strength estimate with uncertainty; updates from every fight; not “per-class” sample size. |
| **Style axes + `min_fights_style_estimate`** | Until ELO-weighted effective fights reach the threshold, axes **blend toward pedigree priors** (`apply_cold_start_prior`). Lower threshold → faster reliance on observed style, **higher variance**; raise → more prior, **stabler** but slower to personalize. |
| **Pedigree fields** | Optional boost when UFC sample is tiny (currently often zeros in scraped profiles). |
| **Physical features** | Reach, height, stance, age diff work from **profiles** even when fight-count is low. |
| **Tier 2/3 fights** | Inform ELO (not regression features today) for cross-promotion history. |

**There is no separate “prediction model” and “explanation model”.** [`main.py`](../main.py) **`predict`** and **`explain`** share the **same** fitted `MultinomialLogisticModel` in `model.pkl`. `explain` only exposes the **linear decomposition** of logits; it does not retrain anything.

---

## Practical knobs to explore (after a time holdout exists)

1. **`min_fights_style_estimate`** (default 3): the main lever for “trust UFC sample sooner vs stay prior-heavy longer.” Tune on holdout log-loss.
2. **`recency_decay_rate`** (style axes): more emphasis on recent fights may help veterans; hurts if the last few fights are unrepresentative.
3. **`l2_lambda`**: stronger shrinkage helps when **feature noise** dominates for thin histories (indirect few-shot effect).
4. **Pedigree / Tier 2–3**: best path to **real** few-shot gains when UFC *n* is tiny but outside data exists.

---

## A/B “symmetry” and matchup interactions

[`matchup/interactions.py`](../src/matchup/interactions.py) states that signed **differences** flip sign when A and B swap. **Multiplicative interaction terms** (`striking_matchup`, `grappling_matchup`, `finish_matchup`) do **not** satisfy `f(A,B) = -f(B,A)` in general (`sa*(1-sb) ≠ -sb*(1-sa)`). So **exact** mirror identities such as `P_A(win KO) = P_B(lose KO)` are **not guaranteed** for the full feature vector. Treat tight symmetry assertions (e.g. `1e-6`) as **invalid** unless interactions are removed or redesigned.

[`scripts/phase2_smoke.py`](../scripts/phase2_smoke.py) uses a **relaxed** tolerance by default; use `--strict-symmetry` only if you are experimenting with a symmetric feature set.

---

## Phase 2 automation

[`scripts/phase2_smoke.py`](../scripts/phase2_smoke.py) runs fast checks: finite `X_train`, class counts, **symmetry** of point probabilities under A/B swap (no bootstrap), and a small **ELO top/tail** snapshot. Full **`predict`** (with CIs) remains `python main.py predict ...`.

---

## See also

- [`TODO.md`](../TODO.md) — ordered “next work bout”
- [`docs/todo.md`](todo.md) — Phase 2.2 symmetry snippet, Phase 3 holdout outline
- [`docs/architecture-decisions.md`](architecture-decisions.md) — scraper / pipeline product decisions
