# ELO modeling — current state and next steps

This document records **where the ELO + Kalman layer stands** after iterative tuning, what is wired into the rest of the pipeline, and **planned improvements** (especially using Kalman uncertainty consistently).

For **knob semantics** (what each parameter does when you change it), see [`elo-tuning-knobs.md`](elo-tuning-knobs.md). For **design intent** (stages, cold start, global layoff clock), see [`architecture.md`](architecture.md) §4 and **ADR-15** in [`architecture-decisions.md`](architecture-decisions.md). For the **layoff-response direction** (why we amplify rather than damp updates after idle time), see **ADR-16** and the full framing in [`elo-kalman-layoff-philosophy.md`](elo-kalman-layoff-philosophy.md).

---

## Implemented today

| Piece | Status |
|--------|--------|
| **Per–weight-class ELO means** | Each `(fighter_id, weight_class)` has a Kalman state: mean (`value`) + variance. |
| **`k_base`** | **100** — scales classical ELO deltas (after method multiplier). |
| **`logistic_divisor`** | **300** in `ELOConfig`, threaded into `expected_score` / `elo_delta` (steeper win-expectancy vs rating gap than the old hardcoded 400). |
| **Finish multipliers** | **KO/TKO** and **submission** both **×1.5** on `k_base`; unanimous decision **×1.0**; split/majority **×0.5**; draw / NC / DQ **×0**. |
| **Kalman process noise** | **0.01** variance per **day** of global inactivity (was **0.0025**; original exploration included **0.05** / **0.10**). |
| **Global layoff clock (ADR-15)** | `kalman_predict` before a bout uses days since the fighter’s **last fight in any division**; per-division `last_fight_date` on `ELOState` remains “last bout in this class.” |
| **Draw / NC / DQ** | No ELO delta; clock and division bookkeeping still advance. |
| **Validation tooling** | [`scripts/chart_elo_distributions.py`](../scripts/chart_elo_distributions.py) — histograms by division, summary stats, **`--top-n`** ranked tables; default chart **`data/elo_by_division.png`**. |

**Source of truth for numeric defaults:** [`src/config.py`](../src/config.py) (`ELOConfig`) and [`src/elo/elo.py`](../src/elo/elo.py) (`_K_SCALE`).

---

## How ELO connects to the multinomial model

- **Regression feature:** Only **`elo_differential`** = `elo_a.elo - elo_b.elo` enters [`build_matchup_features`](../src/matchup/interactions.py) → [`MatchupFeatures`](../src/data/schema.py). That uses the **Kalman mean** at fight date (via `get_state(..., as_of_date=fight_date)`).
- **`ELOState.uncertainty`** (Kalman variance after any forward `predict` for `as_of_date`) is **not** a feature today and does **not** feed prediction CIs.
- **Prediction CIs** come from **bootstrap / Cauchy** on the fitted regression ([`src/confidence/intervals.py`](../src/confidence/intervals.py)), not from propagating ELO Kalman variance.

So: uncertainty is **maintained** inside the ELO filter and affects **how much the mean moves** on the next measurement update (gain \(K = P/(P+R)\)), but the **headline ELO number** does not decay toward 1500 during idle time—only **\(P\)** grows. The **expected-score / surprise** step inside each fight update still uses **point means only**.

[`architecture.md`](architecture.md) §8.4 has been tightened to match this (see that section — ELO Kalman variance is not yet in the feature vector).

---

## Next step — apply Kalman uncertainty to “numeric ELO” end-to-end

**Problem (from design discussion):** After time off we are **less sure** of true skill; that doubt should affect not only internal filter gain but, ideally, how we **treat** the rating in the model—e.g. uncertainty around the scalar ELO and how **wins/losses** are evaluated when beliefs are diffuse.

**Possible directions (pick after experiments / Phase 3):**

1. **Features** — Expose matchup uncertainty (e.g. `sqrt(P_a) + sqrt(P_b)`, or sum of variances) so the regression can down-weight or interact with `elo_differential` when both fighters are poorly observed.
2. **Expected score** — Replace point-mean vs point-mean `expected_score` with an integral over Gaussian beliefs on latent ratings (heavier change; non-standard Elo).
3. **Output uncertainty** — Combine ELO variance with existing bootstrap CIs (variance propagation or simple Bonferroni-style widening for “epistemic” ELO layer).
4. **Explicit mean decay on idle (optional)** — If product goal is “inactive fighters **lower** in point ELO,” that requires a **mean** pull in `kalman_predict` (or a separate prior), not only increasing process noise (which inflates \(P\) but leaves `value` flat until the next fight).

Document outcomes of any experiment in this file or in [`elo-tuning-knobs.md`](elo-tuning-knobs.md).

---

## See also

| Topic | Where |
|--------|--------|
| Knob guide | [`elo-tuning-knobs.md`](elo-tuning-knobs.md) |
| ADR global layoff clock | [`architecture-decisions.md`](architecture-decisions.md) **ADR-15** |
| ADR amplify-on-layoff decision | [`architecture-decisions.md`](architecture-decisions.md) **ADR-16** + [`elo-kalman-layoff-philosophy.md`](elo-kalman-layoff-philosophy.md) |
| Roadmap / Phase gates | [`TODO.md`](../TODO.md), [`todo.md`](todo.md) |
