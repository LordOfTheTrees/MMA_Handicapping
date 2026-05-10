# Displayable data by fight specificity

Concept inventory for a future site: what you can show users, ordered from **least** to **most** specific to a single matchup. Values below mirror what the codebase already defines or computes (`src/data/schema.py`, `src/pipeline.py`, eval modules, architecture docs).

---

## 1. Global (not tied to a fight)

**Model transparency & calibration**

- Training / regression era: `master_start_year`, share of data in the “modern” window (`pct_post_era`-style summaries where computed).
- Confidence interval setup: method label (`bootstrap`, `bootstrap_elo_mc`, `elo_mc`, `cauchy`, `cauchy_wc_debut`), effective sample size (`effective_n`), CI level (from model config, e.g. `ci_alpha`).
- Optional **holdout report** (research / “about the model” page): sample size `n`, mean log-loss, mean Brier, six-way accuracy, macro F1; binary win/loss log-loss, accuracy, F1; finish-vs-decision binary F1; comparisons to uniform-six-way and 50/50 baselines.

**Cross-cutting enumerations** (for legends, filters, copy)

- Weight classes, result methods (KO/TKO, sub, decision flavors, draw, NC, DQ), data tiers (1–4), stance.

---

## 2. Weight-class cohort

**Aggregates sliced by division** (e.g. filters or division landing pages)

- Backtest metrics **per weight class**: `n`, mean log-loss, mean Brier, six-way accuracy, macro F1, W/L metrics, finish-vs-decision F1 (from `WeightClassScoreSlice` / holdout tooling).
- **Population context** (if you precompute): distribution of finish rates, decision rates, or ELO ranges in that division from historical `FightRecord` rows — not a first-class object today, but derivable from loaded fights.

---

## 3. Fighter (career · one athlete)

**Static profile (`FighterProfile`)**

- Name, ids, reach, height, date of birth (→ age at any fight date), stance.
- Pedigree signals: wrestling / boxing / BJJ (0–1), used for cold-start priors.

**Per division state (`ELOState` — one row per fighter × weight class)**

- ELO (Kalman mean), uncertainty (variance), fight count in that division, primary data tier, last fight date **in that division** (with the global layoff clock documented in architecture for uncertainty).

**Per division style (`StyleAxes`)**

- Striker score, grappler score, finish threat, finish vulnerability (each with uncertainty where applicable).
- ELO-weighted effective quality fight count (`n_quality_fights`).

**Trajectory (optional chart)**

- Time series of ELO after each bout when trajectories are recorded: date, ELO, opponent id (labeled with name when profiles exist).

**Fight ledger**

- List of `FightRecord` rows involving this fighter: date, promotion, weight class (canonical + raw label if unknown/catch), opponent, outcome, method, tier; Tier 1 per-fight stats when present (`FightStats`: sig strikes, TDs, control time, sub attempts, fight duration).

---

## 4. Event / card (many bouts on the same show)

*Note: the canonical pipeline centers on `FightRecord`; there is no unified “event” id in the core schema. A card view is a **synthetic** grouping (e.g. same `promotion` + `fight_date`, or external event metadata from scrapers).*

**If you group fights into a card, you can show:**

- Card header: date, promotion, optional event name/URL if you store it.
- Per bout on the card: compact **historical** row (participants, result, method) or **predictive** row (win probabilities, favored side, uncertainty) — each cell links to the single-bout views below.
- Card-level rollups (conceptual): count of finishes vs decisions, average predicted competitiveness, “model favorite” sweep tally — all derived from the list of bouts.

---

## 5. Single bout — historical (completed fight)

**Identity & context**

- `fight_id`, `fighter_a_id`, `fighter_b_id`, `fight_date`, `promotion`, `weight_class` (+ optional `weight_class_raw`), `tier`.

**Outcome**

- `winner_id` (empty for draw / NC / DQ per your rules), `result_method`.

**Tier 1 stats (`FightStats`), per corner**

- Significant strikes landed / attempted / absorbed, takedowns landed / attempted, control time, submission attempts, total fight time.

---

## 6. Single bout — predictive (scheduled or hypothetical)

**Main outputs (`PredictionResult`)**

- Six **mutually exclusive** outcome probabilities (fighter A perspective): win KO/TKO, win sub, win decision, lose decision, lose KO/TKO, lose sub (sum to 1).
- **Derived bars** (same properties as in code): `total_win`, `total_lose`, `finish_win`, `finish_lose`, `go_to_decision`.
- **Confidence intervals** for each of the six classes (lower, upper from prediction CI machinery).
- **Metadata**: `ci_method`, `effective_n`, `pct_post_era` (when populated), fight date, weight class, both fighter ids.

**Matchup feature vector (`MatchupFeatures`) — “why these numbers”**

- ELO differential; striker / grappler / finish threat / finish vulnerability differentials.
- Interaction terms: `striking_matchup`, `grappling_matchup`, `finish_matchup`.
- Physical: reach diff, height diff, stance mismatch flag, age diff (days).

**Interpretability layer (`explain`-style)**

- For each outcome class: predicted probability plus ranked **additive log-odds contributions** from each named feature (exact decomposition from multinomial coefficients × features).

**Optional UI inputs / overlays**

- Hypothetical **days idle** per corner (affects Cauchy ELO Monte Carlo / γ for CI width only, per `predict()` docs).
- Human-readable fighter resolution (exact + fuzzy name match) as in `predict-human`.

**Visualization patterns already prototyped**

- Stacked outcome bars, marginal CI whiskers, combined win-side uncertainty (see `plot_prediction_three_viz`).

---

## Quick “page type” mapping

| Specificity | Example page |
|------------|----------------|
| Global | About / methodology / calibration |
| Weight class | Division leaderboard, division accuracy |
| Fighter | Profile, ELO & style by division, fight history, trajectory |
| Card | “UFC 300” style slate (grouped rows) |
| Bout (past) | Fight detail + stats |
| Bout (future) | Prediction + CIs + feature breakdown + explain |

This list is **backward-looking facts** (records, profiles, states) plus **forward-looking model outputs** (probabilities, intervals, explanations). Anything at “card” or “event” level is a **view** over the same underlying bout- and fighter-level objects.
