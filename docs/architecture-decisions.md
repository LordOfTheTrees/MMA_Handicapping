# Architecture decision record

This document captures **implementation and operations decisions** that emerged while building the UFCStats path and running the pipeline. It does not repeat the modeling design in [`architecture.md`](architecture.md) (stages, ELO math, interpretability goals) or the phased checklist in [`todo.md`](todo.md). Treat this as a **decision log**: each entry states context, the choice we made, and consequences.

Some entries consolidate a longer **implementation thread** (failed-entry logging, canonical `ufcstats_*` naming, same-day skip fix, doctor’s stoppage policy, single global throttle, gap-report overrides, and `refresh_data` scope).

---

## ADR-01: Request spacing for UFCStats (fights and profiles)

**Context.** Full runs issue thousands of HTTP requests. Too aggressive a rate risks harder blocking; too slow wastes wall time. A rejected alternative was **multiple** module-level sleep constants or a separate “rate limit” module that duplicated the same number in several places.

**Decision.** **One authoritative name** only: `REQUEST_DELAY_SEC` at the top of [`src/data/ufcstats_scraper.py`](../src/data/ufcstats_scraper.py), mutated at runtime from the fights CLI `--sleep` (or by assignment before calling into the scraper). The fighter profile scraper **imports** that value when its own `--sleep` is omitted so fights and profiles never drift. The current default in code is **0.2 seconds** (operators historically tried **0.1** and **0.02** for faster runs; **0.2** was chosen as a conservative steady state).

**Consequences.** Full scrapes are intentionally **slow but polite**. No parallel “profile-only” default that disagrees with fights unless you pass an explicit `--sleep` on the profile CLI. For back-of-envelope timing, see **ADR-14**.

---

## ADR-02: Skip legacy fight pages without a modern “totals” table

**Context.** The fights parser locates the main per-fight stats table by requiring a `<thead>` that contains **“Sig. str.”** (see `_find_totals_table` in `ufcstats_scraper.py`). Many **early UFC cards (often 1990s)** on UFCStats do not expose that table or use an incompatible layout, so parsing returns `None` and diagnostics report **`no_totals_table`**.

**Decision.** **Do not** add a dedicated legacy HTML path in the first iteration. Those fights are logged to **`failed_entries.csv`** and omitted from `ufcstats_fights.csv`. For modeling, this is acceptable: **full** UFCStats “totals” tables (Sig. str., etc.) are still sparse on many early cards, while **`Config.master_start_year`** sets which Tier-1 rows *attempt* to enter regression when the loader has usable stats. Ancient cards remain relevant for long-horizon ELO; validate feature coverage if you lower the calendar floor.

**Consequences.** Gap analysis must distinguish **“never scraped”** (use [`ufcstats_gap_report`](../src/data/ufcstats_gap_report.py)) from **“scraped but not parsed”** (use `failed_entries.csv`). If we later need result-only rows for ELO (winner/method, empty stats), that would be a **new** scraper mode and loader rules.

---

## ADR-03: Outcome and method normalization beyond the original label list

**Context.** UFCStats uses long or variant strings (e.g. doctor stoppage phrased as TKO, “Could Not Continue,” DQ wording).

**Decision.** Extend `_normalize_method` and the loader’s method map so that:

- **Draw — two catches (scraper):** (1) **Method line:** `_normalize_method` maps UFCStats method text that contains **`draw`** to **`draw`**. (2) **Banner override:** if `_person_rows` finds exactly two fighters and **both** status flags are **`D`**, **`method_norm` is forced to `draw`** even when the written method still looks decision-like on the page. That second path is the important “catch” — UFCStats can show inconsistent text vs. the **W/L/D** badges.
- **No contest — parallel pattern:** method text such as **Could Not Continue** / **No Contest** normalizes to **`no contest`**; **both** banners **`NC`** forces **`no contest`** the same way as double-**D** for draws. **`winner_id`** stays blank for draw / NC where the pipeline expects no winner.
- **TKO/KO**-prefixed labels (including **doctor’s stoppage**) map to **`ko/tko`** for both scraper and loader so finishes stay consistent with `ResultMethod.KO_TKO`.

**Product rationale (doctor’s stoppage).** Treat these as a **finish credited to the winner**: they materially changed the fight such that the bout was stopped; modeling and ELO use the same **KO/TKO** scale as other stoppages. That implies scraper normalization **and** loader `_parse_method` parity so hand-edited or older CSV rows with long UFCStats strings still load as `KO_TKO`.

**Consequences.** Parser and loader must stay in sync when new site labels appear; extend maps in one place and re-scrape if needed.

---

## ADR-04: Weight class edge cases (catch weight and unknown labels)

**Context.** Titles include **catch weight** wording and non-standard tournament strings (e.g. **Road to UFC** tournament titles) that do not match a fixed division enum.

**Decision.** In the scraper, normalize titles (collapse whitespace, strip common **UFC**/suffix noise such as **Title Bout** / **Tournament Bout** with **Interim Title Bout** ordered before **Title Bout** so interim strips correctly). Detect **catch weight** via substring on the lowercased title before falling back to longest-key substring matching against `WEIGHT_CLASS_MAP`. Map catch-weight bouts to the canonical key **`catch_weight`**. Non-mapped titles are stored as **lower** raw text for CSV fidelity. In the loader, known keys map to enums; unknown cells become **`WeightClass.UNKNOWN`** with **`weight_class_raw`** preserved on `FightRecord`.

**Consequences.** Features and training must treat **`UNKNOWN`** (and catch weight) explicitly where needed. Recurring tournament patterns should be added to mapping logic when they stabilize.

---

## ADR-05: Incomplete cards and same-day events

**Context.** The completed-events index can still list **today’s** card. Scraping those pages before results are final produced **false parse failures** (e.g. `unmapped_method:None`) that looked like a broken parser but were really **incomplete data**.

**Decision.** Skip events whose parsed date is **`>= date.today()`** (today **and** all future dates). Only **`ev_date < date.today()`** is scraped. Apply the same rule in both the main scraper loop and **`iter_expected_fights_from_completed_events`** so tooling stays consistent.

**Consequences.** A short lag around fight night is expected until the card is a **strictly past** date on the site. This wastes fewer requests and keeps `failed_entries.csv` cleaner.

---

## ADR-06: Where failures are recorded

**Context.** Operators need to know why a fight is missing from the CSV, including hundreds of “almost scraped” rows that never become valid pipeline records.

**Decision.**

- **`failed_entries.csv`** (default: next to the fights output, overridable with **`--failed-entries`**): append one row for **every** fight that does not become a CSV row—**bad fight URL**, **HTTP error** on the fight page, or **`parse_fight_page` returned `None`**. Parse failures use **`diagnose_fight_parse_failure`** for a stable **`failure_kind`** / detail string.
- **Live logs:** each failure prints immediately as **`[failed <kind>] fight_id=... | <detail>`** (ASCII-friendly punctuation; Unicode in log lines has caused Windows console mojibake in the past).
- **`ufcstats_gap_report`**: **diff** between site inventory and the CSV (missing `fight_id`s you never successfully ingested). Skipped scrapes do not appear in the fights file, so gap report and failed entries are **complementary**.

**Consequences.** Run **`--check-csv-only`** for structural sanity on rows you already have; use gap report + optional cached event inventory for coverage.

---

## ADR-07: Repository layout vs. local data

**Context.** Fight and profile CSVs and `model.pkl` are large and environment-specific.

**Decision.** **`/data/`** and **`*.pkl`** remain **gitignored**. Operators keep authoritative copies locally; optional archives outside `data/` are a **personal convention**, not part of version control.

**Consequences.** Reproducible collaboration assumes shared **code** and documented **refresh steps**, not committed raw exports.

---

## ADR-08: Pickle persistence of `ELOModel`

**Context.** `MMAPredictor.save` pickles the full pipeline including `ELOModel`. A `defaultdict(lambda: None)` for last-fight dates embeds an **unpicklable** local lambda on some Python versions/platforms.

**Decision.** Use a **module-level** default factory (e.g. `_defaultdict_none`) for that `defaultdict` so the factory is picklable by name.

**Consequences.** Any future `defaultdict` or nested lambdas on persisted objects must follow the same rule.

---

## ADR-09: Windows-friendly CLI output

**Context.** On Windows, the default console encoding (e.g. cp1252) can raise **`UnicodeEncodeError`** on common Unicode punctuation in `print`, and can show **mojibake** for characters like em dashes in scraper progress lines.

**Decision.** Use **ASCII** for high-traffic user-facing strings in [`main.py`](../main.py) (e.g. `->` instead of Unicode arrows in train progress). Apply the same discipline to **scraper** stderr/stdout messages that always run in operator terminals. Docstrings may still contain Unicode; rare paths (e.g. Cauchy CI footnotes) should be tested on cp1252 if they print by default.

**Consequences.** Prefer plain ASCII in CLI `print` paths that always run on first-time setups.

---

## ADR-10: Bootstrap count for prediction-time confidence intervals

**Context.** With bootstrap CIs, each `predict` call refits the multinomial model many times on the training matrix. A **1000**-draw default made interactive **`predict` / `explain`** impractically slow.

**Decision.** Lower the default **`ModelConfig.n_bootstrap`** to **200**, with the understanding that **tighter or more stable intervals** for research or production can raise this after a cost/benefit check.

**Consequences.** Saved **`model.pkl`** embeds the config in effect at **`train`** time; changing the default requires **retraining** (or manual config injection) for old pickles to pick up the new behavior.

---

## ADR-11: Canonical on-disk names (`ufcstats_*` + legacy fallback)

**Context.** Early code used **“tier1”** filenames and CLI flags that did not generalize to “UFCStats is the source” and confused future tiers (Bellator, Sherdog, etc.).

**Decision.** Primary fights output is **`ufcstats_fights.csv`** (constant `DEFAULT_UFCSTATS_FIGHTS_CSV`). Docs and tools use **`ufcstats_gap_report`**, **`ufcstats_event_inventory.csv`**, **`ufcstats_missing_fights.csv`**, etc. The pipeline’s [`load_data`](../src/pipeline.py) tries **`ufcstats_fights.csv` first**, then falls back to legacy **`tier1_ufcstats.csv`** so old trees keep working.

**Consequences.** User-facing instructions should prefer **`--data-dir`** / **`ufcstats_*`** paths; **`tier1_*`** is legacy compatibility only.

---

## ADR-12: Gap report must not clobber the scraper throttle global

**Context.** Inventory and gap tooling reuses the same HTTP patterns as the scraper. Mutating **`REQUEST_DELAY_SEC`** from a side tool risks surprising the operator’s next full scrape.

**Decision.** **`iter_expected_fights_from_completed_events`** accepts an optional **`request_delay_sec`** (or equivalent) passed into **`_throttle`**, so gap-report crawls can space requests **without** assigning the module global. The fights **`main()`** path continues to set **`REQUEST_DELAY_SEC`** from **`--sleep`** only.

**Consequences.** Long-running gap jobs and fight scrapes can use different sleeps in the same Python process if ever orchestrated together (still prefer separate processes for clarity).

---

## ADR-13: `refresh_data` is a thin orchestrator

**Context.** The scrapers expose rich CLIs: **`--max-events`**, **`--max-fights`**, **`--failed-entries`**, **`--sleep`**, profile **`--max-fighters`**, etc.

**Decision.** [`refresh_data`](../src/data/refresh.py) only calls **`scrape_ufcstats_fights_to_csv`** and **`scrape_fighter_profiles_to_csv`** with default paths—**no argument forwarding**. **`main.py train --full-rebuild`** therefore means “full default refresh,” not “cap/smoke refresh.” For capped or custom runs, invoke **`python -m src.data.ufcstats_scraper`** and **`python -m src.data.ufcstats_profiles`** directly (or extend `refresh_data` later with an explicit API).

**Consequences.** Operators should not expect **`--full-rebuild`** to honor ad-hoc scrape limits without code changes.

---

## ADR-14: Planning scrape duration (sleep vs network baselines)

**Context.** Operators want order-of-magnitude wall-clock estimates. A one-off **ICMP ping** to a public DNS (e.g. **~12 ms RTT** to `1.1.1.1`) is easy to run but **does not** measure UFCStats HTML time.

**Decision.** Use a **structural** lower bound when reasoning about throttle contribution: with **`E`** events kept and **`N`** fight pages fetched, **`_throttle()`** runs **`K = E + N`** times (index fetch has **no** leading throttle; **`R = 1 + E + N`** HTTP GETs). A toy bound is **`K · sleep`** plus an optional **~RTT × R** term if you want a numeric floor—**real** runs are dominated by **TLS + server + page size**, often on the order of **seconds per page**, not tens of milliseconds.

**Consequences.** Use ping only as a **generic connectivity** check, not to predict UFCStats latency. For ETA, anchor on observed **per-event** or **per-request** wall times from a short capped run at the chosen **`REQUEST_DELAY_SEC`**.

---

## ADR-15: Kalman layoff clock is per fighter (global), not per weight class

**Context.** ELO **means** are stored per `(fighter_id, weight_class)` because competition level differs by division. The original implementation also keyed **last fight date** only on that pair, so `kalman_predict` saw “days since last fight **in this class**.” A fighter could compete at lightweight, then take a welterweight bout months later, and on return to lightweight the model would still apply a long layoff variance bump to their lightweight state — even though they had just been active in the cage. That conflicts with the intent of the Kalman time update: we are less certain about parameters when we have **not recently observed the athlete**, not when they have not appeared in **this** division.

**Decision.** Maintain **`_last_fight_global[fighter_id]`** — updated on **every** processed fight (all divisions, including draws / NC / DQ where ELO does not move). The Kalman **predict** step before a bout uses **`fight_date - last_global`** for both fighters' states **in the bout's weight class only**. Per-division **`_last_fight[(fighter_id, wc)]`** remains for bookkeeping (e.g. `ELOState.last_fight_date` = last bout **in that class**). **`get_state(..., as_of_date=...)`** applies the same global clock for lookahead-free queries.

**Consequences.** Cross-division activity “refreshes” the time clock for Kalman uncertainty on the next bout in any division. Old **`model.pkl`** files without `_last_fight_global` are migrated on unpickle by taking, per fighter, the **maximum** of known per-division last dates (best effort). **Retrain** after upgrading if you need exact parity with a fresh run.

---

## ADR-16: Kalman gain amplifies post-layoff updates (fast-adjustment over name-retention)

**Context.** With our Kalman filter, variance grows during idle time (`P_pred = P_prev + process_noise × days_idle`) and the applied ELO delta on the next fight is `K × classical_delta` with `K = P_pred / (P_pred + R)`. This means **longer layoffs → larger `K` → a bigger fraction of the classical Elo step lands**. A coherent alternative is the opposite: treat long layoffs as a reason to **damp** the next update (rusty performance is noisy evidence, prior might still be right). Both read "we don't know what happened in the interim" reasonably.

We discussed the two directions explicitly, including which kinds of fighters each favors across a career (see [`elo-kalman-layoff-philosophy.md`](elo-kalman-layoff-philosophy.md)):

- **Damp on layoff** → rating sticky through gaps → model **favors names with history** (returning champs retain rating; rising stars credited slowly; faded legends degrade slowly).
- **Amplify on layoff (our choice)** → stored rating treated as stale → model **favors whoever is currently performing** (returning fighter's result moves rating aggressively in either direction).

**Decision.** Keep the standard Kalman geometry — **amplify** updates after long layoffs. A stale rating is a worse prior than a fresh in-cage result, and a handicapping model is more useful reacting to information than preserving legacy. Current knob values: **`kalman_process_noise = 0.01`**/day, **`kalman_measurement_noise = 1.0`**, so e.g. `K ≈ 0.66` after 3 months idle, `K ≈ 0.82` after 12 months (with `P_prev ≈ 1`, `R = 1`). The full worked example and the scenario table live in [`elo-kalman-layoff-philosophy.md`](elo-kalman-layoff-philosophy.md).

**Consequences.**

- Single rusty return fights can move a rating substantially. That is by design.
- Former champs who take multi-year gaps and then lose will drop out of the elite band quickly in the rating; we do not preserve "rings of honor."
- Rising stars who win a signature fight after a short break are credited immediately.
- Uncertainty grows but the **mean** does not decay toward 1500 during idle time — only `P` grows. Any desired "inactive fighters should drift lower in point ELO" behavior requires a separate change (explicit mean pull in `kalman_predict`, not just more process noise).
- If future calibration shows we are overreacting to return fights, documented flip paths exist: couple `R` to idle time, cap `K` past a threshold, or add a mean pull toward a pool prior. See `elo-kalman-layoff-philosophy.md` §7.

---

## ADR-17: Cauchy prediction intervals for weight-class debuts

**Context.** Bootstrap confidence intervals resample the **global** weighted training set to quantify uncertainty in the **fitted** multinomial coefficients. That captures **model** uncertainty given historical matchups, not a bespoke “this athlete has never fought in this division” epistemic story. For a corner with **no prior bout in the same weight class** in loaded data before the card date, the point estimate still uses cold-start priors and ELO, but bootstrap CIs can look **misleadingly tight** relative to true ignorance about how they perform in-division.

**Decision.** In [`MMAPredictor.predict`](../src/pipeline.py), if **either** fighter has **fewer than one** prior fight in the **same** `WeightClass` with `fight_date` **strictly before** the predicted bout’s date (counting all loaded `FightRecord` rows in that class, any tier), **skip bootstrap** and compute **Cauchy** intervals for **all six** outcome probabilities via [`compute_prediction_ci(..., force_cauchy_wc_debut=True)`](../src/confidence/intervals.py). The returned `ci_method` tag is **`cauchy_wc_debut`** (distinct from generic **`cauchy`** used for sparse ESS / missing bootstrap).

**Explicit Q&A — layoff vs CI width (router vs MC).** The **bootstrap / ESS / debut router** does **not** use idle days. **Layoff-driven widening** (when implemented) is **continuous** via **Cauchy ELO Monte Carlo**: per-corner **γ** grows with calendar idle — see **ADR-19** and `ModelConfig.elo_mc_gamma_*` / `elo_mc_gamma_for_days_idle` in [`src/config.py`](../src/config.py). Kalman layoff still affects **ELO mean path** (ADR-15/16), separate from **γ** sampling.

**Consequences.** Debut-in-division matchups get heavier-tailed, wider nominal intervals around the same point softmax. Fighters with **cross-division** history still count prior bouts **only in the queried weight class**. If the corpus is incomplete, a “debut” may be a data artifact — document data coverage when interpreting.

---

## ADR-19: Cauchy ELO Monte Carlo scales (**γ**)

**Context.** Bootstrap captures uncertainty in **coefficients** `W`. **Epistemic** uncertainty about whether **headline ELO** matches true strength in-division—especially after time off—is better probed by **simulation**: independent **Cauchy** shocks `ε_a`, `ε_b` in ELO points, `elo_draw = μ + ε`, rebuild features, `softmax(Wx)`, percentile intervals. **No Gaussian** draws on ELO; Cauchy absorbs tail events per product preference.

**Decision.** Hyperparameters live on **`ModelConfig`**: `elo_mc_n_draws`, `elo_mc_gamma_min`, `elo_mc_gamma_slope_sqrt_year`, `elo_mc_gamma_max`, with **`elo_mc_gamma_for_days_idle(days_idle)`** implementing  
`γ = min(γ_max, γ_min + slope * sqrt(max(0, days_idle)/365.25))`  
per corner from **global** days since last fight to predict date. **No** discrete layoff threshold routing—wider sampling is **only** from larger **γ**. Distinct from **training** recency row weights (ADR-18).

**Consequences.** Tune **γ** knobs on holdout coverage stratified by layoff. With stored bootstrap **`W`**, `predict` runs **`elo_mc_n_draws`** Cauchy ELO shocks per sample, cycling **`W_b`** rows so coefficient and ELO uncertainty both appear. Set **`elo_mc_n_draws`** to **0** to disable ELO MC and keep bootstrap-only CIs.

---

## ADR-18: Recency leaning (non-stationarity) across training, style axes, and ELO

**Context.** MMA is **non-stationary**; older cards are not exchangeable with modern ones. Several mechanisms **lean on recent evidence**; they are **related in intent** but **not the same mathematics**.

**Decision (documented layering).**

1. **Regression training sample weights** — In [`train_regression`](../src/pipeline.py), each Tier-1 row gets weight `1 / (1 + days_old/365)` relative to **train run date**, so the multinomial fit emphasizes **recent** historical outcomes when estimating **global** coefficients.
2. **Style-axis recency** — [`compute_style_axes`](../src/features/construction.py) applies `FeatureConfig.recency_decay_rate` so **within** a fighter’s history (as of fight date), **recent** bouts contribute more to striker/grappler/finish scores.
3. **Kalman process noise** — Grows posterior variance during **idle calendar time** so the **next** ELO update can move the **mean** more after layoffs (ADR-16). This shapes **ratings**. **Prediction** interval width from layoff uses **Cauchy ELO MC** **γ** (ADR-19), not the Kalman router alone.

**Consequences.** Tuning “how much we trust the past” can move **all three** layers; changes should be justified against holdout metrics (`docs/todo.md` §3.3). Readers should not confuse **training down-weighting of old rows** with **Kalman variance** or with **probability-level Cauchy** / **ELO MC γ** (**ADR-19**) — each addresses a different part of the stack.

---

## ADR-20: Phase 3 walk-forward + per-year random search + frozen winner on pristine

**Context.** Iterative OAT in [`docs/todo.md`](todo.md) §3.4 is too slow to explore a **high-dimensional** joint `Config` space. We needed: (1) an **outer** year-by-year walk-forward over a **selection** block, (2) **inner/forward** log-loss to rank trials, (3) a **pristine** calendar strip (e.g. 2023–2025) that uses **no** in-year tuning — only a **configuration frozen** from the end of the selection block (e.g. **2022** search winner) so “true” OOS is not used to pick hyperparameters.

**Decision.** Implement the harness in [`src/eval/tuning_harness.py`](../src/eval/tuning_harness.py), invoked via **`python -m src.cli.run_phase3_tuning`** ([`src/cli/run_phase3_tuning.py`](../src/cli/run_phase3_tuning.py)) with optional **`--selection-search`** (`--n-trials` per outer year, default **50**; warm-start chain; inner window **`--inner-last-k`** or full inner). Write **`data/phase3_eval/phase3_report.json`**, metrics CSV, plots, and optional **`elo_walkforward_cache.pkl`**. **Serialization choice:** the report stores the **2022 (last selection year) winner** as **`frozen_winner_config`**, **`trial_rows`** (log-loss by trial id only, not full `Config` per trial), and **`selection_campaign`** per-year metadata — **not** a full record of every sampled hyperparameter vector across years.

**Consequences.** A **full** 50-trial/yr run is a **long** wall-time commitment (ELO + repeated multinomial fits). For **A/B** without repeating that cost: run **baseline** walk-forward (no search), or **reduced** `--n-trials` / **narrower** selection years, and compare ranking or pristine deltas to the saved `phase3_metrics.csv` / JSON. **Production ship:** rehydrate **`frozen_winner_config`** (or the nested dict from JSON) into `Config` and run **`train`** with the **intended** deploy holdout / snapshot policy; do not treat `holdout_start_date` inside a frozen copy as binding without re-reading `docs/todo.md` §3.1. **Economic** evaluation (ROI vs book odds) is **out of scope** of this ADR; needs historical lines data.

---

## ADR-21: Abstention framing — EV-based, not confidence-based

**Context.** Once we have historical betting line data, the question arises: should the model abstain from certain predictions — and if so, what triggers abstention? Two intuitive but wrong framings present themselves:

1. **Argmax-probability threshold**: abstain when no single class exceeds some confidence cutoff (e.g. max p < 0.40). This is classification-thinking applied to a probability output.
2. **CI overlap**: abstain when win/loss CIs overlap too much to call a direction. In a 6-class model with bootstrap and Cauchy ELO MC, CIs on *any* single outcome class will always overlap with adjacent classes at realistic sample sizes — this criterion would abstain on nearly every fight.

Both framings optimize for *easy fights*. If we tune an abstention threshold against classification accuracy on the subset of fights the model *chooses* to predict, we learn to cherry-pick mismatches where the model is already extremely lopsided. Reported metrics on that subset will look better than the full-card reality, and the filter becomes a metric-inflation mechanism rather than a genuine decision tool.

**Decision.** Abstention is not a model layer — it is a **downstream financial decision** that the model itself should not own. The model always outputs a full 6-class probability distribution with confidence intervals. Abstention lives in a separate **stake filter** that asks:

> *Given the model's outcome distribution P and the available market line, is there a bet with positive expected value after accounting for margin?*

Formally: abstain (do not stake) unless `max_k [ P(k) × decimal_odds(k) ] > 1 + min_edge`, where `min_edge` is a tunable profitability threshold (e.g. 0.03–0.05 above breakeven). This makes abstention a function of **P × line**, not of P alone. A fight the model finds uncertain may still be bettable if the market is even more uncertain (long odds). A fight the model finds one-sided may not be bettable if the market has already priced it correctly.

**Consequences.**

- Abstention cannot be evaluated or tuned until a reproducible source of historical betting lines (opening, closing, or pre-bell) is available at the fight level. See `TODO.md` §P&L and ADR-20 "Deferred" for scope notes.
- The stake filter is **not** trained jointly with the regression model. It is applied post-hoc to model outputs. Training them jointly would reintroduce the cherry-picking bias.
- **Do not add a confidence threshold to `predict` or `score_tier1_fight_slice`.** Those are model-evaluation surfaces that must score every fight to be honest. A fight the model is uncertain about is still a real fight; excluding it from metrics is dishonest.
- When lines data exists, evaluate abstention on **ROI over all fights** (not accuracy on chosen fights): if the filter skips 40% of cards and the retained set shows positive P&L over a large sample, that is meaningful. If the retained set merely shows higher classification accuracy, it is not.
- The `min_edge` threshold is itself a tuning parameter that should be selected against a holdout period of lines + outcomes, not the same period used to calibrate it.
- **Weight-class and event-type stratification** matters: easy fights tend to cluster in prelims and mismatched debuts. Any abstention analysis should report coverage (fraction of fights staked) alongside ROI so the filter's selectivity is visible.

---

## ADR-22: Split-barrier figure uses whole-number percents only

**Context.** PNGs emitted by **`python -m src.cli.plot_prediction_three_viz`** (stacked bar + total-win badge + marginal-CI strips) are for quick reading; fractional percents on the badge (e.g. “96.5%”) cluttered the focal **total win** line.

**Decision.** Percent **numerals** in that figure are **integers 0–100** (nearest whole percent via `round` on fractional masses; probabilities clamped into `[0, 1]` where appropriate). Whiskers `[lo, hi]` use integer endpoints. This applies to **exported figure copy only**: terminal **`predict`** / **`main.py`** may still display more decimal places where useful.

---

## Deferred (explicitly not decided here)

- **Tier 2/3** promotion ingestion and Sherdog crosswalks.
- **Manual pedigree** fill vs. leaving zeros for cold starts.
- **Legacy result-only** UFCStats rows without sig-strike tables.
- **Production holdout** policy vs tuning scripts (per-run choice; see `todo.md` §3.1, ADR-20) — *Phase 3 walk-forward design* is no longer an open “whether” (ADR-20); **stake/ROI** vs closing lines is still deferred.

---

## See also

| Document | Role |
|----------|------|
| [`architecture.md`](architecture.md) | End-to-end modeling and pipeline design |
| [`todo.md`](todo.md) | Phases, column specs, validation checklist |
| [`../TODO.md`](../TODO.md) | Roadmap, next work bout, gap-report commands |
| [`hyperparameter-tuning.md`](hyperparameter-tuning.md) | Walk-forward search, pristine, case studies (§9) |
| [`validation-and-few-shot.md`](validation-and-few-shot.md) | Time holdout, grouped CV, few-shot / cold-start knobs |
| [`elo-tuning-knobs.md`](elo-tuning-knobs.md) | What each ELO / Kalman parameter does when you change it |
| [`elo-kalman-layoff-philosophy.md`](elo-kalman-layoff-philosophy.md) | Why we amplify (not damp) ELO updates after a layoff — ADR-16 framing |
