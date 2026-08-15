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

**Decision.** [`refresh_data`](../src/data/refresh.py) calls **`scrape_ufcstats_fights_to_csv`**, **`scrape_fighter_profiles_to_csv`**, and **[`scrape_upcoming_cards_to_path`](../src/data/ufcstats_upcoming.py)** ( **`upcoming_cards.json`** — see **ADR-23**) — all with **fixed default paths**, **no argument forwarding**. **`main.py train --full-rebuild`** therefore means “full default refresh,” not “cap/smoke refresh.” For capped or custom runs, invoke **`python -m src.data.ufcstats_scraper`**, **`python -m src.data.ufcstats_profiles`**, or **`python -m src.data.ufcstats_upcoming`** directly (or extend `refresh_data` later with an explicit API).

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

## ADR-23: Upcoming card JSON is site-only — out of scope for training

**Context.** **ADR-05** keeps **today and future** completed-index events **out** of the historical fights CSV so half-finished cards do not corrupt training rows. The product website still needs **scheduled** bout listings.

**Decision.** Maintain a **separate** ingestion path **[`src/data/ufcstats_upcoming.py`](../src/data/ufcstats_upcoming.py)** that reads **http://ufcstats.com/statistics/events/upcoming** and writes **`data/upcoming_cards.json`** (under the same **`/data/`** gitignore). Parsing uses **event-page bout rows only** — **not** **`parse_fight_page`**, which requires finished-fight totals. **`MMAPredictor.load_data`** and **`train_regression`** must **never** read this file; training continues to consume **Tier-1 CSVs only** (`ufcstats_fights.csv`, etc.). Deploy artifacts use **`scripts/export_upcoming_events.py`**.

**Consequences.** One **`--full-rebuild`** refresh updates **both** historical CSVs **and** upcoming listings, but **only** CSV-backed fights feed ELO/regression. Upcoming scrape breakage does not imply training breakage, and vice versa.

---

## ADR-24: Portable deploy JSON + harness (OctagonELO / sibling site)

**Context.** The **`mma.ai`** SPA must run inference **without** shipping **`model.pkl`**, **`data/`** CSV blobs, or `src/` in the consumer image (**ADR-07** stays: local training data gitignored). A frozen Phase 3 configuration (**ADR-20**) still produces one binary **`MMAPredictor`** pickle locally; operators need a **repeatable path** from that pickle to **versioned**, **diffable** JSON the web tier can load.

**Decision.**

1. **Four inference artifacts** — [`scripts/export_artifacts.py`](../scripts/export_artifacts.py) emits **`model_weights.json`**, **`elo_states.json`**, **`style_axes.json`**, **`fighter_profiles.json`** from a trained pickle (`--as-of-date` pins the ELO/style snapshot). Schema version string **`mma-handicapping-export-v1`** on each file.
2. **Upcoming listings** — [`scripts/export_upcoming_events.py`](../scripts/export_upcoming_events.py) maps **`data/upcoming_cards.json`** (**ADR-23**) → **`upcoming_events.json`** for the calendar/card UI (**ADR-23** ingestion remains training-free). This export is now gated on a fresh-scrape signal rather than best-effort — see **ADR-25**.
3. **Snapshot inference in this repo** — [`src/export/json_inference.py`](../src/export/json_inference.py) **`predict_proba_snapshot`**: reconstructs **point** `(6,)` probabilities from JSON **only**, with **`fight_date == as_of_date`** (JSON is a temporal **slice**, not full ELO history).
4. **Harness** — [`tests/test_artifact_parity.py`](../tests/test_artifact_parity.py): strict **equality** pickle **`predict_proba_point_only`** vs **`predict_proba_snapshot`** after **`export_all`** to a temp dir (requires **`data/model.pkl`** or **`MMA_HARNESS_MODEL`**). [`tests/test_site_export_pages.py`](../tests/test_site_export_pages.py): **`JSON_exports/`** structural checks mapped to **`docs/website_elements.md`** page intents. Entrypoint **`python scripts/run_harness.py`** (`quick` / **`site`** / **`integration`** / full discover). Integration docs [`docs/BACKEND_PIPELINE_INTEGRATION.md`](BACKEND_PIPELINE_INTEGRATION.md).

**Consequences.** Production **truth** for point probabilities at the artifact snapshot date stays **defined by this repo** (`json_inference` + parity tests); **`mma.ai`** should match that math or deliberately document drift. Full **`PredictionResult`** (Cauchy, bootstrap intervals, hypothetical idle) stays pickle/API-side until replicated in deploy. Re-run **export** after every material **train**; run **`integration`** (+ **`site`**) before relying on **`JSON_exports/`** in git or copied to **`mma.ai/artifacts/`**.

---

## ADR-25: Upcoming-events export is gated on a fresh scrape signal, not best-effort

**Context.** **ADR-23**/**ADR-24** established `data/upcoming_cards.json` (UFCStats) →
`scripts/export_upcoming_events.py` → `upcoming_events.json` as the site's future-card
pipeline, but `export_upcoming_events.py` was never actually invoked by
`scripts/weekly_update.py` or by the `weekly-model-refresh.yml` / `monthly-model-retrain.yml`
workflows — it was a standalone, manual-only script from the day it was introduced. Separately,
[`refresh_data`](../src/data/refresh.py) treated a UFCStats Cloudflare block (`probe.blocked`)
and any scrape exception as non-fatal, printing a message and returning a "successful"
`RefreshResult` with no field indicating whether `upcoming_cards.json` was actually refreshed
this run. Combined, these two gaps meant `upcoming_events.json` was never produced by CI at all,
and even a local/manual export could silently re-ship a stale `upcoming_cards.json` (carried
over via `ci_restore_data_bundle`) as if it were fresh — with zero errors or warnings anywhere
in the pipeline. UFCStats has in fact Cloudflare-blocked every scheduled CI run for at least the
five most recent weekly runs checked, so this was not a rare edge case.

**Decision.** `RefreshResult` (`src/data/refresh.py`) gains `upcoming_cards_scraped: bool`,
`True` only when `scrape_upcoming_cards_to_path` completes without the probe reporting
`blocked` and without raising. `scripts/weekly_update.py` (`cmd_refresh`/`cmd_retrain`) exports
`upcoming_events.json` — via the same `build_upcoming_events_doc` `export_upcoming_events.py`
uses — immediately after the five inference JSONs, but **only** when this run's own
`refresh_data()` call reported `upcoming_cards_scraped=True`; with `--no-scrape` (CI's
split-step flow) it always skips, since no scrape happened in-process. `scripts/ci_try_refresh_data.py`
writes the same signal as a GitHub Actions step output (`GITHUB_OUTPUT`), and both workflows add
an `export_upcoming_events.py` step gated on it. A blocked or failed scrape now means *no*
`upcoming_events.json` is (re-)produced that run — never a stale one silently passed off as
current. (**ADR-26** extends this same gating mechanism to a second, independent source.)

**Consequences.** `JSON_exports/upcoming_events.json` only exists in a run's bundle when UFCStats
was actually reachable that run, so `sync-json-to-mma-ai.yml` only ships genuinely fresh data (a
run with no fresh scrape simply doesn't touch the file already present downstream, rather than
overwriting it with recycled data under a new timestamp). Operators/CI can distinguish "UFCStats
blocked this run" from "upcoming cards genuinely unchanged" by checking for the presence of the
step's output / the exported file, instead of only reading print statements in logs. Local runs
without `--no-scrape` get this for free; the CI split-step flow needed the parallel
`ci_try_refresh_data.py` → workflow step-output wiring since the scrape and the export happen in
different process invocations there. See [`docs/ufc-com-upcoming-scrape-plan.md`](ufc-com-upcoming-scrape-plan.md)
for the follow-on exploration (ufc.com / extended ESPN ingest) prompted by UFCStats' persistent block.

---

## ADR-26: ESPN `fightcenter` as the preferred upcoming-cards source

**Context.** §0 of [`docs/ufc-com-upcoming-scrape-plan.md`](ufc-com-upcoming-scrape-plan.md)
traced `refresh_espn_fights_incremental`'s existing, already-reliable-in-CI ingest path
(`fetch_fightcenter(event_id)` → `_iter_competitions` → `_competition_is_final` filter in
[`src/data/espn_ingest.py`](../src/data/espn_ingest.py)) and found that the `fightcenter`
response already contains non-final (announced, unplayed) competitions — that is exactly what
`_competition_is_final` filters out for the training path — with each competitor's ESPN athlete
ID and display name embedded directly, no extra network calls needed. A user's live observation
(ESPN's public Fight Center listing an upcoming McGregor/Holloway 2 card, weight class included)
confirmed this at the product level, matching the code-level trace. Given UFCStats' upcoming-cards
scrape has Cloudflare-blocked every checked CI run for over a month (ADR-25's context) while ESPN
has a 5/5 success rate over the same runs, this reprioritizes ESPN above both fixing UFCStats and
the not-yet-built ufc.com scraper.

**Decision.**

1. New module [`src/data/espn_upcoming.py`](../src/data/espn_upcoming.py), output-schema-compatible
   with `ufcstats_upcoming.py`'s `upcoming_cards.json` (so `build_upcoming_events_doc` needs no
   changes). `espn_ingest.py` gained a shared `_resolve_season_events` helper — the season/event
   scan loop factored out of `_collect_incremental_events` — so the training-safe completed-events
   filter (`event_date < today`, unchanged, ADR-05) and `espn_upcoming._collect_future_events`'s
   inverted filter (`event_date >= today`) apply to the **same** resolved event list instead of
   each re-implementing the fetch/parse loop; `_collect_incremental_events` itself is not touched
   beyond calling the extracted helper. `_parse_future_bouts_from_fightcenter` reuses
   `_iter_competitions`/`_competition_is_final` verbatim, keeping non-final competitions and
   reading fighter names/ESPN IDs straight off each competitor's embedded `athlete` object.
2. **Read-only fighter-ID resolution.** `_resolve_known_fighter_id` only checks the existing
   crosswalk (`CrosswalkStore.athlete_to_fighter`) and existing profile name index — it never
   provisions a new hex ID or writes to the crosswalk store, unlike the completed-fight path's
   `resolve_fighter_id`. An announced bout can be cancelled or replaced before fight night, and
   there is no confirmed result yet to justify minting permanent training-facing state; unresolved
   (e.g. debuting) fighters ship name-only (`fighter_id=None`) until they actually fight.
3. **Separate file, independent attempt.** `refresh_data()` (`src/data/refresh.py`) attempts
   `scrape_espn_upcoming_cards_to_path` unconditionally — regardless of UFCStats' block status —
   writing to `data/espn_upcoming_cards.json`, never `data/upcoming_cards.json`, so one source's
   failure can never clobber the other's last-known-good data. `RefreshResult` gains
   `espn_upcoming_cards_scraped: bool` alongside the existing `upcoming_cards_scraped`.
4. **Consolidation at the export step, not a merge.** `weekly_update.py`'s
   `_maybe_export_upcoming_events` and both CI workflows' `export_upcoming_events.py` step now
   pick **whichever single source is fresh this run, ESPN preferred** — not a per-event/per-bout
   merge of both files. `ci_try_refresh_data.py` exposes this as two step outputs
   (`espn_upcoming_scraped`, `ufcstats_upcoming_scraped`, superseding ADR-25's single
   `upcoming_scraped`); both `ci_restore_data_bundle.py` and the workflows' upload/restore file
   lists include `data/espn_upcoming_cards.json`.
5. **No duplicate ESPN requests within a run.** `refresh_data()` now constructs one `ESPNClient`
   and passes it explicitly to `refresh_espn_fights_incremental`, `refresh_espn_profiles_incremental`,
   `run_espn_ingest_audit`, and `scrape_espn_upcoming_cards_to_path` — previously each created its
   own instance. `ESPNClient.get_json` also gained an in-memory memo (`_mem_cache`, checked before
   the existing on-disk cache), so within one process the fights-incremental pass and the
   upcoming-cards pass — which scan overlapping season/event-index URLs — never re-fetch over the
   network (disk cache already prevented that) **and** never redundantly re-read/re-parse the same
   cache file either. `tests/test_refresh_data_wiring.py` asserts the same client instance reaches
   all three call sites.

**Consequences.** The site's future-card data now has a second, independent, currently more
reliable path that requires no new scraping infrastructure or bot-wall risk — it reuses API calls
and helper functions the training ingest already exercises daily. Fighter IDs from this source are
a strict subset of what the crosswalk already knows (never fabricated), so downstream consumers
that already resolve `fighter_id` against `fighter_profiles.csv` degrade gracefully to name-only
display for unresolved fighters rather than erroring. Weight-class availability pre-fight and
`bout_order` display ordering (`_iter_competitions`'s natural card-dict order) are unverified
against live data — flagged in `docs/ufc-com-upcoming-scrape-plan.md` and `test_espn_upcoming_parse.py`
docstrings — and should be confirmed on a real run before treating either as guaranteed-correct.
UFCStats' scrape path (ADR-23/ADR-25) is unchanged and still attempted every run; it becomes the
fallback rather than the primary, and its richer data (hex IDs already resolved, canonical
`event_url`/`location`) is still preferred implicitly whenever ESPN's attempt fails but UFCStats'
doesn't.

---

## ADR-27: Betting evaluation is expected Kelly log growth over the existing holdout

**Context.** ADR-21 settled *when to stake* (EV-based stake filter, not confidence thresholds) but
left open *what number reports whether the strategy works*. Reported metrics to date score against
uninformed baselines — uniform six-way (log 6) and a fair coin — which answer "is the model better
than nothing," not "does this make money." Once fight-level line data exists, the second question
is the only one that matters, and it needs a metric fixed in advance so the analysis is not
designed around whatever the first pass happens to show.

**Decision.** The reporting metric is **expected Kelly log bankroll growth**, evaluated over the
**existing holdout period** — the same fights already scored in
[`model-efficacy-vs-baselines.md`](model-efficacy-vs-baselines.md) §4. This is an evaluation
overlay on a cohort we already have, not a new experimental construct.

The metric connects directly to the objective already in use. For a bet at de-vigged implied
probability *q* when the model's probability is *p*, optimal Kelly staking has expected log growth

> *G\** = *p*·ln(*p*/*q*) + (1−*p*)·ln((1−*p*)/(1−*q*)) = KL(*p* ‖ *q*)

which is exactly **market log-loss minus model log-loss** in expectation. Expected growth per bet
*is* the model's log-loss edge over the market. Log-loss therefore stays the training and selection
objective (ADR-20); what changes is the **reference distribution**, from uniform to market-implied.
Nothing about the model or its fitting procedure needs to change to adopt this metric.

**Reference lines are opening lines and method-market implied probabilities.** The strategy has not
at any point assumed an edge against closing-line moneyline W/L; closing lines are the most
efficient prices in the market and beating them was never the premise. Evaluation against the
closing line may still be reported as a secondary diagnostic, but it is not the success criterion.

**Consequences.**

- **The de-vig method is a modeling choice and must be stated with any result.** Implied
  probabilities sum to more than one; proportional normalization, Shin, and power/logarithmic
  methods distribute the overround differently, and the measured edge moves with the choice. Two-
  and three-way markets carry different overround, so a single method should be applied
  consistently and named in the output.
- **Fractional Kelly, not full.** Full Kelly assumes *p* is known exactly. Model probabilities carry
  estimation error, which makes full Kelly systematically overbet. The bootstrap coefficient stack
  (`_bootstrap_W`, `n_bootstrap=200`) is the existing machinery for quantifying that uncertainty and
  is the natural input to a fraction; note that scoring paths use `predict_proba_point_only` and do
  not consume it, so this is the first analysis that gives those refits a purpose.
- **Report bet count and coverage alongside growth**, per ADR-21. Growth over a staked subset is
  uninterpretable without knowing how many wagers produced it; a positive rate on a small number of
  bets does not separate skill from variance.
- **Metric is fixed before the data arrives**, deliberately. Selecting a betting metric after seeing
  which one flatters the model is the same cherry-picking failure ADR-21 rejects for abstention.
- **Current measured state, for reference when lines arrive.** Post-leak-fix on the pristine
  2023–2025 cohort, six-way log-loss decomposes exactly into a W/L term plus a method-given-side
  term: bespoke **0.6528 + 1.0087 = 1.6615**; the ELO-only baseline with static method priors
  **0.6899 + 1.0191 = 1.7090**; uniform-over-three-methods is ln 3 = **1.0986**. So of the bespoke
  model's 0.048-nat advantage over ELO-only, roughly **0.037 is W/L** and **0.010 is method beyond
  static base rates**. Recorded as a baseline observation, not as a claim about where a market edge
  will or will not be found.
- **Open dependency: fight-level historical prices.** Opening moneylines and method-market prices
  for the holdout period are not in the repo and their historical coverage has not been surveyed.
  Establishing what is actually obtainable — which markets, which fights, how far back — is the
  first task, and no assumption about coverage should be carried into the analysis before that
  survey exists.

---

## Deferred (explicitly not decided here)

- **Tier 2/3** promotion ingestion and Sherdog crosswalks.
- **Manual pedigree** fill vs. leaving zeros for cold starts.
- **Legacy result-only** UFCStats rows without sig-strike tables.
- **Production holdout** policy vs tuning scripts (per-run choice; see `todo.md` §3.1, ADR-20) — *Phase 3 walk-forward design* is no longer an open “whether” (ADR-20). The **betting evaluation metric** is no longer deferred either (ADR-27: expected Kelly log growth over the existing holdout, referenced to opening lines and method-market implied probabilities); what remains deferred is **sourcing the price data** and any claim about market coverage.

---

## See also

| Document | Role |
|----------|------|
| [`architecture.md`](architecture.md) | End-to-end modeling and pipeline design |
| [`todo.md`](todo.md) | Phases, column specs, validation checklist |
| [`../TODO.md`](../TODO.md) | Roadmap, next work bout, gap-report commands |
| [`BACKEND_PIPELINE_INTEGRATION.md`](BACKEND_PIPELINE_INTEGRATION.md) | **`mma.ai`** artifact flow + harness commands |
| [`hyperparameter-tuning.md`](hyperparameter-tuning.md) | Walk-forward search, pristine, case studies (§9) |
| [`validation-and-few-shot.md`](validation-and-few-shot.md) | Time holdout, grouped CV, few-shot / cold-start knobs |
| [`elo-tuning-knobs.md`](elo-tuning-knobs.md) | What each ELO / Kalman parameter does when you change it |
| [`elo-kalman-layoff-philosophy.md`](elo-kalman-layoff-philosophy.md) | Why we amplify (not damp) ELO updates after a layoff — ADR-16 framing |
