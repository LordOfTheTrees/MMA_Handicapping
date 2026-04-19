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

**Decision.** **Do not** add a dedicated legacy HTML path in the first iteration. Those fights are logged to **`failed_entries.csv`** and omitted from `ufcstats_fights.csv`. For modeling, this is acceptable: the regression era begins around **2013**, where full stats exist; ancient cards are mainly relevant for long-horizon ELO context, and missing rows are a small tail.

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

**Decision.** Keep the standard Kalman geometry — **amplify** updates after long layoffs. A stale rating is a worse prior than a fresh in-cage result, and a handicapping model is more useful reacting to information than preserving legacy. Current knob values: **`kalman_process_noise = 0.0025`**/day, **`kalman_measurement_noise = 1.0`**, so e.g. `K ≈ 0.55` after 3 months idle, `K ≈ 0.66` after 12 months (roughly 45% damping vs 34% damping of the classical step with `P_prev ≈ 1`, `R = 1`). The full worked example and the scenario table live in [`elo-kalman-layoff-philosophy.md`](elo-kalman-layoff-philosophy.md).

**Consequences.**

- Single rusty return fights can move a rating substantially. That is by design.
- Former champs who take multi-year gaps and then lose will drop out of the elite band quickly in the rating; we do not preserve "rings of honor."
- Rising stars who win a signature fight after a short break are credited immediately.
- Uncertainty grows but the **mean** does not decay toward 1500 during idle time — only `P` grows. Any desired "inactive fighters should drift lower in point ELO" behavior requires a separate change (explicit mean pull in `kalman_predict`, not just more process noise).
- If future calibration shows we are overreacting to return fights, documented flip paths exist: couple `R` to idle time, cap `K` past a threshold, or add a mean pull toward a pool prior. See `elo-kalman-layoff-philosophy.md` §7.

---

## Deferred (explicitly not decided here)

- **Tier 2/3** promotion ingestion and Sherdog crosswalks.
- **Manual pedigree** fill vs. leaving zeros for cold starts.
- **Legacy result-only** UFCStats rows without sig-strike tables.
- **Holdout design** and calibration tuning (Phase 3+ per `todo.md`).

---

## See also

| Document | Role |
|----------|------|
| [`architecture.md`](architecture.md) | End-to-end modeling and pipeline design |
| [`todo.md`](todo.md) | Phases, column specs, validation checklist |
| [`../TODO.md`](../TODO.md) | Current status, next work bout, gap-report commands |
| [`validation-and-few-shot.md`](validation-and-few-shot.md) | Time holdout, grouped CV, few-shot / cold-start knobs |
| [`elo-tuning-knobs.md`](elo-tuning-knobs.md) | What each ELO / Kalman parameter does when you change it |
| [`elo-kalman-layoff-philosophy.md`](elo-kalman-layoff-philosophy.md) | Why we amplify (not damp) ELO updates after a layoff — ADR-16 framing |
