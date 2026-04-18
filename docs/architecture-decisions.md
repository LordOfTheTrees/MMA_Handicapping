# Architecture decision record

This document captures **implementation and operations decisions** that emerged while building the UFCStats path and running the pipeline. It does not repeat the modeling design in [`architecture.md`](architecture.md) (stages, ELO math, interpretability goals) or the phased checklist in [`todo.md`](todo.md). Treat this as a **decision log**: each entry states context, the choice we made, and consequences.

---

## ADR-01: Request spacing for UFCStats (fights and profiles)

**Context.** Full runs issue thousands of HTTP requests. Too aggressive a rate risks harder blocking; too slow wastes wall time.

**Decision.** A single module-level delay, `REQUEST_DELAY_SEC` in [`src/data/ufcstats_scraper.py`](../src/data/ufcstats_scraper.py), defaults to **0.2 seconds** between requests. The fighter profile scraper imports the same constant when `--sleep` is omitted, so fights and profiles stay aligned without duplicating magic numbers.

**Consequences.** Full scrapes are intentionally **slow but polite**. Operators can still lower `--sleep` for local experiments. Rough planning: wall time is dominated by server and page work, not by ICMP latency to arbitrary hosts; the delay is a small additive term but helps avoid burst patterns.

---

## ADR-02: Skip legacy fight pages without a modern “totals” table

**Context.** The fights parser locates the main per-fight stats table by requiring a `<thead>` that contains **“Sig. str.”** (see `_find_totals_table` in `ufcstats_scraper.py`). Many **early UFC cards (often 1990s)** on UFCStats do not expose that table or use an incompatible layout, so parsing returns `None` and diagnostics report **`no_totals_table`**.

**Decision.** **Do not** add a dedicated legacy HTML path in the first iteration. Those fights are logged to **`failed_entries.csv`** and omitted from `ufcstats_fights.csv`. For modeling, this is acceptable: the regression era begins around **2013**, where full stats exist; ancient cards are mainly relevant for long-horizon ELO context, and missing rows are a small tail.

**Consequences.** Gap analysis must distinguish **“never scraped”** (use [`ufcstats_gap_report`](../src/data/ufcstats_gap_report.py)) from **“scraped but not parsed”** (use `failed_entries.csv`). If we later need result-only rows for ELO (winner/method, empty stats), that would be a **new** scraper mode and loader rules.

---

## ADR-03: Outcome and method normalization beyond the original label list

**Context.** UFCStats uses long or variant strings (e.g. doctor stoppage phrased as TKO, “Could Not Continue,” DQ wording).

**Decision.** Extend `_normalize_method` and the loader’s method map so that:

- Banner-driven **draw** (double **D**) and **no contest** (double **NC** or matching text) set `winner_id` blank where appropriate.
- **TKO/KO**-prefixed labels (including doctor stoppage) map to **`ko/tko`** for both scraper and loader so finishes stay consistent with `ResultMethod.KO_TKO`.

**Consequences.** Parser and loader must stay in sync when new site labels appear; extend maps in one place and re-scrape if needed.

---

## ADR-04: Weight class edge cases (catch weight and unknown labels)

**Context.** Titles include **catch weight** wording and non-standard tournament strings that do not match a fixed division enum.

**Decision.** Map explicit catch-weight titles to a dedicated canonical value (**`catch_weight`**) in the scraper and loader. Non-mapped titles fall through to **raw text** in the CSV; the schema exposes **`WeightClass.UNKNOWN`** (and related fields such as `weight_class_raw` on `FightRecord`) so ingestion does not drop rows silently.

**Consequences.** Features and training must treat **`UNKNOWN`** explicitly where needed. Extending `WEIGHT_CLASS_MAP` remains the preferred fix for recurring labels.

---

## ADR-05: Incomplete cards and same-day events

**Context.** Event pages can list bouts before results exist everywhere on the site.

**Decision.** Skip events whose parsed date is **today or in the future** when scraping completed listings, so incomplete cards are not partially written.

**Consequences.** A short lag around “fight night” is expected until UFCStats marks the card completed and dated in the past.

---

## ADR-06: Where failures are recorded

**Context.** Operators need to know why a fight is missing from the CSV.

**Decision.**

- **`failed_entries.csv`** (beside the fights output): per-fight **HTTP errors** and **parse failures** during `scrape_ufcstats_fights_to_csv`.
- **`ufcstats_gap_report`**: **diff** between site inventory and the CSV (missing `fight_id`s you never successfully ingested). Skipped scrapes do not appear in the fights file, so gap report and failed entries are **complementary**.

**Consequences.** Run **`--check-csv-only`** for structural sanity on rows you already have; use gap report + optional cached event inventory for coverage.

---

## ADR-07: Repository layout vs. local data

**Context.** Fight and profile CSVs and `model.pkl` are large and environment-specific.

**Decision.** **`/data/`** and **`*.pkl`** remain **gitignored**. Operators keep authoritative copies locally; optional snapshots (e.g. under `data/Saved_Runs/`) are a **personal convention**, not part of version control.

**Consequences.** Reproducible collaboration assumes shared **code** and documented **refresh steps**, not committed raw exports.

---

## ADR-08: Pickle persistence of `ELOModel`

**Context.** `MMAPredictor.save` pickles the full pipeline including `ELOModel`. A `defaultdict(lambda: None)` for last-fight dates embeds an **unpicklable** local lambda on some Python versions/platforms.

**Decision.** Use a **module-level** default factory (e.g. `_defaultdict_none`) for that `defaultdict` so the factory is picklable by name.

**Consequences.** Any future `defaultdict` or nested lambdas on persisted objects must follow the same rule.

---

## ADR-09: Windows-friendly CLI output

**Context.** On Windows, the default console encoding (e.g. cp1252) can raise **`UnicodeEncodeError`** on common Unicode punctuation in `print`.

**Decision.** Use **ASCII** for high-traffic user-facing strings in [`main.py`](../main.py) (e.g. `->` instead of Unicode arrows in train progress). Docstrings and rare code paths may still contain Unicode; predict/explain tables that use symbols should be tested on cp1252 if surfaced to users.

**Consequences.** Prefer plain ASCII in CLI `print` paths that always run on first-time setups.

---

## ADR-10: Bootstrap count for prediction-time confidence intervals

**Context.** With bootstrap CIs, each `predict` call refits the multinomial model many times on the training matrix. A **1000**-draw default made interactive **`predict` / `explain`** impractically slow.

**Decision.** Lower the default **`ModelConfig.n_bootstrap`** to **200**, with the understanding that **tighter or more stable intervals** for research or production can raise this after a cost/benefit check.

**Consequences.** Saved **`model.pkl`** embeds the config in effect at **`train`** time; changing the default requires **retraining** (or manual config injection) for old pickles to pick up the new behavior.

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
