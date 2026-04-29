# What to work on next

This file is the **human-facing roadmap**: where the project stands, the **next work bout** in order, then reference chunks. For column specs and phased checklists, see [`docs/todo.md`](docs/todo.md).

**Naming alignment:** UFCStats data paths and CLIs use **`ufcstats_*`** modules and **`ufcstats_fights.csv`**. The pipeline still accepts legacy **`tier1_ufcstats.csv`** if the new file is absent. Loader-era filtering still uses the function **`filter_tier1_post_era()`** (internal training cutoff, not the CSV filename).

---

## Current status (detailed)

**Phase 3 — first full walk-forward + random search (done, Apr 2026):**

- **`scripts/run_phase3_tuning.py --selection-search`** completed on your current `data/` snapshot: **16** selection years (2007–2022, **50** trials/yr, last‑3 inner, seed **42**), then **pristine 2023–2025** with the **frozen 2022 winner** `Config`.
- **Artifacts:** `data/phase3_eval/phase3_metrics.csv`, `phase3_report.json` (**`frozen_winner_config`** = full winning hyperparameters), `log_loss_selection_and_pristine.png`, `pristine_test_yoy.png`, `elo_walkforward_cache.pkl`.
- **Not stored in JSON:** per-trial **hyperparameter vectors** (only `trial_rows` with inner/forward log-loss per trial). **Implication:** the **chain** of year-by-year winner configs is not replayable from the report without code changes or a re-run.
- **Primary read:** pristine mean log-loss **improves** vs end of selection; treat as **supporting** evidence, not a proof — few correlated yearly aggregates.

**In the repo (done):**

- **UFCStats fights:** [`src/data/ufcstats_scraper.py`](src/data/ufcstats_scraper.py) — `scrape_ufcstats_fights_to_csv()`, `parse_fight_page()`, `REQUEST_DELAY_SEC` (set via `--sleep` in `main()` or assign before calling the scraper). Writes **`data/ufcstats_fights.csv`** by default; **`failed_entries.csv`** for fetch/parse failures.
- **Profiles:** [`src/data/ufcstats_profiles.py`](src/data/ufcstats_profiles.py) — `scrape_fighter_profiles_to_csv()`; reads fighter IDs from the fights CSV (`--fights-csv` or `--data-dir` resolution).
- **Refresh:** [`src/data/refresh.py`](src/data/refresh.py) — `refresh_data()` runs fights scrape then profiles (used by `main.py train --full-rebuild`).
- **QA / gaps:** [`src/data/ufcstats_gap_report.py`](src/data/ufcstats_gap_report.py) — `run_gap_report()`, `build_inventory()`; optional cache via [`src/data/tier1_inventory_io.py`](src/data/tier1_inventory_io.py) (`save_inventory_csv` / `load_inventory_csv`).
- **Outcome parsing:** Double **D** → `draw`; double **NC** (and “Could Not Continue” / “No Contest” text) → `no contest`; DQ text → `dq`; `winner_id` blank for draw / NC.

**On disk (`data/`):**

- Whatever **you last wrote** under **`data/`** is the source of truth: a **full refresh in progress**, a **finished** `ufcstats_fights.csv` / `fighter_profiles.csv`, or anything you’ve copied out of `data/` for your own archives. This roadmap does **not** assume a scrape has finished until **you** record it (see §F table).

**ELO + Kalman (modeled, tuned):**

- Per–weight-class ELO with Kalman mean/variance, **global layoff clock** (**ADR-15**). Current defaults and regression wiring are summarized in [`docs/elo-modeling-status.md`](docs/elo-modeling-status.md) (tuned `k_base`, `logistic_divisor`, finish scales, process noise; **next:** use ELO uncertainty in features / expected score / CIs — see that doc).

**Not the focus yet (optional / later):**

- Tier 2–3 promotion CSVs, pedigree manual fill, **one-off** holdout OAT from [`docs/todo.md`](docs/todo.md) §3.3 when comparing single-knob ablations, CI / `elo_mc_*` spot checks in the same doc.

---

## Next work bout (in order)

Do these as the **immediate** slice of work; skip steps that are already satisfied for your tree.

1. **Ship refit (frozen 2022 config + full Tier‑1 data)** — Load hyperparameters from **`data/phase3_eval/phase3_report.json` → `frozen_winner_config`** (or rehydrate into `Config`); run **`main.py train`** (or your train CLI) with the **intended** holdout policy for **production** (see [`docs/hyperparameter-tuning.md`](docs/hyperparameter-tuning.md) + [`docs/todo.md`](docs/todo.md) §3.1). Save a **`.pkl`** for `predict` / deployment.
2. **Fast validation (before another multi-day 50-trial run)** — Use as **A/B** vs the full script or vs `Config()`:
   - **Baseline only:** `run_phase3_tuning.py` **without** `--selection-search` (single `Config` walk-forward) on the same `selection-start`/`end`, compare curves to the saved `phase3_metrics.csv`.
   - **Smaller search:** same script with **`--n-trials 10`–`20`**, and/or **narrower** `--selection-start` / `--selection-end` (e.g. 2018–2022) to see if **ranking** of winners is stable vs the 50-trial run.
   - **OAT / one-knob** generations on **holdout** (§3.4) for cheap sensitivity — does **not** replace walk-forward, but calibrates “how much knob X moves log-loss” on a **locked** data snapshot.
   - **Optimizer cost:** if you re-run long searches, use **`scripts/pilot_lbfgs_stopping.py`** and (later) **tuning-only** `ftol`/`gtol`/`max_iter` *after* a **ranking** spot-check, not on faith alone.
3. **Case studies and examples** — Pristine and selection slices in `phase3_report.json` (per–weight-class). Pull **highest per-fight log-loss** fights for write-ups; see [`docs/hyperparameter-tuning.md`](docs/hyperparameter-tuning.md) §9.
4. **Closing-line / P&L research (future)** — Historical **opening** or **pre-bell** odds (per fight, PIT) would let you test **stake** / ROI vs model probabilities (Kelly, flat stake, etc.). **Out of scope** until you have a **reproducible lines** data source; model metrics alone do not prove profitability.
5. **Abstention / stake filter (future, depends on #4)** — Once lines data exists, a post-model filter can decide which fights to stake on. **The trigger is EV, not model confidence**: abstain unless `P(k) × decimal_odds(k) > 1 + min_edge` for some outcome class k. CIs always overlap in a 6-class model, so "CI overlap" is not a usable abstention criterion and argmax-confidence thresholds cherry-pick easy fights, inflating reported metrics on the chosen subset. See **ADR-21** in [`docs/architecture-decisions.md`](docs/architecture-decisions.md) for the full framing and the key constraint: the stake filter must be evaluated on **ROI over all fights** (coverage + P&L), not accuracy on the fights it chose to predict.

**Data refresh (when you bump UFCStats data):** If `ufcstats_fights.csv` / profiles change material rows, re-run [§A–B](#a-ufcstats-fights--dataufcstats_fightscsv) and treat Phase 3 as a **new campaign** (re-baseline or re-run `run_phase3_tuning` if you need comparability).

---

**Earlier bootstraps (satisfied for many trees; keep for a clean machine):**

1. **Land CSVs you trust** — Full scraper + skip log in [§F](#f-ufcstats-scrape-skips--investigation-log). Profiles after fights file stabilizes.
2. **CSV sanity** — `ufcstats_gap_report --check-csv-only --data-dir ./data`.
3. **Phase 2 smoke** — `main.py train`, `scripts/phase2_smoke.py`, one `predict` / `explain` (`docs/todo.md` §2.1).
4. **Quick gates** — Symmetry + ELO sniff (`docs/todo.md` §2.2–2.3).

### Phase 3 — further tuning (after ship refit or fast A/B)

Use **repeated model generations** for *single-knob* studies: same **`--holdout-start`**, change **one** field in [`src/config.py`](src/config.py), retrain with a **unique `--model-path`**, run **`eval-holdout`**, log metrics. **Full** protocol, walk-forward, and 50-trial search: [`docs/todo.md`](docs/todo.md) §3.4–3.5, [`docs/hyperparameter-tuning.md`](docs/hyperparameter-tuning.md).

```bash
# Example OAT generation (not the multi-day walk-forward search)
python main.py train --data-dir ./data --holdout-start 2023-01-01 --model-path ./data/Saved_Runs/phase3_baseline.pkl
python main.py eval-holdout --model-path ./data/Saved_Runs/phase3_baseline.pkl
```

---

## High level — strategic themes

1. **Data refresh cadence** — Full UFCStats fights scrape is **several hours** (~770 events + fights; README). Re-run after parser or schema changes; profiles after the fights file stabilizes.
2. **Validation before tuning** — Log-loss and era knobs come **after** “train runs, predict runs, symmetry holds.”
3. **Cheap A/B before expensive search** — The **50-trial/yr** walk-forward run is a **reference**, not a weekly habit. **Baseline** walk-forward, **10–20 trials**, or a **shorter** selection window should agree **in spirit** (stable ranking) before you burn another long wall-clock block.
4. **Hardening** — Tests, pinned deps, post-event refresh story—after the model path is trusted.
5. **From probabilities to P&L** — Requires **reproducible** historical **odds** at a defined decision time; out of scope for core modeling until that data exists.

---

## Side projects (low priority)

- **ELO trajectory “never downtrend” scan** — Use recorded ELO trajectories (`build_elo(..., record_trajectories=True)`, `ELOModel.get_trajectory`) and analyze **concavity / segment slopes** (or simpler: consecutive fight-to-fight deltas) to flag fighters whose path in a weight class **never exhibits a downward trend** by whatever operational definition you choose (e.g. no strictly decreasing step between post-fight ELOs; or a minimum-career-length filter). Exploratory curiosity, not part of training or Phase 3 metrics. Implementation could live as a small script under `scripts/` reusing [`scripts/chart_elo_trajectory.py`](scripts/chart_elo_trajectory.py) plumbing.

---

## Soon — detailed starter chunks

Pick one vertical and finish it before starting the next; they’re ordered roughly by dependency.

### A. UFCStats fights → `data/ufcstats_fights.csv`

- **Entrypoint:** `python -m src.data.ufcstats_scraper` → `scrape_ufcstats_fights_to_csv()` in [`src/data/ufcstats_scraper.py`](src/data/ufcstats_scraper.py).
- **Rate limit:** module global `REQUEST_DELAY_SEC`; CLI `--sleep` sets it in `main()`.
- **Canonical `fighter_id`:** hex segment from UFCStats `/fighter-details/<id>` (same in fights CSV and profiles).
- **Event listing → fight IDs:** `statistics/events/completed?page=all`, then each event page, then each `/fight-details/<id>`.
- **Per-fight fields:** outcome (including banner-driven **draw** / **no contest** normalization), method, weight class, date, fight time, A/B stat columns per `docs/todo.md` §1.1.
- **Outputs:** default **`data/ufcstats_fights.csv`**; **`failed_entries.csv`** next to the output for rows that did not parse. Expect **~8k+** fight rows for a full run—use **`ufcstats_gap_report`** if skips are high.

### B. Profiles → `data/fighter_profiles.csv`

- **Entrypoint:** `python -m src.data.ufcstats_profiles` → `scrape_fighter_profiles_to_csv()` in [`src/data/ufcstats_profiles.py`](src/data/ufcstats_profiles.py).
- **Fighter index → profile pages.** Height, reach, DOB, stance; map to the same `fighter_id` as the fights CSV.
- **Pedigree fields** (`wrestling_pedigree`, etc.): start at `0` everywhere if you want speed; refine later for cold starts only.

**Run profiles only** (after `ufcstats_fights.csv` exists — does not re-scrape fights):

```bash
python -m src.data.ufcstats_profiles --data-dir ./data
```

Optional: `--max-fighters N` for a smoke run, `--sleep` (profiles pass; separate from scraper’s `REQUEST_DELAY_SEC`), or `--fights-csv` / `--out` for custom paths.

### C. `refresh_data`

- [`src/data/refresh.py`](src/data/refresh.py) — `refresh_data()` runs `scrape_ufcstats_fights_to_csv()` then `scrape_fighter_profiles_to_csv()` into `data/`. **Smoke test:** `python main.py train --data-dir ./data --full-rebuild` (long), then `python main.py train --data-dir ./data` (second run = model-only from disk).

### D. First pipeline smoke test (after A + B)

- Train: `python main.py train --data-dir ./data`
- Fast checks (finite `X_train`, symmetry, ELO snapshot): `python scripts/phase2_smoke.py`
- Check loaded fight and profile counts, training set size, no non-finite features.
- One `predict` and one `explain` on a known matchup using real IDs from your CSVs.
- Holdout / CV / few-shot notes: [`docs/validation-and-few-shot.md`](docs/validation-and-few-shot.md)

### E. Quick quality gates (before big tuning)

- Add or run a **symmetry check** (swap A/B; win/lose probabilities flip)—see `docs/todo.md` §2.2.
- **ELO sanity:** top/bottom of division vs your intuition; famous names in plausible neighborhoods.

### F. UFCStats scrape skips — investigation log

Use this subsection to record full-run skip counts and follow-ups. The scraper increments **skipped/problem fights** when a fight page fails HTTP, or when `parse_fight_page` returns `None` (no row written). Details are in **`failed_entries.csv`**.

**Gap report:** [`src/data/ufcstats_gap_report.py`](src/data/ufcstats_gap_report.py) — `run_gap_report()` diffs the site fight list vs your CSV; optional diagnosis of **missing** rows. Skipped scrapes never appear in the fights CSV, so you **cannot** infer missing `fight_id`s from that file alone.

**Avoid repeating the slow event crawl:** save inventory once, reuse forever (until UFCStats layout changes):

```bash
# One-time (~770 event requests):
python -m src.data.ufcstats_gap_report --fetch-inventory-only --write-inventory-csv ./data/ufcstats_event_inventory.csv

# Later — no event HTTP; only ~missing fight requests if you diagnose:
python -m src.data.ufcstats_gap_report --data-dir ./data --inventory-csv ./data/ufcstats_event_inventory.csv
```

**Local checks on rows you *do* have** (no network; finds duplicate ids, bad winner_id, etc.):

```bash
python -m src.data.ufcstats_gap_report --check-csv-only --data-dir ./data
```

Other flags: `--no-diagnose` (list missing URLs only), `--out-missing` (default under `--data-dir`: `ufcstats_missing_fights.csv`), `--sleep` (gap report / inventory crawl only; does not change `ufcstats_scraper.REQUEST_DELAY_SEC`).

| Run | Rows written | Skipped | Notes |
|-----|--------------|---------|--------|
| Example full UFCStats fights scrape | ~8.1k+ (line count − header) | varies | Parser fixes (e.g. **no contest**) reduce skips vs early runs; run `ufcstats_gap_report` for a reason breakdown. Extend `WEIGHT_CLASS_MAP` / `_normalize_method` in `ufcstats_scraper.py` when UFCStats adds odd labels. |

---

## Reference

| Topic | Where |
|--------|--------|
| ELO tuning status, Kalman vs regression, next steps | [`docs/elo-modeling-status.md`](docs/elo-modeling-status.md) |
| Why layoffs **amplify** (not damp) the next ELO update — ADR-16 framing | [`docs/elo-kalman-layoff-philosophy.md`](docs/elo-kalman-layoff-philosophy.md) |
| Full phased checklist, schemas, metrics | [`docs/todo.md`](docs/todo.md) |
| Design and stage definitions | [`docs/architecture.md`](docs/architecture.md) |
| Expected CSV filenames / loader behavior | [`src/pipeline.py`](src/pipeline.py) (`load_data`; tries `ufcstats_fights.csv` then legacy `tier1_ufcstats.csv`) |

When you finish a chunk above, tick the matching boxes in `docs/todo.md` so the detailed doc stays the source of truth for progress.
