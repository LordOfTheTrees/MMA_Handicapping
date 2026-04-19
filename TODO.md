# What to work on next

This file is the **human-facing roadmap**: where the project stands, the **next work bout** in order, then reference chunks. For column specs and phased checklists, see [`docs/todo.md`](docs/todo.md).

**Naming alignment:** UFCStats data paths and CLIs use **`ufcstats_*`** modules and **`ufcstats_fights.csv`**. The pipeline still accepts legacy **`tier1_ufcstats.csv`** if the new file is absent. Loader-era filtering still uses the function **`filter_tier1_post_era()`** (internal training cutoff, not the CSV filename).

---

## Current status (detailed)

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

- Tier 2–3 promotion CSVs, pedigree manual fill, **Phase 3 holdout tuning** (`era_cutoff_year` / **2013 boundary**, ELO levers, train–test split, thresholds/weights — full table in [`docs/todo.md`](docs/todo.md) §3.3), CI—see `docs/todo.md`.

---

## Next work bout (in order)

Do these as the **immediate** slice of work; skip steps that are already satisfied for your tree.

1. **Land CSVs you trust** — Let the current UFCStats fights run finish (or start a full `python -m src.data.ufcstats_scraper --data-dir ./data`). Copy **rows written** and **skipped/problem fights** from the scraper’s final log into [§F](#f-ufcstats-scrape-skips--investigation-log). If the fights CSV changed (new fights, parser fix), run **`python -m src.data.ufcstats_profiles --data-dir ./data`** so profiles cover the same ID universe.
2. **CSV sanity** — `python -m src.data.ufcstats_gap_report --check-csv-only --data-dir ./data` (or `--fights-csv ./data/ufcstats_fights.csv`). Optionally inventory diff + diagnose if skips are non-trivial ([§F](#f-ufcstats-scrape-skips--investigation-log)).
3. **Phase 2 smoke test** — `python main.py train --data-dir ./data`: confirm loaded fight/profile counts, training size, no broken features. One **`predict`** and one **`explain`** with real `fighter_id`s from your CSVs (`docs/todo.md` §2.1).
4. **Quick gates** — Symmetry (swap A/B) and a light ELO sniff test (`docs/todo.md` §2.2–2.3).
5. **Then** — Phase 3 (holdout, calibration, knob tuning) only after the smoke test is boringly stable.

---

## High level — strategic themes

1. **Data refresh cadence** — Full UFCStats fights scrape is **several hours** (~770 events + fights; README). Re-run after parser or schema changes; profiles after the fights file stabilizes.
2. **Validation before tuning** — Log-loss and era knobs come **after** “train runs, predict runs, symmetry holds.”
3. **Hardening** — Tests, pinned deps, post-event refresh story—after the model path is trusted.

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
