# Session changelog — catch-up notes (not duplicated elsewhere)

This file is a **sanity-check journal**: decisions and workflow changes that came out of recent working sessions and are **easy to miss** if you only read `architecture.md` / `todo.md`. It is **not** a substitute for those docs or for ADRs in `architecture-decisions.md`.

---

## CLI, training, and data refresh

- **Train from disk (default)** — `python main.py train` / `python -m src.cli.train` loads existing CSVs under `--data-dir` and runs ELO + regression. No scrape unless you opt in.
- **Full rebuild hook** — `--full-rebuild` runs `refresh_data()` from [`src/data/refresh.py`](../src/data/refresh.py) before training (scrapes / exports are implemented there).
- **Root vs package `data/`** — `.gitignore` uses **`/data/`** (repo root only) so **`src/data/`** Python modules stay tracked; an earlier overly broad `data/` rule had hidden `src/data/*.py` from Git.

---

## Scraping and UFCStats hygiene

- **Failed parses / fetch failures** — The UFCStats pipeline logs failures to **`failed_entries.csv`** (alongside the fights CSV unless overridden) and prints a line per failure during the run. This supports gap analysis (`ufcstats_gap_report`, `TODO.md`).
- **Request pacing** — Default scrape sleep was tightened over time for throughput (see current **`REQUEST_DELAY_SEC`** / CLI in [`src/data/ufcstats_scraper.py`](../src/data/ufcstats_scraper.py)); profile scrapes follow the same pattern in [`ufcstats_profiles.py`](../src/data/ufcstats_profiles.py).
- **Naming** — Prefer **`ufcstats_fights.csv`** as the Tier‑1 artifact name in docs and loader; legacy `tier1_ufcstats.csv` remains supported for migration.

---

## GitHub Actions admin (CI scrape + stale fallback)

- **`allow_stale_data`** — Manual **Run workflow** checkbox on **`weekly-model-refresh`** and **`monthly-model-retrain`** (default **false**). **Cron schedules never use it** (always require a live UFCStats scrape).
- **Normal run (`false`)** — `ci_try_refresh_data.py` only; if UFCStats serves a Cloudflare/bot page, the job fails with a grouped log (`::group::UFCStats scrape blocked`). **Sync JSON exports** does not run (trigger requires **success**).
- **Admin/debug run (`true`)** — Restores **`data/*.csv`** from the newest **`weekly-refresh-*`** / **`monthly-retrain-*`** artifact (`ci_restore_data_bundle.py`), then `ci_try_refresh_data.py --allow-stale-data` so rebuild/export can proceed without a live scrape. Use when scraping is broken; data will be as old as that bundle.
- **Scraper guard** — `scrape_ufcstats_fights_to_csv` will not overwrite an existing fights CSV with 0 rows when the index is a bot wall (see `probe_completed_events_index` in [`ufcstats_scraper.py`](../src/data/ufcstats_scraper.py)).
- **Sync artifact name** — Use **`run-bundle`** (not legacy **`mma-json-exports`**). Full ops write-up: **[`docs/BACKEND_PIPELINE_INTEGRATION.md`](BACKEND_PIPELINE_INTEGRATION.md)** § GitHub Actions admin.
- **Local disaster recovery** — Copy-paste restore of **`data/model.pkl`**, Tier‑1 CSVs, and **`JSON_exports/`** from latest CI artifacts: same doc, § **Local disaster recovery (restore last CI build)**.

---

## Planning and status files

- **[`TODO.md`](../TODO.md)** — Short “next work bout” at repo root; defers deep checklists to **`docs/todo.md`**.
- **`docs/todo.md`** — Phase gates (data → smoke → holdout tuning), column specs, Phase 3 knob inventory.

---

## Modeling and evaluation artifacts (local)

These are **machine-local** or **large**; paths are conventional, not always in Git:

| Artifact | Typical path | Notes |
|----------|--------------|--------|
| Phase 3 harness output | `data/phase3_eval/` | `phase3_report.json`, `phase3_metrics.csv`, plots — see `hyperparameter-tuning.md` §8 |
| ELO cache | `data/elo_cache.pkl` | Speeds repeated walk-forward when fight count + `ELOConfig` match |
| Saved runs | `data/Saved_Runs/` | Ad-hoc exports (if present) |
| Committed tuning snapshot | [`docs/first_run_report.json`](first_run_report.json) | Full selection + pristine JSON from a Phase‑3-style run (`generated_utc` inside file) |

---

## Documentation added in-repo (reference)

- **ELO layer** — `elo-modeling-status.md`, `elo-tuning-knobs.md`, `elo-kalman-layoff-philosophy.md`
- **Phase 3 protocol** — `hyperparameter-tuning.md` (selection vs pristine vs ship; random search; inner walk-forward)
- **Validation** — `validation-and-few-shot.md` (time holdout, leakage, symmetry caveat on interactions)
- **Kalman sensitivity** — `k-sensitivity-top15-side-by-side.md` (process-noise comparison tables)

---

## External project registry (non-repo)

- **TreePage `_data/projects.yml`** — One session added a short **MMA Handicapping** blurb next to other portfolio projects (personal site content; not stored in this repo).

---

## How to use this file

- Treat it as **narrative glue** when something feels “we did that in chat but I don’t see it in the architecture doc.”
- When a topic stabilizes, **fold facts into the canonical doc** (`architecture.md`, `pipeline-and-cli.md`, `todo.md`) and trim or cross-link from here so this file stays short.
