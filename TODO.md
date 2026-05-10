# What to work on next

This file is the **human-facing roadmap**: **next work** in order and high-level themes. Column specs and phased checklists stay in [`docs/todo.md`](docs/todo.md).

**Naming alignment:** UFCStats data paths and CLIs use **`ufcstats_*`** modules and **`ufcstats_fights.csv`**. The pipeline still accepts legacy **`tier1_ufcstats.csv`** if the new file is absent. Loader-era filtering still uses **`filter_tier1_post_era()`** (internal training cutoff, not the CSV filename).

**ELO + Kalman:** Defaults and tuning are summarized in [`docs/elo-modeling-status.md`](docs/elo-modeling-status.md); **next modeling step** there is using ELO uncertainty in features / expected score / CIs.

**Website / deploy artifacts (training repo):** Portable JSON for **`mma.ai`**, scrape → **`JSON_exports/`**, **`scripts/run_harness.py`** (`quick` / **`site`** / **`integration`**), pickle vs snapshot parity ([`tests/harness_skip.py`](tests/harness_skip.py), [`tests/test_site_export_pages.py`](tests/test_site_export_pages.py)), and **`docs/BACKEND_PIPELINE_INTEGRATION.md`**. Formal decision log: **ADR-24** in [`docs/architecture-decisions.md`](docs/architecture-decisions.md).

**Deferred (optional):** Tier 2–3 promotion CSVs, pedigree manual fill, one-off holdout OAT from [`docs/todo.md`](docs/todo.md) §3.3 when comparing single-knob ablations, CI / `elo_mc_*` spot checks.

---

## Next work bout (in order)

1. **Case studies and examples** — Pristine and selection slices in **`data/phase3_eval/phase3_report.json`** (per–weight-class). Pull **highest per-fight log-loss** fights for write-ups; see [`docs/hyperparameter-tuning.md`](docs/hyperparameter-tuning.md) §9.
2. **Fight odds + stake / P&L research** — Historical **opening** or **pre-bell** / **closing** odds (per fight, PIT) to test **stake** / ROI vs model probabilities (Kelly, flat stake, **EV-based** filter per **ADR-21**). **Blocked** until a **reproducible lines** data source exists; model metrics alone do not prove profitability.
3. **Fast validation (cheap Phase 3 A/B)** — Before another long walk-forward: use as **A/B** vs saved **`data/phase3_eval/phase3_metrics.csv`** / report:
   - **Baseline only:** `python -m src.cli.run_phase3_tuning` **without** `--selection-search` (single `Config` walk-forward) on the same `selection-start`/`end`, compare curves to the saved metrics.
   - **Smaller search:** same script with **`--n-trials 10`–`20`**, and/or **narrower** `--selection-start` / `--selection-end` (e.g. 2018–2022) to see if **ranking** of winners is stable vs the 50-trial run.
   - **OAT / one-knob** generations on **holdout** ([`docs/todo.md`](docs/todo.md) §3.4) for cheap sensitivity — does **not** replace walk-forward, but calibrates “how much knob X moves log-loss” on a **locked** data snapshot.
   - **Optimizer cost:** if you re-run long searches, use **`scripts/pilot_lbfgs_stopping.py`** and (later) **tuning-only** `ftol`/`gtol`/`max_iter` *after* a **ranking** spot-check, not on faith alone.

**Data refresh:** If `ufcstats_fights.csv` / profiles gain material rows, treat Phase 3 as a **new campaign** (re-baseline or re-run `python -m src.cli.run_phase3_tuning` when you need comparability). Operational steps: scraper / gap report / refresh flows in [`docs/todo.md`](docs/todo.md) §1 and [`README.md`](README.md). After refresh, re-run **export** + **`python scripts/run_harness.py site`** (and **`integration`** if you rely on pickle parity).

---

### Phase 3 — further tuning (after fast A/B or new data)

Use **repeated model generations** for *single-knob* studies: same **`--holdout-start`**, change **one** field in [`src/config.py`](src/config.py), retrain with a **unique `--model-path`**, run **`eval-holdout`**, log metrics. Full protocol and walk-forward: [`docs/todo.md`](docs/todo.md) §3.4–3.5, [`docs/hyperparameter-tuning.md`](docs/hyperparameter-tuning.md).

```bash
# Example OAT generation (not the multi-day walk-forward search)
python main.py train --data-dir ./data --holdout-start 2023-01-01 --model-path ./data/Saved_Runs/phase3_baseline.pkl
python main.py eval-holdout --model-path ./data/Saved_Runs/phase3_baseline.pkl
```

---

## High level — strategic themes

1. **Data refresh cadence** — Full UFCStats fights scrape is **several hours** (~770 events + fights; README). Re-run after parser or schema changes; profiles after the fights file stabilizes.
2. **Validation before tuning** — Log-loss and era knobs come **after** “train runs, predict runs, symmetry holds.”
3. **Cheap A/B before expensive search** — **50-trial/yr** walk-forward is a **reference**, not a weekly habit. Baseline walk-forward, **10–20 trials**, or a **shorter** selection window should agree **in spirit** (stable ranking) before another long wall-clock run.
4. **Hardening** — Harness + export parity + site-page JSON checks in repo; widen tests as needed.
5. **From probabilities to P&L** — Requires **reproducible** historical **odds** at a defined decision time; out of scope for core modeling until that data exists.

---

## Side projects (low priority)

- **ELO trajectory “never downtrend” scan** — Use recorded ELO trajectories (`build_elo(..., record_trajectories=True)`, `ELOModel.get_trajectory`) and analyze **concavity / segment slopes** (or consecutive fight-to-fight deltas) to flag fighters whose path in a weight class **never exhibits a downward trend** by your operational definition. Exploratory; not part of training or Phase 3 metrics. Starting point: [`src/cli/chart_elo_trajectory.py`](src/cli/chart_elo_trajectory.py) (`python -m src.cli.chart_elo_trajectory`) and [`src/elo/`](src/elo/).

---

## Reference

| Topic | Where |
|--------|--------|
| ELO tuning status, Kalman vs regression, next steps | [`docs/elo-modeling-status.md`](docs/elo-modeling-status.md) |
| Why layoffs **amplify** (not damp) the next ELO update — ADR-16 | [`docs/elo-kalman-layoff-philosophy.md`](docs/elo-kalman-layoff-philosophy.md) |
| Full phased checklist, schemas, metrics | [`docs/todo.md`](docs/todo.md) |
| Design and stage definitions | [`docs/architecture.md`](docs/architecture.md) |
| Expected CSV filenames / loader behavior | [`src/pipeline.py`](src/pipeline.py) (`load_data`; tries `ufcstats_fights.csv` then legacy `tier1_ufcstats.csv`) |

When you finish a chunk, mirror progress in [`docs/todo.md`](docs/todo.md).
