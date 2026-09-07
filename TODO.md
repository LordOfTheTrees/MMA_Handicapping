# What to work on next

This file is the **human-facing roadmap**: **next work** in order and high-level themes. Column specs and phased checklists stay in [`docs/todo.md`](docs/todo.md).

**Naming alignment:** UFCStats data paths and CLIs use **`ufcstats_*`** modules and **`ufcstats_fights.csv`**. The pipeline still accepts legacy **`tier1_ufcstats.csv`** if the new file is absent. Loader-era filtering still uses **`filter_tier1_post_era()`** (internal training cutoff, not the CSV filename).

**ELO + Kalman:** Defaults and tuning are summarized in [`docs/elo-modeling-status.md`](docs/elo-modeling-status.md); **next modeling step** there is using ELO uncertainty in features / expected score / CIs.

**Website / deploy artifacts (training repo):** Portable JSON for **`mma.ai`**, scrape → **`JSON_exports/`**, **`scripts/run_harness.py`** (`quick` / **`site`** / **`integration`**), pickle vs snapshot parity ([`tests/harness_skip.py`](tests/harness_skip.py), [`tests/test_site_export_pages.py`](tests/test_site_export_pages.py)), and **`docs/BACKEND_PIPELINE_INTEGRATION.md`**. Formal decision log: **ADR-24** in [`docs/architecture-decisions.md`](docs/architecture-decisions.md).

**Deferred (optional):** Tier 2–3 promotion CSVs, pedigree manual fill, one-off holdout OAT from [`docs/todo.md`](docs/todo.md) §3.3 when comparing single-knob ablations, CI / `elo_mc_*` spot checks.

**CI / scrape ops (documented):** [`docs/BACKEND_PIPELINE_INTEGRATION.md`](docs/BACKEND_PIPELINE_INTEGRATION.md) § GitHub Actions admin; local restore: § **Local disaster recovery**.

---

## Most important (blocking data + CI)

Do these before treating weekly refresh or new modeling work as “healthy.” Detail and checklists: [`docs/todo.md`](docs/todo.md) § **P0 — Scrape recovery & incremental data**.

### 1. Fix the re-scrape problem (multiple paths)

UFCStats completed-events scrape is **blocked** from CI/GitHub (and often locally) by a Cloudflare bot wall — full re-scrape writes **0 rows** and breaks the pipeline. Pursue **more than one** path; keep what works in parallel.

| Path | Role |
|------|------|
| **A. ESPN API (primary)** | Incremental ingest + crosswalk — see [`docs/data-sources-espn.md`](docs/data-sources-espn.md). UFCStats hex IDs preserved via `espn_crosswalk_*.csv`. |
| **A′. UFCStats + HTTP** | Legacy HTML path; blocked in CI. Parser kept as reference / fallback. |
| **B. Official UFC site** | Ingest **upcoming** cards and bout matchups from **ufc.com** (or UFC’s public event pages) for the calendar / `upcoming_events.json` path — does **not** replace historical fight stats unless we add a separate results feed. Complements [`src/data/ufcstats_upcoming.py`](src/data/ufcstats_upcoming.py) (today: UFCStats upcoming only). |
| **C. Alternate finished-fight source** | Sherdog / other tier-2/3 CSVs, or manual gap-fill for **new** fights only if UFCStats stays blocked. |
| **D. Bridge (now)** | CI **`allow_stale_data`** + artifact restore; local **disaster recovery** script bundle — unblocks export/retrain on **last good** snapshot until A–C land. |

**Success:** scheduled weekly/monthly can complete **without** admin stale mode when new fights need to land in **`ufcstats_fights.csv`**.

### 2. Stop re-scraping the same data every run (incremental stitch)

Today a “refresh” rediscovers **~770 events** and re-fetches thousands of fight pages (~hours). Target behavior:

1. **Seed** CI (and local refresh) from **existing artifacts**: `mma-model-state`, run-bundle CSVs (`ufcstats_fights.csv`, profiles, `upcoming_cards.json`) — see disaster-recovery doc.
2. **Incremental ingest**: maintain inventory / max `fight_id` or last event date; fetch **only new** events and fight-detail pages; **append** rows (and **new** profile IDs only).
3. **Upload** updated bundle after success so the next run seeds again — no cold full scrape unless explicitly requested.

**Success:** typical weekly job time scales with **new fights since last run**, not full history.

---

## Next work bout (after P0 scrape items)

1. **Case studies and examples** — Pristine and selection slices in **`data/phase3_eval/phase3_report.json`** (per–weight-class). Pull **highest per-fight log-loss** fights for write-ups; see [`docs/hyperparameter-tuning.md`](docs/hyperparameter-tuning.md) §9.
2. **Fight odds + stake / P&L research** — Post-hoc walk-forward book: **`python -m src.eval.market_book`** ([`docs/pipeline-and-cli.md`](docs/pipeline-and-cli.md) §4.1). Odds never enter training. ADR-21 **`min_edge` is still not searched** (stake rule is every posted `e > 0`). First PIT book: projected ≫ realized; next maps (de-vig, simultaneous Kelly, blend, contract binaries) are **ADR-28** — display softmax stays.
3. **Fast validation (cheap Phase 3 A/B)** — Before another long walk-forward: use as **A/B** vs saved **`data/phase3_eval/phase3_metrics.csv`** / report:
   - **Baseline only:** `python -m src.cli.run_phase3_tuning` **without** `--selection-search` (single `Config` walk-forward) on the same `selection-start`/`end`, compare curves to the saved metrics.
   - **Smaller search:** same script with **`--n-trials 10`–`20`**, and/or **narrower** `--selection-start` / `--selection-end` (e.g. 2018–2022) to see if **ranking** of winners is stable vs the 50-trial run.
   - **OAT / one-knob** generations on **holdout** ([`docs/todo.md`](docs/todo.md) §3.4) for cheap sensitivity — does **not** replace walk-forward, but calibrates “how much knob X moves log-loss” on a **locked** data snapshot.
   - **Optimizer cost:** if you re-run long searches, use **`scripts/dev/pilot_lbfgs_stopping.py`** and (later) **tuning-only** `ftol`/`gtol`/`max_iter` *after* a **ranking** spot-check, not on faith alone.

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

1. **Data refresh cadence** — **P0:** incremental stitch + scrape recovery (above). Until then, full UFCStats fights scrape is **several hours** (~770 events + fights) when it works at all; re-run full only after parser/schema changes or explicit `--full-rescrape`.
2. **Validation before tuning** — Log-loss and era knobs come **after** “train runs, predict runs, symmetry holds.”
3. **Cheap A/B before expensive search** — **50-trial/yr** walk-forward is a **reference**, not a weekly habit. Baseline walk-forward, **10–20 trials**, or a **shorter** selection window should agree **in spirit** (stable ranking) before another long wall-clock run.
4. **Hardening** — Harness + export parity + site-page JSON checks in repo; widen tests as needed.
5. **From probabilities to P&L** — Local verification book is **`python -m src.eval.market_book`**; still not a training objective. ADR-21 `min_edge` is not searched. Softmax-for-humans vs contract pricing: **ADR-28**.

---

## Engineering quality (from code review, 2026-09-07)

Codebase-health items rather than modeling or data work, ordered by impact. Each is
verified against the tree with the measurement that justifies it. Deploy-repo items live
in **`mma.ai`** [`TODO.md`](https://github.com/LordOfTheTrees/MMA.AI/blob/main/TODO.md);
item 4 below is cross-repo and appears in both.

### 1. CI exists here but has never actually run

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs `ruff check .` + `pytest -q`
(126 tests) on Python 3.12 and 3.13, plus a job that regenerates the cross-repo parity
fixture and diffs it against the committed copy. It triggers on `pull_request` and
`push: main` — and no PR has ever opened, so **not one step has executed**. Every step was
verified locally in a clean venv built from `requirements-dev.txt`, which is not the same
as having run.

- [ ] Open a PR for the current branch so the workflow executes at least once. Cheapest
      item on this list by a wide margin.

### 2. `print()` everywhere, no `logging`

**264** `print()` calls in `src/`; **zero** modules import `logging`. The monthly retrain
runs unattended for up to 12 hours (`timeout-minutes: 720`) and everything it emits is
undifferentiated stdout — no levels, no timestamps, no way to filter for the failure.

- [ ] Move `src/` to `logging` with a module-level logger, keeping the existing message
      text. CLI entry points configure a handler; library code just logs.
- [ ] Progress counters (`matrix_progress_every`, bootstrap ticks) stay as-is or move to
      `tqdm` — those are genuinely for a human watching a terminal.

*(`mma.ai` already does this correctly — `logging` throughout — so the target style is
settled.)*

### 3. `market_book.py` is 1,503 lines with ~120 duplicated

`_kelly_path` / `_kelly_path_simul` are ~95% identical: the drawdown, ruin and
Brownian-approximation block is copy-pasted character for character, as is the return
dict. Same pattern for `_flat_1u_path` / `_flat_1u_path_simul` and
`rollup_picks` / `rollup_simul` (54 `_simul` references in the file).

- [ ] Collapse the three pairs into one wealth-path walker parameterised by a per-item
      `(log_growth, multiplier, stake)` callback.
- [ ] Split the module into a package while in there — parsing, staking, accumulation and
      CLI are four separable concerns in one file.
- [ ] [`tests/test_market_book.py`](tests/test_market_book.py) (376 lines) covers this
      well enough to refactor against.

### 4. Generated artifacts are committed into `mma.ai` *(cross-repo)*

[`.github/workflows/sync-json-to-mma-ai.yml`](.github/workflows/sync-json-to-mma-ai.yml)
pushes `JSON_exports/` into the deploy repo on every refresh. There, 20 of 52 commits are
artifact syncs rewriting ~41,000 lines each, and `.git` is 20 MB against ~11k LOC of
source. The producer is here, so the fix is partly here.

- [ ] Publish the bundle as release assets / object storage; have the deploy image fetch it.
- [ ] Interim: stop pretty-printing `bootstrap_W` in
      [`scripts/export_artifacts.py`](scripts/export_artifacts.py) — 200 matrices at one
      float per line is most of the 33k-line churn in `model_weights.json`.

### 5. Dependencies are floor-pinned in a repo that produces model artifacts

All six entries in [`requirements.txt`](requirements.txt) are `>=` with no upper bound and
no lockfile, including `numpy` and `scipy`. The monthly retrain resolves them fresh, so a
numeric result is not reproducible from the repo alone. `mma.ai` pins exactly.

- [ ] Pin exact versions here too, or add a lockfile.
- [ ] Record the resolved versions in the export manifest so an artifact says what built it.

### 6. Widen the lint scope one group at a time

[`pyproject.toml`](pyproject.toml) selects only `F`, `E9`, `W` — deliberately narrow so the
first CI run was green rather than buried under ~1,200 style findings. That initial set
already earned itself: it found an undefined `List` in
[`src/eval/tuning_plots.py`](src/eval/tuning_plots.py) (latent only because the module has
`from __future__ import annotations`), 12 unused imports and two dead assignments.

Remaining groups, counted under the pinned `ruff==0.15.8` — each wants its own cleanup
commit:

- [ ] `RET` (2) — return-path simplification
- [ ] `C4` (11) — comprehension simplification
- [ ] `B` (25) — bugbear; worth reading individually, some are real
- [ ] `I` (42) — import sorting
- [ ] `UP` (1,162) — pyupgrade; large but mechanical

Counts move with the ruff version, so re-measure before starting:
`ruff check src scripts tests main.py --select <GROUP> --statistics`.

### 7. Decide on `standardize_features` (opt-in, currently off)

[`ModelConfig.standardize_features`](src/config.py) fits on column-scaled features and
transforms coefficients back to raw space, so `W` and every consumer are unchanged. It is
**off by default** because enabling it changes what `l2_lambda` means — the shipped tuned
`huber_delta` / `l2_lambda` no longer apply, and flipping it silently would ship an
untuned model out of the unattended retrain.

Measured on a matrix built through the real feature construction: `cond(X'X)` 2.9e13 →
1.2e06, and L-BFGS-B went from hitting the 10,000-iteration cap **without converging** to
converging in ~305. The `grappling_matchup` column is currently penalised ~4e11 times as
hard as `age_diff_days` by the shared `l2_lambda`.

- [ ] Run [`scripts/dev/benchmark_feature_scaling.py`](scripts/dev/benchmark_feature_scaling.py)
      on the real corpus. It reports both regimes and says whether a re-tune is mandatory.
- [ ] If held-out log-loss is flat → enable and re-tune `l2_lambda` for the new meaning.
- [ ] If it moves → `l2_lambda` was doing accidental feature selection through the scale
      disparity. Worth understanding *what* it was selecting before tuning it away.
- [ ] Separately: even after scaling, conditioning lands near 1e6 because
      `striker_score_diff` and `striking_matchup` are near-collinear by construction. Only
      worth chasing if the numbers say it costs something.

### Not an issue after a closer look

`TODO.md` and [`docs/todo.md`](docs/todo.md) were flagged in review as duplicated backlog.
They are not — this file is the roadmap, that one holds phased checklists and column
specs, and each points at the other. The split is deliberate and documented. Noted here so
it does not get "fixed."

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
