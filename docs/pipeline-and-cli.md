# Pipeline and CLI reference

This document describes how to invoke the **CLI** (`main.py`, **`python -m src.cli.train`**, and other **`python -m src.cli.*`** modules), the **programmatic** [`MMAPredictor`](../src/pipeline.py) API, and related **arguments**. For architecture and design, see [`architecture.md`](architecture.md). Tunables live in [`src/config.py`](../src/config.py).

---

## 1. Invocation pattern

The primary entry point is **`python main.py`** with a **required subcommand**:

```text
python main.py [GLOBAL_OPTIONS] <subcommand> [SUBCOMMAND_OPTIONS] [POSITIONAL_ARGS]
```

### Global option

| Argument | Default | Meaning |
|----------|---------|---------|
| `--model-path PATH` | `./model.pkl` | **`train`:** destination pickle after fit. **`predict`**, **`explain`**, **`eval-holdout`**, **`predict-human`:** model to load. |

Subcommands below can override inference paths where noted.

---

## 2. Subcommands (`main.py`)

### 2.1 `train`

Loads CSVs from `--data-dir`, builds or loads ELO, fits multinomial regression (with optional holdout), and **saves** the pickle to **`--model-path`** (global flag on `main.py`). Same behavior as **`python -m src.cli.train`** (see [§3](#3-dedicated-train-module-python--m-srcclitrain)).

| Argument | Meaning |
|----------|---------|
| `--data-dir DIR` | Root directory for `ufcstats_fights.csv` (or legacy `tier1_ufcstats.csv`), optional tier2/3 CSVs, `fighter_profiles.csv`. Default `./data`. |
| `--full-rebuild` | If set with default scrape path: runs `refresh_data()` to re-fetch CSVs (see `--no-scrape` / `--skip-refresh-if-present`). |
| `--no-scrape` | With `--full-rebuild`: skip network refresh; use existing UFCStats CSVs in `--data-dir`. |
| `--skip-refresh-if-present` | With `--full-rebuild`: if `ufcstats_fights.csv` already exists, skip refresh. |
| `--elo-cache PATH` | After ELO build (or if cache valid): read/write a pickle so the next run can **skip ELO rebuild** when fight count and [`ELOConfig`](../src/config.py) match. |
| `--no-holdout` | Train regression on **all** Tier‑1 post-era rows (no date cut). Use for production refits; default path reserves a holdout for evaluation. |
| `--holdout-start YYYY-MM-DD` | Exclude Tier‑1 fights with `fight_date >=` this date from **regression** fitting. ELO still uses full history. Overrides default `Config.holdout_start_date` (default `2023-01-01`). |

**Stages (conceptual):** `load_data` → `build_elo` (or cache) → `train_regression` → [`save`](../src/pipeline.py).

---

### 2.2 `eval-holdout`

Scores the model on **Tier‑1 decisive fights** with `fight_date >= config.holdout_start_date` (fighter A perspective). Requires a model trained **with** a non-`None` holdout.

| Argument | Meaning |
|----------|---------|
| `--model-path PATH` (subparser) | If set, overrides the **global** `--model-path` for this command only. |

Output includes mean log-loss, Brier, accuracy, and a block comparing metrics to **uniform 6-class** and **binary coin-flip** baselines (see [`holdout_metrics`](../src/eval/holdout_metrics.py)).

---

### 2.3 `predict`

Loads the model and calls [`MMAPredictor.predict()`](../src/pipeline.py) once (full table + CIs when applicable).

| Positional | Meaning |
|------------|---------|
| `fighter_a` | Fighter **A** `fighter_id` (must match CSV / profile IDs). Probabilities are from **A’s perspective** (win KO/sub/dec vs lose …). |
| `fighter_b` | Fighter **B** `fighter_id`. |
| `weight_class` | Weight class name or alias (see [§5](#5-weight-class-aliases)). |

| Option | Default | Meaning |
|--------|---------|---------|
| `--date YYYY-MM-DD` | Today | Fight **calendar** date used for point-in-time ELO and style axes. |

---

### 2.4 `explain`

Same positional arguments as `predict`; calls [`MMAPredictor.explain()`](../src/pipeline.py): prints **exact** per-feature log-odds contributions per outcome class (no intervals).

---

### 2.5 `predict-human`

Interactive prediction by **display name** from `fighter_profiles.csv` (embedded in the pickle). Resolves names (exact → **fuzzy** via `difflib` if needed), optionally picks among **past fights** between the two fighters in loaded data, then runs `predict` or `explain`.

| Positional | Meaning |
|------------|---------|
| `NAME_A` | Optional. First fighter (corner **A**). If omitted, you are prompted. |
| `NAME_B` | Optional. Second fighter (corner **B**). |

| Option | Meaning |
|--------|---------|
| `--explain` | Use `explain` instead of `predict`. |
| `--model-path PATH` | Model pickle (overrides global `--model-path` if given). |
| `--date`, `--weight-class` | **Hypothetical** matchup when **no** fight exists in loaded data for the pair—or with `--force-context` to ignore DB fights (both must be supplied for hypothetical shortcut). |
| `--force-context` | Use `--date` and `--weight-class` **even if** fights exist in DB (scores a hypothetical at that weight/date). |

**Semantics:** Corner order is **your** order (first name = first argument to `predict`). Scraped CSV rows use lexicographic **sorted** fighter IDs for A/B positions; interactive mode follows **human** ordering for readability.

---

## 3. Dedicated train module (`python -m src.cli.train`)

**Same logic** as `main.py train`; the module’s parser includes **`--model-path`** directly (defaults to `model.pkl`), equivalent to the global `--model-path` on `main.py`.

```text
python -m src.cli.train --data-dir ./data --model-path ./out/model.pkl [TRAIN_OPTIONS...]
```

`TRAIN_OPTIONS` are the same flags as [`register_train_arguments`](../src/cli/train.py): `--full-rebuild`, `--elo-cache`, `--holdout-start`, `--no-holdout`, etc.

---

## 4. Other CLIs (overview)

Implementations live under [`src/cli/`](../src/cli/). From the repo root, run with **`python -m src.cli.<module>`** so `src` resolves as a package (`PYTHONPATH`/cwd is typically the repo root).

| Module | Role |
|--------|------|
| [`src/cli/run_phase3_tuning.py`](../src/cli/run_phase3_tuning.py) | Phase‑3 walk‑forward / pristine evaluation CSV+JSON (`docs/hyperparameter-tuning.md`). |
| [`src/cli/plot_prediction_three_viz.py`](../src/cli/plot_prediction_three_viz.py) | Split-barrier and related single-fight prediction figures; **percent labels are whole numbers** (see ADR-22). |
| [`src/cli/plot_training_feature_histograms.py`](../src/cli/plot_training_feature_histograms.py) | Builds the training matrix (`train_regression(fit_model=False)`) and writes per-feature PNG histograms. |
| [`scripts/pilot_lbfgs_stopping.py`](../scripts/pilot_lbfgs_stopping.py) | Experiments L-BFGS-B stopping tolerances on the training matrix. |
| [`scripts/phase2_smoke.py`](../scripts/phase2_smoke.py) | Phase‑2 smoke checks. |
| [`src/cli/chart_elo_trajectory.py`](../src/cli/chart_elo_trajectory.py), [`src/cli/chart_elo_distributions.py`](../src/cli/chart_elo_distributions.py) | ELO visualization helpers. |

---

## 5. Weight class aliases

Resolved in [`src/cli/common.py`](../src/cli/common.py) (`resolve_weight_class`). Examples (non-exhaustive; see source for full list):

- `lightweight`, `light`, `welterweight`, `welter`, `middleweight`, `middle`, `lhw`, `light_heavyweight`, `heavyweight`, `heavy`, `hw`
- Women’s: `w_strawweight`, `w_flyweight`, `w_bantamweight`, `w_featherweight` (short forms exist)

Normalization: lowercased; hyphens and spaces mapped to underscores.

---

## 6. Programmatic API: [`MMAPredictor`](../src/pipeline.py)

Typical notebook or script flow:

```python
from pathlib import Path
from datetime import date
from src.config import Config
from src.pipeline import MMAPredictor
from src.data.schema import WeightClass

cfg = Config()  # or customize holdout_start_date, model.l2_lambda, ...
p = MMAPredictor(cfg)
p.load_data(Path("./data"))
p.build_elo()
p.train_regression()
p.save(Path("model.pkl"))
```

### 6.1 Construction and config

| Member / call | Meaning |
|---------------|---------|
| `MMAPredictor(config=None)` | `config` defaults to `Config()` from [`src/config.py`](../src/config.py): ELO / features / regression / `master_start_year`, `holdout_start_date`, L-BFGS and bootstrap knobs, etc. |

### 6.2 Data and ELO

| Method | Arguments | Meaning |
|--------|-----------|---------|
| `load_data(data_dir)` | `Path` | Loads fights + profiles from CSVs under `data_dir`. |
| `load_fights_direct(fights, profiles=None)` | Lists / dict | In-memory fights (sorted); optional profiles dict for tests. |
| `try_load_elo_from_cache(path)` | `Path` | Loads cached ELO if fight count + ELO hash match; returns bool. |
| `build_elo(...)` | `elo_progress_every=1000`, `record_trajectories=False` | Fits ELO over `self.fights`. Trajectories for charting optional. |
| `save_elo_cache(path)` | `Path` | Writes cache after `build_elo`. |

### 6.3 Training regression

| Method | Arguments | Meaning |
|--------|-----------|---------|
| `train_regression(matrix_progress_every=500, fit_model=True)` | | Builds `X`,`y`,`weights`; if `fit_model`, runs L‑BFGS and optional weighted bootstrap CI storage. **`fit_model=False`** builds matrices only (diagnostics). |
| *(after train)* `training_regression_audit` | | Optional dict with coefficient importance (raw + std-scaled blocks when training matrix present). |

### 6.4 Style axes (advanced)

| Method | Meaning |
|--------|---------|
| `get_style_axes(fighter_id, wc, as_of_date)` | ELO‑weighted style axes strictly before `as_of_date`. |

### 6.5 Prediction

| Method | Arguments | Meaning |
|--------|-----------|---------|
| `predict_proba_point_only(a_id, b_id, wc, fight_date)` | | `(6,)` probs only—no bootstrap / Cauchy overhead; good for scoring loops. |
| `predict(a_id, b_id, wc, fight_date, verbose=True)` | | Full **PredictionResult** with CIs (bootstrap stack, Cauchy, or legacy path depending on ESS and pickles). |
| `explain(a_id, b_id, wc, fight_date)` | | Prints decomposition to stdout. |

Swapping **A** and **B** flips the six-way distribution between win/loss sides (see architecture notes on symmetry).

### 6.6 Persistence

| Method | Meaning |
|--------|---------|
| `save(path)` | Pickle entire predictor (`fights`, `profiles`, `config`, regression weights, optional `_bootstrap_W`, etc.). |
| `MMAPredictor.load(path)` | Classmethod; restores; older pickles get field defaults via [`__setstate__`](../src/pipeline.py). |

---

## 7. Fighter name helpers (library use)

[`src/data/fighter_names.py`](../src/data/fighter_names.py) exposes **exact**, case-insensitive matching from profile `name` to `fighter_id`:

- `fighter_ids_for_exact_name(name, profiles) -> List[str]` (0, 1, or many IDs)
- `resolve_fighter_id`, `require_fighter_id` — convenience when uniquely resolvable.

`predict-human` adds **fuzzy** matching on top; programmatic callers can replicate or call into the same `_fuzzy_candidates` pattern.

---

## 8. Quick command cheat sheet

```text
# Train
python -m src.cli.train --data-dir ./data --model-path ./data/model.pkl --elo-cache ./data/elo_cache.pkl

# Holdout metrics
python main.py --model-path ./data/model.pkl eval-holdout

# IDS
python main.py --model-path ./data/model.pkl predict <id_a> <id_b> lightweight --date 2024-06-01
python main.py --model-path ./data/model.pkl explain <id_a> <id_b> lightweight --date 2024-06-01

# Names (interactive)
python main.py --model-path ./data/model.pkl predict-human --explain "Sean O'Malley" "Petr Yan"
```
