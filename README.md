# MMA Handicapping

Pre-fight model for **UFC-style matchups**: a calibrated **six-outcome** distribution (win/lose × KO-TKO / submission / decision), built from historical fight data, **ELO**, **style axes**, and **interpretable** matchup features—with **uncertainty** (confidence intervals), not just point estimates.

Design detail lives in [`docs/architecture.md`](docs/architecture.md). **CLI flags, scripts, and the `MMAPredictor` programmatic API** are documented in **[`docs/pipeline-and-cli.md`](docs/pipeline-and-cli.md)** (§9 = JSON export scripts for **`mma.ai`**). Roadmap and data tasks: [`TODO.md`](TODO.md) and [`docs/todo.md`](docs/todo.md).

**Production site (OctagonELO)** ships from sibling repo **`mma.ai`**. See **[Website export](#website-export-mmaai)** below for copy-paste export commands and [`docs/BACKEND_PIPELINE_INTEGRATION.md`](docs/BACKEND_PIPELINE_INTEGRATION.md) for integration notes.

---

## Approach (short)

1. **Data tiers** — Tier 1 (UFCStats) supplies full per-fight stats for the modern era; optional tiers 2–3 extend ELO with other promotions / regional records. Everything is CSV-backed and loaded into typed records in `src/data/`.

2. **ELO** — Per **weight class**, with K scaled by **result certainty**. ELO both **weights** past performance when building style features and enters the model as a **direct** quality-gap feature (`elo_differential`).

3. **Style axes** — ELO-weighted, recency-aware summaries (striker/grappler, finish threat, etc.) feed **matchup interaction** terms so the feature vector respects **A vs B** symmetry.

4. **Outcome model** — **Multinomial logistic regression** over the six labels; **confidence intervals** (bootstrap draws at train time, Cauchy fallbacks where appropriate).

5. **Constraints** — **Pre-fight only**. **Interpretability** is a stated goal (`explain`, coefficient reporting).

---

## Repository layout

| Path | Role |
|------|------|
| [`main.py`](main.py) | CLI: **`train`**, **`predict`**, **`explain`**, **`eval-holdout`**, **`predict-human`** |
| [`src/pipeline.py`](src/pipeline.py) | `MMAPredictor`: load data → ELO → features → train → predict |
| [`src/cli/train.py`](src/cli/train.py) | Dedicated train entrypoint: **`python -m src.cli.train`** (same flags as `main.py train` + top-level `--model-path`) |
| `src/data/` | Schemas, loaders, UFCStats scrapers, `refresh_data` |
| `src/elo/`, `src/features/`, `src/matchup/`, `src/model/`, `src/confidence/` | Stages |
| [`scripts/`](scripts/) | **`run_harness.py`** wraps unittest (`quick` / `integration` / full `discover`). Also **JSON export for `mma.ai`** (`export_artifacts.py`, `export_upcoming_events.py`, `copy_exports_to_mma_ai.py` — see [Website export](#website-export-mmaai)); core model CLIs live under **`src/cli/`** |
| [`src/cli/plot_prediction_three_viz.py`](src/cli/plot_prediction_three_viz.py) | Optional **split-barrier** PNG for a fight; chart copy uses **integer %** (ADR-22) |
| `data/` | Local CSVs and artifacts (gitignored where appropriate; see `.gitignore`) |

---

## Quick usage

```bash
pip install -r requirements.txt

# Train (writes model pickle)
python -m src.cli.train --data-dir ./data --model-path ./data/model.pkl

# Inference by UFCStats fighter_id
python main.py --model-path ./data/model.pkl predict <fighter_a_id> <fighter_b_id> lightweight --date 2024-06-01

# Interactive: names → fuzzy lookup → optional fight picker
python main.py --model-path ./data/model.pkl predict-human "Fighter One" "Fighter Two"

# Score held-out Tier‑1 fights (needs model trained with a time holdout)
python main.py --model-path ./data/model.pkl eval-holdout
```

Weight-class aliases (`lhw`, `w_flyweight`, …), every train flag, and the full **Python API** (`MMAPredictor.load`, `predict`, `train_regression`, …) are listed in **`docs/pipeline-and-cli.md`** (**§9** references the site JSON scripts). Copy-paste **`mma.ai` export**: **[Website export](#website-export-mmaai)**.

---

## Data and scraping

Expected inputs under `data/` (see [`src/pipeline.py`](src/pipeline.py) / [`src/data/loader.py`](src/data/loader.py)):

- **`ufcstats_fights.csv`** — scraped Tier‑1 UFC fights (`scrape_ufcstats_fights_to_csv` in `src/data/ufcstats_scraper.py`). Loader also accepts legacy **`tier1_ufcstats.csv`** if the new file is missing.
- **`fighter_profiles.csv`** — same `fighter_id` keys: names, dimensions, stance, pedigree columns (`src/data/ufcstats_profiles.py`).

Commands (network refresh; multi-hour runs are normal):

```bash
python -m src.data.ufcstats_scraper --data-dir ./data
python -m src.data.ufcstats_profiles --data-dir ./data
```

`python main.py train --data-dir ./data --full-rebuild` runs `refresh_data()` (Tier‑1 fights CSV, fighter profiles, and **`data/upcoming_cards.json`**) before training unless you bypass with **`--no-scrape`** / **`--skip-refresh-if-present`**. The upcoming file is **for the website export path only** and is **not** read when fitting the model (see **ADR-23** in [`docs/architecture-decisions.md`](docs/architecture-decisions.md)). Timing, **`failed_entries.csv`**, and scrape caps: [`TODO.md`](TODO.md), [`docs/todo.md`](docs/todo.md).

---

## Test harness (`scripts/run_harness.py`)

Wraps **`unittest`** so you don’t need discovery flags. Repo root = directory with **`main.py`**.

```bash
python scripts/run_harness.py                     # all modules under tests/
python scripts/run_harness.py quick               # offline only (no model.pkl)
python scripts/run_harness.py integration         # export smoke + pickle vs JSON parity
python scripts/run_harness.py integration --model path/to/model.pkl
python scripts/run_harness.py -q integration      # quieter (no -v)
```

Model lookup for **`integration`** matches **`tests/harness_skip.py`**: **`MMA_HARNESS_MODEL`**, then **`data/model.pkl`**, then **`tests/fixtures/parity/model.pkl`**. More detail: **`docs/BACKEND_PIPELINE_INTEGRATION.md`** (Harness).

---

## Website export (mma.ai)

Portable **JSON** snapshots for the sibling deploy repo **`mma.ai`** (OctagonELO). Run everything from **this repo root** (`MMA_Handicapping/`).

**Prerequisites**

- Trained pickle: e.g. **`data/model.pkl`** (from `python -m src.cli.train ...` — see [Quick usage](#quick-usage)).
- Optional: **`data/upcoming_cards.json`** — created by a full data refresh or by the upcoming scraper below.
- Clone layout: default copy targets assume **`mma.ai`** sits **next to** this repo (same parent folder), e.g. `Personal Coding/MMA_Handicapping` and `Personal Coding/mma.ai`. Override paths with flags if yours differs.
- **`JSON_exports/`** — recommended staging folder for the five **`*.json`** files before or after syncing to **`mma.ai`**. It is **not** listed in `.gitignore` so you **can commit** snapshots if desired (unlike **`data/`** and **`*.pkl`**).

**Artifacts**

| Output (this repo) | Role |
|--------------------|------|
| **`JSON_exports/model_weights.json`** | Regression **W**, bootstrap draws, inference config slice |
| **`JSON_exports/elo_states.json`** | Per fighter × division ELO snapshot |
| **`JSON_exports/style_axes.json`** | Style axes snapshot |
| **`JSON_exports/fighter_profiles.json`** | Names, reach, stance, pedigree, … |
| **`JSON_exports/upcoming_events.json`** | Scheduled cards (after `export_upcoming_events.py`) |

Production loads these from **`mma.ai/artifacts/`** (same filenames). Do **not** ship **`model.pkl`** or raw CSVs to the web repo.

### One-time / repeat: full manual sequence

From a Unix-style shell or **PowerShell** (`cd` quoted paths with spaces). Repo root = directory that contains **`main.py`** and **`scripts/`**.

```bash
# 0) Repo root, deps
cd /path/to/MMA_Handicapping
pip install -r requirements.txt

# 1) (Optional) Refresh UFCStats data + upcoming card listing under data/
python -m src.data.ufcstats_scraper --data-dir ./data
python -m src.data.ufcstats_profiles --data-dir ./data
python -m src.data.ufcstats_upcoming --data-dir ./data
# Or use train --full-rebuild, which runs refresh_data() (fights + profiles + upcoming_cards.json).

# 2) Export the four inference JSON files from your pickle
python scripts/export_artifacts.py --model-path ./data/model.pkl --out-dir ./JSON_exports

# 3) Build deploy upcoming JSON (requires data/upcoming_cards.json from step 1)
python scripts/export_upcoming_events.py --cards ./data/upcoming_cards.json --out ./JSON_exports/upcoming_events.json

# 4) Copy all JSON_exports/*.json into sibling mma.ai/artifacts/
python scripts/copy_exports_to_mma_ai.py --src ./JSON_exports
```

### Shorter path: copy while exporting

Append **`--copy-to-mma-ai`** to push straight to **`../mma.ai/artifacts`** (still writes local files first where applicable):

```bash
python scripts/export_artifacts.py --model-path ./data/model.pkl --out-dir ./JSON_exports --copy-to-mma-ai
python scripts/export_upcoming_events.py --cards ./data/upcoming_cards.json --out ./JSON_exports/upcoming_events.json --copy-to-mma-ai
```

If **`mma.ai`** lives elsewhere:

```bash
python scripts/copy_exports_to_mma_ai.py --src ./JSON_exports --dest /path/to/mma.ai/artifacts
python scripts/export_artifacts.py --model-path ./data/model.pkl --out-dir ./JSON_exports --copy-to-mma-ai --mma-ai-artifacts-dir /path/to/mma.ai/artifacts
```

Optional: pin ELO/style **`as_of_date`** on export:  
`python scripts/export_artifacts.py ... --as-of-date 2026-05-01`

Contract details for **`mma.ai`** consumers: **`docs/BACKEND_PIPELINE_INTEGRATION.md`** and **`mma.ai/docs/`**.

---

## Requirements

- Python **3.10+** recommended.
- `pip install -r requirements.txt` (scraping: `curl_cffi`, `beautifulsoup4`; modeling: `numpy`, `scipy`; plots: optional `matplotlib` for scripts).

---

## License

See [`LICENSE`](LICENSE).
