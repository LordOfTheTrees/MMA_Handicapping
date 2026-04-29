# MMA Handicapping

Pre-fight model for **UFC-style matchups**: a calibrated **six-outcome** distribution (win/lose × KO-TKO / submission / decision), built from historical fight data, **ELO**, **style axes**, and **interpretable** matchup features—with **uncertainty** (confidence intervals), not just point estimates.

Design detail lives in [`docs/architecture.md`](docs/architecture.md). **CLI flags, scripts, and the `MMAPredictor` programmatic API** are documented in **[`docs/pipeline-and-cli.md`](docs/pipeline-and-cli.md)**. Roadmap and data tasks: [`TODO.md`](TODO.md) and [`docs/todo.md`](docs/todo.md).

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
| [`scripts/train_model.py`](scripts/train_model.py) | Same train flow as `main.py train` with explicit **`--model-path`** on the script |
| `src/data/` | Schemas, loaders, UFCStats scrapers, `refresh_data` |
| `src/elo/`, `src/features/`, `src/matchup/`, `src/model/`, `src/confidence/` | Stages |
| [`scripts/`](scripts/) | Auxiliaries (Phase‑3 tuning, ELO plots, training-feature histograms, etc.) |
| `data/` | Local CSVs and artifacts (gitignored where appropriate; see `.gitignore`) |

---

## Quick usage

```bash
pip install -r requirements.txt

# Train (writes model pickle)
python scripts/train_model.py --data-dir ./data --model-path ./data/model.pkl

# Inference by UFCStats fighter_id
python main.py --model-path ./data/model.pkl predict <fighter_a_id> <fighter_b_id> lightweight --date 2024-06-01

# Interactive: names → fuzzy lookup → optional fight picker
python main.py --model-path ./data/model.pkl predict-human "Fighter One" "Fighter Two"

# Score held-out Tier‑1 fights (needs model trained with a time holdout)
python main.py --model-path ./data/model.pkl eval-holdout
```

Weight-class aliases (`lhw`, `w_flyweight`, …), every train flag, and the full **Python API** (`MMAPredictor.load`, `predict`, `train_regression`, …) are listed in **`docs/pipeline-and-cli.md`**.

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

`python main.py train --data-dir ./data --full-rebuild` runs `refresh_data()` (scrapes fights then profiles) before training unless you bypass with **`--no-scrape`** / **`--skip-refresh-if-present`**. Timing and **`failed_entries.csv`** are discussed in **`TODO.md`** and `docs/todo.md`.

---

## Requirements

- Python **3.10+** recommended.
- `pip install -r requirements.txt` (scraping: `curl_cffi`, `beautifulsoup4`; modeling: `numpy`, `scipy`; plots: optional `matplotlib` for scripts).

---

## License

See [`LICENSE`](LICENSE).
