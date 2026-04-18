# MMA Handicapping

Pre-fight model for **UFC-style matchups**: a calibrated **six-outcome** distribution (win/lose × KO-TKO / submission / decision), built from historical fight data, **ELO**, **style axes**, and **interpretable** matchup features—with **uncertainty** (confidence intervals), not just point estimates.

Design detail lives in [`docs/architecture.md`](docs/architecture.md). Roadmap and data tasks: [`TODO.md`](TODO.md) and [`docs/todo.md`](docs/todo.md).

---

## Approach (short)

1. **Data tiers** — Tier 1 (UFCStats) supplies full per-fight stats for the modern era; optional tiers 2–3 extend ELO with other promotions / regional records. Everything is CSV-backed and loaded into typed records in `src/data/`.

2. **ELO** — Per **weight class**, with K scaled by **result certainty** (e.g. finish vs. split decision). ELO both **weights** past performance when building style features and enters the model as a **direct** quality gap feature.

3. **Style axes** — ELO-weighted, recency-aware summaries (striker/grappler, finish threat, etc.) feed **matchup interaction** terms so the feature vector respects **A vs B** symmetry.

4. **Outcome model** — **Multinomial logistic regression** over the six labels; **confidence intervals** summarize epistemic uncertainty when data are thin.

5. **Constraints** — **Pre-fight only**: no in-fight or live data. **Interpretability** is a stated goal: coefficients and `explain` should stay human-readable.

---

## Repository layout

| Path | Role |
|------|------|
| `main.py` | CLI: `train`, `predict`, `explain` |
| `src/pipeline.py` | `MMAPredictor`: load data → ELO → features → train → predict |
| `src/data/` | Schemas, loaders, UFCStats scrapers, `refresh_data` |
| `src/elo/`, `src/features/`, `src/matchup/`, `src/model/`, `src/confidence/` | Stages of the pipeline |
| `data/` | Local CSVs and artifacts (gitignored; see `.gitignore`) |

---

## Data and scraping

Expected inputs under `data/` (see `src/pipeline.py` / `src/data/loader.py`):

- **`ufcstats_fights.csv`** — UFCStats fights (`scrape_ufcstats_fights_to_csv` in `src/data/ufcstats_scraper.py`): hex `fight_id` / `fighter_*_id` from site URLs, outcome, weight class, date, sig str / TD / control / subs, etc. The loader also accepts legacy **`tier1_ufcstats.csv`** if the new file is missing.
- **`fighter_profiles.csv`** — Same `fighter_id` as the fights CSV: name, height/reach (cm), DOB, stance, optional pedigree columns (`scrape_fighter_profiles_to_csv` in `ufcstats_profiles.py`).

**Scrapers** (Chrome impersonation via `curl_cffi`, referer chain; fights scraper uses module global `REQUEST_DELAY_SEC`, set by `--sleep` in `ufcstats_scraper.main`, default **0.1** s):

```bash
pip install -r requirements.txt

# UFCStats fights: completed-events index (?page=all) → each event → each fight-details page
python -m src.data.ufcstats_scraper --data-dir ./data

# Profiles: unique fighter IDs from the fights CSV → /fighter-details/<id>
python -m src.data.ufcstats_profiles --data-dir ./data
```

**How long:** A **full fights scrape** walks on the order of **~770 completed events** and **thousands of fight pages**. With the default delay, expect **several hours** end to end depending on network and UFCStats response times. Fighter profiles are **one request per distinct ID**—usually shorter than the fights pass, still plan for a long run if you have many fighters. **`failed_entries.csv`** records fight pages that did not ingest; **`python -m src.data.ufcstats_gap_report`** diffs the site vs your CSV (see `TODO.md` §F).

**Outcomes in `ufcstats_fights.csv`:** The parser maps UFCStats quirks into the loader’s `method` strings: both banners **D** → `draw` (even when the method line still says a decision type); both **NC** → `no contest` (including when the method line says **Could Not Continue**); disqualification text → `dq`. After any parser fix, re-run `ufcstats_scraper` so rows that previously failed to parse (e.g. NCs) are written.

`python main.py train --data-dir ./data --full-rebuild` runs `refresh_data()` in `src/data/refresh.py`, which **re-scrapes fights then profiles** in sequence—treat it like an overnight or background job.

---

## Train, predict, explain

```bash
# Train from existing CSVs (writes model.pkl by default)
python main.py train --data-dir ./data

# Predict / explain: use real fighter_id strings from your CSVs
python main.py predict <fighter_a_id> <fighter_b_id> lightweight
python main.py explain <fighter_a_id> <fighter_b_id> lightweight --date YYYY-MM-DD
```

Weight-class aliases include `lhw`, `w_flyweight`, etc. (see `main.py` help).

---

## Requirements

- Python 3.10+ recommended.
- Install: `pip install -r requirements.txt` (scraping: `curl_cffi`, `beautifulsoup4`; training: `numpy`, `scipy`).

---

## License

See [`LICENSE`](LICENSE).
