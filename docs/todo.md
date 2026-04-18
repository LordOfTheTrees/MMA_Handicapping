# Project TODO
## MMA Pre-Fight Prediction Model — Next Steps

Steps are ordered chronologically. Detail decreases the further out they are.

### Roadmap vs this doc

- **[`TODO.md`](../TODO.md)** — **Current status** (what is implemented), **next work bout** (ordered immediate steps), and the UFCStats scrape skip / `ufcstats_gap_report` log. Read that first for “where we are / what to do next.”
- **This file** — Column specs, phased checklists (Phase 1–5), and deeper validation / tuning notes.

**Gate:** Phase 2 (below) is the next **milestone** once refreshed `ufcstats_fights.csv` (or legacy `tier1_ufcstats.csv`) and `fighter_profiles.csv` are in `data/`. Phase 3+ assumes Phase 2 smoke tests pass.

---

## Phase 1 — Data Acquisition & Ingestion
*Primary UFC CSVs are produced by the in-repo scrapers; remaining work is refresh cadence, gap checks, and optional tiers 2–3.*

### 1.1 UFCStats fights scraper (`scrape_ufcstats_fights_to_csv`)

**Status:** Implemented in [`src/data/ufcstats_scraper.py`](../src/data/ufcstats_scraper.py) (`scrape_ufcstats_fights_to_csv`, `parse_fight_page`, …). Discovery uses the completed-events page with `?page=all` (~770 events), then each event page and each `fight-details` URL. HTTP uses `curl_cffi` with Chrome impersonation and a referer chain.

**Runtime:** A full scrape issues thousands of requests; spacing is **`REQUEST_DELAY_SEC`** in that module (CLI `--sleep` in `main()`, default **0.2 s** in current code). Plan for **several hours** wall time depending on network. Fighter profiles (`scrape_fighter_profiles_to_csv` in `ufcstats_profiles.py`) are a separate pass over **unique fighter IDs** from the fights CSV.

**Outcome normalization (UFCStats quirks):**
- Both fighter banners **D** → `method` = `draw` (method line may still read like a decision).
- Both banners **NC**, or method text such as **Could Not Continue** / **No Contest** → `no contest`; `winner_id` blank.
- Disqualification wording → `dq` (winner inferred from W/L when the site shows one fighter **L**).

The model needs per-fight striking and grappling stats for every UFC fight from 2013 onward, plus outcome/method data for older fights used in ELO construction.

**Target columns for `data/ufcstats_fights.csv`** (pipeline also accepts legacy `tier1_ufcstats.csv`):

| Column | Description |
|---|---|
| `fight_id` | Unique fight identifier (e.g. UFCStats URL slug) |
| `fighter_a_id` | Consistent fighter ID (same fighter = same ID across all CSVs) |
| `fighter_b_id` | — |
| `winner_id` | fighter_a_id or fighter_b_id; blank for draw/NC/DQ |
| `method` | One of: `ko/tko`, `submission`, `unanimous decision`, `split decision`, `majority decision`, `draw`, `no contest`, `dq` |
| `weight_class` | One of the strings in `src/data/loader.py WEIGHT_CLASS_MAP` |
| `date` | `YYYY-MM-DD` |
| `fight_time_sec` | Total fight time in seconds |
| `a_sig_str_landed` | Fighter A significant strikes landed |
| `a_sig_str_attempted` | — |
| `a_sig_str_absorbed` | Fighter A sig strikes absorbed (= B's landed) |
| `a_td_landed` | Takedowns landed |
| `a_td_attempted` | — |
| `a_ctrl_time_sec` | Control time in seconds |
| `a_sub_attempts` | Submission attempts |
| *(same b_ columns for Fighter B)* | — |

**Action items:**
- [x] UFCStats scraper (`curl_cffi` + BeautifulSoup). URL pattern: `ufcstats.com/fight-details/<fight_id>` (hex id).
- [x] Full listing from `ufcstats.com/statistics/events/completed?page=all` → per-event fight links.
- [x] Stable `fighter_id`: hex segment from `/fighter-details/<id>` (same id in fights CSV and profiles).
- [x] Scraper writes `data/ufcstats_fights.csv` by default (via `--data-dir` or `--out`); failures logged to `failed_entries.csv`.
- [x] Row count: full runs are on the order of **8k+** fights; exact count grows with new events.

**Skip investigation**  
Skipped fights never appear in the CSV. Use [`TODO.md`](../TODO.md) §F and `python -m src.data.ufcstats_gap_report` with optional cached event inventory (`tier1_inventory_io`). Early runs saw hundreds of skips; extending `_normalize_method` / weight-class mapping in `ufcstats_scraper.py` fixes most systematic gaps (re-scrape after parser changes).

### 1.2 Fighter Profiles (Physical Attributes)

**Status:** [`src/data/ufcstats_profiles.py`](../src/data/ufcstats_profiles.py) (`scrape_fighter_profiles_to_csv`) reads fighter IDs from `ufcstats_fights.csv` (or legacy `tier1_ufcstats.csv` when using `--data-dir`) and writes `fighter_profiles.csv`. Run after the fights CSV is updated.

UFCStats also has reach, height, date of birth, and stance per fighter.

**Target columns for `data/fighter_profiles.csv`:**

| Column | Notes |
|---|---|
| `fighter_id` | Must match IDs used in fight CSVs |
| `name` | Human-readable display name |
| `reach_cm` | Reach in centimetres (convert from inches if needed) |
| `height_cm` | Height in centimetres |
| `date_of_birth` | `YYYY-MM-DD` |
| `stance` | `orthodox`, `southpaw`, or `switch` |
| `wrestling_pedigree` | 0.0–1.0 signal (see below) |
| `boxing_pedigree` | 0.0–1.0 signal |
| `bjj_pedigree` | 0.0–1.0 signal |

**Pedigree signal encoding (initial approach — refine later):**
- `1.0` = elite collegiate or Olympic-level (D1 All-American wrestler, professional boxer, black belt BJJ competitor)
- `0.7` = strong regional or D3/NAIA level
- `0.4` = amateur competitive background
- `0.0` = no documented background in that discipline

Pedigree only affects cold-start ELO and style axis priors for fighters with fewer than ~3 observed fights. It has zero influence once real UFC data is available for the fighter.

- [x] Scrape profile pages from UFCStats (`/fighter-details/<id>` for each ID seen in the fights CSV).
- [ ] Populate pedigree signals manually for debut fighters, or leave at `0.0` and let the ELO and style axes update from real fights. Manual entry can be deferred.

**Standalone profile scrape** (IDs derived from the fights CSV; run after `scrape_ufcstats_fights_to_csv` is complete):

```bash
python -m src.data.ufcstats_profiles --data-dir ./data
```

### 1.3 Tier 2 — Major Promotions (Optional but Valuable for Cold Start)

Useful primarily for fighters who debut in the UFC with significant Bellator/ONE/PFL/RIZIN history.

**Target: `data/tier2_bellator.csv`, `data/tier2_one.csv`, etc.**

Same schema as Tier 1 minus the stats columns (outcomes + method + date is sufficient). Tapology is a convenient source for Bellator/ONE results. Sherdog also covers these promotions.

- [ ] Identify source for Bellator, ONE, PFL, RIZIN fight outcomes.
- [ ] Build a scraper or use an existing dataset (Tapology or Sherdog).
- [ ] Cross-reference fighter IDs with UFCStats — fighters who have fought in both need the **same** `fighter_id` string. A fighter name normalisation step is required to handle name variants.

### 1.4 Tier 3 — Sherdog Regional Records

Used for ELO construction only — outcome and method, no stats needed.

**Target: `data/tier3_sherdog.csv`**

Sherdog Fight Finder covers the vast majority of professional MMA fights globally. Key challenge is ID consistency: Sherdog uses its own numeric IDs, UFCStats uses **hex** IDs from profile/fight URLs. A mapping table (`data/id_crosswalk.csv`) linking Sherdog IDs to UFCStats fighter IDs is needed for fighters who appear in both datasets.

- [ ] Download or scrape Sherdog fight records for fighters who appear in the UFC.
- [ ] Build `data/id_crosswalk.csv` with columns `sherdog_id`, `fighter_id` (UFCStats canonical).
- [ ] Add a lookup step in `src/data/loader.py::load_sherdog_fights()` to apply the crosswalk before writing `fighter_a_id` / `fighter_b_id`.

---

## Phase 2 — Pipeline Smoke Test
*Run the full pipeline on real data for the first time. Goal: no crashes, plausible outputs. This is the **next milestone** after UFCStats fights + profiles are refreshed—see **`TODO.md` → Next work bout** for the ordered steps into this phase.*

### 2.1 End-to-End Run

With `data/ufcstats_fights.csv` (or legacy `tier1_ufcstats.csv`) and `data/fighter_profiles.csv` in place:

```bash
python main.py train --data-dir ./data
```

Expected output:
```
Stage 1: Loading data from data/ ...
  7,241 fight records loaded.
  3,892 fighter profiles loaded.
Stage 2: Building ELO model ...
  ELO construction complete.
Stages 3–5: Constructing features and training regression ...
  Regression trained on 4,103 fights.
Saving model → model.pkl
Done.
```

- [ ] Verify training fight count is plausible (post-2013 UFC fights with a decisive result ≈ 4,000–5,000).
- [ ] Verify no `NaN` or `inf` in the feature matrix (`np.isfinite(predictor._X_train).all()`).
- [ ] Run a test prediction on a known recent fight and check that the output probabilities sum to 1.0.

```bash
python main.py predict <fighter_id_a> <fighter_id_b> lightweight --date 2024-06-01
```

### 2.2 Symmetry Check

Swap A and B: mirrored outcome classes should **roughly** align (e.g. A’s win-KO vs B’s lose-KO). **Exact** `1e-6` equality is **not** expected with the current **multiplicative matchup interaction** features (`striking_matchup`, etc.) — see [`validation-and-few-shot.md`](validation-and-few-shot.md).

Use point probabilities only (no bootstrap) and a **relaxed** tolerance, e.g. [`scripts/phase2_smoke.py`](../scripts/phase2_smoke.py), or:

```python
from src.pipeline import MMAPredictor
from src.data.schema import WeightClass
from datetime import date

p = MMAPredictor.load("model.pkl")
wc, fd = WeightClass.LIGHTWEIGHT, date(2024, 6, 1)
p1 = p.predict_proba_point_only("fighter_a", "fighter_b", wc, fd)
p2 = p.predict_proba_point_only("fighter_b", "fighter_a", wc, fd)
assert abs(p1[0] - p2[4]) < 0.12  # example tolerance; tune after validation strategy is fixed
```

- [ ] Add a unit test with the chosen tolerance in `tests/test_symmetry.py`.

### 2.3 Sanity Checks on ELO

- [ ] Print the top-10 and bottom-10 ELO fighters per weight class. Champions and recent title challengers should cluster near the top.
- [ ] Verify Jon Jones, Khabib Nurmagomedov, Israel Adesanya etc. are in the expected ELO neighbourhood for their weight classes.
- [ ] Verify that fighters with no recorded fights are initialised at 1500 ± pedigree boost.

---

## Phase 3 — Holdout Validation
*The model runs. Now check whether it's actually making good predictions.*

### 3.1 Build a Holdout Set

- Reserve UFC fights from a fixed recent window (e.g., 2023–2024) as the holdout. They are excluded from regression training but ELO is still updated through them.
- Add an `era_cutoff_year` parameter to `filter_tier1_post_era()` plus a separate `holdout_start_year` parameter for this purpose.

### 3.2 Accuracy and Calibration

- **Log-loss** on holdout outcomes: primary metric. Measures how well the full probability distribution is calibrated.
- **Class-level accuracy**: what fraction of the most-probable class calls are correct?
- **Calibration plot**: bin predictions by predicted probability; plot predicted vs. observed frequency. A well-calibrated model lies on the diagonal.
- **Brier score**: mean squared error of the predicted probability vector against the one-hot true outcome.

### 3.3 Tune Open Design Parameters Against Holdout

Work through the parameters from architecture Section 10 one at a time, fixing all others:

1. **Era cutoff year** — try 2010, 2012, 2013, 2015. Expect a U-shape: too early includes non-stationary data; too late throws away useful data.
2. **K-factor base** — try 16, 24, 32, 40. Affects how quickly ELO responds to results.
3. **Recency decay rate λ** — try 0.05, 0.10, 0.20. Affects how much recent fights dominate style axes.
4. **Cross-promotion discount** — tune Tier 2 and Tier 3 discounts independently.
5. **Cauchy fallback threshold** — tune against CI coverage: what fraction of true outcomes fall inside the stated CI?

---

## Phase 4 — Model Hardening
*Calibration is acceptable. Now make the system robust and maintainable.*

- Add `tests/` directory with unit tests for ELO update math, feature symmetry, and CI coverage.
- Add a `requirements.txt` with pinned versions of `numpy`, `scipy`.
- Handle weight class changes properly: when a fighter moves weight class, their ELO in the new class should initialise from their prior class ELO with an appropriate discount (architecture deferred this).
- Validate CI coverage empirically: check that 95% CIs contain the true outcome in approximately 95% of holdout fights.
- Add a `--verbose` flag to expose intermediate state (ELO values, raw feature vector) for debugging predictions.

---

## Phase 5 — Ongoing Data Pipeline
*The model is validated. Maintain it going forward.*

- Automate weekly UFC results ingestion after each event.
- Incrementally update ELO after each fight rather than full reprocessing.
- Decide on a regression refitting cadence (e.g. monthly full refit vs. online updates).
- Consider a simple web interface or API endpoint for prediction queries.
