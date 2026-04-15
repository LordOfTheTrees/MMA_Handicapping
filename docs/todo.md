# Project TODO
## MMA Pre-Fight Prediction Model — Next Steps

Steps are ordered chronologically. Detail decreases the further out they are.

---

## Phase 1 — Data Acquisition & Ingestion
*Everything downstream is blocked until there is real data in the correct CSV format.*

### 1.1 UFCStats Scraper (Tier 1)

The model needs per-fight striking and grappling stats for every UFC fight from 2013 onward, plus outcome/method data for older fights used in ELO construction.

**Target columns for `data/tier1_ufcstats.csv`:**

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
- [ ] Write or adapt a UFCStats scraper (Python `requests` + `BeautifulSoup`). UFCStats has a consistent URL pattern: `ufcstats.com/fight-details/<fight_id>`.
- [ ] Scrape the full fight listing from `ufcstats.com/statistics/events/completed` to get all fight IDs.
- [ ] Assign stable `fighter_id` values — use the UFCStats fighter URL slug (e.g. `jon-jones-xyz123`) as the canonical ID. This ensures the same fighter maps to the same ID across all data sources.
- [ ] Save scraper output as `data/tier1_ufcstats.csv`.
- [ ] Verify row count: as of 2024 there are approximately 7,000 UFC fights on record.

### 1.2 Fighter Profiles (Physical Attributes)

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

- [ ] Scrape fighter profile pages from UFCStats (the fighter index at `ufcstats.com/statistics/fighters`).
- [ ] Populate pedigree signals manually for debut fighters, or leave at `0.0` and let the ELO and style axes update from real fights. Manual entry can be deferred.

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

Sherdog Fight Finder covers the vast majority of professional MMA fights globally. Key challenge is ID consistency: Sherdog uses its own numeric IDs, UFCStats uses slug-based IDs. A mapping table (`data/id_crosswalk.csv`) linking Sherdog IDs to UFCStats fighter IDs is needed for fighters who appear in both datasets.

- [ ] Download or scrape Sherdog fight records for fighters who appear in the UFC.
- [ ] Build `data/id_crosswalk.csv` with columns `sherdog_id`, `fighter_id` (UFCStats canonical).
- [ ] Add a lookup step in `src/data/loader.py::load_sherdog_fights()` to apply the crosswalk before writing `fighter_a_id` / `fighter_b_id`.

---

## Phase 2 — Pipeline Smoke Test
*Run the full pipeline on real data for the first time. Goal: no crashes, plausible outputs.*

### 2.1 End-to-End Run

With `data/tier1_ufcstats.csv` and `data/fighter_profiles.csv` in place:

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

Swap A and B and confirm win probabilities become lose probabilities:

```python
from src.pipeline import MMAPredictor
from src.data.schema import WeightClass
from datetime import date

p = MMAPredictor.load("model.pkl")
r1 = p.predict("fighter_a", "fighter_b", WeightClass.LIGHTWEIGHT, date(2024, 6, 1), verbose=False)
r2 = p.predict("fighter_b", "fighter_a", WeightClass.LIGHTWEIGHT, date(2024, 6, 1), verbose=False)

# Should be near-zero
assert abs(r1.p_win_ko_tko   - r2.p_lose_ko_tko)   < 1e-6
assert abs(r1.p_win_decision  - r2.p_lose_decision) < 1e-6
```

- [ ] Add this as a unit test in `tests/test_symmetry.py`.

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
