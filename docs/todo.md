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

The regression needs per-fight striking and grappling stats for Tier-1 UFC fights with `year >= Config.master_start_year` (default **2005**) wherever the scraper/loader yields a complete feature row. UFCStats **full totals** coverage is still uneven in early years—expect more skips or thin stats before the ~2010s–2013 window; outcome/method data for older fights continues to support ELO construction.

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

- [ ] Verify training fight count is plausible for your **`master_start_year`** (rough sanity: post-~2013 decisive Tier-1 UFC rows often ≈ 4k–5k; earlier floors need explicit counts and missing-stat checks).
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

### 3.1 Build a Train / Test Split

- **Holdout (evaluation-only):** Reserve UFC fights from a fixed **recent calendar window** (e.g. fights on or after **2023-01-01**, or years **2023–2024**) for scoring only. They are **excluded from regression training**; **ELO** (and style history) should still be built **chronologically** so features at each holdout fight use only past information (same lookahead discipline as today).
- **Implementation:** [`Config.holdout_start_date`](../src/config.py) + filter in [`train_regression`](../src/pipeline.py) (fights with `fight_date >= holdout_start_date` excluded from the multinomial fit). Train CLI: `python main.py train --holdout-start YYYY-MM-DD`. Metrics: `python main.py eval-holdout` (mean log-loss, Brier, accuracy on holdout rows). **`Config.master_start_year`** is the **regression-era floor** (default **2005**); independent of the holdout window.
- **Protocol:** Prefer **time-based** holdout (see [`validation-and-few-shot.md`](validation-and-few-shot.md)). Optional **walk-forward** (expand training end, score the next chunk) for drift.
- **Do not** use IID random splits across fight rows (fighter leakage).

### 3.2 Accuracy and Calibration

- **Log-loss** on holdout outcomes: primary metric. Measures how well the full probability distribution is calibrated.
- **Class-level accuracy**: what fraction of the most-probable class calls are correct?
- **Calibration plot**: bin predictions by predicted probability; plot predicted vs. observed frequency. A well-calibrated model lies on the diagonal.
- **Brier score**: mean squared error of the predicted probability vector against the one-hot true outcome.

### 3.3 Phase 3 tuning inventory (evaluate on holdout)

Tune **one knob at a time** (or small factorial only after baselines), fixing others; compare **holdout log-loss** (primary), Brier, and calibration. Source of truth for semantics: [`src/config.py`](../src/config.py), [`docs/elo-tuning-knobs.md`](elo-tuning-knobs.md), [`src/elo/elo.py`](../src/elo/elo.py) (`_K_SCALE`).

#### Train / era boundary

| Item | Where | Role in Phase 3 |
|------|--------|-----------------|
| **`master_start_year`** (default **2005**) | `Config` | **Regression training floor** — Tier 1 fights before this calendar year are excluded from the **multinomial** fit (and anchors the first expanding walk-forward training year). **Tune on holdout** (e.g. try 2005, 2010, 2013, 2015): too early → non-stationary sport / thin stats; too late → less data. **Single source of truth** — do not duplicate year cutoffs elsewhere. |
| **`holdout_start_date`** | `Config` | **Evaluation-only** from this calendar date onward; excluded from `X_train`. Set via `--holdout-start` on train or in code. |
| **Training row weights** | `train_regression` (recency: `1 / (1 + days_old/365)`) | Implicit **sample weights** on post-era rows; document sensitivity in ablations if needed. |

#### ELO layer (`ELOConfig` + method scale)

| Item | Where | Role in Phase 3 |
|------|--------|-----------------|
| **`k_base`** | `ELOConfig` | Scales ELO step size; tune vs holdout through features + ELO differential. |
| **`logistic_divisor`** | `ELOConfig` | Win-expectancy curve vs rating gap; tune with `k_base`. |
| **`initial_elo`** | `ELOConfig` | Usually fixed at 1500; revisit only if rescaling the whole system. |
| **`tier_discount`** (Tiers 1–4) | `ELOConfig` | Cross-promotion transfer; tune Tier 2/3 (and policy for cold imports). |
| **`kalman_process_noise`** | `ELOConfig` | Layoff → variance → Kalman gain; tune with **`kalman_measurement_noise`**. |
| **`kalman_measurement_noise`** | `ELOConfig` | Damps / amplifies Kalman updates globally. |
| **`_K_SCALE`** (KO/sub/decision multipliers) | `elo.py` | Method-of-finish scaling on `k_base`; tune if finish vs decision balance matters on holdout. |

#### Feature construction (`FeatureConfig`)

| Item | Where | Role in Phase 3 |
|------|--------|-----------------|
| **`recency_decay_rate`** (λ) | `FeatureConfig` | Style-axis recency decay; try e.g. 0.05, 0.10, 0.20. |
| **`min_fights_style_estimate`** | `FeatureConfig` | Cold-start vs UFC-data trust for style axes. |

#### Multinomial / CI (`ModelConfig`)

| Item | Where | Role in Phase 3 |
|------|--------|-----------------|
| **`l2_lambda`** | `ModelConfig` | Coefficient shrinkage; affects generalization. |
| **`huber_delta`** | `ModelConfig` | Robust loss tail; outlier sensitivity. |
| **`n_bootstrap`** | `ModelConfig` | Count of stored bootstrap draws (train cost vs CI stability). |
| **`bootstrap_seed`** | `ModelConfig` | Reproducibility of bootstrap draws. |
| **`ci_alpha`** | `ModelConfig` | CI width (default **0.10** → 90% two-sided). |
| **`cauchy_fallback_threshold`** | `ModelConfig` | ESS below this → Cauchy CIs; tune vs empirical CI coverage on holdout. |
| **`cauchy_scale`** | `ModelConfig` | Width of Cauchy fallback intervals (probability-level Cauchy, not ELO MC). |
| **`elo_mc_n_draws`** | `ModelConfig` | Draw count for **Cauchy ELO Monte Carlo** at `predict` (nested with stored bootstrap `W` when available). |
| **`elo_mc_gamma_min`** | `ModelConfig` | **γ** floor (ELO points) for `Cauchy(0, γ)` shocks at zero idle; see `elo_mc_gamma_for_days_idle`. |
| **`elo_mc_gamma_slope_sqrt_year`** | `ModelConfig` | **γ** growth with `sqrt(idle_years)`; larger → wider MC for long layoffs. |
| **`elo_mc_gamma_max`** | `ModelConfig` | Cap on **γ** per corner so extreme idle does not explode sampling. |

**Legacy note:** Section 3.3 previously listed a shorter subset; the table above is the **authoritative Phase 3 checklist** (`master_start_year`, all ELO levers above, train/test split parameters once implemented, thresholds and weights).

### 3.4 Iterative tuning loop (model generations)

Phase 3 tuning is **not** a one-shot train: you run **repeated model generations** — each generation is a **full train** on the same data snapshot, a **saved model artifact**, and a **holdout score** so runs sort cleanly by **mean log-loss** (primary), then Brier / accuracy.

**Lock the evaluation slice.** Pick **`holdout_start_date`** once for the whole campaign (e.g. `2023-01-01` via `python main.py train --data-dir ./data --holdout-start 2023-01-01`). Changing the holdout between generations makes metrics incomparable unless you explicitly start a **new** tuning campaign and re-baseline.

**Per iteration (one generation):**

1. **Change one knob** (§3.3) in [`src/config.py`](../src/config.py) — or in the relevant module if the constant is not yet on `Config`. Record what changed (commit message, lab notebook, or run log).
2. **Train:**  
   `python main.py train --data-dir ./data --holdout-start YYYY-MM-DD`  
   Use a **distinct** `--model-path` per generation so baselines are not overwritten, e.g.  
   `--model-path ./data/Saved_Runs/phase3_run04_l2_1e-3.pkl`.
3. **Score holdout:**  
   `python main.py eval-holdout --model-path ./data/Saved_Runs/phase3_run04_l2_1e-3.pkl`  
   (Top-level `--model-path` before `eval-holdout` also works.)
4. **Record** mean log-loss, Brier, accuracy, wall time, and a short label (knob + value). A spreadsheet or a markdown table under `data/Saved_Runs/` is enough.

**Rules of thumb:**

- **One knob at a time** until the baseline is stable; then optional small factorials on **related** pairs (e.g. `k_base` × `logistic_divisor`).
- **Same `data/` directory** across generations unless you intentionally refresh UFCStats — a data refresh starts a **new** campaign; re-train a baseline before comparing new knobs.

**`elo_mc_*` / γ:** [`eval-holdout`](../main.py) uses **point** probabilities only, so it **does not** directly optimize interval width. For **γ**, still use the same retrain loop; add **`predict`** spot checks or interval/coverage analysis (Phase 4) when tuning layoff-sensitive CIs.

**Optional outer loop:** After a single holdout is stable, add **expanding walk-forward** (train through year *Y*, score *Y+1*) for drift — see [`validation-and-few-shot.md`](validation-and-few-shot.md). That is separate from the first pass of §3.4.

---

## Phase 4 — Model Hardening
*Calibration is acceptable. Now make the system robust and maintainable.*

- Add `tests/` directory with unit tests for ELO update math, feature symmetry, and CI coverage.
- Add a `requirements.txt` with pinned versions of `numpy`, `scipy`.
- Handle weight class changes properly: when a fighter moves weight class, their ELO in the new class should initialise from their prior class ELO with an appropriate discount (architecture deferred this).
- Validate CI coverage empirically: check that intervals at **`1 - ci_alpha`** (default **90%** with `ci_alpha = 0.10`) contain the true outcome at roughly that rate on holdout fights.
- Add a `--verbose` flag to expose intermediate state (ELO values, raw feature vector) for debugging predictions.

---

## Phase 5 — Ongoing Data Pipeline
*The model is validated. Maintain it going forward.*

- Automate weekly UFC results ingestion after each event.
- Incrementally update ELO after each fight rather than full reprocessing.
- Decide on a regression refitting cadence (e.g. monthly full refit vs. online updates).
- Consider a simple web interface or API endpoint for prediction queries.

---

## Side projects (low priority)

- **ELO trajectory concavity / “never downtrend”** — After [`build_elo(..., record_trajectories=True)`](../src/pipeline.py) and [`ELOModel.get_trajectory`](../src/elo/elo.py), analyze per-division ELO sequences (see [`scripts/chart_elo_trajectory.py`](../scripts/chart_elo_trajectory.py)) using **concavity**, segment-wise slopes, or consecutive deltas to identify fighters whose trajectory **never shows a downtrend** under a clear rule (exploratory; define thresholds and minimum fights). Not a core model deliverable.
