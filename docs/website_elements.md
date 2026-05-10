# Website Documentation — Fight Model Display Site

## Pages (in order)

### 1. Main / Home Page
- **Primary feature:** Calendar of upcoming events
- **Secondary feature:** Search bar to look up stats for a particular fighter — uses `predict-human` fuzzy name matching (exact + fuzzy resolution)

#### Event / Fight Card View
- Each upcoming event expands to list every individual bout on the card
- Each bout displays a **model prediction** based on current metrics (as of the most recent Monday backend run)
- **Backend refresh cadence:** re-run every Monday; predictions reflect that week's ELO and fighter state
- Events beyond the next upcoming one are locked behind subscription (see Subscription Model)

##### Per-Bout Prediction Display (3-part layout)
Modeled after `2023-03-04_split_barrier.png`:

**Part 1 — Win Probability Bar (top)**
- Horizontal stacked bar split between the two fighters
- Each fighter's side subdivided by method: **KO/TKO**, **Submission**, **Decision** (color-coded, blue tones for Fighter A, red tones for Fighter B)
- Each segment labeled with its percentage
- Total win % annotated below the split point (e.g. "total win 30.8%")
- Derived bars also available: `total_win`, `total_lose`, `finish_win`, `finish_lose`, `go_to_decision`

**Part 2 — Marginal CI Summary (middle)**
- Displays **Best** and **Worst** marginal confidence interval sums per fighter
- CI metadata surfaced: `ci_method`, `effective_n`, `pct_post_era`
- **⚠ Cauchy Warning (yellow):** rendered when `ci_method` indicates Cauchy applies — states the specific insufficiency (e.g. "insufficient in-division bout history for Jon Jones in heavyweight") and communicates that probability ranges are correspondingly wide / unstable. Matches the annotation pattern visible in `2023-03-04_split_barrier.png`.

**Part 3 — Confidence Ranges Bar Chart (bottom)**
- Bar chart of all 6 mutually exclusive outcomes (Fighter A perspective): win KO/TKO, win sub, win decision, lose decision, lose KO/TKO, lose sub — sum to 1
- Y-axis: probability 0–100%
- Each bar shows a **90% confidence interval** marker (open circle at point estimate, bar spanning CI lower/upper)
- X-axis labeled with outcome names
- *Reference: `2023-03-04_split_barrier.png` / `plot_prediction_three_viz`*

---

### Single Bout Prediction Page (scheduled or hypothetical)

#### Main Outputs (`PredictionResult`)
| Field | Notes |
|---|---|
| Six outcome probabilities | Fighter A perspective: win KO/TKO, win sub, win decision, lose decision, lose KO/TKO, lose sub — mutually exclusive, sum to 1 |
| `total_win` / `total_lose` | Derived aggregate bars |
| `finish_win` / `finish_lose` | Derived finish-side bars |
| `go_to_decision` | Derived bar |
| Confidence intervals | Lower + upper per outcome class, from CI machinery |
| `ci_method` | Method used for CI construction |
| `effective_n` | Effective sample size |
| `pct_post_era` | % of training data post-era (when populated) |
| Fight date | |
| Weight class | |

> **⚠ Cauchy Warning (yellow banner):** When `ci_method` is Cauchy, a yellow warning is displayed stating the specific insufficiency per fighter and noting that the resulting confidence intervals are wide and the prediction is unstable. Triggered per-fighter — one or both corners may carry the flag independently.

#### Matchup Feature Vector (`MatchupFeatures`) — "Why These Numbers"
| Feature | Type |
|---|---|
| `elo_differential` | Core signal |
| `striker_score_diff` | Pedigree differential |
| `grappler_score_diff` | Pedigree differential |
| `finish_threat_diff` | Pedigree differential |
| `finish_vulnerability_diff` | Pedigree differential |
| `striking_matchup` | Interaction term |
| `grappling_matchup` | Interaction term |
| `finish_matchup` | Interaction term |
| `reach_diff_cm` | Physical |
| `height_diff_cm` | Physical |
| `stance_mismatch` | Physical (binary flag) |
| `age_diff_days` | Physical |

#### Interpretability Layer (`explain`-style)
- Positioned **at the bottom of the page** (below the fold; user scrolls to reach it)
- Dominated by **ELO feature axis bars** — ranked additive log-odds contributions per named feature, visualized as a bar chart per outcome class
- For each of the 6 outcome classes: predicted probability + ranked feature contributions
- Exact decomposition from multinomial coefficients × feature values
- No need to describe this section prominently high on the page — it's supplementary detail for users who want to dig in

#### Page Layout — Vertical Order (Single Bout Prediction)
1. **Days Idle (top)** — displayed per corner at the very top of the page
   - Shows days since last fight for each fighter
   - Includes **percentile position in the all-time cumulative days idle distribution** (e.g. "top 12% of layoffs all time") so the user understands how unusual the layoff is
   - Affects CI width via Cauchy ELO Monte Carlo / γ; does not shift point estimate
2. **⚠ Cauchy Warning** (yellow, if applicable)
3. **Win Probability Bar** (Part 1)
4. **Marginal CI Summary** (Part 2)
5. **Confidence Ranges Bar Chart** (Part 3)
6. **Matchup Feature Vector** — "Why These Numbers"
7. **Interpretability Layer** — ELO feature axis bars (bottom, scroll to reach)

#### Optional UI Inputs / Overlays
- **Human-readable fighter resolution** — supports exact match and fuzzy name match (as in `predict-human`)

---

### 2. Rankings / ELO Browser
- Filtered by **weight class** and **gender**
- Displays **active ELOs** for fighters in the selected division

---

### 3. Hypothetical Bout *(subscribers only)*
- **Top-level navigation tab** — visible to all users but gated; non-subscribers are prompted to subscribe on click
- Allows a user to input any two fighters and a hypothetical weight class and receive a full `PredictionResult` output
- Same page layout as the Single Bout Prediction page (days idle → Cauchy warning → probability bar → CI summary → feature vector → interpretability layer)
- **Days idle** is user-adjustable per corner for scenario modeling
- Fighter input supports fuzzy name resolution (`predict-human`)

---

### Fighter Profile Page (accessible via search or rankings)

#### Static Profile (`FighterProfile`)
| Field | Notes |
|---|---|
| Name | Display name |
| IDs | Fighter identifiers |
| Reach | |
| Height | |
| Date of Birth | Computed → age at any given fight date |
| Stance | |
| Pedigree Signals | Wrestling / Boxing / BJJ scores (0–1 scale), used for cold-start priors |

#### Per-Division State (`ELOState` — one row per fighter × weight class)
| Field | Notes |
|---|---|
| ELO | Kalman mean |
| Uncertainty | Variance |
| Fight Count | In this division |
| Primary Data Tier | |
| Last Fight Date | In this division; feeds the global layoff clock (see ADR for uncertainty decay) |

#### Trajectory (Optional Chart)
- Line chart of **ELO (Kalman mean)** over time, one point per bout
- X-axis: date (e.g. `2019-01`, `2022-01`); Y-axis: ELO value
- Each data point is labeled with the **opponent's name**
- **Fight ledger interaction:** selecting a fight in the ledger highlights the corresponding data point on the trajectory chart
- **Free user behavior:** chart renders as a blurred teaser with a lock icon overlay; interaction and data are hidden until subscribed
- *Reference: `zabit_magomedsharipov_elo_trajectory.png`, `anthony_smith_lhw_elo_trajectory.png`*

#### Fight Ledger (`FightRecord` rows)
| Field | Notes |
|---|---|
| Date | |
| Promotion | |
| Weight Class | Canonical label + raw label if unknown/catch weight |
| Opponent | |
| Outcome | |
| Method | |
| Tier | Data tier for this fight |
| Per-Fight Stats | Tier 1 only (`FightStats`): sig strikes, TDs, control time, sub attempts, fight duration |

---

### Second to Last — About the Model Construction
- Contains all **Architecture Decision Record (ADR)** data
- Documents how the model was built out, including key architectural decisions made during development
- **Reference visualizations available** (to be embedded or linked in the ADR page):
  - `elo_by_division.png` — ELO distribution histograms across all divisions (as of 2026-04-19), showing spread and clustering per weight class
  - `data/figures/division_elo_histograms/*.png` — One histogram per weight class (from `python -m src.cli.chart_elo_distributions`)
  - **Precomputed chart data:** **`reference_distributions.json`** (mma.ai contract: `matchup_features` + `division_elo` quantile grids; optional nested **`chart_histograms`** for bin/count payloads) — export next to other deploy JSONs from `export_artifacts.py`
  - `histogram_all_grid.png` — Full grid of all 12 regression training features (n=6,366 bouts)
  - Individual feature histograms: `elo_differential`, `striker_score_diff`, `grappler_score_diff`, `finish_threat_diff`, `finish_vulnerability_diff`, `striking_matchup`, `grappling_matchup`, `finish_matchup`, `reach_diff_cm`, `height_diff_cm`, `stance_mismatch`, `age_diff_days`

### Last — Contact Me
- Standard contact page

---

## Subscription Model

### Access Tiers
| Content | Free | Subscribed |
|---|---|---|
| Upcoming event calendar | Next event only | All upcoming events |
| Predictions | Next event only (not yet started) | All upcoming events |
| Fighter search & profiles | ✅ Full access | ✅ Full access |
| ELO rankings | ✅ Full access | ✅ Full access |
| ELO trajectory chart | ❌ Blurred teaser | ✅ Unlocked |
| Hypothetical Bout tab | ❌ Locked (subscribe prompt on click) | ✅ Full access |

### Prediction Locking Rules
- **Only the next event that has not yet started** is open for free predictions
- All events beyond that are **locked behind subscription**
- Once an event starts, the prediction window closes

### Payment Provider
- **Decision pending** — Stripe is the leading candidate (industry standard, well-documented)
- To be evaluated comparatively before implementation
- *Pin: schedule a comparative evaluation of payment providers (Stripe vs. alternatives)*

---

## Global Elements

### Right Margin — Subscription Widget (all pages)
- Persistent sidebar element visible on every page
- Reflects the user's current auth/subscription state dynamically
- For free users: summarizes what is locked and what is accessible
- Promotes the value of subscribing with a list of gated features (full event calendar, all predictions, ELO trajectory, etc.)
- Contains a **Subscribe / Upgrade CTA button**
- **Mobile behavior:** collapses to a compact form (e.g., icon + minimal text) but remains always visible — does not fully hide

### Bottom Banner
- Low-impact advertisement for **CombatCognition** (user's webapp)
- Positioned at the bottom of all pages

---

## Automated checks (`MMA_Handicapping` repo)

Structural validation that **`JSON_exports/`** JSON aligns with **this** page inventory (home/upcoming, rankings/ELO snapshot, fighter profile fields, hypothetical/single-bout **`predict_proba_snapshot`**, about-model **`model_weights`**, **`reference_distributions.json`**) runs via:

```bash
python scripts/run_harness.py site
```

Optional env **`MMA_SITE_EXPORT_DIR`** if artifacts live outside **`JSON_exports/`**. Subscription/chrome and Contact pages have no artifact contract in the training repo.

---

*This document will be updated as additional pages and elements are described.*
