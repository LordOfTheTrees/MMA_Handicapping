# MMA Predictive Modeling Architecture
## Pre-Fight Win & Finish Probability Model

---

## 1. Problem Framing

### 1.1 Goal
A pre-fight handicapping model that produces a calibrated probability distribution over six mutually exclusive fight outcomes for any given matchup, using only information available before the fight occurs.

### 1.2 Output Space
A 6-class softmax producing probabilities that sum to 1:

| Class | Label |
|---|---|
| 0 | Win by KO/TKO |
| 1 | Win by Submission |
| 2 | Win by Decision |
| 3 | Lose by Decision |
| 4 | Lose by KO/TKO |
| 5 | Lose by Submission |

Derived aggregates are trivially computable from the 6-vector:

```
total_win%       = P(0) + P(1) + P(2)
finish_win%      = P(0) + P(1)
finish_lose%     = P(4) + P(5)
go_to_decision%  = P(2) + P(3)
```

### 1.3 Core Design Principles
- **Interpretability is non-negotiable.** Every coefficient in the model must have a readable, defensible meaning. No black box components anywhere in the pipeline.
- **Assumptions must be explicit and minimal.** Where assumptions are required they are stated openly. Distributional assumptions on outcomes are avoided wherever possible.
- **Uncertainty is first-class.** The model communicates what it does not know through confidence intervals that widen honestly with data sparsity, not through false precision.
- **Pre-fight only.** No in-fight, in-round, or real-time data enters the model. All features are computable before the opening bell.

---

## 2. Pipeline Overview

The architecture is a strict sequential dependency chain. Each stage is a prerequisite for the next and cannot be reordered.

```
Stage 1:  Data Collection & Tiering
              ↓
Stage 2:  ELO Model Construction
              ↓
Stage 3:  ELO-Weighted Feature Construction
              ↓
Stage 4:  Matchup Interaction Terms
              ↓
Stage 5:  Multinomial Logistic Regression
              ↓
Stage 6:  Confidence Interval Computation
```

---

## 3. Data Architecture

### 3.1 Data Tiers

The data pipeline has four tiers reflecting data quality and availability. Each tier has a distinct role in the pipeline.

| Tier | Source | Coverage | Stats Available | Pipeline Role |
|---|---|---|---|---|
| 1 | UFCStats.com | All UFC fights | Full per-fight, per-round stats | ELO + feature construction |
| 2 | Bellator, ONE, PFL, RIZIN | Major promotions | Outcomes + partial stats | ELO + partial feature construction |
| 3 | Sherdog Fight Finder | Regional and minor promotions globally | Outcomes only | ELO construction only |
| 4 | Combat sports background | Wrestling rankings, boxing records, BJJ results | Signals only | Starting ELO prior adjustment |

### 3.2 Era Cutoff

The sport has changed structurally and is not a stationary process across its full history. Athletic composition, training methodology, finish rates, and judging criteria have all shifted materially. To avoid training the regression on data drawn from a different underlying sport:

- **Regression training data:** Tier 1 fights from approximately 2013 onward only. This represents the modern era of MMA-specific athletic preparation, stabilized judging criteria, and current finish rate norms. The exact cutoff year is a tunable parameter but should be validated against holdout performance.
- **ELO construction:** All available historical data across all tiers, regardless of era. Pre-cutoff fights inform ELO initialization but never enter the regression training set.

### 3.3 Cross-Promotion ELO Transfer

Fighter ELO constructed from Tier 2 and Tier 3 data transfers to the UFC ELO system with a discount reflecting genuine uncertainty about how quality translates across promotions. The discount is a tunable parameter, with stronger regional promotions receiving a smaller discount than minor regional shows.

---

## 4. ELO Model

### 4.1 Scope
ELO is constructed **per weight class**. A fighter's ELO in one weight class carries no information about their performance in another. Weight class changes trigger a full ELO reset in the destination weight class, initialized from whatever cross-promotion prior is available.

### 4.2 Result Certainty Scale

The ELO K-factor is scaled by result certainty. This reflects that a dominant finish is a stronger signal of relative quality than a disputed split decision, and that quality of wins is already embedded in the update mechanism.

```
KO/TKO win                     →  K × 1.5
Submission win                 →  K × 1.5
Unanimous decision win         →  K × 1.0
Split or Majority decision win →  K × 0.5
Draw / No Contest / DQ         →  K × 0.0
Split or Majority decision loss→  -K × 0.5
Unanimous decision loss        →  -K × 1.0
KO/TKO loss                    →  -K × 1.5
Submission loss                →  -K × 1.5
```

**Rationale for collapsing Split and Majority:** With three judges in MMA, both a split decision (2-1) and a majority decision (2-0-1) represent exactly one judge dissenting. The signal strength is equivalent regardless of whether that dissent is expressed as a vote for the opponent or a draw score. False precision in distinguishing them is avoided.

**Rationale for Draw/NC/DQ at zero:** These results carry no meaningful information about relative fighter quality. No Contest voids a result procedurally. DQ results from a foul, not combat outcome — adjudicating fight state at the time of the foul is subjective and inconsistent across events. All three receive zero ELO movement.

### 4.3 ELO as a Dual-Purpose Mechanism

ELO performs two distinct and non-conflicting jobs in the pipeline:

**Job 1 — Upstream quality weight (Stage 3):** Every fighter stat used in feature construction is weighted by the ELO of the opponent against whom it was produced. A finish rate built against high-ELO opposition is a stronger signal than the same finish rate built against low-ELO opposition.

**Job 2 — Direct regression feature (Stage 5):** The ELO differential between the two fighters entering a fight is itself a feature in the regression model. It represents the residual quality component — ring generalship, composure, adaptability — that style axes do not fully capture. It is the best scalar summary of relative overall quality.

### 4.4 Cold Start Handling

Fighters debuting in the UFC are not initialized at a flat 1500. Regional fight history from Tier 2 and Tier 3 sources provides a meaningful prior:

```
Fighter with strong Tier 2 record     → ELO transferred with small cross-promotion discount
Fighter with Tier 3 record only       → ELO transferred with larger discount
Fighter with combat sports background → 1500 baseline adjusted upward modestly by pedigree signal
Complete unknown                       → 1500 flat, high uncertainty
```

### 4.5 Time Uncertainty (Kalman Integration)

Fighter parameters are not stationary — a fighter observed last month is better characterized than one returning from a long layoff. A lightweight Kalman filter maintains uncertainty estimates on each fighter's ELO and style axis scores with the following minimal design:

- **Uncertainty grows** with time elapsed since the fighter's **last professional fight in any weight class**. ELO means and updates remain **per division**, but the time-update before a bout uses a **global** layoff clock: a training camp and a cage appearance at another weight still mean we have a recent observation of the athlete, so uncertainty should not stay artificially tight in one division while they were active elsewhere. (See [`architecture-decisions.md`](architecture-decisions.md) **ADR-15**.)
- **Uncertainty shrinks** with each new observation. The quality of that observation — already encoded in the K-factor scaling — determines how much the uncertainty contracts.
- **Layoff direction (ADR-16).** Standard Kalman gain (`K = P / (P + R)`) is retained: longer idle time → larger `P` → **larger** fraction of the classical ELO step is applied on the next fight. We chose this **fast-adjustment** behavior over a **damp-on-layoff** alternative because a stale stored rating is a worse prior than a fresh cage result; the tradeoff and the rejected alternative are written up in [`elo-kalman-layoff-philosophy.md`](elo-kalman-layoff-philosophy.md).

No further assumptions are made about the mechanics of parameter change. There are no aging curves, no durability decay functions, no career stage priors. The data determines uncertainty through observation frequency and quality alone.

---

## 5. Feature Construction

### 5.1 Symmetry Requirement

The model receives a matchup (Fighter A vs. Fighter B). Swapping A and B must produce a prediction where all win probabilities become lose probabilities and vice versa. This antisymmetry is enforced by construction through signed difference and ratio features.

For any stat, the feature entering the model is:
```
A_stat - B_stat                        (signed difference)
A_stat / (A_stat + B_stat)             (relative share)
```

When A and B are swapped, differences negate and predictions flip accordingly.

### 5.2 Style Axes

The two primary axes represent the domains in which a fighter operates. Both are computed as ELO opponent-weighted, recency-decayed aggregates over the fighter's fight history in the current weight class, subject to the era cutoff.

**Striker Score**
How much does this fighter operate in the striking domain? Derived from significant strike rate differential — offensive strikes landed relative to strikes absorbed, quality-adjusted by opponent ELO at fight time.

**Grappler Score**
How much does this fighter operate in the grappling domain? Derived from takedown rate, control time rate, and submission attempt rate, quality-adjusted by opponent ELO at fight time.

Striker and Grappler scores are treated as approximately orthogonal axes. A fighter can be high on both (well-rounded), high on one (specialist), or low on both (brawler). They are not constrained to sum to any value.

### 5.3 Finish Overlays

Two finish tendency measures sit on top of the style axes. They are independent of style — a grappler and a striker can both have high finish threat — and are derived directly from career outcome distributions, quality-adjusted by opponent ELO.

**Finish Threat**
Historical rate at which this fighter finishes opponents, opponent-ELO-weighted. Captures offensive finishing ability regardless of method.

**Finish Vulnerability**
Historical rate at which this fighter is finished, opponent-ELO-weighted. Captures defensive durability regardless of finish type. A high finish vulnerability is a strong input to both lose-by-KO and lose-by-submission prediction classes.

### 5.4 Physical Features

Directly observable pre-fight, requiring no aggregation or quality adjustment:

```
Reach differential           → reach_A - reach_B
Height differential          → height_A - height_B
Stance mismatch              → orthodox vs. southpaw indicator (categorical)
Age differential             → age_A - age_B at fight date
```

### 5.5 Full Feature Vector

```
ELO differential             → residual quality
Striker score differential   → striking domain advantage
Grappler score differential  → grappling domain advantage
Finish threat differential   → offensive finishing advantage
Finish vulnerability diff    → defensive durability relative position
striking_matchup             → explicit interaction term (see Section 6)
grappling_matchup            → explicit interaction term (see Section 6)
finish_matchup               → explicit interaction term (see Section 6)
Reach differential
Height differential
Stance mismatch
Age differential
```

---

## 6. Matchup Interaction Terms

Style axes capture each fighter's individual tendencies. Matchup interactions capture how those tendencies combine. They are explicit products of opponent axis values — not learned by the model, but specified from the causal theory of fight outcomes.

```
striking_matchup  = striker_score_A × (1 - striker_score_B)
                    → how much A's striking advantage is enabled by B's striking weakness

grappling_matchup = grappler_score_A × (1 - grappler_score_B)
                    → how much A's grappling advantage is enabled by B's grappling weakness

finish_matchup    = finish_threat_A × finish_vulnerability_B
                    → how much A's finishing ability is enabled by B's durability weakness
```

These terms are what allow the model to distinguish a striker-vs-grappler matchup from a striker-vs-striker matchup at equal ELO — a distinction raw style axis differentials alone cannot fully capture.

---

## 7. Regression Model

### 7.1 Specification

**Multinomial logistic regression** with a robust loss specification. The model is linear in the feature vector with a softmax output over 6 classes. Every coefficient has a direct interpretation: the signed contribution of that feature to the log-odds of each outcome class.

The regression is trained exclusively on Tier 1 data post-era-cutoff. This is the only data stratum assumed to be drawn from approximately the same distribution as the fights being predicted.

### 7.2 Interpretability

The why of any prediction is fully decomposable:

```
P(Win by KO) is high because:
  - ELO differential is positive (+0.8 contribution)
  - Striking matchup term is high (+1.2 contribution)
  - Opponent finish vulnerability is high (+0.9 contribution)
  - Reach differential favors Fighter A (+0.3 contribution)
```

No approximation of the explanation is required. The decomposition is exact because the model is additive.

### 7.3 Robust Loss Specification

Standard maximum likelihood logistic regression implicitly assumes well-behaved errors. Given the non-stationarity of individual fighter careers and the era heterogeneity in training data, a heavy-tailed robust loss function is used in estimation. This downweights the influence of outlier fights and produces more conservative coefficient estimates in regions of the feature space where training data is sparse.

### 7.4 Style Axis Cold Start

For fighters with no Tier 1 data, style axis scores cannot be estimated from UFC stats. The following hierarchy applies:

1. **Infer from finish method distribution** in regional data — a fighter with 6 KO wins and 0 submission wins receives a weak striker prior.
2. **Combat sports background** — a D1 wrestler receives a weak grappler prior.
3. **Weight class average** — when neither is available, initialize at weight class mean with maximum uncertainty.

In all cold start cases, the Kalman uncertainty on that fighter's style axes is set wide, propagating directly into wider prediction confidence intervals.

---

## 8. Confidence Intervals

### 8.1 Philosophy

Confidence intervals communicate what the model does not know. They widen automatically and honestly when data is sparse, eras are mixed, or fighters are poorly observed. False precision is not acceptable.

### 8.2 Primary Method: Bootstrap

When sufficient recent reference data exists, confidence intervals are computed via percentile bootstrap on the **softmax probabilities** implied by bootstrap draws of the coefficient matrix:

1. At **train** time: weighted bootstrap resamples of the training matrix (recency weights), refit the multinomial model on each resample, and **store** the resulting coefficient matrices on the saved predictor.
2. At **predict** time: for the current feature vector `x`, evaluate `softmax(W_b x)` for each stored draw `W_b` (no refit). Report percentile intervals across those probability vectors.

Older `model.pkl` files without stored draws fall back to refitting on each `predict` call until retrained.

CI width emerges naturally from the effective sample size, the consistency of outcomes in similar historical matchups, and the era composition of the reference data.

### 8.3 Fallback: Cauchy Default

When the effective sample is too small for bootstrap CIs to be trustworthy — sparse feature region, heavy era mixing, fighter with very few quality-adjusted observations — the model falls back to a Cauchy-distribution-based conservative interval. The Cauchy makes no strong assumptions about outcome distribution shape and produces heavy-tailed, appropriately wide intervals that reflect genuine ignorance.

The trigger for Cauchy fallback is a tunable threshold on effective sample size after weighting. Below the threshold, the model explicitly communicates that it lacks sufficient reference class data.

### 8.4 Kalman Uncertainty Contribution

Time elapsed since the fighter's **last bout in any weight class** grows the **posterior variance** on that fighter’s ELO Kalman state for the division being queried (§4.5). That variance controls **Kalman gain** on the next fight update (how aggressively the stored **mean** moves toward the classical `value + delta` target) and appears in **`ELOState.uncertainty`** for diagnostics (e.g. chart summaries).

**Regression today:** The multinomial feature vector uses **`elo_differential`** from the Kalman **mean** only ([`build_matchup_features`](src/matchup/interactions.py)). ELO Kalman variance does **not** yet enter features or the bootstrap/Cauchy CIs (those reflect coefficient/data uncertainty on the fitted logit, not ELO epistemics). Closing that gap — features, expected-score under doubt, and/or CI propagation — is tracked in [`docs/elo-modeling-status.md`](docs/elo-modeling-status.md).

### 8.5 Output Format

```
                    Point Est.   95% CI          Method
Win by KO/TKO         0.24      [0.17, 0.31]    Bootstrap n=47
Win by Submission      0.09      [0.04, 0.14]    Bootstrap n=47
Win by Decision        0.28      [0.21, 0.35]    Bootstrap n=47
Lose by Decision       0.21      [0.15, 0.28]    Bootstrap n=47
Lose by KO/TKO         0.13      [0.08, 0.19]    Bootstrap n=47
Lose by Submission     0.05      [0.02, 0.09]    Bootstrap n=47

Reference: 47 similar fights | 89% post-2019 | Era: Modern
```

Sparse case:
```
                    Point Est.   95% CI          Method
Win by KO/TKO         0.26      [0.04, 0.61]    Cauchy — sparse reference class
...

Reference: 6 similar fights | Mixed eras | ⚠ Interpret with caution
```

---

## 9. What The Model Explicitly Does Not Do

| Omission | Reason |
|---|---|
| In-fight or round-by-round data | Pre-fight model only |
| Aging curves or physical decline functions | Unobservable, assumption-heavy |
| Durability decay modeling | Already captured in finish vulnerability axis |
| Career stage classification | Arbitrary step functions on a continuous process |
| Parametric distributional assumptions on outcomes | Avoided in favor of bootstrap and Cauchy fallback |
| Gradient boosting or neural network components | Black box — interpretability is non-negotiable |
| Global ELO across weight classes | Quality does not transfer across weight classes |
| Pre-era-cutoff regression training data | Non-stationary with modern sport |

---

## 10. Open Design Questions

The following parameters require **empirical tuning against holdout prediction performance** (Phase 3) and are not specified a priori. The **authoritative Phase 3 inventory** — including the **2013-class `era_cutoff_year` boundary**, **all ELO levers**, **train/holdout split** (once implemented), and **model thresholds/weights** — lives in [`docs/todo.md`](docs/todo.md) §**3.3 Phase 3 tuning inventory**.

Summary (see §3.3 for the full table):

- **`era_cutoff_year`** (default 2013): regression-era floor; tune on holdout, not fixed forever.
- **Holdout window** / train–test split parameters (to be added to the pipeline).
- **ELO:** `k_base`, `logistic_divisor`, `tier_discount`, Kalman noises, `_K_SCALE` method multipliers.
- **Features:** `recency_decay_rate`, `min_fights_style_estimate`.
- **Regression / CIs:** `l2_lambda`, `huber_delta`, `n_bootstrap`, `ci_alpha`, Cauchy threshold/scale, bootstrap seed.
