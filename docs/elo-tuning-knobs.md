# ELO tuning knobs — what changing each thing means

Defaults live in [`src/config.py`](../src/config.py) (`ELOConfig`). Method scaling is in [`src/elo/elo.py`](../src/elo/elo.py) (`_K_SCALE`). **`logistic_divisor`** in config feeds `expected_score` (default **300**).

**Sequential tuning:** Change **one** knob at a time when experimenting; **retrain** after changes. Current repo defaults: **`k_base` = 100**, **`logistic_divisor` = 300** (baseline `k_base` was 32, divisor was 400).

Changing ELO dynamics changes **`elo_differential`** (and uncertainty) fed into the regression — after substantive tweaks, **retrain** the multinomial model and re-check calibration.

---

## 1. `k_base` (currently **60**; baseline was **32**)

**Role:** Scales every decisive fight’s ELO step:

`delta_a = k_base * result_k_scale(method) * (actual_a - expected_a)`  
(see `elo_delta` in [`elo.py`](../src/elo/elo.py)).

| Change | Effect |
|--------|--------|
| **Increase** (e.g. 32 → 50) | **Faster** rating movement; **wider** spread in a division (more separation between elite and median); **noisier** trajectory fight-to-fight; upsets and streaks swing ratings more. |
| **Decrease** | **Slower** movement; ratings stay **closer to 1500**; more “inertia”; needs more fights to reflect true level. |

**Intuition:** This is the main dial for “are we moving enough?” without touching the logistic curve.

---

## 2. `logistic_divisor` in `expected_score` (currently **300**)

**Role:**  
`expected_a = 1 / (1 + 10 ** ((elo_b - elo_a) / logistic_divisor))`

Same rating **gap** implies a **steeper** win-expectancy curve if you **lower** the divisor, and a **flatter** curve if you **raise** it.

| Change | Effect |
|--------|--------|
| **Lower divisor** (e.g. 400 → 300) | Same **+100** gap means **higher** expected win % for the favorite; favorites “should” win more often for a given ELO difference; updates can feel **more** decisive because `(actual - expected)` is often larger in magnitude for upsets/favorites. |
| **Higher divisor** (e.g. 400 → 500) | **Gentler** curve; need **larger** ELO gaps for the same win probability; ratings can **spread wider** in points before implying extreme win % (depending how you pair with `k_base`). |

**Implementation:** `ELOConfig.logistic_divisor` → `elo_delta` → `expected_score(..., divisor=...)`.

---

## 3. `result_k_scale` / `_K_SCALE` (KO vs decision, etc.)

**Role:** Multiplies `k_base` per `ResultMethod` (KO/TKO **1.5**, submission **1.5**, unanimous decision **1.0**, split/majority **0.5**, draw/NC/DQ **0**).

| Change | Effect |
|--------|--------|
| **Raise finish multipliers** | Finishes move ratings **more** than decisions; rewards “quality of win” in **ELO steps** (not in the regression’s method breakdown). |
| **Flatten multipliers** (all closer to 1) | Outcomes treated more **uniformly**; more movement rides on **opponent strength** and **surprise** than on method label. |
| **Cut decision scale further** | Decisions barely move ELO; system becomes **finish-heavy**. |

---

## 4. `initial_elo` (default `1500`)

**Role:** Starting point for new `(fighter, weight_class)` states.

| Change | Effect |
|--------|--------|
| **Shift** (e.g. 1500 → 1600) | **Whole scale shifts**; **relative** gaps and `expected_score` **if both sides shift** unchanged for pairs already near baseline — usually you keep 1500 fixed and interpret “+150” as “150 above pool baseline.” Changing baseline without rethinking is rarely useful. |

---

## 5. `kalman_process_noise` (default `0.01` per **day** of inactivity; was `0.0025`, earlier `0.10`, originally `0.05`)

**Role:** Before each fight, variance grows: `P += process_noise * days_since_last_fight` ([`kalman_predict`](../src/elo/kalman.py)). **Days** are since the fighter's last bout **in any weight class** (global layoff clock; [`architecture.md`](architecture.md) §4.5, **ADR-15**).

| Change | Effect |
|--------|--------|
| **Increase** | After layoffs, **uncertainty** is larger → **larger Kalman gain** on the next update → the post-fight mean can move **more** toward the classical `value + delta` target after long gaps. **Point ELO** paths: slightly more reactive **returning** fighters. |
| **Decrease** | Filter stays **confident** through layoffs; less variance inflation; **smaller** gain; updates **damp** after time off. |

The direction of layoff response (amplify vs damp) was a deliberate product choice — see **ADR-16** in [`architecture-decisions.md`](architecture-decisions.md) and the full framing in [`elo-kalman-layoff-philosophy.md`](elo-kalman-layoff-philosophy.md).

**Does not** by itself widen the **steady-state spread** of ratings across the roster the way `k_base` does; it mostly shapes **time-off behavior** and **posterior uncertainty** (used in features / CIs).

---

## 6. `kalman_measurement_noise` (default `1.0`, call it **R**)

**Role:** In `kalman_update`, gain `K = P / (P + R)`. The “observation” is the classical post-fight ELO `value + delta`. Effective move toward that target is scaled by **K**.

| Change | Effect |
|--------|--------|
| **Increase R** | **Smaller** gain → **dampens** how much the **reported** ELO mean moves each fight toward the raw ELO step; smoother, **slower** mean trajectory. |
| **Decrease R** | **Larger** gain → each fight pulls the mean **closer** to the classical one-step update; **snappier** mean. |

Interact with **P** (layoff-inflated or not): after a long layoff, **P** is big, so **K** can still be large unless **R** is huge.

---

## 7. `tier_discount` (cross-promotion)

**Role:** When bringing in outside promotion ELO (`transfer_from_tier` in [`elo.py`](../src/elo/elo.py)), how much of the gap above 1500 is kept.

| Change | Effect |
|--------|--------|
| **Lower discounts** | Outside records **pull UFC ELO less**; more **regression toward 1500** when entering UFC. |
| **Higher discounts** | Outside strength **transfers more**; faster differentiation for imports. |

**Data vs code:** Tier **2/3** fights appear in memory only if you ship `tier2_*.csv` / `tier3_sherdog.csv` under `data/` (see [`pipeline.load_data`](../src/pipeline.py)). UFCStats rows are always **Tier 1**. **`transfer_from_tier` is not called from the pipeline today**, so `tier_discount` affects nothing until that hook is wired; non-UFC bouts still update ELO through normal fight processing when those CSVs are present.

---

## Quick “what to turn first?”

1. **`k_base`** — “ratings should move more / less.” (**Now 100.**)  
2. **`logistic_divisor`** — “what does +100 ELO **mean** in win %?” (**Now 300.**)  
3. **`_K_SCALE`** — how much **method** vs **opponent surprise** drives steps. (**KO/submission** multipliers **1.5**.)  
4. **Kalman `R` and process noise** — **smoothing vs reactivity** and **uncertainty** bands. (**Process noise** **0.01**/day.)

---

## See also

- [`elo-modeling-status.md`](elo-modeling-status.md) — tuned defaults, regression wiring, next steps for Kalman × ELO  
- [`elo-kalman-layoff-philosophy.md`](elo-kalman-layoff-philosophy.md) — why long layoffs **amplify** (not damp) the next update (ADR-16)  
- [`architecture.md`](architecture.md) — ELO + Kalman design intent  
- [`docs/architecture-decisions.md`](architecture-decisions.md) — product decisions log  
- [`src/cli/chart_elo_distributions.py`](../src/cli/chart_elo_distributions.py) (`python -m src.cli.chart_elo_distributions`) — visualize per-division ELO spread after changes (`--top-n` for ranked tables); default PNG is `data/elo_by_division.png` (existing file is deleted then rewritten each run; copy aside from `data/` first if you want to keep a snapshot)
