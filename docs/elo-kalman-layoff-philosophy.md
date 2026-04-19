# Kalman on layoff: fast-adjustment vs name-retention

This doc captures the **product-philosophy decision** behind how our Kalman filter responds to a fighter returning from time off. It pairs with **ADR-16** in [`architecture-decisions.md`](architecture-decisions.md) and the mechanical knob guide in [`elo-tuning-knobs.md`](elo-tuning-knobs.md).

The short version: when a fighter's last cage appearance is far in the past, our filter **amplifies** the next ELO update rather than damping it. That is a deliberate choice in favor of a model that **adjusts swiftly to new data** over one that **protects names through gaps**.

---

## 1. The question

When a fighter comes back from a long layoff, what should happen to their rating on that first fight back? Two readings of "we don't know what happened in the interim" both sound reasonable:

- **Damp moves on layoff** — the result might be noisy (ring rust, injury, one-off matchup); don't overcommit to it. Keep the prior and wait for more evidence.
- **Amplify moves on layoff** — the stored rating is stale and therefore a poor prior; let the fresh measurement dominate.

Standard Kalman geometry picks door #2. With predict-step variance growth `P_pred = P_prev + process_noise × days_idle` and measurement noise `R`, the update gain is

```
K = P_pred / (P_pred + R)
applied_delta = K × classical_elo_delta
```

Longer idle → bigger `P_pred` → larger `K` → larger fraction of the classical Elo step is applied. That is what our code does today.

---

## 2. How each direction tilts the model

| Situation | Damp moves (not used) | Amplify moves (current) |
|---|---|---|
| Dominant champ idle ~1 yr, returns & **wins** | Small rating growth | Rating jumps higher |
| Dominant champ idle ~1 yr, returns & **loses** | Small drop; still top | Big drop; can fall out of elite |
| Rising star idle ~3 mo, wins a signature fight | Small credit; needs repetition | Solid credit immediately |
| Faded legend idle for years | Stays highly rated longer than reality | Crashes the moment they lose |
| Underdog beats an idle favorite | Underdog inches up, favorite inches down | Underdog leaps, favorite drops hard |
| Two actively fighting equals | Nearly identical either way | Nearly identical either way |

**One-line framing:**

- Damp → protects reputation through gaps → **favors names with history**.
- Amplify (current) → discounts reputation through gaps → **favors whoever is currently performing**.

---

## 3. Worked example (current behavior)

Using `kalman_process_noise = 0.01`, `kalman_measurement_noise = 1.0`, and a steady-state `P_prev ≈ 1.0`:

```
P_pred(3 months idle)  = 1.0 + 0.01 × 90   = 1.90   → K ≈ 0.655
P_pred(12 months idle) = 1.0 + 0.01 × 365  = 4.65  → K ≈ 0.823
```

Take a favorite vs. underdog matchup with classical Elo steps of **+31.7 / −31.7** on an expected win, or **−68.3 / +68.3** on the upset (same illustrative gap as before, scaled to **`k_base` = 100** vs 60; exact steps depend on ELO gap, `logistic_divisor`, and method scale).

| Scenario | Higher-Elo fighter | Lower-Elo fighter |
|---|---|---|
| Higher (1 yr off) **wins**, Lower (3 mo off) loses | +26.1 (0.823 × +31.7) | −20.8 (0.655 × −31.7) |
| Higher (3 mo off) **wins**, Lower (1 yr off) loses | +20.8 (0.655 × +31.7) | −26.1 (0.823 × −31.7) |
| Higher (1 yr off) **loses**, Lower (3 mo off) wins (upset) | −56.2 (0.823 × −68.3) | +44.8 (0.655 × +68.3) |
| Higher (3 mo off) **loses**, Lower (1 yr off) wins (upset) | −44.8 (0.655 × −68.3) | +56.2 (0.823 × +68.3) |

The **longer-idle side** always gets the **larger fraction** of the classical step on the next fight. That is the "amplify on return" behavior, seen in numbers.

---

## 4. Why we picked amplify (the current behavior)

- A stale rating on an idle fighter is a **worse prior** than a fresh in-cage result. The stored ELO reflects what we knew a year ago; the new fight reflects what is true tonight.
- MMA careers contain real decline, real improvement, real style pivots. We want those surfacing quickly in the rating rather than being masked by reputation.
- The model is a **handicapping tool**. Its value comes from **reacting** to information, not from preserving an internally consistent legacy.
- Volatility on return fights is an **acceptable cost** relative to the alternative — which is that the model quietly over-rates fighters who simply haven't been tested lately.

---

## 5. Tradeoffs we are accepting

- One rusty return performance can move a rating a lot. That is by design; the cage result is the evidence we have.
- We do not preserve "rings of honor" — former champs who take gaps and look poor on return will fall in the rating fast.
- Rising stars who break through after a short break will be credited quickly (this is mostly upside).
- The amplify direction reacts aggressively to what is sometimes a **small sample** (one fight). That is a known epistemic cost of the choice.

---

## 6. What the current knobs actually control

- **`kalman_process_noise`** (currently **0.01/day**) — how quickly variance grows during idle time. Bigger value → more amplification of the return-fight update.
- **`kalman_measurement_noise`** (currently **1.0**) — the denominator brake on gain. Bigger value would pull K back down and **damp** updates universally, not just after layoffs.
- **`k_base`** (currently **100**) and **`_K_SCALE`** (KO/sub **×1.5**) — set the size of the classical step that `K` then scales.

See [`elo-tuning-knobs.md`](elo-tuning-knobs.md) for mechanical semantics and [`elo-modeling-status.md`](elo-modeling-status.md) for how these feed (or don't yet feed) into the regression.

---

## 7. How we'd flip this later (not taken)

If we ever want the damp-on-layoff behavior instead, there are clean paths that don't require rewriting the filter:

1. **Couple measurement noise to idle time** — e.g. `R_effective = R × (1 + α × days_idle)` so long gaps inflate `R` faster than they inflate `P`, pulling `K` down.
2. **Cap `K` after a threshold layoff** — hard ceiling on the applied fraction once idle time exceeds ~9–12 months.
3. **Explicit mean pull** in `kalman_predict` toward a pool prior (e.g. 1500 or a weight-class mean) scaled by idle time — changes the stored **mean** during downtime instead of just the variance.

Each of these is a viable future experiment but sits behind (1) actually observing degraded calibration from the amplify direction, and (2) integrating `ELOState.uncertainty` into features / CIs end-to-end first (see `elo-modeling-status.md`).

---

## See also

- [`architecture-decisions.md`](architecture-decisions.md) — **ADR-16** (this decision), **ADR-15** (global layoff clock)
- [`architecture.md`](architecture.md) §4.5 — Kalman design intent
- [`elo-tuning-knobs.md`](elo-tuning-knobs.md) — per-knob semantics
- [`elo-modeling-status.md`](elo-modeling-status.md) — current wiring into the regression and planned uncertainty integration
