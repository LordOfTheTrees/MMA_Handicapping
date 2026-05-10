# Days-off (layoff) histogram for OctagonELO / `mma.ai`

**Goal:** Ship a **static histogram PNG** of **global days idle** (same semantics as **`reference_distributions.json` → `global_days_idle`**: ADR‑15 global clock — calendar days from last recorded bout in any division to the scoring snapshot, or the empirical definition used for that quantile grid).

**Public site expects (committed in `mma.ai`):**

| Deliverable | Path in `mma.ai` repo |
|-------------|------------------------|
| Histogram PNG | `frontend/public/model-viz/feature_histograms/histogram_global_days_idle.png` |
| About UI slice | `frontend/src/pages/about/dataDistribution.ts` → `FEATURE_HISTOGRAM_SLICES` (`id: 'global-days-idle'`) |

Copy/sync the PNG from this repo’s figure output when you run the chart step (below).

---

## Implementation (this repo)

1. **Sample vector**  
   See [`src/export/reference_distributions_export.py`](../src/export/reference_distributions_export.py): `collect_global_days_idle_training_corners` — two values per decisive training bout (corners A and B), days since the fighter’s prior bout in **any** division (`days_idle_global_at_fight_date` from full fight chronology; not `ELOModel._last_fight_global` terminal state).

2. **PNG**  
   From repo root, use the **same cohort** as JSON export (artifacts stay under **this repo** only)::

       python -m src.cli.plot_training_feature_histograms --model-path ./data/model.pkl

   Writes `histogram_global_days_idle.png` under `<repo>/data/figures/feature_histograms/` (default). The layoff figure uses a fixed **1000-day** x-axis (see `--idle-x-max-days`). Copy into the site repo manually when deploying (`frontend/public/model-viz/feature_histograms/`; see `dataDistribution.ts`).

3. **`global_days_idle` in `reference_distributions.json`**  
   Emitted by `build_reference_distributions_document` (101-point quantiles + `chart_histograms.global_days_idle` histogram block).


## Sibling repo

`mma.ai` holds a short pointer: **`docs/mma-handicapping-days-idle-histogram.md`**.
