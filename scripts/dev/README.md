# Optional dev / research scripts

Tools that are **not** part of the normal deploy path (`weekly_update`, `export_*`, `copy_exports_to_mma_ai`, `run_harness`). Run from repo root, e.g.:

```bash
python scripts/dev/phase2_smoke.py
python scripts/dev/validate_fighter_id_lex_order.py
```

| Script | Purpose |
|--------|---------|
| `phase2_smoke.py` | Fast matrix / symmetry / ELO sanity after train or data changes |
| `validate_fighter_id_lex_order.py` | Assert `fighter_a_id` \< `fighter_b_id` lexicographic on fight CSVs |
| `smoke_bootstrap_timing.py` | Estimate full-train bootstrap wall time from a short probe |
| `pilot_lbfgs_stopping.py` | L-BFGS-B stopping grid on the training matrix |
| `benchmark_xgboost_vs_holdout.py` | Optional XGBoost baseline (see `requirements-benchmark.txt`) |
| `predict_cold_corners.py` | Synthetic vs synthetic / known-ID “cold corner” predictions |
| `audit_lex_id_age_cohort.py` | Lex-ID vs age cohort bias audit (figures + stats) |

One-off parsers with fragile inputs live under **`scripts/archive/`**.
