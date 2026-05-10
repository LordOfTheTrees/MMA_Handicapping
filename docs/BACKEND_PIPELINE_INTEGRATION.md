# Backend pipeline integration checklist

Mirrors **`mma.ai`** `docs/BACKEND_PIPELINE_INTEGRATION.md`; keep edits in sync so trainers and frontend deploy stay aligned.

**Where this repo sits:** **`MMA_Handicapping`** owns training + **artifact export**. **OctagonELO** ships from sibling repo **`mma.ai`** — FastAPI, React SPA, **`api/inference.py`**, and frontend mocks until the pipeline is wired.

---

## Operator quickstart (human-run export)

Copy-paste flows from **`MMA_Handicapping` repo root**; fuller context in **[README.md](../README.md#website-export-mmaai)**.

**A. Export four inference JSONs** (from **`data/model.pkl`**):

```bash
python scripts/export_artifacts.py --model-path ./data/model.pkl --out-dir ./JSON_exports
```

**B. Build `upcoming_events.json`** (needs **`data/upcoming_cards.json`** from `refresh_data`, `train --full-rebuild`, or `python -m src.data.ufcstats_upcoming --data-dir ./data`):

```bash
python scripts/export_upcoming_events.py --cards ./data/upcoming_cards.json --out ./JSON_exports/upcoming_events.json
```

**C. Push JSON to `mma.ai/artifacts`** (default: sibling **`../mma.ai/artifacts`**):

```bash
python scripts/copy_exports_to_mma_ai.py --src ./JSON_exports
```

Or combine **A** / **B** with **`--copy-to-mma-ai`** (and optional **`--mma-ai-artifacts-dir PATH`**) on the export scripts.

---

## Canonical contract docs (live in sibling deploy repo)

If you cloned both repos under the same parent (e.g. `Personal Coding/`), browse **`../mma.ai/docs/`**.

| Topic | Path in **mma.ai** |
|-------|---------------------|
| JSON export shape + manifest | **`mma.ai/docs/export-artifacts-spec.md`** |
| 12-vector order, 6-way outcomes, **`POST /api/predict`** JSON | **`mma.ai/docs/inference-and-api-contract.md`** |
| Feature / interaction parity | **`mma.ai/docs/feature-engineering-port.md`** |
| Layoff / Cauchy / MC display semantics | **`mma.ai/docs/display-semantics-adrs.md`** |
| Deploy layout (two-repo sketch) | **`mma.ai/docs/training-repo.md`**, **`mma.ai/docs/website-architecture.md`** |

---

## Current state — **mma.ai** deploy repo (not implemented here)

| Piece | Notes |
|-------|------|
| **`mma.ai/api/inference.py`** | Loads four JSON files; **`predict`** / **`build_features`** / search TODO |
| **`mma.ai/api/routes/predict.py`** | **`POST /api/predict`** → **503** until inference wired |
| **`mma.ai/api/routes/events.py`**, **`fighters.py`** | **503** stubs |
| **Frontend mocks** | **`mma.ai/frontend/src/data/mock/`**; TypeScript **`PredictionPayload`** matches wire subset |
| **“Why these numbers”** | UI uses **`featureInterpretability.ts`** mocks until API returns marginal / percentile fields |

Training repo has **zero** obligation to mirror those paths in git — integration is behavioral (artifacts + parity), not subtree copy.

---

## Phase 1 — **This repo (MMA_Handicapping): export scripts** (implemented)

Entry points:

| Script | Output |
|--------|--------|
| **`scripts/export_artifacts.py`** | **`model_weights.json`**, **`elo_states.json`**, **`style_axes.json`**, **`fighter_profiles.json`** (under **`--out-dir`**, typically **`JSON_exports/`**) |
| **`scripts/export_upcoming_events.py`** | **`upcoming_events.json`** |
| **`scripts/copy_exports_to_mma_ai.py`** | Copies **`*.json`** into **`mma.ai/artifacts/`** |

Details:

1. **`model_weights.json`** — **`W`** (6×12), bootstrap draws / config for CI routing (**`ModelConfig`**, **`ci_alpha`**, bootstrap count, elo_MC / Cauchy switches per training). **`export_manifest`** includes `git_sha_training`, `exported_at`, schema version.
2. Loads the **same** shipped **`MMAPredictor`** pickle as **`python main.py predict`** / **`explain`**.
3. **`elo_states.json`**, **`style_axes.json`**, **`fighter_profiles.json`** — canonical field names in **`mma.ai/docs/export-artifacts-spec.md`** (sibling checkout).
4. **Smoke parity test** (recommended): CLI **`predict`** vs JSON-only reconstruction on a frozen triple (**TODO harness**).

**Exit:** four + upcoming JSON files from a manual or CI run; parity test passes when wired.

---

## Phase 2 — Hand off JSON to **mma.ai**

1. Drop files into **`mma.ai/artifacts/`** (PR, deploy hook, or copy).
2. Optional: GitHub Action **here** that opens PRs against **`mma.ai`** when a model tag bumps.
3. Never ship pickles, **`data/`** CSV blobs, or `src/` into the web image — JSON only (`mma.ai/docs/training-repo.md`).

---

## Phase 3 — **mma.ai** — standalone `InferenceEngine`

Implemented only in **`mma.ai`**.

- **`build_matchup_features`** mirrors **`FEATURE_NAMES`** / **`features_to_array`** from **`mma.ai/docs/inference-and-api-contract.md`** and interaction math ported from **`src/matchup/`** (copy logic; **do not** `import MMA_Handicapping` from production).
- **`predict_point`**, CI routing (**bootstrap**, **`elo_mc`**, Cauchy…) from embedded config.
- Layoffs (**`days_idle_*`**) from profile / ELO **`last_fight_date`** semantics (**`mma.ai/docs/display-semantics-adrs.md`**).
- **`rapidfuzz`** search over **`fighter_profiles.json`** for **`GET /api/fighters?q=`**.

**Training validation:** rerun Phase‑1 parity after **`mma.ai`** implements engine (recommended cross-repo QA step).

---

## Phase 4 — **mma.ai** FastAPI routes

| Route | Role |
|-------|------|
| **`POST /api/predict`** | Body IDs + **`weight_class`** + **`fight_date`** → full prediction JSON |
| **`GET /api/fighters`**, **`/api/fighters/{id}`** | Search + card payload |
| **`GET /api/events/upcoming`** | Precomputed card JSON (**export**) and/or predict-on-demand (**mma.ai** chooses A/B/C) |

---

## Phase 5–7 — Frontend, ops, subscriptions

Handled entirely in **`mma.ai`** (SPA fetch, **`VITE_*`**, rate limits on predict, Stripe later). Optionally extend **`POST /api/predict`** response with interpretability fields matching **`FeatureBreakdown`**.

See **`mma.ai/docs/BACKEND_PIPELINE_INTEGRATION.md`** for full wording.

---

## Verification matrix

| Check | Owner |
|-------|-------|
| 12 features + class order vs **`inference-and-api-contract`** | Train + **`mma.ai`** parity test |
| Six **`probs`** sum ~1 | **`mma.ai`** inference |
| Parity CLI vs **`POST /api/predict`** on golden triple | **MMA_Handicapping** export test + **`mma.ai`** harness |
| CIs monotone vs point mass | **`mma.ai`** inference + training semantics |
| **`/health`** reports loaded manifest version | **`mma.ai`** |

---

## Where to refine next (**MMA_Handicapping**)

- **Parity harness:** CLI **`predict`** vs JSON-only reconstruction (verification matrix below)
- **`scripts/export_artifacts.py`**, **`export_upcoming_events.py`**, **`copy_exports_to_mma_ai.py`**
- Optional: GitHub Action to push **`mma.ai/artifacts`** after export
