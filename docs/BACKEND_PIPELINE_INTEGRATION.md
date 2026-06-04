# Backend pipeline integration checklist

Mirrors **`mma.ai`** `docs/BACKEND_PIPELINE_INTEGRATION.md`; keep edits in sync so trainers and frontend deploy stay aligned.

**Where this repo sits:** **`MMA_Handicapping`** owns training + **artifact export**. **OctagonELO** ships from sibling repo **`mma.ai`** — FastAPI, React SPA, **`api/inference.py`**, and frontend mocks until the pipeline is wired.

---

## Operator quickstart (human-run export)

Copy-paste flows from **`MMA_Handicapping` repo root**; fuller context in **[README.md](../README.md#website-export-mmaai)**.

**A. Export five inference JSONs** (from **`data/model.pkl`**):

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

### CI artifacts → local **`mma.ai`** (summary)

**GitHub Actions** (`weekly-model-refresh`, `monthly-model-retrain`) upload:

| Artifact | Contents |
|----------|-----------|
| **`mma-model-state`** | **`data/model.pkl`** for the next CI run |
| **`weekly-refresh-<run_id>`** / **`monthly-retrain-<run_id>`** | **`JSON_exports/`**, **`data/ufcstats_fights.csv`**, **`fighter_profiles.csv`**, **`espn_crosswalk_*.csv`**, **`espn_ingest_state.json`**, **`upcoming_cards.json`** — see [**data-sources-espn.md**](data-sources-espn.md). |

**After you download an artifact zip locally:** unzip and point **`--src`** at the folder that contains the five inference **`*.json`** files (usually **`JSON_exports/`** inside the bundle). From **`MMA_Handicapping` repo root**:

```bash
python scripts/copy_exports_to_mma_ai.py --src "/path/to/unzipped/JSON_exports"
```

Default **`--dest`** is sibling **`../mma.ai/artifacts`** (override with **`--dest`** if needed).

**Automated push:** **`sync-json-to-mma-ai.yml`** runs after weekly/monthly **succeed** on the default branch (downloads newest **`weekly-refresh-*`** / **`monthly-retrain-*`** artifact via **`run-bundle`**). Manual **Run workflow** still works. Requires **`MMA_AI_SYNC_PAT`**. Only copies what was in **`JSON_exports/`** at upload time (**`upcoming_events.json`** only if generated into **`JSON_exports/`** before packaging).

### GitHub Actions admin

Record of operator controls and failure modes for **[`weekly-model-refresh.yml`](../.github/workflows/weekly-model-refresh.yml)**, **[`monthly-model-retrain.yml`](../.github/workflows/monthly-model-retrain.yml)**, and **[`sync-json-to-mma-ai.yml`](../.github/workflows/sync-json-to-mma-ai.yml)**.

#### Default weekly/monthly data path (restore + ESPN delta)

Every scheduled and normal manual run:

1. `python scripts/ci_restore_data_bundle.py` — seed fights, profiles, crosswalk, ingest state, upcoming cards from the newest run bundle.
2. `python scripts/ci_try_refresh_data.py` — ESPN **incremental** only. **Fails** if ESPN is down or if **zero fights** were updated/added (weekly goal not met).

**Manual dispatch:** `skip_espn_refresh` skips step 2. `--allow-stale-data` (CLI only) succeeds without new ESPN rows (debug).

**Quiet week (no UFC card):** ingest may correctly update 0 fights and the job will fail; use `--allow-stale-data` manually that week or skip the workflow.

Full history rescrape (UFCStats HTML, multi-year ESPN crosswalk) is **operator-only**, not CI.

#### Refresh failure (ESPN / network)

CI uses **[`scripts/ci_try_refresh_data.py`](../scripts/ci_try_refresh_data.py)** → ESPN incremental ingest (see **[data-sources-espn.md](data-sources-espn.md)**). Failures log `::group::ESPN ingest failed`.

With **`allow_stale_data: false`**, the job **fails**; no new run bundle is uploaded. Ingest never wipes a non-empty **`ufcstats_fights.csv`** with zero rows.

**One-time local setup:** after restoring an artifact bundle, run crosswalk for recent seasons so new ESPN rows keep UFCStats hex IDs:

```bash
python -m src.data.espn_ingest crosswalk --data-dir ./data --season 2024 --season 2025
```

UFCStats HTML remains a **fallback** (often Cloudflare-blocked); Sherdog/Tapology are documented fallbacks only — not wired in CI.

#### Sync after failed refresh

**`sync-json-to-mma-ai`** triggers on **`workflow_run` completed** but the job runs only when the upstream workflow **`conclusion == success`**. A failed weekly/monthly run produces a **skipped** sync — **`mma.ai`** keeps the last successful push.

You can still **manually** run **Sync JSON exports** with **`artifact_name: run-bundle`** to republish the newest stored bundle (which may be weeks old). That does not re-run the model; it only copies JSON from the artifact zip.

#### Sync artifact selection

- Prefer **`run-bundle`** (newest **`weekly-refresh-*`** / **`monthly-retrain-*`**).
- Legacy name **`mma-json-exports`** is aliased to **`run-bundle`** in **`scripts/ci_download_latest_artifact.py`** (that standalone artifact upload was removed May 2026).
- On **`workflow_run`** sync, **`TRIGGERING_WORKFLOW_RUN_ID`** prefers the bundle from the workflow that just finished (when present).

#### Secrets and artifacts

| Item | Notes |
|------|--------|
| **`MMA_AI_SYNC_PAT`** | PAT with **Contents: write** on **`mma.ai`** for sync push |
| **`mma-model-state`** | Pickle forwarded between weekly/monthly runs |
| **Run bundles** | Expire per GitHub retention; stale fallback quality depends on newest non-expired bundle |

#### Local disaster recovery (restore last CI build)

Use when a laptop is fresh, **`data/`** was wiped, or UFCStats scraping is broken but you still need the **last good** pickle, Tier‑1 CSVs, and deploy JSONs. Pulls the newest non-expired **`mma-model-state`** and **`weekly-refresh-*`** / **`monthly-retrain-*`** artifacts (same selection as **`run-bundle`**).

**Prerequisites:** [`gh`](https://cli.github.com/) logged in (`gh auth login`), repo root, deps installed (`pip install -r requirements.txt`).

```bash
cd /path/to/MMA_Handicapping
export GITHUB_REPOSITORY=LordOfTheTrees/MMA_Handicapping   # or your fork
export GITHUB_TOKEN="$(gh auth token)"

python scripts/ci_restore_model_artifact.py      # -> data/model.pkl
python scripts/ci_restore_data_bundle.py         # -> data/ufcstats_fights.csv, fighter_profiles.csv, upcoming_cards.json

python scripts/ci_download_latest_artifact.py run-bundle ./_artifact_restore
mkdir -p JSON_exports
cp -a ./_artifact_restore/JSON_exports/*.json JSON_exports/
rm -rf ./_artifact_restore
```

**Verify:**

```bash
python -c "import csv; print('fights', sum(1 for _ in csv.DictReader(open('data/ufcstats_fights.csv'))))"
ls -lh data/model.pkl JSON_exports/*.json
python -c "import json; print(json.load(open('JSON_exports/elo_states.json'))['export_manifest'])"
```

**Work offline** (no scrape): `python scripts/weekly_update.py refresh --data-dir ./data --model-path ./data/model.pkl --out-dir ./JSON_exports --no-scrape`

**Local workflow smoke** (ESPN sample + audit + ELO/export): `python scripts/weekly_update.py refresh --smoke-test --data-dir ./data --model-path ./data/model.pkl`

**Note:** `data/` is gitignored (`/data/` in `.gitignore`). Restored files stay local until the next restore or a manual scrape. Artifact expiry follows GitHub Actions retention (check Actions → Artifacts if downloads start failing).

---

## Harness (pickle vs JSON snapshot)

**Purpose:** prove that the five exported inference JSONs reproduce **`MMAPredictor.predict_proba_point_only`** when evaluated at the artifacts’ snapshot date.

**Date contract:** **`elo_states.json`** and **`style_axes.json`** store a single timeline slice: **`as_of_date`**. Snapshot inference ([`src/export/json_inference.py`](../src/export/json_inference.py)) only matches the pickle when **`fight_date == as_of_date`** for that export (same as [`scripts/export_artifacts.py`](../scripts/export_artifacts.py) `--as-of-date`). The pickle can differ for other dates because it runs full temporal ELO/style.

**Enable integration tests (export smoke + parity):** a trained **`model.pkl`** must exist:

1. **`MMA_HARNESS_MODEL`** — optional path override **only if** that file exists  
2. **`data/model.pkl`** at repo root (**default** after train)

**Commands** (from repo root):

```bash
# One entrypoint (recommended): see scripts/run_harness.py
python scripts/run_harness.py quick
python scripts/run_harness.py integration
python scripts/run_harness.py                         # full discover

# Raw unittest (same suites the script calls)
python -m unittest tests.test_json_snapshot_inference tests.test_upcoming_events_export tests.test_upcoming_bouts_parse -v
python -m unittest tests.test_site_export_pages -v
python -m unittest tests.test_export_artifacts_smoke tests.test_artifact_parity -v
```

**Console output:** The unittest **`skipped '…'`** line embeds **`HARNESS_SKIP_REASON`** (`tests.harness_skip`): env path and **`data/model.pkl`**, each with **`exists=`**. Loading those modules still prints the stderr banner (`print_harness_integration_preamble`).

If parity fails, **`assert_point_probs_match_pkl`** prints **per-class** pickle vs JSON values and **max_abs_delta** (treat as exporter/loader drift until fixed).

### Site page contracts (committed JSON vs `website_elements.md`)

**`tests/test_site_export_pages.py`** checks **`JSON_exports/*.json`** (override with **`MMA_SITE_EXPORT_DIR`**) against the SPA page inventory in **`docs/website_elements.md`**: home/upcoming calendar, rankings snapshot, fighter profile keys, bout/hypothetical inference via **`predict_proba_snapshot`**, about-model **`model_weights`** fields, and **`reference_distributions.json`** (same contract as **`mma.ai`** **`api/reference_distributions.py`**). **`python scripts/run_harness.py site`**. Subscription UI and Contact have no artifact contract here.

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
| **`mma.ai/api/inference.py`** | Loads five core JSON files (weights, ELO, style, profiles, **`reference_distributions.json`**); **`predict`** / **`build_features`** / search TODO |
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
| **`scripts/weekly_update.py`** | Operator path: **`refresh_data()`** by default (scrape; **`--no-scrape`** to skip), reload **`data/`**, **`build_elo`**, **`train_regression`** (`refresh` keeps regression **W**; `retrain` refits), write five JSONs, optional pickle update — see **`README.md`**. |
| **`scripts/export_artifacts.py`** | Pickle only → same five inference JSONs (no data reload); optional **`--rebuild-elo-for-trajectories`**. |
| **`scripts/export_upcoming_events.py`** | **`upcoming_events.json`** |
| **`scripts/copy_exports_to_mma_ai.py`** | Copies **`*.json`** into **`mma.ai/artifacts/`** |

Optional diagnostics and research CLIs: **`scripts/dev/`** ([`scripts/dev/README.md`](../scripts/dev/README.md)).

Details:

1. **`model_weights.json`** — **`W`** (6×12), bootstrap draws / config for CI routing (**`ModelConfig`**, **`ci_alpha`**, bootstrap count, elo_MC / Cauchy switches per training). **`export_manifest`** includes `git_sha_training`, `exported_at`, schema version.
2. **`reference_distributions.json`** — **`matchup_features`**: 101-point empirical quantiles per regression feature (percentiles 0…100). Optional **`global_days_idle`**. **`division_elo`**: per–weight-class ELO quantiles at snapshot. Training repo may add **`chart_histograms`** (bins/counts) for static charts; **`mma.ai`** preserves these keys after validation. **Layoff histogram + export contract:** [`docs/days-idle-histogram-for-mma-ai.md`](days-idle-histogram-for-mma-ai.md).
3. **`elo_states.json`**, **`style_axes.json`**, **`fighter_profiles.json`** — canonical field names in **`mma.ai/docs/export-artifacts-spec.md`** (sibling checkout). **`fighter_profiles`**: static fields from CSV plus optional **`elo_trajectories`**: `{ "<weight_class>": [ { "fight_date", "elo", "opponent_fighter_id" }, ... ] }` when the model was built with ELO trajectory recording (`weekly_update` default; or `export_artifacts.py --rebuild-elo-for-trajectories`).
4. Loads the **same** shipped **`MMAPredictor`** pickle as **`python main.py predict`** / **`explain`**.
5. **Parity harness:** [`tests/test_artifact_parity.py`](../tests/test_artifact_parity.py) reloads the temp export and compares to **`predict_proba_point_only`** (see **Harness** above).

**Exit:** five inference JSON files plus **`upcoming_events.json`** from a manual or CI run; parity tests run when **`data/model.pkl`** exists **or** **`MMA_HARNESS_MODEL`** points at a file that exists.

---

## Phase 2 — Hand off JSON to **mma.ai**

1. Drop files into **`mma.ai/artifacts/`** (manual copy, **`scripts/copy_exports_to_mma_ai.py`**, or CI artifact unzip — see **CI artifacts → local mma.ai** above).
2. **`sync-json-to-mma-ai.yml`** pushes after weekly/monthly CI ( **`run-bundle`** = newest weekly/monthly artifact) or manual run — needs **`MMA_AI_SYNC_PAT`**.
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

- **Implemented:** **`tests/test_artifact_parity.py`** (pickle **`predict_proba_point_only`** vs [`src/export/json_inference.py`](../src/export/json_inference.py)); optional **`mma.ai`** **`POST /api/predict`** cross-check remains a separate QA step.
- **`scripts/weekly_update.py`**, **`export_artifacts.py`**, **`export_upcoming_events.py`**, **`copy_exports_to_mma_ai.py`**
- Optional: GitHub Action to push **`mma.ai/artifacts`** after export
