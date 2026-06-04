# Data sources — ESPN (primary) and fallbacks

## Primary: ESPN public APIs (Tier 1 stats)

UFC per-fight totals (significant strikes, takedowns, control time, submissions) are ingested from ESPN’s undocumented MMA endpoints. Requests are:

- **Cached** under `data/cache/espn/` (repeat runs do not re-hit the network).
- **Rate-limited** via `ESPN_REQUEST_DELAY_SEC` in [`src/data/espn_client.py`](../src/data/espn_client.py) (default **0.45 s** between live HTTP calls).

### Operator commands

```bash
# Incremental merge into data/ufcstats_fights.csv (keeps UFCStats IDs via crosswalk)
python -m src.data.espn_ingest incremental --data-dir ./data

# Build / extend ID crosswalk from existing CSV + ESPN (run before heavy ingest)
python -m src.data.espn_ingest crosswalk --data-dir ./data --season 2024 --season 2025
# Per-event progress is on by default; use --quiet to hide. --log-network for each live GET.

# Profiles for crosswalk-linked fighters only
python -m src.data.espn_profiles --data-dir ./data
```

[`refresh_data()`](../src/data/refresh.py) and [`scripts/ci_try_refresh_data.py`](../scripts/ci_try_refresh_data.py) call the ESPN incremental path by default.

**Incremental window:** reads the latest `date` in `ufcstats_fights.csv` (`max_date`), then logs only **`pulling N event(s) on/after that date`**. Older cards in ESPN’s index are counted as skipped, not listed per event. Progress lines look like `[2/4] 2026-05-16 UFC Fight Night | …`.

### ESPN-derived files (explicit source in name)

| File | Purpose |
|------|---------|
| `data/espn_crosswalk_fights.csv` | `ufcstats_fight_id` ↔ `espn_competition_id` |
| `data/espn_crosswalk_fighters.csv` | `ufcstats_fighter_id` ↔ `espn_athlete_id` |
| `data/espn_ingest_state.json` | Scraped competition IDs, seasons touched, `last_run` new IDs |
| `data/espn_ingest_audit.json` | Post-ingest audit (fuzzy matches, rookie check, Method 5) |
| `data/cache/espn/*.json` | Raw API response cache |

### Duplicate / rookie audit

After incremental ingest, [`src/data/espn_audit.py`](../src/data/espn_audit.py) reviews every **new** `espn_*` fighter/fight from `last_run` in ingest state:

- **Method 1:** fuzzy name suggestions vs `fighter_profiles.csv`
- **Auto-reject:** fuzzy score ≥ 0.85 **and** identical `height_cm` + `reach_cm`
- **Rookie check:** ESPN athlete eventlog must show **exactly one** UFC bout while career W/L can be higher (pre-UFC)
- **Method 5:** fight-level consistency (hex fighters + `espn_*` fight id, opponent history, etc.)

[`refresh_data()`](../src/data/refresh.py) runs the audit after every ingest (used by **`weekly_update`**, **`main.py train --full-rebuild`**, and CI). Rejects fail the run unless `--allow-audit-failures` (weekly_update debug). CI sets `require_fight_updates=True` via [`ci_try_refresh_data.py`](../scripts/ci_try_refresh_data.py). Human-readable lines: [`scripts/espn_weekly_audit_report.py`](../scripts/espn_weekly_audit_report.py). Scheduled report workflow: `.github/workflows/espn-weekly-audit.yml`.

Local workflow smoke (caps ESPN, still runs audit on new ids from sample):

```bash
python scripts/weekly_update.py refresh --smoke-test --data-dir ./data --model-path ./data/model.pkl
```

Training artifacts keep **UFCStats column names and filenames** (`ufcstats_fights.csv`, `fighter_profiles.csv`). New fights that cannot be matched to historical rows may temporarily use `espn_{competition_id}` / `espn_{athlete_id}` until a full backfill crosswalk is run.

### GitHub Actions

- **Weekly / monthly:** `ci_try_refresh_data.py` → ESPN incremental (not a full-history scrape).
- **Artifacts:** Continue uploading `ufcstats_fights.csv` and `fighter_profiles.csv`; crosswalk files are included in the run bundle when present.
- **mma.ai sync:** Unchanged — `JSON_exports/` from the pickle/export step; fighter IDs in exports remain whatever is in the CSVs (crosswalk preserves hex IDs for matched history).

Scheduled **full retrain** (monthly) should use the same refresh path, then `weekly_update.py retrain --no-scrape` after CI refresh — see [BACKEND_PIPELINE_INTEGRATION.md](BACKEND_PIPELINE_INTEGRATION.md).

## Fallbacks (not implemented in ingest yet)

Documented for the next outage; do **not** scrape aggressively.

| Source | Role |
|--------|------|
| **UFCStats HTML** | Historical reference; blocked by Cloudflare in CI. Parser remains in `ufcstats_*` modules. |
| **Sherdog** | Tier 3 outcomes only (`tier3_sherdog.csv`) for ELO — no strike-level stats. |
| **Tapology** | Calendar / results only; not suitable for Tier 1 regression stats. |

## Working up to a full re-scrape

1. Ship crosswalk + incremental ESPN (current).
2. Optionally run `espn_ingest crosswalk` season-by-season off-peak (cached; still rate-limited).
3. When ready, run a controlled full ESPN backfill (extend `incremental` or add `full` with season caps) and then **monthly retrain** on Actions to refresh `model.pkl` and mma.ai JSON.
