# What to work on next

This file is the **human-facing roadmap**: big picture first, then the concrete slices you can start this week. For exhaustive checklists, column specs, and phase-by-phase notes, see [`docs/todo.md`](docs/todo.md).

---

## High level — what matters next

1. **Real data in the box**  
   The training code path exists, but the project is still blocked on **populating `data/`** with the CSVs the loader expects (`tier1_ufcstats.csv`, `fighter_profiles.csv`, optional tier2/3). Until that lands, everything downstream is theoretical.

2. **Wire refresh into “full rebuild”**  
   `python main.py train --full-rebuild` calls `refresh_data()` in [`src/data/refresh.py`](src/data/refresh.py). Right now it is a stub—once you have scrapers or export scripts, **implement that function** (or delegate to them) so one command refreshes files and retrains.

3. **First honest run on real data**  
   After data exists: train end-to-end, spot-check counts, run a prediction, confirm symmetry (A vs B). Goal is *confidence the pipeline doesn’t lie*, not perfect accuracy yet.

4. **Validation, then tuning**  
   Holdout split, log-loss / calibration, then tune era cutoff and ELO knobs one at a time—as laid out in the architecture and `docs/todo.md`.

5. **Hardening and upkeep**  
   Tests, `requirements.txt`, weight-class moves, CI coverage checks, and eventually a repeatable “after each event” refresh story.

---

## Soon — detailed starter chunks

Pick one vertical and finish it before starting the next; they’re ordered roughly by dependency.

### A. Tier 1 UFCStats → `data/tier1_ufcstats.csv`

- **Decide canonical `fighter_id`.** Strong default: UFCStats fighter URL slug, same string in every CSV.
- **Event listing → fight IDs.** Crawl completed events, collect fight detail URLs or IDs.
- **Per-fight scrape.** For each fight: outcome, method, weight class, date, fight time, and A/B stat columns expected by the loader (see table in `docs/todo.md` §1.1).
- **Write CSV** to `data/tier1_ufcstats.csv` and sanity-check row count (order of thousands of UFC fights for the modern era).

### B. Profiles → `data/fighter_profiles.csv`

- **Fighter index → profile pages.** Height, reach, DOB, stance; map to the same `fighter_id` as Tier 1.
- **Pedigree fields** (`wrestling_pedigree`, etc.): start at `0.0` everywhere if you want speed; refine later for cold starts only.

### C. Hook up `refresh_data`

- **Implement** [`src/data/refresh.py`](src/data/refresh.py) so it: ensures `data/` exists, runs your scrape/export pipeline, writes/updates the CSVs above (and optional tier 2/3 when you add them).
- **Smoke test:** `python main.py train --data-dir ./data --full-rebuild` then `python main.py train --data-dir ./data` (second run = model-only from disk).

### D. First pipeline smoke test (after A + B)

- Train: `python main.py train --data-dir ./data`
- Check loaded fight and profile counts, training set size, no non-finite features.
- One `predict` and one `explain` on a known matchup using real IDs from your CSVs.

### E. Quick quality gates (before big tuning)

- Add or run a **symmetry check** (swap A/B; win/lose probabilities flip)—see `docs/todo.md` §2.2.
- **ELO sanity:** top/bottom of division vs your intuition; famous names in plausible neighborhoods.

---

## Reference

| Topic | Where |
|--------|--------|
| Full phased checklist, schemas, metrics | [`docs/todo.md`](docs/todo.md) |
| Design and stage definitions | [`docs/architecture.md`](docs/architecture.md) |
| Expected CSV filenames / loader behavior | [`src/pipeline.py`](src/pipeline.py) (`load_data`) |

When you finish a chunk above, tick the matching boxes in `docs/todo.md` so the detailed doc stays the source of truth for progress.
