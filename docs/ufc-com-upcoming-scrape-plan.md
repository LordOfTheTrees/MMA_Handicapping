# UFC.com upcoming-card ingest — exploration & design plan

**Status:** Not implemented. Exploration only — write-up so work can proceed locally
(this sandbox's outbound egress policy blocks `ufc.com`, `ufcstats.com`, and even
ESPN's API directly — see "Why this is a planning doc, not code" below).

Related: [`docs/todo.md`](todo.md) P0.1 ("UFC.com (official) upcoming"), **ADR-23**
in [`architecture-decisions.md`](architecture-decisions.md).

---

## 0. Does ESPN already have future-fight info? (yes — it's filtered out on purpose)

Before adding a third scrape target, it's worth being precise about what ESPN's API
already returns, because if it includes announced-but-unplayed bouts, extending the
**already-reliable** ESPN path is far cheaper than standing up a new scraper against
a site we haven't tested.

Looking at [`src/data/espn_ingest.py`](../src/data/espn_ingest.py) `_collect_incremental_events`:

```python
for event_ref in espn.list_event_refs(year):
    event = espn.fetch_event(event_ref)
    ...
    if event_date is None or event_date >= today:
        skipped += 1
        continue
```

`list_event_refs(season_year)` hits ESPN's **season-level** event index
(`.../seasons/{year}/types/2/events`) — this is the same index that backs ESPN's own
public UFC schedule page, which lists upcoming events with announced matchups well
before fight night. The code above resolves every event in that index (past *and*
future) and then explicitly **discards** anything dated today-or-later. That's a
deliberate choice tied to training-data hygiene (mirrors ADR-05: only completed
fights belong in `ufcstats_fights.csv`), **not** a limitation of the ESPN endpoint.

**Unverified but likely, pending a local check:** `fetch_event(event_ref)` on a future
event probably still returns its competitions (booked matchups) even though those
competitions have no result yet — ESPN's site surfaces "announced" fight cards this
way. If true, a **new, separate** function (not touching
`_collect_incremental_events`, which must stay training-safe) could resolve the same
season index, keep only `event_date >= today`, and pull competitor names from each
future event's competitions — reusing `ESPNClient.fetch_event` /
`list_competitor_refs` / `fetch_competitor` that already exist.

**Why this matters for prioritization:** ESPN has a 100% success rate across every
CI run checked (5/5 recent weekly runs) — zero Cloudflare issues — while UFCStats has
failed every single one of those same runs. If ESPN's future-event data holds up,
it's a lower-risk, lower-effort win than a brand-new ufc.com scraper, and should be
tried first. Treat §2 below (ufc.com) as a **parallel/fallback** effort, not a
replacement for checking this.

**Action (local, since this sandbox can't reach ESPN's API either — see below):**
fetch one `event_ref` for a known future event and confirm competitions/competitors
are populated before writing any code against this hypothesis.

---

## Why this is a planning doc, not code

This sandbox's outbound network goes through an org-level egress proxy that denies
(403 at the CONNECT level, not a site-side block) every host tried during this
session: `www.ufc.com`, `sports.core.api.espn.com`. That's a policy decision for
*this* session, separate from whether GitHub Actions runners or your local machine
can reach these hosts. Concretely, this means:

- I cannot inspect ufc.com's real markup from here to pin down selectors.
- I cannot confirm the ESPN future-event hypothesis in §0 from here either.

Both need a local (or GH Actions `workflow_dispatch` smoke) check before code is
written against assumed selectors/response shapes.

---

## 1. Comparative rationale: three candidate sources

| Source | CI reliability (observed) | Data richness | Effort | Risk |
|---|---|---|---|---|
| **UFCStats** (current) | 0/5 recent weekly runs — Cloudflare-blocked every time | Full: hex fighter IDs, weight class, bout order | Already built | Site-side bot wall shows no sign of clearing |
| **ESPN API, extended** (§0) | 5/5 — proven reliable channel | Needs local check: matchups yes, but ESPN IDs only (crosswalk to hex IDs already exists) | Small — reuses `ESPNClient`, new non-training code path | Untested for future events specifically; ESPN could change the schedule endpoint |
| **ufc.com** (§2, this doc) | Unknown — untested from any environment so far | Names + weight class; **no** fighter IDs of any kind (see §2.3) | Medium — new module, new parser, new tests | Unknown bot-protection posture; even if your laptop can reach it, GitHub Actions' shared runner IPs might get the same Cloudflare treatment UFCStats gives them (verify separately — do not assume local success implies CI success) |

Recommended sequencing: verify §0 locally first (cheap, high-confidence win if it
pans out); build §2 in parallel as a second, independent source since even a
name-only fallback is strictly better than the current "nothing" when both UFCStats
and ESPN's future-event data are unavailable for some reason.

---

## 2. UFC.com scraper design

### 2.1 Target and open structural questions (must confirm locally first)

Starting point per your message: `https://www.ufc.com/events#events-list-upcoming`.
The `#events-list-upcoming` fragment is a client-side anchor — it does not change
what the server returns, so the first thing to confirm locally (view-source, or
curl with a browser UA) is:

- **Is the full upcoming-events list server-rendered in the initial HTML**, or does
  the site (ufc.com runs on Drupal) load additional events via an AJAX "load more" /
  Views pagination call (typically a `POST` to something like `/views/ajax`)? This
  determines whether the **outer loop** (see 2.2) is a single parse or a
  fetch-then-paginate loop.
- What does an **event card** in the listing look like (wrapping tag/class), and
  what does its **detail-page URL** look like (e.g. `/event/ufc-3xx-name-vs-name`)?
- What does an **individual event page's fight-card markup** look like — one
  consistent list/table of bouts, or split into "main card" / "prelims" / "early
  prelims" sections that need separate selectors?
- Is there any bot-challenge / CAPTCHA page (Cloudflare, Akamai, PerimeterX) served
  to non-browser clients, and does it look like UFCStats' Cloudflare page or
  something else? This determines the shape of the `probe`-equivalent (2.4).

None of this should be guessed — capture 2-3 real HTML snapshots (listing page +
2 event detail pages, ideally one fully-announced and one with TBD undercard slots)
and save them as test fixtures (2.6) before writing the parser.

### 2.2 Nested-loop structure

Mirrors the shape you described — outer loop over the listing, inner loop over each
event's own page:

```
outer loop: for each upcoming event found on the listing page(s)
    (if listing paginates via AJAX: keep requesting next page until empty)
    → collect event_url, event_title, event_date, location from the card

    inner loop: for each bout row found in this event's detail page
        → fetch event_url once
        → parse announced fighter_a_name, fighter_b_name, weight_class, bout_order
        → tolerate rows with a TBD/unannounced opponent (partial cards are normal
          for far-out events) — skip or mark such rows, do not error out
```

This is structurally identical to `src/data/ufcstats_upcoming.py`'s existing
`scrape_upcoming_cards()` (outer loop over `iter_completed_event_urls`, inner call to
`parse_upcoming_bouts_from_event_soup` per event) — reuse that shape and, where
possible, the same throttling helpers (`_throttle`, a `REQUEST_DELAY_SEC`-style
module constant) from `src/data/ufcstats_scraper.py` rather than inventing new
rate-limiting.

### 2.3 Module & output schema (integration point)

**New module:** `src/data/ufc_upcoming.py`, exposing the same public shape as
`ufcstats_upcoming.py` so it's a drop-in alternate source behind an identical
interface:

- `scrape_upcoming_cards(...) -> Dict[str, Any]`
- `scrape_upcoming_cards_to_path(path, ...) -> Path`
- `parse_upcoming_bouts_from_event_soup(soup) -> List[Dict[str, Any]]`
- A `probe_*`-style pre-check (2.4)

**Output schema:** reuse the *exact* `upcoming_cards.json` shape (`schema_version`,
`source`, `scraped_at`, `events[].{event_url, event_id, event_title, event_date,
location, bouts[]}`, `bouts[].{bout_order, fighter_a_name, fighter_b_name,
weight_class, weight_class_raw}`), just with `source` set to the ufc.com URL. Reusing
the schema means `src/export/upcoming_events_doc.py`'s `build_upcoming_events_doc`
needs **zero** changes to consume either source.

**Key gap vs UFCStats:** ufc.com bout listings will not carry UFCStats hex
`fighter_a_id` / `fighter_b_id` — there's no such ID system on that site. Two
options, not mutually exclusive:

1. **Ship name-only bouts** (`fighter_a_id`/`fighter_b_id` = `None`) for the
   ufc.com-sourced file. Degraded but still useful for the site's calendar/card UI,
   which can fall back to name display when no ID-linked profile exists.
2. **Resolve names against existing fighters** using the crosswalk name-matching
   already built for ESPN ingest — `build_name_index` /
   `build_name_index_from_profiles` in `src/data/espn_crosswalk.py` — rather than
   writing new fuzzy-matching logic. Only attempt this for fighters who already have
   a profile; newly-debuting fighters stay name-only until UFCStats/ESPN catches up
   post-fight.

Recommend (1) first (cheap, always available) with (2) as a follow-up enhancement.

### 2.4 Failure handling (non-negotiable, given what just broke)

This is the part of the original bug we're not allowed to repeat: `refresh_data()`'s
UFCStats path used to swallow both a Cloudflare block *and* any scrape exception
with a bare `print(...)`, so nothing ever surfaced as a CI failure or even a
distinguishable "this run didn't get fresh data" signal until we added
`RefreshResult.upcoming_cards_scraped` (see the weekly/monthly pipeline changes
already landed on this branch). Any new source must follow the same pattern from
day one:

- A `probe_ufc_com_upcoming_index()`-style function that makes one request and
  returns a typed result (`blocked: bool`, `detail: str`) distinguishing "bot
  challenge page" from "empty listing" from "network error" — same shape as
  `probe_completed_events_index` in `ufcstats_scraper.py`.
- The top-level scrape function must let the caller know definitively whether it
  produced fresh data this run (return a bool / raise a typed exception) — no bare
  `except Exception: print(...)` that looks like success to everything downstream.
- Whatever calls this from `refresh_data()` must feed the **same**
  `upcoming_cards_scraped`-style gating already wired into `weekly_update.py` /
  `ci_try_refresh_data.py` / the CI workflows, extended to be `True` if *either*
  UFCStats or ufc.com produced fresh data this run — not a second, parallel signal
  that the export step has to separately remember to check.

### 2.5 Integration point in `refresh_data()`

Proposed: inside the existing `if ufcstats_gap_fill:` block in
[`src/data/refresh.py`](../src/data/refresh.py), after the UFCStats attempt, add a
fallback attempt against ufc.com **only when UFCStats itself didn't produce fresh
data this run** (blocked or exception) — write to a separate file
(`data/upcoming_cards_ufc_com.json`) rather than overwriting UFCStats' output, so a
bad scrape from one source can never corrupt or silently replace good data from the
other. `export_upcoming_events.py` (or a small wrapper) picks whichever file has the
newer `scraped_at`, or merges by event date if both are present and non-stale.
Keeping sources in separate files also makes it trivial to compare them side-by-side
while ufc.com's reliability is still unproven.

### 2.6 Testing

Same approach as [`tests/test_upcoming_bouts_parse.py`](../tests/test_upcoming_bouts_parse.py):
commit real captured HTML as fixtures (listing page + at least 2 event detail pages,
including one with an unannounced/TBD slot) and test the parser against those
fixtures. This keeps the test suite fully offline — it doesn't matter whether CI can
reach ufc.com, since the parser logic is verified against known-good saved markup,
exactly like the existing UFCStats bot-challenge tests
(`tests/test_ufcstats_bot_challenge.py`) do.

### 2.7 Verification checklist before this counts as "done"

1. Confirm the listing page's pagination mechanism and one event detail page's fight
   card markup locally (2.1).
2. Build the module + fixtures + tests locally; get it working end-to-end from a
   machine with normal internet access.
3. **Separately** verify ufc.com is reachable from an actual GitHub Actions runner —
   add a `workflow_dispatch`-only smoke step that just probes ufc.com and reports
   blocked/ok, run it once, before wiring it into the real weekly/monthly jobs. Do
   not assume "works on my laptop" implies "works in CI" — that gap is exactly what
   went unnoticed with UFCStats for the past month.
4. Wire into `refresh_data()` per §2.5, extend the `upcoming_cards_scraped` gating,
   and confirm (via a real run, not just code review) that `JSON_exports/upcoming_events.json`
   actually gets produced when UFCStats is blocked but ufc.com isn't.

---

## Open questions

- Does ufc.com's fight-card markup expose a stable per-bout "card segment" (main
  card / prelims) worth capturing, or is `bout_order` (as UFCStats already does)
  sufficient for the site's rendering needs?
- If both ESPN's future-event data (§0) and ufc.com (§2) pan out, is there value in
  keeping both long-term (cross-check / redundancy), or should one become primary
  once proven and the other dropped to reduce maintenance surface?
