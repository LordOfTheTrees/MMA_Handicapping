# MMA Predictions Website — Architecture & Wireframes

## Overview

A React SPA frontend served by a FastAPI backend, deployed on Render.com with a custom domain. The production server contains **zero model training code** — only exported trained artifacts (JSON) and a standalone inference module. All training stays in this private repo.

---

## Goals

- Interactive, visually rich matchup tool and fighter explorer
- Custom domain (**production:** **`https://octagonelo.com`** on Render HTTPS), SEO-friendly (Bing/Google indexable)
- ~$7/month hosting (Render.com Web Service)
- Model IP fully protected — methodology never leaves this private repo

---

## Two-Repo Architecture

### Repo 1: `MMA_Handicapping` (this repo, private)
Training code, data, scrapers, pipeline — everything stays here. Gets one new script:
- `scripts/export_artifacts.py` — serializes trained model state to portable JSON files

### Repo 2: `mma-predictions-web` (new Render deploy repo)
Only what production needs:

```
mma-predictions-web/
├── api/
│   ├── app.py                  # FastAPI app — serves React build + /api/* routes
│   ├── inference.py            # Standalone predictor (~150 lines, no src/ imports)
│   └── routes/
│       ├── predict.py          # POST /api/predict
│       ├── fighters.py         # GET /api/fighters?q=  and  GET /api/fighters/{id}
│       └── events.py           # GET /api/events/upcoming
├── frontend/                   # React SPA (Vite build)
│   └── src/
│       ├── components/
│       │   ├── MatchupBuilder.jsx
│       │   ├── ProbabilityBars.jsx
│       │   ├── FighterProfile.jsx
│       │   └── EventCard.jsx
│       └── pages/
├── artifacts/                  # Pushed here by GH Actions after each model refit
│   ├── model_weights.json      # W matrix (6×12) + 200 bootstrap draws
│   ├── elo_states.json         # Per-fighter, per-weight-class ELO state
│   ├── style_axes.json         # Per-fighter, per-weight-class style scores
│   └── fighter_profiles.json   # Name, reach, height, stance, pedigree
├── render.yaml
└── requirements.txt            # fastapi, uvicorn, numpy, scipy, rapidfuzz
```

---

## What Gets Exported (Artifacts)

`scripts/export_artifacts.py` loads a trained `MMAPredictor` pickle and writes:

| File | Contents | Est. Size |
|------|----------|-----------|
| `model_weights.json` | Regression W matrix (6×12) + 200 bootstrap draws + config | ~500 KB |
| `elo_states.json` | `{fighter_id: {weight_class: {elo, uncertainty, last_fight_date, n_fights}}}` | ~2 MB |
| `style_axes.json` | `{fighter_id: {weight_class: {striker_score, grappler_score, finish_threat, finish_vulnerability}}}` | ~2 MB |
| `fighter_profiles.json` | `{fighter_id: {name, reach_cm, height_cm, dob, stance}}` | ~1 MB |
| `reference_distributions.json` | **`matchup_features`** (101-point quantiles per feature), **`division_elo`**, optional **`global_days_idle`**; optional **`chart_histograms`** for SPA | ~150–800 KB |

No pipeline code, no training data, no scraping logic — only the learned numeric state.

---

## Standalone Inference Module

`api/inference.py` (~150 lines, `numpy` + `scipy` + `rapidfuzz` only):

1. Load core JSON artifacts at startup (cached in memory): weights, ELO, style, profiles, and **`reference_distributions.json`** (quantile grids + optional chart histograms)
2. `fuzzy_search(query)` → ranked fighter list
3. `build_matchup_features(a_id, b_id, weight_class, date)` → 12-element vector
4. `predict(features)` → softmax(W @ x) → 6-class probabilities
5. `bootstrap_ci(features, alpha=0.10)` → percentile CIs from stored 200 draws

---

## API Endpoints

```
POST /api/predict
  { fighter_a_id, fighter_b_id, weight_class, date }
  → PredictionJSON

GET  /api/fighters?q=name&limit=10
  → [{id, name, elo, weight_classes}]

GET  /api/fighters/{id}
  → full profile + ELO by weight class + style axes

GET  /api/events/upcoming
  → card + pre-computed matchup predictions
```

---

## Self-Running Refresh (GitHub Actions, this private repo)

```
Weekly cron (Monday 10am UTC)
  → src/data/refresh.py         (scrape latest UFC data)
  → src.cli.train               (refit with frozen Phase 3 config)
  → scripts/export_artifacts.py (serialize to artifacts/*.json)
  → git push artifacts → mma-predictions-web repo (deploy key)
  → Render auto-deploys on push
```

Training never leaves this private repo. Only JSON artifacts cross to the web repo.

---

## Wireframes

Design language: **dark background, glassmorphism cards, accent color (red/gold UFC palette), smooth Framer Motion transitions.** Tailwind CSS for layout.

---

### Page 1 — Home / Matchup Builder

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  ⚔  MMA PREDICTIONS                              [Events]  [Fighters]  [↗]  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│           MATCHUP BUILDER                                                   │
│           ──────────────────────────────────────────────────                │
│                                                                             │
│   ┌──────────────────────┐   VS   ┌──────────────────────┐                 │
│   │ 🔍 Search fighter A  │        │ 🔍 Search fighter B  │                 │
│   │ ──────────────────── │        │ ──────────────────── │                 │
│   │ Islam Makhachev  ✓   │        │ Charles Oliveira ✓   │                 │
│   │ ──────────────────── │        │ ──────────────────── │                 │
│   │ ELO: 1847  LW  ████  │        │ ELO: 1721  LW  ███   │                 │
│   │ Striker:  ▓▓▓▓▓░░░░  │        │ Striker:  ▓▓▓░░░░░░  │                 │
│   │ Grappler: ▓▓▓▓▓▓░░░  │        │ Grappler: ▓▓▓▓░░░░░  │                 │
│   │ Finish ↑: ▓▓▓▓░░░░░  │        │ Finish ↑: ▓▓▓▓▓▓░░░  │                 │
│   └──────────────────────┘        └──────────────────────┘                 │
│                                                                             │
│   Weight Class:  [ Lightweight ▼ ]     Date: [ 2026-06-07 ]                │
│                                                                             │
│                    [ PREDICT  ▶ ]                                           │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│  PREDICTION RESULT                                    CI: 90%  [Explain ↓] │
│                                                                             │
│  Islam Makhachev wins                              65%                      │
│                                                                             │
│  Win KO/TKO      ████████░░░░░░░░░░░░░░    14%  [9–20%]                    │
│  Win Submission  ██████████████░░░░░░░░    18%  [12–25%]                   │
│  Win Decision    █████████████████████░    33%  [25–41%]   ← most likely   │
│  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─               │
│  Lose Decision   ██████████░░░░░░░░░░░░    16%  [10–23%]                   │
│  Lose KO/TKO     ████████░░░░░░░░░░░░░░    10%  [5–16%]                    │
│  Lose Submission ██████░░░░░░░░░░░░░░░░     9%  [5–15%]                    │
│                                                                             │
│  ┌────────────────────────────────────────────────────────────────────┐     │
│  │  [Explain: feature log-odds breakdown ▼]                           │     │
│  │  ELO differential (+125.7)     → +0.38 log-odds toward win        │     │
│  │  Grappling matchup (0.31 edge) → +0.22 log-odds toward submission │     │
│  │  Striker score (−0.18)         → −0.09 log-odds against KO win    │     │
│  └────────────────────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Notes:**
- Fighter cards update live as user types (debounced fuzzy search, 300ms)
- Probability bars animate in on result load (Framer Motion stagger)
- CI ranges shown as faded bar extensions
- Explain panel collapses/expands with animation
- Swapping A↔B flips the prediction (antisymmetry indicator in UI)

---

### Page 2 — Fighter Profile

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  ⚔  MMA PREDICTIONS                              [Events]  [Fighters]  [↗]  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ← Back     ISLAM MAKHACHEV                                                 │
│             Lightweight  •  Orthodox  •  Age 32                             │
│             Reach: 178cm  •  Height: 171cm                                  │
│                                                                             │
│  ┌─────────────────────────┐  ┌─────────────────────────────────────────┐  │
│  │   STYLE RADAR           │  │   ELO TRAJECTORY  (Lightweight)         │  │
│  │                         │  │                                         │  │
│  │       Striker           │  │  1900 ─                          ╭──    │  │
│  │         ▲               │  │  1800 ─               ╭────────╯        │  │
│  │    ╱‾‾‾‾‾‾‾‾‾╲          │  │  1700 ─     ╭────────╯                  │  │
│  │   │   ██████  │         │  │  1600 ─╭────╯                           │  │
│  │Fin│ ██      ██│Grp      │  │        2019  2021  2023  2025           │  │
│  │   │   ██████  │         │  │                                         │  │
│  │    ╲___________╱        │  │  Current ELO: 1847  (±62 uncertainty)   │  │
│  │       Vuln              │  └─────────────────────────────────────────┘  │
│  └─────────────────────────┘                                               │
│                                                                             │
│  [ USE IN MATCHUP BUILDER ]                                                 │
│                                                                             │
│  RECENT FIGHTS                                                              │
│  ─────────────────────────────────────────────────────────────────────      │
│  Jun 2024  vs Dustin Poirier     W Decision  ELO +18   LW                  │
│  Oct 2023  vs Alexander Volkan.  W Submission ELO +31  LW                  │
│  Feb 2022  vs Charles Oliveira   W Submission ELO +42  LW                  │
│  ...                                                                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Notes:**
- Radar chart: 4-axis (Striker, Grappler, Finish Threat, Finish Vulnerability) using Recharts
- ELO trajectory: line chart with uncertainty band (shaded ±1σ)
- ELO shown per weight class (tab selector if fighter competed in multiple)
- "Use in Matchup Builder" button pre-populates home page fighter slot

---

### Page 3 — Upcoming Events / Card View

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  ⚔  MMA PREDICTIONS                              [Events]  [Fighters]  [↗]  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  UFC 314  •  June 7, 2026  •  Las Vegas                                     │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  MAIN EVENT                                              [ ↓ More ] │   │
│  │                                                                     │   │
│  │  Islam Makhachev          65%  ████████████████░░░░  35%            │   │
│  │                     VS         LW Championship                      │   │
│  │  Charles Oliveira         35%  ░░░░████████████████  65%            │   │
│  │                                                                     │   │
│  │  Most likely: Makhachev wins by Decision (33%)                      │   │
│  │  Finish probability: 41%  |  [Full Breakdown ↓]                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  CO-MAIN EVENT                                           [ ↓ More ] │   │
│  │  Fighter A                58%  ███████████████░░░░░  42%            │   │
│  │                     VS         Featherweight                        │   │
│  │  Fighter B                42%  ░░░░░███████████████  58%            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  [ + 8 more bouts ]                                                         │
│                                                                             │
│  ─────────────────────────────────────────────────────────────────────      │
│  Past Events   [ UFC 313 ▶ ]  [ UFC 312 ▶ ]  [ UFC 311 ▶ ]                │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Notes:**
- Cards collapsed by default, expand to show full 6-class breakdown
- Win probability shown as a split bar (A left, B right)
- "Most likely outcome" label beneath the bar
- Past events show model accuracy retroactively (if fight has occurred)

---

## SEO Strategy

Canonical origin: **`https://octagonelo.com`**.

Fighter and event pages are pre-rendered as HTML (server-side via FastAPI Jinja2 templates or a static generation step) so Bing can index them without running JavaScript.

- `https://octagonelo.com/fighters/islam-makhachev` — indexable fighter page
- `https://octagonelo.com/events/ufc-314` — indexable event page
- `https://octagonelo.com/sitemap.xml` — auto-generated from fighter profiles + event list
- Each page has `<title>`, `<meta description>`, and JSON-LD structured data

---

## Implementation Phases

| Phase | Work | Est. Time |
|-------|------|-----------|
| 1 | `scripts/export_artifacts.py` + round-trip validation | 1 day |
| 2 | `api/inference.py` standalone predictor + FastAPI skeleton | 1–2 days |
| 3 | Render deploy + **`octagonelo.com`** custom domain + HTTPS | 1 day |
| 4 | React frontend (matchup builder, probability viz) | 3–5 days |
| 5 | Fighter profiles + event card pages | 2–3 days |
| 6 | SEO (pre-render, sitemap) + GH Actions self-runner | 1–2 days |

---

## Existing Code to Reuse (from this repo)

| What | File | For |
|------|------|-----|
| `MMAPredictor.load()` / `._regression_W` / `._bootstrap_W` | `src/pipeline.py` | Export script |
| `ELOState`, `StyleAxes`, `FighterProfile`, `PredictionResult` | `src/data/schema.py` | Serialization |
| `src/data/fighter_names.py` fuzzy lookup | `src/data/fighter_names.py` | Reference impl for `inference.py` |
| `resolve_weight_class()` | `src/cli/common.py` | Reference impl for API input parsing |
| `src/data/refresh.py` | `src/data/refresh.py` | GH Actions scrape step |

---

## Verification Checklist

- [ ] `scripts/export_artifacts.py` produces 4 valid JSON files from trained pickle
- [ ] `api/inference.py` prediction matches CLI `python main.py predict` within float tolerance
- [ ] `POST /api/predict` returns 6 probs summing to 1.0
- [ ] Fighter fuzzy search returns correct results within 300ms
- [ ] React SPA loads and matchup builder works end-to-end in browser
- [ ] **`octagonelo.com`** resolves with HTTPS on Render
- [ ] Bing Webmaster Tools confirms sitemap indexed
- [ ] GitHub Actions refresh workflow completes on manual trigger
