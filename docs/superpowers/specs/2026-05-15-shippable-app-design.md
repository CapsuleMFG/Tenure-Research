# Shippable App — Design (v1.0)

**Date:** 2026-05-15
**Status:** Approved (autonomous brainstorming under `/goal`)
**Working name:** `LivestockBrief`
**Goal:** Turn the sandbox into a shippable, presentable app that brings value to non-technical users.

> **Note on CLAUDE.md.** The repo's `CLAUDE.md` says "No hosted/multi-user dashboard." The user's `/goal` directive explicitly overrides that line for v1.0. All other non-negotiables (classical methods only, tidy parquet schema, seeded RNG, idempotent ingest, laptop-friendly) remain in force.

---

## 1. Audience & value proposition

**Audience:** Cattle and hog producers, ag analysts, commodity-curious readers. Not data scientists, not forecaster tuners.

**Value prop:** *"USDA-grounded livestock price forecasts in plain English, with honest uncertainty."*

A casual visitor should, in <30 seconds, know:
- What current livestock prices are doing (latest, MoM, YoY).
- Where the next 6 months are forecast to go and with what uncertainty.
- Why they should trust the numbers (USDA source, classical methods, calibrated PIs).

---

## 2. What's in scope for v1.0

1. **Home / Market Brief page** — auto-generated headline + commodity cards.
2. **Commodity / series detail view** — restyled deep dive.
3. **Forecast page** — simplified default with an `Advanced` expander for analysts.
4. **Methodology page** — plain-English explanation of data + models + uncertainty.
5. **About page** — who, why, version, data freshness, link to GitHub.
6. **Visual identity** — custom CSS theme, typography, color palette, favicon.
7. **Precomputed forecast cache** — all winning-model forward forecasts baked weekly so the Home page is instant.
8. **Scheduled data refresh** — GitHub Actions cron pulls ERS + futures + reruns precompute weekly, commits the parquet + cache to a branch.
9. **Streamlit Community Cloud deployment configuration** — `streamlit_app.py` entrypoint at repo root, `requirements.txt` exported from `uv`, `.streamlit/config.toml` for theming.
10. **Error / empty states** — graceful UI when data is loading, missing, or a forecaster fails.
11. **README + deploy instructions** updated for production posture.

## 3. What's out of scope for v1.0

- User accounts, auth, watchlists, persistence per user.
- Email / push alerts.
- Paid tier, Stripe, any monetization.
- Mobile-native (we will be responsive, not native).
- Additional data sources beyond USDA ERS + CME futures via yfinance.
- An API.
- Additional forecasting models (no NeuralProphet/Chronos/TFT — violates CLAUDE.md "classical only").
- The interactive backtest UI as a top-level page (it stays, but moves behind `Advanced` on the Forecast page).

---

## 4. Information architecture

```
Sidebar (slim):
  ├── Brief                      (home)
  ├── Commodities                (group landing pages)
  ├── Series detail              (single-series deep dive — current Visualize)
  ├── Forecast                   (simplified + Advanced)
  ├── Methodology
  └── About
```

Streamlit's native multipage system stays. Files in `dashboard/pages/` will be renamed and reordered to drive this nav.

### 4.1 Home / Brief page

Three blocks, top to bottom:

**Block A — Headline brief (auto-generated):**
A 2-3 sentence deterministic template per featured anchor series, filled with current values:

> *"{series_name} closed {month} at {latest_price} per {unit}, {direction_phrase_mom} {abs_mom_pct}% MoM and {direction_phrase_yoy} {abs_yoy_pct}% YoY. Our {horizon}-month {winner_model} forecast expects {forward_point} per {unit} by {forward_date}, with 80% confidence the price lands between {pi_lower} and {pi_upper}."*

Implementation: `dashboard/components/brief.py::compose_brief(series_id)` consumes the precomputed forecast cache and returns a string.

**Block B — Commodity cards** (3 columns desktop, 1 column mobile):
One card per "headline series":
- **Cattle (fed)** — `cattle_steer_choice_nebraska`
- **Cattle (feeder)** — `cattle_feeder_steer_750_800`
- **Hogs** — `hog_barrow_gilt_natbase_51_52`
- **Beef wholesale** — `boxed_beef_cutout_choice`
- **Pork wholesale** — `pork_cutout_composite`
- **Lamb** — `lamb_slaughter_choice_san_angelo`

Each card shows:
- Big latest price ($X.XX / unit)
- 24-month sparkline
- MoM and YoY badges (colored: green = up for prices, red = down — semantic for producers selling)
- One-line forecast hint: "→ ~$233 by Oct '26 (±$15)"
- Click anywhere on card → series detail page

**Block C — Methodology hint footer:** one-line link to `Methodology` page.

### 4.2 Series detail page

The existing `pages/2_Visualize.py` polished:
- Replace `st.title` and `st.subheader` with custom-styled headers (CSS classes).
- Move the catalog `notes` field to a prominent intro card instead of a footnote line.
- Keep the chart, YoY chart, seasonal decomposition.
- Add at the bottom: "**Forecast for this series →**" link to Forecast page pre-loaded with this series_id.

### 4.3 Forecast page

**Default state** (what 95% of users see):
- Series picker (slim).
- "What we think the next 12 months look like" headline.
- The winning model's 12-month forecast chart with conformal-calibrated 80% PI.
- A plain-English readout block: forecast trajectory in words, with a small uncertainty caveat.
- Source line: "Forecast: {winner_model} (selected from bake-off over 12 CV windows). Calibrated 80% PI."

**Advanced expander** (analysts only):
- Wraps the existing controls: horizon slider, CV windows slider, model multiselect.
- "Run backtest" button.
- Scoreboard, CV overlay chart, residual diagnostics — i.e., everything from today's Forecast page.

All today's Forecast page logic is preserved verbatim, just collapsed into the expander.

### 4.4 Methodology page

A single markdown page (with a couple of `st.expander` blocks) covering:
- Where the data comes from (USDA ERS Livestock and Meat Domestic Data + CME futures via yfinance).
- Refresh cadence (weekly on Sunday night UTC).
- The three forecasters in two sentences each (AutoARIMA, Prophet, LightGBM).
- What "80% prediction interval" means + how conformal calibration works in plain English.
- A "Known limits" list (regime shifts, slaughter shocks, etc. — pulled from the README "what I learned" section).

### 4.5 About page

- One paragraph: who built it, why.
- "Last data refresh: {timestamp}" pulled live from `observations.parquet` mtime.
- Version: from `usda_sandbox.__version__`.
- Link to GitHub repo.
- Disclaimer: not financial advice; educational and informational.

---

## 5. Visual identity

### 5.1 Theme

`.streamlit/config.toml`:
```toml
[theme]
base = "light"
primaryColor = "#B4521E"          # clay/rust accent
backgroundColor = "#FAF7F2"       # parchment
secondaryBackgroundColor = "#F0EBE1"
textColor = "#1F1F1F"
font = "serif"
```

Custom CSS injected once via `dashboard/components/theme.py::inject_global_css()`:
- Headlines: serif (system serif fallback chain — no remote fonts to keep it laptop-friendly).
- Body: sans (Inter / system sans).
- Commodity card style: white background, 1px border in muted clay, 12px border-radius, 16px padding, hover lift.
- "Brief" headline block: large serif, slightly larger leading.
- Sidebar branding header.

### 5.2 Favicon and page title

- Replace 🐂 emoji with a small wordmark favicon (`dashboard/static/favicon.png`) and "LivestockBrief" as page title.
- Per-page `page_title` set explicitly.

---

## 6. Data flow

```
GitHub Actions (cron, weekly Sunday 21:00 UTC)
  ↓
  sync_downloads() → data/raw/*.xlsx
  sync_continuous_futures() → data/raw/futures_continuous/*.csv
  clean_all() → data/clean/observations.parquet
  precompute_forecasts() → data/clean/forecasts.json
  git commit, push to `data` branch
  ↓
Streamlit Community Cloud pulls `data` branch automatically
  ↓
Pages read from observations.parquet + forecasts.json
```

### 6.1 `forecasts.json` cache schema

```json
{
  "generated_at": "2026-05-12T21:03:11Z",
  "by_series": {
    "cattle_steer_choice_nebraska": {
      "winner_model": "AutoARIMA",
      "winner_metrics": {"mape": 6.29, "smape": 6.46, "mase": 2.94},
      "horizon": 6,
      "n_windows": 12,
      "forward": [
        {"period_start": "2026-04-01", "point": 233.57, "lower_80": 215.4, "upper_80": 251.7},
        ...
      ],
      "latest_actual": {"period_start": "2026-03-01", "value": 221.0},
      "scoreboard": [{"model": "AutoARIMA", "mape": 6.29, ...}, ...]
    }
  }
}
```

Generated by a new module `src/usda_sandbox/precompute.py::build_forecast_cache()`. The Home and Forecast (default) pages read this cache instead of fitting models live. The Forecast page's Advanced expander still fits live so analysts can play with parameters.

---

## 7. Module changes

**New files:**
- `dashboard/components/theme.py` — `inject_global_css()`, `set_page_chrome()`.
- `dashboard/components/brief.py` — `compose_brief(series_id, cache)`, `compose_commodity_card(...)`.
- `dashboard/components/cache.py` — `load_forecast_cache()`, cache helpers.
- `dashboard/pages/0_Brief.py` — replaces `app.py` landing as the new home (Streamlit sorts numerically; `0_` keeps it first).
- `dashboard/pages/4_Methodology.py`
- `dashboard/pages/5_About.py`
- `dashboard/static/favicon.png`
- `src/usda_sandbox/precompute.py` — bake forecasts.json from observations.parquet.
- `streamlit_app.py` (repo root) — Streamlit Cloud entrypoint; thin shim that imports `dashboard/app.py`.
- `requirements.txt` (repo root, generated from `uv export`).
- `.streamlit/config.toml`.
- `.github/workflows/refresh-data.yml`.

**Changed files:**
- `dashboard/app.py` — becomes a redirect/landing that routes to Brief, or is repurposed as Brief itself.
- `dashboard/pages/1_Explore.py` — kept, repositioned as "Catalog" (Advanced/Methodology adjunct, not headline).
- `dashboard/pages/2_Visualize.py` → `2_Series.py` — restyled, notes promoted, link to Forecast added.
- `dashboard/pages/3_Forecast.py` — default view simplified; current UI moves into Advanced expander.
- `dashboard/components/sidebar.py` — slim sidebar, branding title.
- `README.md` — production-posture rewrite with deploy instructions.

**Unchanged:**
- All of `src/usda_sandbox/` except for the new `precompute.py`.
- The observations.parquet schema.
- Tests in `tests/`.
- Notebooks.

---

## 8. Error handling & edge cases

- **No observations.parquet yet** (fresh clone, never refreshed): All pages show a single banner: "Initial data sync in progress — check back in 5 minutes." No stack traces.
- **No forecasts.json yet** (precompute never ran): Home page falls back to "Data loaded; forecast snapshot generating, refresh shortly." Forecast page Advanced expander still works (fits live).
- **yfinance fetch failure during refresh**: GitHub Action fails loudly. The old `observations.parquet` on the `data` branch stays in place. Site keeps serving stale-but-correct data. About page surfaces last refresh timestamp so users know.
- **A series goes dark in source** (e.g., lamb after 2018): Existing null-tolerant pipeline handles it. Brief page guards against `NaN`/missing latest values — skips that card with a soft "data temporarily unavailable" placeholder.
- **Series with too little history for the configured precompute horizon**: precompute logs and skips; site only shows commodity cards for series that have a fresh cache entry.

---

## 9. Testing

- **Unit:** `tests/test_precompute.py` for `build_forecast_cache()` round-trip (build cache from a small fixture parquet, assert schema, assert values reproducible with seed).
- **Unit:** `tests/test_brief.py` for `compose_brief()` template formatting across edge cases (positive/negative MoM, missing forecast, etc.).
- **Smoke:** `tests/test_dashboard.py` extended to import each new page module (catches syntax/import errors without running Streamlit).
- **Manual:** browse the running app on a laptop (smoke check golden path + each page).

---

## 10. Deployment

1. User connects the repo to Streamlit Community Cloud (one-time, manual — needs their GitHub credentials).
2. Cloud points at `streamlit_app.py` on `main` for the app and `data` branch for the parquet.
3. GitHub Action runs weekly; pushes refreshed `observations.parquet` and `forecasts.json` to `data` branch.
4. Streamlit Cloud auto-redeploys on push.

Public URL: assigned by Streamlit Cloud (`<repo>-<hash>.streamlit.app`). Custom domain optional, not in v1.0 scope.

---

## 11. Implementation order (the plan-writing skill will turn this into tasks)

1. Visual identity: `.streamlit/config.toml`, `theme.py`, favicon, page chrome.
2. `precompute.py` + `forecasts.json` cache + tests.
3. New `Brief` page (home).
4. Commodity-card component.
5. Series detail polish.
6. Forecast page Default + Advanced split.
7. Methodology page.
8. About page.
9. Sidebar slim-down.
10. `streamlit_app.py` + `requirements.txt` export.
11. GitHub Action workflow.
12. README rewrite for production posture.
13. Manual browser smoke pass.
14. Commit + tag v1.0.

---

## 12. Non-goals reaffirmed

- This is still a single-tenant, no-auth public dashboard.
- Forecasts are educational, not financial advice. The About page says so.
- No deep learning. No LLMs. The "plain English" briefs are *deterministic templates*, not generative.
