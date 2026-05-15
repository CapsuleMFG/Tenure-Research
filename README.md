# LivestockBrief

> **A free public dashboard for direct-market cattle ranchers. Plan your
> operation, track today's input costs, set your freezer-beef pricing.**

LivestockBrief is built for farms that **raise, finish, and (often)
slaughter their own cattle**, selling freezer beef directly to consumers
as quarters, halves, wholes, or retail cuts.

The headline tools (v3.0):

- **Plan** — three modes (cow-calf, stocker, finish-and-direct). Pure
  cost-stack math; pasture-shaped, not feedlot-shaped. Per-head margin,
  annual operation P&L, breakeven $/cwt — with the relevant market
  signals (feeders, live cattle, corn) surfaced inline.
- **Costs** — today's feed grains (CME corn, soybean meal, oats), feeder
  cattle (GF futures + Oklahoma auction), and a hay reference with a
  local-price override that flows back into Plan.
- **Pricing** — research-derived ranges for hanging-weight pricing
  (grain-finished / grass-finished / premium branded), share sizing
  (quarter / half / whole), and per-cut retail. With a calculator that
  turns your hanging weight + $/lb into share-size revenue.

Underneath sit the v1.0/v2.0 forecasting layer (AutoARIMA, Prophet,
LightGBM with conformally calibrated 80% PIs over USDA ERS cash + CME
futures) and the v2.0 commodity tools (Decide, Feedlot Breakeven, Basis),
kept available for users who want them. All v1.0/v2.0 series IDs and
APIs unchanged.

## Live app

> **Deployed to Streamlit Community Cloud.** Connect this repo at
> [share.streamlit.io](https://share.streamlit.io) → "New app" → point at
> `streamlit_app.py` on the `main` branch. Streamlit Cloud will auto-redeploy
> when the data refresh workflow pushes new artifacts to the `data` branch.

## Run locally

```bash
# Install (Python 3.11 via uv)
uv sync

# Pull raw USDA ERS XLSX files + CME continuous futures
uv run python -c "from usda_sandbox.ingest import sync_downloads; sync_downloads()"
uv run python -c "from usda_sandbox.futures_continuous import sync_continuous_futures; sync_continuous_futures()"

# Clean into the tidy parquet store
uv run python -c "from usda_sandbox.clean import clean_all; clean_all('data/catalog.json', 'data/raw', 'data/clean/observations.parquet')"

# Pull daily front-month futures (LE / GF / HE)
uv run python -c "from usda_sandbox.futures_daily import sync_daily_futures, append_daily_to_observations; sync_daily_futures(); append_daily_to_observations()"

# Bake the forecast cache (the Brief page reads this)
uv run python -m usda_sandbox.precompute

# Launch
uv run streamlit run streamlit_app.py
```

Opens at `http://localhost:8501`. On a deployed instance, the data sync,
daily futures refresh, and forecast cache rebake are all done by GitHub
Actions (weekly full refresh, daily futures-only refresh).

## Pages

**Primary (v3.0 direct-market tools):**

* **Brief** *(home)* — today's daily futures (feeder, live cattle, corn),
  intro tagline, commodity cards reframed for direct-market context.
* **Plan** — the central operation modeler. Three tabs:
  *cow-calf* (breeding herd, sell weaned calves), *stocker* (buy weaned,
  graze, sell to feedlot), *finish-and-direct* (finish + freezer beef
  sales with commodity-floor sanity check).
* **Costs** — daily feed grains + feeder cattle prices + hay reference
  + cull-cow proxy via boxed-beef cutout trend.
* **Pricing** — share-size and per-cut pricing reference with a
  hanging-weight calculator.

**Explore data (kept):**

* **Catalog** — every series in the store, row counts, date ranges,
  null-span data quality summary.
* **Series** — single-series deep dive with basis card.
* **Forecast** — cached winner per series + Advanced live-backtest UI.

**Commodity tools (v2.0, kept for fed-cattle / feedlot users):**

* **Decide (commodity)** — sell-now / hold tool for commodity producers.
* **Feedlot breakeven** — KSU-style feedlot cost calc.

**Reference:** Methodology, About.

## Architecture

```
streamlit_app.py            Streamlit Cloud entrypoint (loads dashboard/app.py)
.streamlit/config.toml      Brand theme (parchment background, clay accent)
.github/workflows/
  refresh-data.yml          Weekly cron: refresh data + rebake forecasts.json
dashboard/
  app.py                    Brief (home) — landing
  components/
    theme.py                Brand chrome + global CSS
    brief.py                Plain-English brief composer + commodity cards
    cache.py                Streamlit-cached forecast-cache accessors
    sidebar.py              Branding + (optional) series picker + admin
    plots.py                Plotly figure builders (chart, YoY, CV, residuals)
  pages/
    1_Explore.py            Catalog & coverage
    2_Series.py             Single-series deep dive
    3_Forecast.py           Simplified default + Advanced backtest
    4_Methodology.py        How LivestockBrief works
    5_About.py              Credits + version + disclaimer
src/usda_sandbox/
  ingest.py                 ERS XLSX downloader (idempotent, manifest-keyed)
  futures_continuous.py     yfinance MONTHLY continuous front-month
  futures_daily.py          yfinance DAILY (livestock + grains, v3.0)
  catalog.py                SeriesDefinition (Pydantic)
  clean.py                  Tidy-parquet builder
  store.py                  polars / DuckDB accessors
  forecast.py               AutoARIMA / Prophet / LightGBM + backtest
  calibration.py            Conformal PI calibration (per-horizon)
  precompute.py             Bake forecasts.json for the Brief / Forecast UI
  basis.py                  Cash-to-futures basis math (v2.0)
  breakeven.py              Feedlot cost-of-production calculator (v2.0)
  decision.py               Sell-now / hold rule engine (v2.0)
  direct_market.py          Cow-calf / stocker / finish-direct economics (v3.0)
  direct_pricing.py         Freezer-beef pricing reference + yield (v3.0)
data/
  catalog.json              All series definitions (in repo)
  raw/                      gitignored — local downloads
  clean/                    gitignored on main; published to the `data` branch
notebooks/                  Research artifacts (not user-facing)
tests/                      67 tests over ingest/clean/store/forecast/dashboard
```

## Data flow

```
weekly cron (GitHub Actions, Sunday 21:00 UTC) — full refresh + rebake
  ├── sync_downloads()                   → data/raw/*.xlsx
  ├── sync_continuous_futures()          → data/raw/futures_continuous/*.parquet
  ├── clean_all()                        → data/clean/observations.parquet
  └── precompute.build_forecast_cache()  → data/clean/forecasts.json
                                          → push both to `data` branch

daily cron (GitHub Actions, 22:00 UTC) — daily prices only
  ├── sync_daily_futures()               → data/raw/futures_daily/*.parquet
  ├── append_daily_to_observations()     → merge into observations.parquet
  └── push observations.parquet          → `data` branch → Cloud redeploys
```

The web app reads `observations.parquet` for charts and history, and
`forecasts.json` for the cached winner forecasts shown on the Brief and
Forecast pages.

## Deploy your own

1. **Fork the repo.**
2. **Run the workflow once manually** (Actions → "Refresh data + rebake
   forecasts" → "Run workflow"). This populates the `data` branch.
3. **Sign in to [share.streamlit.io](https://share.streamlit.io).**
4. **New app** → pick your fork → entrypoint `streamlit_app.py` → main branch.
5. *(Optional)* Configure secrets, custom domain.

The GitHub Actions cron is opt-in — schedule will not fire until you have
push access enabled on your fork's `data` branch.

## What's in the cleaned store

12 forecastable series (8 cattle, 3 hogs, 1 sheep/lamb), 39 supporting
CME futures contracts (LE / HE / GF continuous front-month + 1-12 month
deferred). Monthly cash prices back to 2000; quarterly WASDE
supply/disappearance back to 2021. See `data/catalog.json` for the full
manifest.

## Limits we know about

* MASE typically lands 2-7 across the board — forecasts are useful for
  *level and range*, not for tactical week-to-week timing.
* The lamb series effectively stops in 2018-04 when AMS changed reporting.
* Quarterly WASDE series live on Catalog and Series only; they're not
  forecast.
* Not financial advice. See the **About** page for the full disclaimer.

## Non-negotiables (carried over from the sandbox phase)

* Classical methods only. No deep learning, no LLMs, no fine-tuning.
* All data flows through the tidy `observations.parquet` schema.
* Ingest is idempotent. Re-running it does not duplicate rows.
* Every random source is seeded.
* No GPU, no cloud accounts, no paid APIs.

## Tests / lint / types

```bash
uv run pytest        # 67 tests
uv run ruff check .
uv run mypy
```

## Why daily futures and not daily AMS cash?

A producer might reasonably ask: "where's the daily AMS LMR cash data?"
We considered it and didn't ship it.

* **USDA MARS API requires eAuth Level 2** (in-person USDA identity
  proofing). That's too much friction for an open-source side project.
* **Public AMS PDF scraping is fragile.** USDA tweaks report layouts
  every few quarters; a scrape breaks silently and the dashboard quietly
  serves stale numbers. We won't ship infrastructure we can't trust to
  fail loudly.
* **Daily futures cover the need.** LE / GF / HE incorporate AMS daily
  cash via the basis-and-arbitrage relationship that the industry
  actually uses to read the market. A producer looking at LE=$247 with
  a historical Nebraska basis of -$5 has a defensible read on cash. The
  Decide tool's math doesn't depend on AMS cash directly.

If a future contributor (a producer or extension agent) wants to add a
proper AMS LMR pipeline — regional basis, slaughter weights, dressing
percent, export volume — that's a real v3.0 project worth doing. It just
needs an owner who'll maintain it.

## Version

v3.0 — "direct-market rancher" release: reorients the app around farms
that raise, finish, and direct-sell. Adds the Plan / Costs / Pricing
trio, plus daily grain futures (corn, SBM, oats). v2.0 commodity tools
(Decide, Feedlot Breakeven) remain. See
`docs/superpowers/specs/2026-05-15-v3-direct-market-design.md` for the
v3.0 rationale; v1.0 and v2.0 design docs are alongside.

## Disclaimer

LivestockBrief is **educational and informational only**. The Decide tool
is a reasoning aid: a transparent, deterministic synthesis of public
prices, your inputs, and a calibrated forecast. It does not account for
your working capital, pen space, hedge position, tax situation, or local
basis dynamics beyond what appears in the cash series. Use it to structure
the question, then make your own call. Not financial advice.
