# LivestockBrief

> **Daily livestock prices + USDA-grounded forecasts + a sell-now decision
> tool. Plain English. Honest uncertainty.**

LivestockBrief is a free public dashboard for cattle and hog producers.
It combines USDA Economic Research Service monthly cash, daily CME front-
month futures, three classical forecasting models with conformally
calibrated 80% prediction intervals, and three v2.0 producer tools:

- **Basis** — cash-to-nearby-futures gap, current + 5-year band
- **Breakeven** — feedlot cost-of-production calculator (KSU defaults)
- **Decide** — a *sell now / hold / hedge* recommendation that synthesizes
  today's cash, today's futures, your basis, your breakeven, and the
  6-month forecast into a deterministic, auditable suggestion

Every recommendation comes with the inputs that drove it and the rule
that fired. No LLM, no opaque score. Not financial advice.

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

* **Brief** *(home)* — today's front-month futures strip + auto-generated
  market headline + commodity cards across cattle, hogs, beef wholesale,
  pork wholesale, and lamb.
* **Decide** *(v2.0)* — pick a commodity + region, enter your breakeven,
  get a deterministic *sell now / hold / hedge* recommendation grounded
  in today's prices and the 6-month forecast.
* **Breakeven** *(v2.0)* — feedlot cost-of-production calculator; result
  feeds the Decide tool.
* **Catalog** — every series in the store, with row counts, date ranges,
  and null-span data quality summary.
* **Series** — single-series deep dive: time series, YoY change, seasonal
  decomposition, **basis card** (cash − nearby futures), jump to forecast.
* **Forecast** — default = the winner from the weekly bake-off, 12-month
  forward forecast, calibrated 80% PI. Advanced expander = the live-backtest
  UI (horizon, CV windows, model selection, residual diagnostics).
* **Methodology** — plain-English explanation of data, models, PIs,
  basis, breakeven, and the Decide tool's rules.
* **About** — credits, version, last-refresh status, GitHub link, disclaimer.

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
  futures_daily.py          yfinance DAILY continuous front-month (v2.0)
  ams_lmr.py                AMS LMR optional ingest (needs AMS_API_KEY, v2.0)
  catalog.py                SeriesDefinition (Pydantic)
  clean.py                  Tidy-parquet builder
  store.py                  polars / DuckDB accessors over observations.parquet
  forecast.py               AutoARIMA / Prophet / LightGBM + backtest
  calibration.py            Conformal PI calibration (per-horizon)
  precompute.py             Bake forecasts.json for the Brief / Forecast UI
  basis.py                  Cash-to-futures basis math (v2.0)
  breakeven.py              Feedlot cost-of-production calculator (v2.0)
  decision.py               Sell-now / hold rule engine (v2.0)
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
  ├── (optional) ams_lmr.sync_ams_daily  → if AMS_API_KEY secret is set
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

## Version

v2.0 — "actually useful" release: adds daily futures, basis, breakeven
calculator, and the Decide tool. See `docs/superpowers/specs/2026-05-15-v2-actually-useful-design.md`
for the v2.0 rationale; v1.0 design is alongside.

## Disclaimer

LivestockBrief is **educational and informational only**. The Decide tool
is a reasoning aid: a transparent, deterministic synthesis of public
prices, your inputs, and a calibrated forecast. It does not account for
your working capital, pen space, hedge position, tax situation, or local
basis dynamics beyond what appears in the cash series. Use it to structure
the question, then make your own call. Not financial advice.
