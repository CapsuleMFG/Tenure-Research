# Dashboard Design — USDA Livestock Sandbox

**Status:** Draft, pending user review
**Date:** 2026-05-08
**Author:** Tyler + Claude (brainstormed)

## Goal

A laptop-local Streamlit app that turns the existing notebooks into an
interactive explorer — pick a series, change a horizon, watch a backtest
run. The dashboard is a *view layer* over the existing
`src/usda_sandbox/` package; no new computation logic, just UI on top of
what already works.

Concretely, after this lands the user can:

- Open `streamlit run dashboard/app.py` from the project root
- See dataset coverage, catalog, and data-quality on an Explore page
- Pick any catalog series and view its time-series + seasonal
  decomposition + YoY change on a Visualize page
- Configure and run a 6-month backtest on a Forecast page, watching
  per-model / per-window progress stream in, then see the scoreboard,
  CV overlay, residual diagnostics, and a 12-month forward forecast

Out of scope for v1 of the dashboard (kept for later): authentication,
multi-user, scheduled refresh, persisted backtest history, custom
catalog editing through the UI, exogenous regressors.

## Architecture

```
dashboard/                          # new top-level dir
├── app.py                          # entrypoint: sidebar + page registration
├── pages/
│   ├── 1_Explore.py                # leading number drives nav order
│   ├── 2_Visualize.py
│   └── 3_Forecast.py
├── components/
│   ├── __init__.py
│   ├── sidebar.py                  # data-status panel + refresh button
│   └── plots.py                    # plotly figure builders shared across pages
└── README.md                       # how to run, troubleshooting

src/usda_sandbox/                   # existing — UNCHANGED
```

Why a `dashboard/` directory and not `src/usda_sandbox/dashboard/`:
the dashboard is a UI consumer of the package, not part of the package
itself. Keeping it sibling means the `src/usda_sandbox/` package stays
the data/forecast contract and the dashboard is replaceable.

`components/plots.py` extracts the plotly figures that already live in
the notebooks (CV overlay, residual diagnostics, forward forecast band)
into pure functions: `build_cv_overlay(history, cv_df) -> go.Figure`.
This serves two ends: the dashboard pages stay short, and the same
functions can be imported by future notebooks instead of copy-pasted.

## Page-by-page spec

### Sidebar (visible on every page)

Implemented in `dashboard/components/sidebar.py`, called from each
page's top-of-file. Streamlit's auto-generated page nav goes above it.

Contents, top-to-bottom:

1. **Series picker** — `st.selectbox` populated from `list_series()`,
   stored in `st.session_state["series_id"]` so the selection persists
   across pages. Shown as `series_id` with the human-readable
   `series_name` in a tooltip.
2. **Data status block** — three lines:
   - `N series · M rows`
   - `Range: 2000-01-01 .. 2026-03-01`
   - `Last cleaned: HH:MM ago` (mtime of `observations.parquet`)
3. **"Refresh data" button** — clicking opens an `st.status()` panel and
   runs `sync_downloads()` then `clean_all()` with progress messages.
   On success, clears `@st.cache_data` for `list_series` and
   `read_observations` so the rest of the UI picks up the new data.
   See *Live computation* below for the streaming pattern.

The sidebar gracefully degrades if `data/clean/observations.parquet`
doesn't exist yet: it shows a "No cleaned data yet — click Refresh
to bootstrap" message and disables the page nav (or keeps it enabled
but pages render an empty-state).

### Page 1 — Explore

Read-only mirror of `notebooks/01_explore.ipynb` content but with
sortable / filterable tables instead of static markdown.

- **Header KPIs** (three `st.metric` cards): total series, total rows,
  date range
- **Catalog table** — `list_series()` rendered with `st.dataframe`,
  with a multi-select filter for commodity above it
- **By-commodity rollup** — DuckDB query (same as notebook), shown
  as a `st.dataframe` with column highlights
- **Null-span summary** — only series with `n_nulls > 0`, sorted descending,
  with the date range of the null span

No widgets that mutate state — this page is pure read.

### Page 2 — Visualize

Single-series deep dive. Series chosen from the sidebar picker; this
page only adds horizon controls.

- **Series header** — name, commodity, unit, frequency, coverage range
- **Date range slider** — `st.slider` over the series' actual range,
  controls all charts on this page (default: full range)
- **Time series chart** — plotly line chart with the same five event
  markers from the notebook (2008 GFC, 2014-15 cattle peak, 2020 COVID,
  2022 commodity spike, 2024-25 cycle high). Event markers visible only
  when they fall inside the slider range.
- **Seasonal decomposition** — multiplicative, period=12 (configurable
  via a small expander, default hidden). Falls back to additive if
  the series has any non-positive values.
- **Year-over-year change** — 12-month percent change line chart with
  zero reference line.

This page does no model fitting, so it should re-render in well under
a second on every interaction.

### Page 3 — Forecast

The interactive backtest. Series chosen from the sidebar picker; this
page adds the run controls plus the live progress UI.

**Inputs (top of page, in a row):**

- Horizon (`st.slider`, default 6, range 1-12)
- Number of CV windows (`st.slider`, default 12, range 2-24)
- Models to include (`st.multiselect`, default all three)
- "Run backtest" button (primary)

**Live progress panel (only visible while running):**

An `st.status("Running backtest...")` container with one `st.write()`
per fitted window: `"AutoARIMA window 4/12 done — MAPE so far: 5.8%"`.
Updates flush as each model+window completes. On success the status
collapses to a green check; on error it stays expanded with the
traceback. See *Live computation* below for the implementation pattern.

**Results (only shown after a successful run):**

- **Scoreboard** — per-model metrics table (MAPE, sMAPE, MASE)
  highlighting the best per metric
- **CV-overlay plot** — actuals + per-model forecast segments per
  window, identical to notebook 03 (delegated to
  `components/plots.build_cv_overlay`)
- **Residual diagnostics** — three-panel figure for the winning model
  (histogram, MAE-by-step, Q-Q vs normal)
- **Forward 12-month forecast** — winner refit on full series, plotly
  with shaded 80% PI; numeric forecast table below

The full result `BacktestResult` is stored in
`st.session_state[f"backtest:{series_id}:{horizon}:{n_windows}"]` so
that flipping pages and coming back doesn't re-run the computation.

## Live computation

Two surfaces. Both follow the same pattern: a long-running function
that yields progress events, consumed by Streamlit via
`st.status()` + `st.write()` calls inside the iteration loop.

### 1. Forecast backtest progress

The existing `run_backtest()` is one-shot — it returns when fully done.
For dashboard use we add a **generator variant**, not a replacement:

```python
# src/usda_sandbox/forecast.py — new function, additive
def iter_run_backtest(
    series_id, horizon, n_windows, *, obs_path=None, seed=42
) -> Iterator[BacktestProgress | BacktestResult]:
    """Same inputs/outputs as run_backtest() but yields BacktestProgress
    events as each (model, window) completes, then yields the final
    BacktestResult last."""
```

`BacktestProgress` is a small frozen dataclass: `model: str`,
`window: int`, `n_windows: int`, `elapsed_s: float`,
`running_mape: float | None`. The dashboard's Forecast page consumes
the iterator and writes one line per event into the `st.status()`.

This keeps the existing `run_backtest()` unchanged (so all 56 existing
tests stay green), and the dashboard becomes a thin consumer.

### 2. Sidebar refresh button

```python
with st.status("Refreshing data...") as status:
    status.write("Discovering ERS download URLs...")
    sync_downloads()  # already chatty via prints; we'll redirect to status
    status.write("Cleaning all catalog series...")
    clean_all("data/catalog.json", "data/raw", "data/clean/observations.parquet")
    status.update(label="Refresh complete", state="complete")
read_observations.clear()  # bust the cache decorator
list_series.clear()
```

We don't add per-file ingest progress streaming in v1 — `sync_downloads`
already prints filenames as they're processed, and capturing those
into Streamlit means an extra refactor we don't need yet.

## Caching

Two `@st.cache_data` decorators wrap the read paths so flipping pages
doesn't re-read parquet:

```python
@st.cache_data(ttl=300)  # bust on Refresh button click via .clear()
def cached_list_series(obs_path_str: str) -> pl.DataFrame:
    return list_series(Path(obs_path_str))

@st.cache_data(ttl=300)
def cached_read_series(series_id: str, obs_path_str: str) -> pl.DataFrame:
    return read_series(series_id, Path(obs_path_str))
```

Backtest results live in `st.session_state` keyed on the input tuple,
not in `@st.cache_data`, because polars DataFrames + a custom dataclass
nested inside `BacktestResult` interact awkwardly with Streamlit's
cache hashing.

## Dependencies and launch

Add to `pyproject.toml` `[project.dependencies]`:

```toml
"streamlit>=1.32",
```

Launch (documented in README):

```bash
uv sync                           # picks up the new dep
uv run streamlit run dashboard/app.py
```

Streamlit defaults to `localhost:8501`. The README gets a new
"Dashboard" section pointing here.

## Error handling

Edge cases the dashboard must handle without crashing:

| Case | Behavior |
|---|---|
| `observations.parquet` missing | Sidebar shows "no data — click Refresh"; pages render empty-state copy |
| `data/raw/` empty when Refresh clicked | `sync_downloads()` populates it; status panel reflects the work |
| Backtest for a series with too few non-null obs | `run_backtest` already raises `ValueError`; we catch in the page and show `st.error()` with the message |
| User flips pages mid-backtest | `st.session_state` survives nav; the in-flight iterator is interrupted by Streamlit's rerun (acceptable for v1) |
| Network error during Refresh | `sync_downloads` raises; `st.status` captures, page shows the error |
| Plotly figure renders empty | Should not happen given input validation, but if it does, page shows the underlying DataFrame instead so the user can see what's wrong |

## Testing

The Streamlit app itself is not unit-tested in v1 — Streamlit testing is
awkward and the value is low for a viewer. **What we test instead** is
the new computation surface that the dashboard consumes:

- `iter_run_backtest()` — new generator variant — gets a test that
  verifies it yields N progress events plus one result, in order, and
  that the final result equals what `run_backtest()` returns for the
  same inputs (deterministic with seed)
- `components/plots.py` — figure builders get small smoke tests that
  call them with synthetic data and assert the returned object is a
  `plotly.graph_objects.Figure`

Existing 56 tests stay green; this adds ~3-4 new tests.

## What this is not

- Not a product. It's a development tool for understanding the data.
- Not a forecast pipeline. It runs forecasts on demand, doesn't
  schedule them or persist their results to disk.
- Not multi-user. `st.session_state` is per browser session.
- Not a replacement for the notebooks. The notebooks are the polished
  artifact (the README references them); the dashboard is the
  exploration tool.
- Not a Tenure pricing module. When that gets built, the dashboard
  isn't what gets lifted — the `src/usda_sandbox/` package is.

## Open questions for the user

None — every decision has a default. If something feels wrong on review,
flag it inline and we'll iterate.
