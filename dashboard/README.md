# Dashboard

Local Streamlit app over the cleaned `data/clean/observations.parquet` store.

## Launch

```bash
uv sync
uv run streamlit run dashboard/app.py
```

Opens at http://localhost:8501.

## Pages

* **Explore** — catalog table (filterable by commodity), per-commodity
  rollup via DuckDB, and a null-span summary. Read-only.
* **Visualize** — single-series view chosen via the sidebar series picker.
  Date-range slider, time-series with event markers, year-over-year
  percent change, and statsmodels seasonal decomposition (multiplicative
  by default; falls back to additive for series with non-positive values).
* **Forecast** — interactive backtest. Configure horizon (1-12 months),
  number of CV windows (2-24), and which models to include
  (AutoARIMA / Prophet / LightGBM). Click **Run backtest** to watch
  per-model / per-window progress stream in. Results: scoreboard with
  MAPE / sMAPE / MASE, CV-overlay chart, residual diagnostics for the
  winner, and a 12-month forward forecast with shaded 80% prediction
  interval.

## Sidebar

* **Series picker** — populated from `list_series()`; selection persists
  across pages via `st.session_state`.
* **Data status** — series count, total rows, date range, and how
  recently `observations.parquet` was rebuilt.
* **Refresh data** — runs `sync_downloads()` and `clean_all()` in
  sequence with progress streamed into a status panel; clears the
  cached series list on success.

## Architecture

```
dashboard/
├── app.py                # entrypoint + landing page
├── components/
│   ├── plots.py          # plotly figure builders (pure functions, smoke-tested)
│   └── sidebar.py        # series picker / data status / refresh button
└── pages/
    ├── 1_Explore.py
    ├── 2_Visualize.py
    └── 3_Forecast.py
```

The dashboard imports from `src/usda_sandbox/` but never modifies it. The
package is the contract; the dashboard is a replaceable view layer.

## Caching

* `cached_list_series` and `cached_dataset_overview` use
  `@st.cache_data(ttl=300)`; both are cleared on Refresh.
* Backtest results live in `st.session_state` keyed on
  `(series_id, horizon, n_windows, models)` — flipping pages doesn't
  re-run the computation.

## Known limitations (v1)

* Single-user (`st.session_state` is per-browser-session).
* Backtest cancellation isn't supported — flipping pages mid-run interrupts
  the iterator (Streamlit reruns the script). Acceptable: just rerun.
* No persisted backtest history; nothing is written to disk by the dashboard.
* No exogenous regressors (CME futures, weather) — those are out of scope
  for v0.1; see top-level README "What's not in v0.1".
