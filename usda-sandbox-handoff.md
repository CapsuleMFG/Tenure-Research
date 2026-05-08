# USDA Livestock Sandbox — Claude Code Handoff

A small, focused Python project that downloads USDA livestock and meat data, cleans it into a tidy form, makes forecasts, and shows you what's in it. Not a product. A sandbox that teaches you the data and seeds the future Tenure pricing dashboard.

Hand this to Claude Code in a fresh repo.

---

## Why this project exists

Three reasons, in order of importance:

1. **You need to actually understand what's in the USDA livestock data** before you design the pricing module that ships with Tenure. Reading the ERS website doesn't teach you that — touching the data does.
2. **Classical time-series forecasting on real economic data is a worthwhile skill** that you haven't done much of, and it's the foundation underneath any "AI for ranchers" feature.
3. **The code is reusable.** When the Tenure ops platform gets a pricing module, this repo's ingest layer and forecasting layer drop in.

What this is *not*: a product, a deep-learning experiment, a fine-tuned LLM, a dashboard, or anything connected to the Tenure ops platform yet.

---

## What the finished project looks like

```
usda-livestock-sandbox/
├── data/
│   ├── raw/              # downloaded XLSX/ZIP files (gitignored)
│   ├── clean/            # tidy parquet files (gitignored)
│   └── catalog.json      # what we know about each series
├── notebooks/
│   ├── 01_explore.ipynb       # what's in the data
│   ├── 02_visualize.ipynb     # historical plots
│   └── 03_forecast.ipynb      # forecasting and CV
├── src/usda_sandbox/
│   ├── __init__.py
│   ├── ingest.py         # download + parse ERS XLSX files
│   ├── clean.py          # tidy transforms
│   ├── catalog.py        # series metadata
│   ├── store.py          # parquet/DuckDB read+write
│   ├── forecast.py       # AutoARIMA / Prophet / LightGBM
│   └── viz.py            # plotting helpers
├── tests/
│   └── test_ingest.py
├── pyproject.toml        # uv-managed
├── CLAUDE.md             # repo guidance
├── README.md
└── .gitignore
```

Three notebooks at the end. A clean parquet store. Code that can be re-run any time the USDA publishes new data.

---

## Stack

Mirroring your CapsulePC patterns where it matters; using best-in-class small-data tools where it's better.

- **Python 3.11**, managed with `uv` (matches your existing pattern)
- **polars** for data wrangling (memory-efficient, much faster than pandas, sane API)
- **DuckDB** for analytical SQL when polars isn't enough
- **statsforecast** (Nixtla) for AutoARIMA, AutoETS, AutoTheta — the modern classical-forecast library
- **prophet** for seasonal decomposition forecasts (compare against statsforecast)
- **lightgbm** for gradient-boosted forecasts with lag features
- **plotly** for interactive plots in notebooks; matplotlib as fallback
- **pyarrow** for parquet I/O
- **httpx** for downloads, **openpyxl** for XLSX parsing
- **jupyter** + VS Code notebook UI
- **pytest** for tests
- **ruff** + **mypy** for lint and types

No deep learning libraries. No GPU. No fine-tuning. If a future notebook wants to try a foundation model like Chronos, that's a separate experiment, not part of v1.

---

## The data — what we're actually downloading

From <https://www.ers.usda.gov/data-products/livestock-and-meat-domestic-data>, the downloadable XLSX files cover, roughly:

- **Livestock prices** — monthly average prices for cattle, calves, hogs, sheep, lambs, broilers, turkeys, eggs
- **Wholesale prices** — boxed beef, pork, lamb wholesale values
- **Meat supply and disappearance** — production, imports, exports, ending stocks, per-capita consumption (red meat and poultry)
- **Feeder cattle outside feedlots** — quarterly inventory estimates
- **High Plains cattle feeding simulator** — feed costs, breakeven, projected returns
- **Various per-capita and aggregate consumption series**

Most series are monthly back to the 1970s. Some are quarterly or annual. Each XLSX has its own quirks: multi-row headers, embedded notes, hidden columns, mid-table subtotals.

The cleaning code lives in `src/usda_sandbox/clean.py` and is the part that takes the most actual thought. Everything downstream depends on getting it right.

---

## The output schema (what `clean/` contains)

A single tidy parquet table called `observations.parquet`:

| column          | type      | example                              |
|-----------------|-----------|--------------------------------------|
| series_id       | string    | "cattle_price_avg_choice_steers"     |
| series_name     | string    | "Choice Steers, 5-area weighted avg" |
| commodity       | string    | "cattle"                             |
| metric          | string    | "price"                              |
| unit            | string    | "USD/cwt"                            |
| frequency       | string    | "monthly"                            |
| period_start    | date      | 2024-03-01                           |
| period_end      | date      | 2024-03-31                           |
| value           | float64   | 187.42                               |
| source_file     | string    | "LivestockMeatDomesticData.xlsx"     |
| source_sheet    | string    | "Choice steers"                      |
| ingested_at     | timestamp | 2026-05-08T14:22:11Z                 |

Plus `catalog.json` with one entry per `series_id` describing what it is, where it came from, and its known quirks.

This shape is the contract. Every downstream piece of code reads this table.

---

## CLAUDE.md (drop in repo root)

```markdown
# USDA Livestock Sandbox — Claude Code Guidance

## What this is
A learning sandbox that ingests USDA ERS livestock data, cleans it into a tidy
parquet store, and runs classical time-series forecasts. Code here may be
lifted into the Tenure ops platform later as the data layer behind the
pricing dashboard.

## Non-negotiables
- No fine-tuning, no LLM training, no deep learning. Classical methods only.
- All data flows through the tidy `observations.parquet` schema. Don't invent
  parallel schemas. If the schema needs to change, change it everywhere.
- Ingest is idempotent. Re-running it should not duplicate rows. Keyed on
  (series_id, period_start, source_file).
- Cleaning logic is testable. Every transform that turns a messy XLSX into
  tidy rows has a small fixture file and a test.
- All randomness is seeded. Forecasting code that uses any RNG (LightGBM,
  Prophet's MCMC option) sets a seed.
- The project is laptop-friendly. Don't pull in libraries that need a GPU,
  cloud accounts, or paid APIs.

## Patterns
- polars first, pandas only where a third-party library forces it
- DuckDB for SQL-shaped analytical queries
- statsforecast for the primary forecasting baseline; Prophet and LightGBM
  for comparison
- Notebooks are for exploration and final visuals. Reusable code lives in
  src/usda_sandbox.
- One series_id = one time series. Don't pivot wide unless a downstream
  function specifically requires it.

## What to build first
See INITIAL_PROMPTS.md. Work through them in order.
```

---

## INITIAL_PROMPTS.md (your first sessions with Claude Code)

These are sized to be one Claude Code session each. Run them in order.

### Prompt 1 — Repo setup

```
Initialize a Python 3.11 project managed by uv. Create the directory layout
from the handoff doc (data/, notebooks/, src/usda_sandbox/, tests/).

Add these dependencies via uv:
  polars, duckdb, pyarrow, httpx, openpyxl, statsforecast, prophet,
  lightgbm, plotly, matplotlib, jupyter, pytest, ruff, mypy

Add a .gitignore that excludes data/raw/, data/clean/, .venv/, __pycache__,
.ipynb_checkpoints, and the usual Python noise.

Add the CLAUDE.md from the handoff doc to the repo root.

Add a minimal pyproject.toml with ruff and mypy config (line length 100,
target python 3.11, mypy strict on src/usda_sandbox).

Add a README.md with a one-paragraph description and a "how to run" section
with uv sync and jupyter lab.

Confirm `uv sync` works and `uv run python -c "import polars, duckdb,
statsforecast, prophet, lightgbm"` runs without error, then stop.
```

### Prompt 2 — Ingest layer

```
Build src/usda_sandbox/ingest.py.

The ERS Livestock and Meat Domestic Data product page is at:
https://www.ers.usda.gov/data-products/livestock-and-meat-domestic-data

It links to several XLSX downloads. Write a function that:
1. Fetches the product page
2. Discovers all XLSX/ZIP download URLs on it
3. Downloads each to data/raw/ with the original filename
4. Records a manifest at data/raw/manifest.json with url, filename,
   sha256, and downloaded_at for every file

Make it idempotent: if a file already exists with the same sha256 in the
manifest, skip it. If the remote sha256 differs, replace and update the
manifest entry.

Add tests in tests/test_ingest.py that mock the HTTP layer and verify the
manifest behavior.

Do not parse the XLSX files yet. That's the next prompt.
```

### Prompt 3 — Catalog and cleaning

```
Build src/usda_sandbox/catalog.py and src/usda_sandbox/clean.py.

For catalog.py:
- Define a Pydantic model SeriesDefinition with fields:
  series_id, series_name, commodity, metric, unit, frequency,
  source_file, source_sheet, header_rows_to_skip, value_columns,
  date_column, notes
- Maintain catalog.json with one entry per known series. Start by manually
  cataloging 5-10 representative series from the ERS files (a mix of cattle
  prices, wholesale prices, supply/disappearance). Inspect the actual XLSX
  files in data/raw/ to fill in the right header offsets and column names.

For clean.py:
- A function clean_series(series_def, raw_path) -> polars.DataFrame
  that reads the specified sheet with the right header offsets and produces
  rows matching the observations schema in the handoff doc.
- A function clean_all(catalog_path, raw_dir, out_path) that processes
  every series in the catalog and writes a single tidy
  observations.parquet to data/clean/.

Add tests with small fixture XLSX files in tests/fixtures/ that exercise
the header-skipping, date parsing, and unit normalization. Aim for the
fixtures to cover the awkward shapes you actually find in the real files.

Run clean_all against the real downloaded data and report:
- Total rows in observations.parquet
- Date range covered
- Series count
- Any rows where value is null and why
```

### Prompt 4 — Storage helpers

```
Build src/usda_sandbox/store.py.

Provide thin helpers:
- read_observations() -> polars.LazyFrame
- read_series(series_id: str) -> polars.DataFrame
- list_series() -> polars.DataFrame  # one row per series with metadata
- duckdb_connection() -> duckdb.DuckDBPyConnection
  attached to the parquet file as a virtual table named `obs`

These are convenience wrappers. The notebooks should reach for them instead
of opening parquet directly.

Add a quick test that all four helpers work against the cleaned data.
```

### Prompt 5 — Exploration notebook

```
Create notebooks/01_explore.ipynb. The notebook should:
1. Load the catalog and the observations table using src/usda_sandbox/store
2. Print summary statistics: total series, total rows, date range per
   commodity, frequency mix
3. Show a series list grouped by commodity with row counts and date ranges
4. Plot a representative sample (5-10 series) with plotly on shared axes
   so the relative scales are visible
5. Highlight any data quality issues found during ingest: missing months,
   sudden series breaks, unit changes

Keep it readable. Markdown cells explaining what each section is showing.
This notebook is the artifact you'd show someone to answer "what's in this
data?"
```

### Prompt 6 — Forecasting module

```
Build src/usda_sandbox/forecast.py.

Implement three forecasters with a common interface:
- StatsForecastAutoARIMA (uses statsforecast.AutoARIMA)
- ProphetForecaster (uses prophet)
- LightGBMForecaster (lag features 1, 2, 3, 6, 12, 24 months;
  rolling means; calendar features)

Common interface:
  fit(df: polars.DataFrame) -> None
  predict(horizon: int) -> polars.DataFrame
    # returns columns: period_start, point, lower_80, upper_80
  cross_validate(df, horizon, n_windows) -> polars.DataFrame
    # returns one row per (window, period_start) with point and actual

Add a top-level run_backtest(series_id, horizon=6, n_windows=8) function
that runs all three forecasters with cross-validation and returns a
combined results table plus per-model MAPE / sMAPE / MASE.

Seed every random source. The function should be deterministic given the
same inputs.
```

### Prompt 7 — Visualization notebook

```
Create notebooks/02_visualize.ipynb. The notebook should:
1. Pick 4-5 high-interest series (cattle prices, boxed beef cutout, hog
   prices, broiler prices, beef per-capita disappearance — adjust based on
   what's actually in the data)
2. Plot each on its own with annotations for major events (e.g., 2014-15
   cattle price peak, 2020 COVID disruption, 2022-24 cattle cycle)
3. Show seasonal decomposition (trend + seasonal + residual) using
   statsmodels.tsa.seasonal_decompose for each
4. Show year-over-year change plots
5. Show a correlation heatmap across the selected series at monthly
   frequency

Output should look like a polished mini-report. The plots should be ones
you'd actually screenshot and put in front of Chad and Wade.
```

### Prompt 8 — Forecasting notebook

```
Create notebooks/03_forecast.ipynb. The notebook should:
1. Pick one or two anchor series (e.g., choice steer prices, boxed beef
   cutout)
2. Run forecast.run_backtest on each, with horizon=6 months and
   n_windows=12
3. Show a table comparing AutoARIMA / Prophet / LightGBM on
   MAPE / sMAPE / MASE
4. Plot the actuals vs. each model's forecasts across all CV windows
5. Plot the in-sample residual diagnostics for the best model
6. Generate and plot the forward 12-month forecast from the best model
   with 80% prediction intervals
7. End with a short markdown takeaway: which model wins, by how much,
   and what the data tells you about predictability of the series

The takeaway is the artifact. It is the answer to "is forecasting useful
on this data?"
```

### Prompt 9 — Polish and document

```
Pass over the whole repo:
- Make sure ruff and mypy pass cleanly
- Make sure pytest passes
- Make sure the three notebooks run top-to-bottom without manual
  intervention given a clean data/clean/observations.parquet
- Update the README with the actual results: how many series, date
  range, best forecaster on the anchor series, MAPE numbers
- Add a "what I learned" section to the README that summarizes the
  takeaway notebooks 1-3 produced

Tag this as v0.1 and stop. The project is finished as a sandbox.
```

---

## What you walk away with

- A working repo that re-runs end-to-end any time the USDA refreshes the data
- A tidy parquet store of all USDA livestock and meat series cataloged
- Three notebooks: what's in the data, what it looks like over time, and how forecastable each series is
- Numbers — actual MAPE / sMAPE / MASE results — on cattle prices and boxed beef cutout
- A clear sense of which series are forecastable and which aren't, which directly informs the Tenure pricing module design later
- Code (`ingest.py`, `clean.py`, `store.py`, `forecast.py`) that can be lifted into the Tenure ops platform when the pricing module gets built

---

## Practical notes for working with Claude Code on this

1. **Resist scope creep.** Claude Code will sometimes propose adding scrapers for AMS LMR, CME futures, weather data. All worthwhile, none in scope for v0.1. Tell it no and stay focused on ERS.

2. **The cleaning step is where the project lives or dies.** When Claude Code wants to skip writing fixture-based tests for `clean.py`, push back. Each XLSX shape that breaks should produce a fixture and a regression test. This is the boring discipline that makes the data layer reusable later.

3. **Keep notebooks reproducible.** No "data already loaded" cells, no hardcoded paths, no commented-out exploration. If a cell isn't needed for the final story, delete it. The three notebooks are deliverables, not scratchpads.

4. **Don't let the forecasting wing turn into a leaderboard.** Three baseline models (ARIMA, Prophet, LightGBM) is enough. Adding NeuralProphet, N-BEATS, TFT, etc. is the wrong direction for v0.1 — that's later research, not this sandbox.

5. **The Yoga is your dev machine for this.** All of the work fits comfortably on the i7-10750H + 16 GB RAM. Don't let Claude Code talk you into Colab unless you specifically want to play with foundation-model time-series stuff later.

Good luck. This one should be quick — a couple of evening sessions if you want to push through it.
