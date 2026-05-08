# USDA Livestock Sandbox

A small Python project that downloads USDA ERS livestock and meat data, cleans it into a tidy parquet store, runs classical time-series forecasts, and surfaces the results in three notebooks. It is a sandbox for learning the data ahead of building Tenure's pricing module — not a product. See [CLAUDE.md](CLAUDE.md) for project guidance and constraints.

## How to run

```bash
# Install Python 3.11 and dependencies into a managed venv
uv sync

# 1. Pull raw XLSX/ZIP files from ers.usda.gov into data/raw/
uv run python -c "from usda_sandbox.ingest import sync_downloads; sync_downloads()"

# 2. Clean every catalog series into data/clean/observations.parquet
uv run python -c "from usda_sandbox.clean import clean_all; clean_all('data/catalog.json', 'data/raw', 'data/clean/observations.parquet')"

# 3. Open the notebooks
uv run jupyter lab

# Tests / lint / types
uv run pytest
uv run ruff check .
uv run mypy
```

### Launch the dashboard

```bash
uv run streamlit run dashboard/app.py
```

Opens at http://localhost:8501. Three pages: **Explore** (catalog and
coverage), **Visualize** (single-series time-series + decomposition + YoY),
and **Forecast** (live backtest with per-window progress + scoreboard +
12-month forward forecast). See [`dashboard/README.md`](dashboard/README.md)
for details.

## Results — what's in the cleaned store

12 series, 2,895 observations.

| dimension | value |
|---|---|
| Series count | 12 (8 cattle, 3 hogs, 1 sheep/lamb) |
| Frequencies | Monthly (9 series) and quarterly (3 series) |
| Monthly date range | 2000-01-01 → 2026-03-01 (315 months per series) |
| Quarterly date range | 2021 Q1 → 2025 Q4 (20 quarters per series) |
| Total rows | 2,895 |
| Null values | 113 (all from real source-side reporting gaps) |
| Missing rows | 0 — every series has a row for every period in its range |

Where the nulls live:

| Series | Nulls | Cause |
|---|---|---|
| `lamb_slaughter_choice_san_angelo` | 96 | Series effectively discontinued in source after 2018-04 |
| `hog_barrow_gilt_natbase_51_52` | 12 | Recent months 2025-04+ not yet populated |
| `cattle_steer_choice_tx_ok_nm` | 4 | 2019-11/12, 2020-01 (COVID-era reporting gap), 2025-12 |
| `cattle_steer_choice_nebraska` | 1 | Single missing month at 2022-01 |

## Results — forecast bake-off

Two anchor series, 6-month horizon, 12 rolling-origin CV windows. Per-anchor winners (lowest MAPE):

| Anchor | Best model | MAPE | sMAPE | MASE |
|---|---|---|---|---|
| **Nebraska Choice 65-80% steers** | **AutoARIMA** | **6.29%** | 6.46% | 2.94 |
| **Boxed beef cutout, Choice** | **Prophet** | **8.08%** | 8.81% | 3.06 |

Full scoreboard (sorted by MAPE within anchor):

| Anchor | Model | MAPE | sMAPE | MASE |
|---|---|---|---|---|
| Boxed beef cutout — Choice | Prophet | 8.08% | 8.81% | 3.06 |
| Boxed beef cutout — Choice | LightGBM | 8.56% | 8.85% | 3.12 |
| Boxed beef cutout — Choice | AutoARIMA | 8.57% | 8.99% | 3.10 |
| NE Steers (Choice 65-80%) | AutoARIMA | 6.29% | 6.46% | 2.94 |
| NE Steers (Choice 65-80%) | LightGBM | 7.73% | 7.91% | 3.52 |
| NE Steers (Choice 65-80%) | Prophet | 14.72% | 16.39% | 6.72 |

Forward 12-month point forecasts from the per-anchor winners (full numeric tables in `notebooks/03_forecast.ipynb`):

- **NE Steers (AutoARIMA):** $233.57/cwt at 2026-04 rising toward ~$245/cwt by 2027-02; 80% PI widens from ±$6 to ±$20 across the horizon
- **Boxed Beef Choice (Prophet):** Oscillates seasonally around $363-381/cwt; 80% PI ±$25

## What I learned

Notebook 01 — *what's in the data*:

- **Coverage is dense.** Every monthly series has a row for every month between its first and last observation. Where data is genuinely unavailable, the cleaner emits a null rather than dropping the row — so coverage gaps are visible in the data instead of hidden by parsing.
- **Most nulls cluster in two known issues.** Lamb slaughter at San Angelo went dark in the source after 2018-04; hog 51-52% lean is sparse in the most recent months because the live release isn't fully populated yet. These will need a substitution strategy before forecasting.
- **MoM shocks line up with real events.** 17 monthly observations have >25% month-over-month change — and they map cleanly to known cattle-cycle peaks and the 2020 COVID disruption, not to parsing artifacts.
- **No series changes units mid-stream.** Downstream code can treat each `series_id` as a single, consistent variable.

Notebook 02 — *what it looks like over time*:

- **The cattle complex moves as one.** Fed steers, feeder steers, and both boxed-beef cutouts (Choice and Select) correlate at 0.90 to 0.97 across 315 monthly observations. Choice ↔ Select boxed beef is essentially the same series at 0.996.
- **Pork is on its own clock.** Pork cutout correlates at only 0.65 to 0.74 with the cattle complex; its cycles (PED outbreaks, 2020 COVID slaughter shock) dominate.
- **Seasonality is real but modest.** Multiplicative seasonal factor stays inside ±5% for fed cattle; pork seasonality is more pronounced (grilling-season demand pattern is visible).
- **Three macro shocks dominate the residuals:** 2014-15 (drought + tight cattle inventory), 2020 COVID (slaughter capacity collapse, biggest in pork), 2022-23 (commodity / inflation spike).
- **YoY framing flatters the cycle.** Plotting 12-month percent change makes the cattle cycle's amplitude obvious — multi-year regimes of double-digit positive YoY change followed by stretches near zero or negative.

Notebook 03 — *is forecasting useful?*:

- **No single model wins both anchors.** AutoARIMA dominates the NE Steers anchor at 6.29% MAPE. Prophet wins boxed beef cutout, but only barely (all three within ~0.5 percentage points). Pick the model per series, not project-wide.
- **Prophet is high-variance.** It's the boxed-beef winner but the worst fit on Nebraska steers by a wide margin (14.72% vs. AutoARIMA's 6.29%) — likely because Prophet's changepoint detection over-reacts to the post-2020 cattle-cycle regime shift in fed steers. If a series has had a recent durable level-change, AutoARIMA is the safer default.
- **MASE > 1 across the board (2.9 — 6.7).** None of the three models beats a naive one-step forecast on absolute scaled error. Six months out, model errors are ~3x the typical month-to-month price change. The forecasts are useful for level/range, not tactical timing.
- **Residuals are roughly centered but mildly heavy-tailed.** The 80% PI is too tight during shock periods. Read the interval as "80% under business-as-usual," not as a hard probability statement.
- **Errors grow with horizon, modestly.** Month-1-ahead MAE is roughly half of month-6-ahead MAE — useful information stays useful at 6 months but degrades.

## What's not in v0.1

By design — see [CLAUDE.md](CLAUDE.md) non-negotiables:

- No deep learning, no LLMs, no fine-tuning, no GPU
- No external data: AMS LMR, CME futures, weather, feed costs are out of scope
- No live dashboard or API — this is a sandbox

Worth running next (out of scope for v0.1):

1. Add CME futures as exogenous regressors (ARIMAX, LightGBM with futures features)
2. Hybrid model: AutoARIMA point forecast + LightGBM residual model
3. Conformal prediction on the CV residuals to calibrate the 80% PI
4. Per-series model selection baked into any pricing-module downstream

## Layout

```
data/
  raw/               gitignored — sync_downloads writes here
  clean/             gitignored — clean_all writes observations.parquet here
  catalog.json       checked in — 12 series definitions
notebooks/
  01_explore.ipynb   what's in the data
  02_visualize.ipynb what it looks like over time
  03_forecast.ipynb  the bake-off + forward 12-month forecasts
src/usda_sandbox/
  ingest.py          discover + download ERS XLSX/ZIP files (idempotent)
  catalog.py         SeriesDefinition (Pydantic)
  clean.py           parse XLSX → tidy observations.parquet
  store.py           polars + DuckDB accessors
  forecast.py        AutoARIMA / Prophet / LightGBM with shared interface
tests/               56 tests across ingest / clean / store / forecast
pyproject.toml       uv-managed, ruff + mypy strict on src/usda_sandbox
CLAUDE.md            project guidance and non-negotiables
```
