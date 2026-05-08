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
