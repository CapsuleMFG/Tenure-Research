# CME Futures as Exogenous Regressors (v0.2b)

**Status:** Draft, pending user review
**Date:** 2026-05-08
**Author:** Tyler + Claude (brainstormed)

## Goal

Cut forecast MAPE on the cattle and pork series by feeding each forecaster
the **CME futures curve** as exogenous regressors. The market's
6-month-deferred Live Cattle price is the best single predictor of the
6-month-ahead cash steer price; we've been forecasting without it. This
spec adds that signal — uniformly across AutoARIMA, Prophet, and
LightGBM — and ships it behind the existing dashboard with no new UI
controls.

After this lands, the Forecast page on `cattle_steer_choice_nebraska`
should show a MAPE drop from v0.2a's 6.3% to something in the 3-5%
range (literature benchmark for ARIMAX cattle forecasts with futures
regressors). The improvement is expected to be smaller on cutout series
because cash-futures coupling is weaker for wholesale beef and pork
than for live cattle.

This is v0.2b. Conformal calibration from v0.2a continues to wrap the
output — calibration will recompute against the new (tighter) CV
residuals and produce a smaller scale factor automatically.

## Scope

**In scope:**

- CME Live Cattle (`LE`) and Lean Hogs (`HE`) — the two contracts that
  cover 8 of the 9 monthly cash series in the catalog.
- yfinance as the data source.
- Horizon-matched deferred continuous series: at each month-end date
  `t` and each horizon `h ∈ {1, ..., 12}`, a series whose value is the
  price of the contract maturing closest to `t + h months`.
- All three forecasters (AutoARIMA, Prophet, LightGBM) gain
  exogenous-regressor support.
- Catalog-driven activation: a cash series's `exogenous_regressors`
  field determines whether it gets the futures treatment.

**Explicitly out of scope** (called out for v0.2c+):

- Feeder cattle (`GF`) futures — adds two more series (`cattle_feeder_*`)
  to the futures path. Worth doing after v0.2b proves out.
- Basis decomposition (separate forecast of `cash − futures`) — A in
  brainstorming; we ruled it out for v0.2b.
- Quantile / term-structure features (curve slope, calendar spreads,
  contango indicator).
- Lamb futures — no public US lamb futures market exists.
- Real-time daily refresh of futures data — same monthly cadence as
  ERS cash data.

## Architecture

One new module (`futures.py`), small extensions to five existing files
(`catalog.py`, `clean.py`, `forecast.py`, `data/catalog.json`,
`dashboard/pages/3_Forecast.py`), no big refactors:

```
src/usda_sandbox/
├── ingest.py          # unchanged
├── catalog.py         # +2 optional fields on SeriesDefinition
├── clean.py           # dispatch by source_file prefix
├── store.py           # unchanged
├── forecast.py        # +exog regressor plumbing
├── calibration.py     # unchanged (v0.2a)
└── futures.py         # NEW
```

New disk layout under `data/raw/futures/`:

```
data/raw/futures/
├── manifest.json                              # SHA + downloaded_at per contract
├── LE_G_2000.parquet  LE_J_2000.parquet  ...  # one parquet per contract
├── LE_G_2001.parquet  LE_J_2001.parquet  ...
└── HE_G_2000.parquet  HE_J_2000.parquet  ...
```

Contract files store `date, open, high, low, close, volume` for that
contract's lifetime. `futures.py` reads from this raw cache and writes
the 24 derived deferred series into `observations.parquet`.

### Catalog growth

12 → 36 entries. Two new optional fields on `SeriesDefinition`:

```python
class SeriesDefinition(BaseModel):
    ...existing fields...
    exogenous_regressors: list[str] = Field(default_factory=list)
    forecastable: bool = True
```

The 9 monthly cash series get `exogenous_regressors` populated with
the 12 deferred-h ids of their matching commodity. The 24 new futures
series_ids get added with `forecastable: false`. The 3 quarterly WASDE
series are untouched.

`clean_all()` becomes a dispatcher: if `source_file.startswith("futures:")`,
route to `futures.build_and_append_deferred_for(commodity)`. Otherwise
route to the existing XLSX path. Existing tests stay green because the
XLSX path is unchanged.

## The `futures.py` module

Four functions, in execution order:

```python
@dataclass(frozen=True)
class FuturesManifestEntry:
    commodity: str          # "LE" or "HE"
    month_code: str         # "G" "J" "K" "M" "N" "Q" "V" "Z"
    year: int
    delivery_date: date     # last business day of contract month
    sha256: str
    downloaded_at: str      # ISO-8601


# 1. Discovery & download (idempotent — SHA-keyed manifest)
def sync_futures(
    commodities: Sequence[str] = ("LE", "HE"),
    *,
    raw_dir: Path = Path("data/raw/futures"),
    start_year: int = 1999,
    client: Any = None,  # yfinance.Ticker or compatible — injectable for tests
) -> dict[str, FuturesManifestEntry]: ...


# 2. Contract calendar
def contract_months(commodity: str) -> tuple[str, ...]:
    """Live cattle (LE): Feb/Apr/Jun/Aug/Oct/Dec → ('G','J','M','Q','V','Z').
    Lean hogs (HE): Feb/Apr/May/Jun/Jul/Aug/Oct/Dec → ('G','J','K','M','N','Q','V','Z')."""


def contract_delivery_date(commodity: str, code: str, year: int) -> date:
    """Last business day of the contract's delivery month.
    Used as the contract's 'expiration anchor' for term-structure math."""


# 3. Build one deferred-h continuous series
def build_deferred_series(
    commodity: str,
    horizon_months: int,             # 1..12
    raw_dir: Path = Path("data/raw/futures"),
) -> pl.DataFrame:
    """Returns a DataFrame matching observations.parquet's schema, with
    series_id = f'{commodity_to_name(commodity)}_deferred_{horizon_months}mo',
    one row per month-end from start_year onward.
    Each row: closing price of the contract maturing closest to (t + h months),
    linearly interpolated between adjacent contracts when no contract
    matches exactly."""


# 4. Top-level pipeline: build all 24 series and append to observations.parquet
def append_futures_to_observations(
    obs_path: Path = Path("data/clean/observations.parquet"),
    raw_dir: Path = Path("data/raw/futures"),
    commodities: Sequence[str] = ("LE", "HE"),
    horizons: range = range(1, 13),
) -> None:
    """Build all (commodity, horizon) combinations and merge into
    observations.parquet. Idempotent: replaces existing rows for the
    affected series_ids (keyed on series_id + period_start)."""
```

### Contract calendars (CME month codes)

| Commodity | Months traded | Settlement |
|---|---|---|
| Live Cattle (`LE`) | Feb, Apr, Jun, Aug, Oct, Dec | Last business day of contract month, physical delivery |
| Lean Hogs (`HE`) | Feb, Apr, May, Jun, Jul, Aug, Oct, Dec | 10th business day before last business day, cash settlement |

For our purposes (monthly cadence), settlement specifics don't matter
beyond: "what is the contract's effective delivery month, and what
month-end closing price does that contract have on date `t`?"

### Deferred-h interpolation logic

For each month-end date `t` and each horizon `h ∈ {1..12}`:

1. Target delivery month = `t + h months`.
2. Find the two adjacent active contracts on date `t`:
   - Contract A maturing ≤ target_delivery
   - Contract B maturing > target_delivery
3. Interpolate:
   ```
   weight = (target_delivery - delivery_A) / (delivery_B - delivery_A)
   price(t, h) = price_A(t) + weight * (price_B(t) - price_A(t))
   ```
4. If target ≤ earliest active contract: use earliest contract's price
   (rare; only happens at the very start of the observation window).
5. If target > latest active contract: use latest contract's price and
   record a warning. In practice CME contracts trade ~18+ months out,
   so for h ≤ 12 this almost never triggers.

`price_A(t)` and `price_B(t)` are the month-end closing prices of
contracts A and B as observed on date `t`.

### yfinance fetching

Tickers follow the convention `<commodity><month_code><yy>.CME`, e.g.,
`LEZ24.CME` for Live Cattle Dec 2024. `sync_futures()` enumerates all
(commodity, month_code, year) tuples from `start_year` to current, attempts
to fetch each via `yfinance.Ticker(symbol).history(period="max")`, and
writes the resulting OHLCV parquet to disk.

Idempotency: like `sync_downloads()` in `ingest.py`, the manifest tracks
SHA + timestamp per file. Re-running `sync_futures()` only re-fetches
contracts that don't have a matching SHA on disk, or whose remote data
has changed.

Older contracts may be unavailable on Yahoo. The manifest records
missing contracts; `build_deferred_series` skips them and warns. If a
deferred-h series ends up with > 10% null rows in the post-2000 window,
log a clear error so the user knows to investigate.

## Forecaster modifications

`BaseForecaster` gets an `exog` parameter on `fit`, `predict`, and
`cross_validate_iter`. Default `None` preserves v0.2a behavior bit-for-bit.

```python
class BaseForecaster:
    def fit(
        self, df: pl.DataFrame, exog: pl.DataFrame | None = None
    ) -> None: ...

    def predict(
        self, horizon: int, exog_future: pl.DataFrame | None = None
    ) -> pl.DataFrame: ...

    def cross_validate_iter(
        self,
        df: pl.DataFrame,
        horizon: int,
        n_windows: int,
        exog: pl.DataFrame | None = None,
    ) -> Iterator[tuple[int, pl.DataFrame]]: ...
```

`exog` schema: `period_start` (Date) + one column per regressor. Must
align with `df` on `period_start`. Rows with any null regressor are
dropped (and `df` rows for those periods are dropped in lockstep).
`exog_future` must have exactly `horizon` rows.

### Per-forecaster wiring

**`StatsForecastAutoARIMA`** — `statsforecast.StatsForecast.fit(train)`
already accepts extra columns alongside `unique_id, ds, y` and routes
them to AutoARIMA as the `X_df` exogenous regressor matrix. At predict
time, pass `X_df=exog_future_pandas`.

**`ProphetForecaster`** — before `model.fit(train)`, call
`model.add_regressor(col)` for each exog column. The training DataFrame
includes the regressor columns. The `future` DataFrame passed to
`model.predict(future)` must carry the same columns. Default
`prior_scale=10.0` for each regressor (Prophet's default; tuning is
out of scope).

**`LightGBMForecaster`** — exog columns are appended to the feature
matrix in `_build_features()` alongside the existing lags/rollings/
calendar columns. The recursive predict step uses the **actual
observed deferred-h price** for each future step (no extrapolation,
because the deferred-h regressor IS the market's expectation for that
step, observed today).

### `iter_run_backtest` upgrade

```python
def iter_run_backtest(
    series_id: str,
    horizon: int = 6,
    n_windows: int = 8,
    *,
    obs_path: Path | str | None = None,
    seed: int = DEFAULT_SEED,
    models: Sequence[str] | None = None,
    catalog_path: Path | str | None = None,  # NEW — defaults to data/catalog.json
) -> Iterator[BacktestProgress | BacktestResult]: ...
```

New behavior: at the start of the function, load the catalog and look
up the target series's `SeriesDefinition`. If `exogenous_regressors`
is non-empty, fetch those series from `observations.parquet`, pivot
wide on `period_start` (one column per regressor), and pass through
to each forecaster's `cross_validate_iter`. If empty (e.g., lamb),
the path is identical to v0.2a — no exog passed.

For the forward forecast on the dashboard (refit winner on full
history, predict 12 months ahead): pull the most recent 12 deferred
rows as `exog_future`. The dashboard's existing refit code path becomes:

```python
fcst.fit(history, exog=exog_history)
forward = fcst.predict(horizon=12, exog_future=exog_future)
```

### Backward compatibility

All 85 existing tests pass unchanged. The new `exog` parameter defaults
to `None` everywhere; when `None`, the code paths are guarded with
`if exog is None: # v0.2a path` and behave identically.

## Catalog updates

`data/catalog.json` schema gains two optional fields. Existing entries
get the defaults (`exogenous_regressors=[]`, `forecastable=True`) and
become forwards-compatible.

The 9 monthly cash series get their `exogenous_regressors` field set:

| Cash series | `exogenous_regressors` |
|---|---|
| `cattle_steer_choice_nebraska` | 12 entries: `cattle_lc_deferred_1mo` ... `cattle_lc_deferred_12mo` |
| `cattle_steer_choice_tx_ok_nm` | same 12 |
| `cattle_feeder_steer_500_550` | same 12 (will switch to `GF` deferred when v0.2c lands) |
| `cattle_feeder_steer_750_800` | same 12 |
| `boxed_beef_cutout_choice` | same 12 |
| `boxed_beef_cutout_select` | same 12 |
| `pork_cutout_composite` | 12 entries: `hogs_he_deferred_1mo` ... `hogs_he_deferred_12mo` |
| `hog_barrow_gilt_natbase_51_52` | same 12 |
| `lamb_slaughter_choice_san_angelo` | `[]` (no public lamb futures) |

24 new SeriesDefinitions get appended for the futures themselves:

```json
{
  "series_id": "cattle_lc_deferred_1mo",
  "series_name": "Live Cattle futures, 1-month deferred (continuous)",
  "commodity": "cattle",
  "metric": "futures_price",
  "unit": "USD/cwt",
  "frequency": "monthly",
  "source_file": "futures:LE",
  "source_sheet": "",
  "header_rows_to_skip": 0,
  "value_columns": [],
  "date_column": "",
  "notes": "Continuous series: at each month-end, the price of the LE contract maturing 1 month ahead. Linear interpolation between adjacent contracts when no single contract maps exactly. Built from per-contract data pulled via yfinance.",
  "exogenous_regressors": [],
  "forecastable": false
}
```

The XLSX-shaped fields (`source_sheet`, `header_rows_to_skip`,
`value_columns`, `date_column`) are placeholders for futures entries —
`clean_all()`'s dispatcher checks `source_file.startswith("futures:")`
to skip the XLSX parsing path.

## Dashboard impact

Only one change: a new caption appears under the Forecast page's
scoreboard when the chosen series has `exogenous_regressors`:

```python
target_def = _series_def_by_id[series_id]
if target_def.exogenous_regressors:
    commodity_label = (
        "Live Cattle" if target_def.commodity == "cattle" else "Lean Hogs"
    )
    st.caption(
        f"This series was forecast with **{len(target_def.exogenous_regressors)} "
        f"exogenous regressors**: deferred {commodity_label} futures (1-12 "
        f"months ahead). Each forecaster (AutoARIMA, Prophet, LightGBM) sees "
        f"these alongside the cash history."
    )
```

The sidebar series picker on the Forecast page is filtered to
`forecastable == True` (already filtered to `monthly` for the existing
v0.2a constraint). So the user sees the same 9 cash series they see
today — not 33.

The Explore page lists all series including futures (no filter), which
serves as a sanity check on the deferred-curve construction.

The Visualize page picker also gets the `forecastable == True` filter
applied — users go to Visualize to look at *cash* series. If they want
to inspect the futures curves themselves, the Explore page is the
right home for that.

The Forecast page's existing "How to read these results" expander
keeps the existing copy. Adding a sentence about regressors would
bloat it; the new caption under the scoreboard carries the load.

## Tests

Suite goes 85 → ~117. Three new test files / extensions:

### `tests/test_futures.py` (~18 new tests)

```python
# Contract calendar
test_contract_months_live_cattle()                    # ('G','J','M','Q','V','Z')
test_contract_months_lean_hogs()                       # ('G','J','K','M','N','Q','V','Z')
test_contract_months_unknown_commodity_raises()
test_contract_delivery_date_handles_month_boundaries()

# Deferred-h interpolation
test_deferred_series_uses_exact_contract_when_target_matches()
test_deferred_series_linear_interpolates_between_contracts()
test_deferred_series_uses_earliest_when_target_before_curve()
test_deferred_series_warns_and_uses_latest_when_target_past_curve()
test_deferred_series_monthly_cadence()                 # one row per month-end
test_deferred_series_handles_missing_contract_data_with_nulls()

# Sync (yfinance mocked)
test_sync_futures_idempotent_on_unchanged_data()
test_sync_futures_replaces_changed_contract_data()
test_sync_futures_records_manifest_correctly()
test_sync_futures_skips_unavailable_contracts_gracefully()

# Top-level pipeline
test_append_futures_to_observations_writes_24_series()
test_append_futures_to_observations_is_idempotent()
test_append_futures_to_observations_preserves_existing_cash_series()
test_clean_all_dispatcher_routes_futures_correctly()
```

### Extensions to `tests/test_forecast.py`

```python
# Backward compat — exog=None paths identical to today
test_fit_predict_without_exog_unchanged_for_all_three_forecasters()
test_iter_run_backtest_without_exog_path_unchanged()

# With exog
test_autoarima_with_exog_returns_correct_shape()
test_prophet_with_exog_returns_correct_shape()
test_lightgbm_with_exog_returns_correct_shape()
test_iter_run_backtest_loads_catalog_exog_when_present()
test_iter_run_backtest_skips_exog_when_empty_regressor_list()

# Alignment edge cases
test_fit_drops_rows_where_any_exog_column_is_null()
test_predict_raises_when_exog_future_length_mismatches_horizon()
```

### Extensions to `tests/test_clean.py` (and the inline catalog tests)

```python
# Catalog
test_series_definition_exogenous_regressors_defaults_empty()
test_series_definition_forecastable_defaults_true()
test_series_definition_with_regressors_round_trips()

# clean_all dispatcher
test_clean_all_dispatches_futures_prefix_to_futures_path()
test_clean_all_xlsx_path_unchanged_for_existing_series()
```

## Edge cases (explicit handling)

| Case | Behavior |
|---|---|
| `exog` shorter than `df` | Truncate `df` to overlap; log a warning |
| `exog` has null values in some rows | Drop those rows from both `exog` and `df` in lockstep |
| `exog_future` length != `horizon` | `ValueError("exog_future must have <horizon> rows")` |
| yfinance returns empty data for an old contract | Skip; record in manifest as `missing: true`; do not raise |
| Target delivery beyond latest available contract | Use latest contract's price; log a warning |
| Cash series starts before any futures data | Train+CV restricted to overlap; warned. NE steers (2000-01) + LE futures (1964-) → no truncation in practice |
| User picks a series with empty `exogenous_regressors` (lamb) | v0.2a path runs unchanged; no exog passed, no scoreboard caption shown |
| yfinance rate limits / network error during `sync_futures` | Retry once; if still failing, log and skip; allow `append_futures_to_observations` to proceed with whatever's already in `data/raw/futures/` |
| `data/raw/futures/` doesn't exist | `sync_futures` creates it. `append_futures_to_observations` raises if called before sync |

## Interactions with v0.2a (conformal calibration)

No code interaction — v0.2a's `calibration.py` is unchanged. But the
empirical effect chains:

1. Better point forecasts (lower MAPE) → tighter CV residuals
2. Tighter CV residuals → smaller conformal scale factor
3. Smaller scale factor → tighter calibrated 80% PI

The dashboard caption currently says "scaled by 2.10x..." for NE Steers
on AutoARIMA. After v0.2b ships, expect that number to drop into the
1.2-1.7x range as the model becomes less overconfident in the first
place. The conformal caption will surface the change without any code
changes to it.

## Open questions for the user

None — every decision has a default chosen during brainstorming. If
something feels wrong on review, flag it inline and we'll iterate.
