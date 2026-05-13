# Continuous Front-Month Futures as Single Regressor (v0.2c)

**Status:** Shipped (see post-implementation note below)
**Date:** 2026-05-13
**Author:** Tyler + Claude (brainstormed)

## Post-implementation note (2026-05-13, after implementation)

Two empirical surprises landed during implementation; both were
patched in-branch.

1. **Stooq closed its free CSV endpoint.** The original plan called for
   `https://stooq.com/q/d/l/?s={symbol}&i=m`; this URL now returns an
   instruction to register for an API key (captcha-gated). The pivot:
   use yfinance's continuous-front-month symbols (`LE=F`, `HE=F`,
   `GF=F`) instead. yfinance returns ~25 years of monthly closes per
   symbol — enough for our purposes. The architecture, catalog wiring,
   and dispatcher path are unchanged; only the symbol naming
   (`le.c` → `LE=F`), on-disk filename pattern (`=` → `_`), and fetcher
   implementation differ from what's described below.

2. **yfinance drops random months in its monthly resampling.** Verified
   empirically: e.g., `LE=F` returns no row for 2001-04, 2026-02, and
   2026-03, even though those months had full trading activity. The
   fix is a forward-fill pass at append-time (`_fill_monthly_gaps`):
   reindex to a contiguous month-start range and carry forward the
   most recent close. The on-disk cache stays a faithful record of
   yfinance's response.

End-to-end results (horizon=6, n_windows=8) confirm the v0.2c design
delivers the spec's promise:
- `cattle_steer_choice_nebraska`: Prophet, MAPE 2.38%
- `cattle_feeder_steer_500_550`: LightGBM, MAPE 7.48%
- `pork_cutout_composite`: Prophet, MAPE 4.46%

Nebraska steers comes in below the spec's promised 3-5% range.

The rest of this document describes the original design. Mentions of
Stooq, `le.c`/`he.c`/`gf.c` symbols, `_default_stooq_fetcher`, and
`_parse_stooq_csv` should be read as their yfinance equivalents
(`LE=F`/`HE=F`/`GF=F`, `_default_fetcher`, no parser needed).

## Goal

Cut forecast MAPE on the 8 monthly cash series by feeding each
forecaster the **continuous front-month** CME futures price for the
matching commodity as an exogenous regressor. Replaces v0.2b's
"12 deferred-h regressors per cash series via yfinance" design, which
end-to-end testing showed doesn't survive real data:

1. yfinance serves ~5 years of per-contract history → ~29 months of
   overlap between cash and all-12-regressors-non-null → too few rows
   for any reasonable CV window count.
2. 12 deferred-h columns of the same commodity have effective rank
   ~2 (level + curve slope) → AutoARIMA's rank check raises
   "xreg is rank deficient" even when row count is sufficient.

After v0.2c lands, the Forecast page on `cattle_steer_choice_nebraska`
should show a MAPE drop from v0.2a's ~6.3% into the 3-5% range
(literature benchmark for ARIMAX cattle forecasts with a single
futures regressor). The improvement is expected to be smaller on
cutout series because cash-futures coupling is weaker for wholesale
beef and pork than for live cattle.

## Scope

**In scope:**

- Stooq as the data source. Free CSV endpoint, no key, ~30 years of
  monthly continuous front-month closes per commodity.
- Two new continuous series: `cattle_lc_front` (LE) and `hogs_he_front`
  (HE). GF continuous (`gf.c`) is opportunistic — if Stooq serves it,
  add `cattle_feeder_front`; if not, cattle feeder series use
  `cattle_lc_front` (LE leads the feeder cycle anyway).
- 8 cash series get exactly one regressor each, pointing at the
  matching continuous front-month.
- The 36 existing deferred-h catalog entries stay. They're a useful
  diagnostic view on Explore but no longer feed forecasters. The
  per-contract `sync_futures()` flow is opt-in, not part of the
  default Refresh.
- Conformal calibration (v0.2a + per-horizon from v0.2c[3]) continues
  to wrap forecasts unchanged.

**Out of scope:**

- Re-enabling the per-contract / deferred-h path as a forecaster
  regressor. Requires a deeper data source we don't have on free terms.
- Term-structure features (curve slope, spreads, contango). The
  continuous front-month is one column; no curve information.
- Lamb futures — no public US market.
- Daily refresh — same monthly cadence as ERS cash.

## Architecture

One new module (`futures_continuous.py`), three touched files
(`catalog.json`, `dashboard/components/sidebar.py`,
`dashboard/pages/3_Forecast.py`), one new test file. The existing
`futures.py` is untouched but de-prioritized in the default flow.

```
src/usda_sandbox/
├── ingest.py             # unchanged
├── catalog.py            # unchanged (schema already supports exog)
├── clean.py              # +1 dispatcher branch for "futures_continuous:" prefix
├── store.py              # unchanged
├── forecast.py           # unchanged (exog plumbing already done in v0.2b)
├── calibration.py        # unchanged
├── futures.py            # unchanged — still available, just opt-in to sync
└── futures_continuous.py # NEW
```

New disk layout under `data/raw/futures_continuous/`:

```
data/raw/futures_continuous/
├── manifest.json                  # SHA + downloaded_at per commodity
├── le.c.parquet                   # monthly Stooq CSV → parquet
├── he.c.parquet
└── gf.c.parquet                   # only if Stooq returns data
```

## The `futures_continuous.py` module

Two functions and a small dataclass.

```python
@dataclass(frozen=True)
class ContinuousManifestEntry:
    symbol: str                # "le.c", "he.c", "gf.c"
    sha256: str
    downloaded_at: str         # ISO-8601
    missing: bool = False      # True if Stooq returned no data

def sync_continuous_futures(
    *,
    symbols: Iterable[str] = ("le.c", "he.c", "gf.c"),
    raw_dir: Path = Path("data/raw/futures_continuous"),
    fetcher: Callable[[str], pl.DataFrame] | None = None,
) -> dict[str, ContinuousManifestEntry]:
    """Download monthly continuous front-month closes from Stooq.

    Idempotent: SHA-keyed manifest, parallel to sync_futures().
    Records missing symbols (e.g., if gf.c isn't served) without
    raising so the rest of the sync proceeds.
    """

def append_continuous_to_observations(
    *,
    obs_path: Path = Path("data/clean/observations.parquet"),
    raw_dir: Path = Path("data/raw/futures_continuous"),
    symbols: Iterable[str] = ("le.c", "he.c", "gf.c"),
) -> None:
    """Read cached parquets, produce one row per (symbol, month-end),
    merge into observations.parquet. Idempotent on
    (series_id, period_start, source_file)."""
```

### Stooq fetcher

```python
def _default_stooq_fetcher(symbol: str) -> pl.DataFrame:
    url = f"https://stooq.com/q/d/l/?s={symbol}&i=m"
    # urllib + StringIO + polars.read_csv with explicit schema.
    # Returns a DataFrame with columns: period_start (Date), close (Float64).
    # If Stooq returns "No data" or HTTP error, return empty DataFrame.
```

CSV columns are `Date, Open, High, Low, Close, Volume`. We keep
`Date → period_start` and `Close → close`; everything else is dropped.
Monthly cadence with `i=m` returns month-end values directly, no
resampling needed.

### Series-id mapping

| Stooq symbol | series_id              | series_name                                  |
|---|---|---|
| `le.c`       | `cattle_lc_front`      | Live Cattle front-month futures (continuous) |
| `he.c`       | `hogs_he_front`        | Lean Hogs front-month futures (continuous)   |
| `gf.c`       | `cattle_feeder_front`  | Feeder Cattle front-month futures (cont.)    |

Each appended row uses:
- `commodity` → "cattle" / "hogs" / "cattle"
- `metric` → "futures_price"
- `unit` → "USD/cwt"
- `frequency` → "monthly"
- `source_file` → `f"futures_continuous:{symbol}"`
- `source_sheet` → ""

`clean.py` gets one new dispatcher branch parallel to the existing
`futures:` branch. The behavior mirrors v0.2b's pattern:

1. The per-series loop in `clean_all()` skips any series whose
   `source_file.startswith("futures_continuous:")` (don't try to
   parse an XLSX that doesn't exist).
2. After the XLSX loop, if any series_def was a futures_continuous
   series, `clean_all()` calls `append_continuous_to_observations()`
   to inject those rows into observations.parquet.

This keeps the call-graph the same: the dashboard sidebar invokes
`sync_continuous_futures()` then `clean_all()`, and `clean_all()` is
responsible for routing each catalog series to its source-of-truth
appender.

## Catalog updates

`data/catalog.json` gains 2-3 new entries (the GF continuous is
conditional on Stooq serving it; we add the entry preemptively and
let the sync record it as missing if needed) and rewires 8 cash
series' regressors.

### New entries

```json
{
  "series_id": "cattle_lc_front",
  "series_name": "Live Cattle front-month futures (continuous)",
  "commodity": "cattle",
  "metric": "futures_price",
  "unit": "USD/cwt",
  "frequency": "monthly",
  "source_file": "futures_continuous:le.c",
  "source_sheet": "",
  "header_rows_to_skip": 0,
  "value_columns": ["X"],
  "date_column": "X",
  "notes": "Stooq's back-adjusted continuous front-month series. Used as a single exogenous regressor for the cattle cash series. The deferred-h elaboration from v0.2b lives in the catalog (cattle_lc_deferred_*mo) for diagnostic viewing on Explore but is no longer fed to forecasters.",
  "exogenous_regressors": [],
  "forecastable": false
}
```

Analogous entries for `hogs_he_front` and `cattle_feeder_front`.

### Cash-series rewire

| Cash series                          | New `exogenous_regressors` |
|---|---|
| `cattle_steer_choice_nebraska`       | `["cattle_lc_front"]` |
| `cattle_steer_choice_tx_ok_nm`       | `["cattle_lc_front"]` |
| `cattle_feeder_steer_500_550`        | `["cattle_lc_front"]` (or `cattle_feeder_front` if `gf.c` lands) |
| `cattle_feeder_steer_750_800`        | `["cattle_lc_front"]` (same) |
| `boxed_beef_cutout_choice`           | `["cattle_lc_front"]` |
| `boxed_beef_cutout_select`           | `["cattle_lc_front"]` |
| `pork_cutout_composite`              | `["hogs_he_front"]` |
| `hog_barrow_gilt_natbase_51_52`      | `["hogs_he_front"]` |
| `lamb_slaughter_choice_san_angelo`   | `[]` (unchanged) |

The feeder→GF rewire happens automatically as part of the implementation
if `gf.c` returns data on first sync. If it doesn't, feeder stays on
`cattle_lc_front` (LE leads the feeder cycle by 12-18 months, so this
is economically defensible).

## Dashboard changes

### Sidebar Refresh button

Drop `sync_futures()` from the default chain. Add `sync_continuous_futures()`.

```python
# components/sidebar.py — _render_refresh_button
sync_downloads(raw_dir=DEFAULT_RAW_DIR)                              # unchanged
sync_continuous_futures(raw_dir=DEFAULT_RAW_DIR / "futures_continuous")  # NEW
clean_all(DEFAULT_CATALOG_PATH, DEFAULT_RAW_DIR, obs_path)           # unchanged
```

`clean_all()` will pick up the futures_continuous rows via the new
dispatcher branch and write them into observations.parquet.

### Forecast page caption

The existing "X exogenous regressors: deferred {commodity} futures
(1-12 months ahead)" caption is replaced by a one-regressor variant
when the cash series has a single front-month regressor:

```python
if len(target_def.exogenous_regressors) == 1:
    reg = target_def.exogenous_regressors[0]
    commodity_label = {
        "cattle_lc_front": "Live Cattle",
        "cattle_feeder_front": "Feeder Cattle",
        "hogs_he_front": "Lean Hogs",
    }.get(reg, "futures")
    st.caption(
        f"This series was forecast with a single exogenous regressor: "
        f"the continuous front-month **{commodity_label}** futures price. "
        f"Each forecaster (AutoARIMA, Prophet, LightGBM) sees it alongside "
        f"the cash history."
    )
```

The existing multi-regressor branch is kept for forward compatibility
(if the deferred-h path is re-enabled later) but won't fire on the
shipped catalog.

### Slider math

No changes. The exog-overlap-aware bounds added in v0.2b's revert
commit naturally produce `effective_n_obs ≈ n_obs` when the overlap
with one continuous regressor is decades deep.

## Tests

Suite goes from 143 → ~155.

### `tests/test_futures_continuous.py` (~10 new tests)

```python
# Fetcher round-trip
test_fetcher_parses_stooq_csv_correctly()
test_fetcher_returns_empty_on_no_data_response()
test_fetcher_returns_empty_on_http_error()

# Sync idempotency + manifest
test_sync_writes_parquet_and_manifest()
test_sync_idempotent_when_sha_unchanged()
test_sync_records_missing_symbols_without_raising()
test_sync_skips_known_missing_symbols_on_rerun()

# Append
test_append_writes_one_row_per_month_end()
test_append_preserves_existing_observations()
test_append_idempotent_on_rerun()
test_append_dispatcher_routes_futures_continuous_prefix()
```

All tests inject a synthetic fetcher; no network calls in the test
suite.

### Existing test files

Unchanged. `test_futures.py`, `test_calibration.py`, `test_forecast.py`,
`test_clean.py` all pass without modification — `futures.py` is
untouched, calibration is untouched, forecast.py's exog plumbing is
untouched.

## Edge cases

| Case | Behavior |
|---|---|
| Stooq returns "No data" for a symbol (e.g., `gf.c` not served) | Record `missing=True` in manifest; skip; do not raise |
| HTTP error mid-sync | Record `missing=True` for that symbol; continue with others |
| `data/raw/futures_continuous/` doesn't exist | `sync_continuous_futures` creates it |
| `append_continuous_to_observations` runs before sync | Raise `FileNotFoundError` with a clear message |
| Cash series with `exogenous_regressors=["cattle_lc_front"]` but `cattle_lc_front` not in observations.parquet | Forecast.py's `_load_exog_for_target` returns None gracefully (already handled in v0.2b) → falls back to no-exog path |
| User wants the deferred-h Explore view back | Call `sync_futures()` and `append_futures_to_observations()` manually; documented in dashboard README |

## Interactions with v0.2a + v0.2c[3]

- Per-horizon conformal calibration (v0.2c[3]) is unchanged. It now
  has more rows to calibrate against (decades vs. 29 months), so
  scales should be more stable.
- v0.2a's scalar conformal fallback also benefits from the wider
  calibration set.

## Open questions

None. The `gf.c` question resolves at first sync; the catalog rewire
is deterministic in either case.
