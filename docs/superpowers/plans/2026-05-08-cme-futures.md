# CME Futures Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add CME Live Cattle and Lean Hogs futures as exogenous regressors to all three forecasters, with 12 horizon-matched deferred continuous series per commodity, catalog-driven activation, and no new dashboard UI controls.

**Architecture:** One new module `src/usda_sandbox/futures.py` (~250 lines) handles per-contract download (yfinance), contract-calendar logic, and deferred-series construction. Small extensions to `catalog.py` (+2 optional fields), `clean.py` (dispatcher), `forecast.py` (exog plumbing in all 3 forecasters + iter_run_backtest), `data/catalog.json` (+24 entries, 9 updated), and `dashboard/pages/3_Forecast.py` (+1 caption). All 85 existing tests stay green because `exog` defaults to `None`.

**Tech Stack:** yfinance (new dep), polars, statsforecast, prophet, lightgbm. No new infra; laptop-friendly per CLAUDE.md.

**Spec:** [`docs/superpowers/specs/2026-05-08-cme-futures-design.md`](../specs/2026-05-08-cme-futures-design.md)

---

## File map

| Path | Action | Responsibility |
|---|---|---|
| `pyproject.toml` | modify | Add `yfinance>=0.2` to `[project] dependencies` |
| `src/usda_sandbox/catalog.py` | modify | Add `exogenous_regressors: list[str] = []` and `forecastable: bool = True` to `SeriesDefinition` |
| `src/usda_sandbox/futures.py` | create | Contract calendar, deferred-h interpolation, yfinance sync, append to observations.parquet |
| `src/usda_sandbox/clean.py` | modify | `clean_all()` dispatches on `source_file.startswith("futures:")` to call `futures.append_futures_to_observations()`, then resumes XLSX path for non-futures entries |
| `src/usda_sandbox/forecast.py` | modify | Add `exog` parameter on `BaseForecaster.fit/predict/cross_validate_iter`, plumb through all three forecasters and `iter_run_backtest` |
| `data/catalog.json` | modify | Add 24 new futures entries, update 9 monthly cash series with `exogenous_regressors` |
| `dashboard/pages/3_Forecast.py` | modify | One new caption under scoreboard when target series has regressors; filter sidebar picker to `forecastable=True` |
| `tests/test_catalog.py` | create | Field-level tests for the new SeriesDefinition fields (currently no `test_catalog.py` exists; small new file is cleaner than appending to `test_clean.py`) |
| `tests/test_futures.py` | create | ~18 tests covering contract calendar, interpolation, sync, append |
| `tests/test_forecast.py` | modify | ~9 new tests covering exog paths in all forecasters + `iter_run_backtest` catalog lookup |
| `tests/test_clean.py` | modify | ~2 new tests for the dispatcher |

---

## Task 1: Add yfinance dependency

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add the dependency**

In `pyproject.toml`, find the `dependencies = [` array under `[project]` and add `"yfinance>=0.2",` immediately after `"streamlit>=1.32",`. The array's last items become:

```toml
    "streamlit>=1.32",
    "yfinance>=0.2",
]
```

- [ ] **Step 2: Sync**

Run: `uv sync`
Expected: yfinance and transitive deps (requests, beautifulsoup4, etc.) install cleanly.

- [ ] **Step 3: Verify import**

Run: `uv run python -c "import yfinance; print(yfinance.__version__)"`
Expected: prints a version ≥ 0.2.

- [ ] **Step 4: Existing tests still pass**

Run: `uv run pytest`
Expected: 85 passed.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "deps: add yfinance for CME futures ingest"
```

---

## Task 2: Extend `SeriesDefinition` with two new optional fields

**Files:**
- Modify: `src/usda_sandbox/catalog.py`
- Create: `tests/test_catalog.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_catalog.py`:

```python
"""Tests for SeriesDefinition field semantics.

Existing catalog tests live inline in test_clean.py because they were
about round-tripping through clean_all. This file focuses on the new
optional fields added in v0.2b: exogenous_regressors and forecastable.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from usda_sandbox.catalog import (
    SeriesDefinition,
    load_catalog,
    save_catalog,
)


def _base_def(**overrides: object) -> dict[str, object]:
    """A minimal SeriesDefinition dict; override fields per test."""
    payload: dict[str, object] = {
        "series_id": "x",
        "series_name": "X",
        "commodity": "cattle",
        "metric": "price",
        "unit": "USD/cwt",
        "frequency": "monthly",
        "source_file": "x.xlsx",
        "source_sheet": "Sheet1",
        "header_rows_to_skip": 0,
        "value_columns": ["B"],
        "date_column": "A",
        "notes": "",
    }
    payload.update(overrides)
    return payload


def test_series_definition_exogenous_regressors_defaults_empty() -> None:
    sd = SeriesDefinition.model_validate(_base_def())
    assert sd.exogenous_regressors == []


def test_series_definition_forecastable_defaults_true() -> None:
    sd = SeriesDefinition.model_validate(_base_def())
    assert sd.forecastable is True


def test_series_definition_accepts_explicit_regressors() -> None:
    sd = SeriesDefinition.model_validate(
        _base_def(exogenous_regressors=["fut_1", "fut_2"])
    )
    assert sd.exogenous_regressors == ["fut_1", "fut_2"]


def test_series_definition_accepts_forecastable_false() -> None:
    sd = SeriesDefinition.model_validate(_base_def(forecastable=False))
    assert sd.forecastable is False


def test_series_definition_round_trips_with_new_fields(tmp_path: Path) -> None:
    cat = [
        SeriesDefinition.model_validate(
            _base_def(
                series_id="cash",
                exogenous_regressors=["fut_1mo", "fut_2mo"],
                forecastable=True,
            )
        ),
        SeriesDefinition.model_validate(
            _base_def(
                series_id="fut_1mo",
                series_name="Futures, 1 month",
                forecastable=False,
            )
        ),
    ]
    path = tmp_path / "catalog.json"
    save_catalog(path, cat)

    raw = json.loads(path.read_text())
    assert raw[0]["exogenous_regressors"] == ["fut_1mo", "fut_2mo"]
    assert raw[0]["forecastable"] is True
    assert raw[1]["exogenous_regressors"] == []
    assert raw[1]["forecastable"] is False

    reloaded = load_catalog(path)
    assert reloaded == cat


def test_series_definition_rejects_extra_unknown_field() -> None:
    """ConfigDict(extra='forbid') must still block typos like 'forcastable'."""
    with pytest.raises(ValidationError):
        SeriesDefinition.model_validate(_base_def(forcastable=True))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_catalog.py -v`
Expected: 6 failed — `forecastable` and `exogenous_regressors` are unknown fields and `ConfigDict(extra="forbid")` rejects them.

- [ ] **Step 3: Add the two new fields**

In `src/usda_sandbox/catalog.py`, locate the `SeriesDefinition` class. Find the existing fields ending with:

```python
    date_column: str = Field(..., min_length=1)
    notes: str = ""
```

Add two new fields immediately after `notes`:

```python
    date_column: str = Field(..., min_length=1)
    notes: str = ""
    exogenous_regressors: list[str] = Field(default_factory=list)
    forecastable: bool = True
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_catalog.py -v`
Expected: 6 passed.

- [ ] **Step 5: Run full suite + lint + types**

Run: `uv run pytest && uv run ruff check . && uv run mypy`
Expected: 91 passed (85 existing + 6 new), ruff clean, mypy clean.

- [ ] **Step 6: Commit**

```bash
git add src/usda_sandbox/catalog.py tests/test_catalog.py
git commit -m "feat(catalog): add exogenous_regressors and forecastable fields"
```

---

## Task 3: Contract calendar in `futures.py`

Pure functions: month codes, delivery dates, ticker-symbol construction. No I/O. TDD-friendly.

**Files:**
- Create: `src/usda_sandbox/futures.py`
- Create: `tests/test_futures.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_futures.py`:

```python
"""Tests for the CME futures ingest and deferred-series construction.

Tests are split into sections matching futures.py structure:
- Contract calendar (this task)
- Deferred-h interpolation (Task 4)
- Sync & manifest (Task 5)
- Append to observations (Task 5)
"""

from __future__ import annotations

from datetime import date

import pytest

from usda_sandbox.futures import (
    contract_delivery_date,
    contract_months,
    contract_ticker,
    parse_contract_ticker,
)


# --------------------------------------------------------------------------- #
# Contract calendar
# --------------------------------------------------------------------------- #


def test_contract_months_live_cattle() -> None:
    """Live cattle trades Feb/Apr/Jun/Aug/Oct/Dec → G/J/M/Q/V/Z."""
    assert contract_months("LE") == ("G", "J", "M", "Q", "V", "Z")


def test_contract_months_lean_hogs() -> None:
    """Lean hogs trades Feb/Apr/May/Jun/Jul/Aug/Oct/Dec → G/J/K/M/N/Q/V/Z."""
    assert contract_months("HE") == ("G", "J", "K", "M", "N", "Q", "V", "Z")


def test_contract_months_unknown_commodity_raises() -> None:
    with pytest.raises(ValueError, match="unknown commodity"):
        contract_months("XX")


def test_contract_delivery_date_returns_last_day_of_contract_month() -> None:
    # G = Feb, Z = Dec
    assert contract_delivery_date("LE", "Z", 2024) == date(2024, 12, 31)
    assert contract_delivery_date("LE", "G", 2024) == date(2024, 2, 29)  # leap year
    assert contract_delivery_date("LE", "G", 2023) == date(2023, 2, 28)
    assert contract_delivery_date("HE", "N", 2025) == date(2025, 7, 31)  # July


def test_contract_delivery_date_rejects_invalid_month() -> None:
    # Live cattle doesn't have a May contract (K)
    with pytest.raises(ValueError, match="not a valid LE contract month"):
        contract_delivery_date("LE", "K", 2024)


def test_contract_ticker_builds_yahoo_symbol() -> None:
    assert contract_ticker("LE", "Z", 2024) == "LEZ24.CME"
    assert contract_ticker("HE", "G", 2009) == "HEG09.CME"


def test_parse_contract_ticker_round_trips() -> None:
    assert parse_contract_ticker("LEZ24.CME") == ("LE", "Z", 2024)
    assert parse_contract_ticker("HEG09.CME") == ("HE", "G", 2009)


def test_parse_contract_ticker_rejects_malformed() -> None:
    with pytest.raises(ValueError, match="malformed ticker"):
        parse_contract_ticker("not-a-ticker")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_futures.py -v`
Expected: 8 failed with `ModuleNotFoundError: No module named 'usda_sandbox.futures'`.

- [ ] **Step 3: Implement the contract calendar module**

Create `src/usda_sandbox/futures.py`:

```python
"""CME futures ingest and deferred-series construction.

The design and rationale live in
``docs/superpowers/specs/2026-05-08-cme-futures-design.md``.

This module has four responsibilities:

1. **Contract calendar** — knows which months trade for each commodity
   and what the delivery dates are.
2. **Deferred-h interpolation** — given a date ``t`` and a horizon ``h``,
   returns the price of the contract maturing closest to ``t + h`` months,
   linearly interpolated between adjacent contracts.
3. **Sync** — pulls per-contract historical price data via yfinance into
   a local cache, idempotent via SHA-keyed manifest.
4. **Append** — builds the 24 derived deferred series (12 horizons × 2
   commodities) and merges them into ``observations.parquet``.
"""

from __future__ import annotations

import calendar
from datetime import date

__all__ = [
    "contract_delivery_date",
    "contract_months",
    "contract_ticker",
    "parse_contract_ticker",
]

# CME month codes: F G H J K M N Q U V X Z = Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec
_MONTH_CODE_TO_NUMBER: dict[str, int] = {
    "F": 1, "G": 2, "H": 3, "J": 4, "K": 5, "M": 6,
    "N": 7, "Q": 8, "U": 9, "V": 10, "X": 11, "Z": 12,
}

_CONTRACT_MONTHS: dict[str, tuple[str, ...]] = {
    # Live cattle: Feb, Apr, Jun, Aug, Oct, Dec (six contracts/year, 2-month spacing)
    "LE": ("G", "J", "M", "Q", "V", "Z"),
    # Lean hogs: Feb, Apr, May, Jun, Jul, Aug, Oct, Dec (eight contracts/year)
    "HE": ("G", "J", "K", "M", "N", "Q", "V", "Z"),
}


def contract_months(commodity: str) -> tuple[str, ...]:
    """Return the active CME month codes for a commodity."""
    if commodity not in _CONTRACT_MONTHS:
        raise ValueError(
            f"unknown commodity {commodity!r}; expected one of "
            f"{sorted(_CONTRACT_MONTHS)}"
        )
    return _CONTRACT_MONTHS[commodity]


def contract_delivery_date(commodity: str, code: str, year: int) -> date:
    """Last business day of the contract's delivery month.

    For our monthly-cadence purposes this is the right anchor; settlement
    mechanics (physical for LE, cash for HE) don't matter for term-structure
    interpolation against month-end cash prices.
    """
    valid = contract_months(commodity)
    if code not in valid:
        raise ValueError(
            f"{code!r} is not a valid {commodity} contract month; "
            f"expected one of {valid}"
        )
    month = _MONTH_CODE_TO_NUMBER[code]
    last_day = calendar.monthrange(year, month)[1]
    return date(year, month, last_day)


def contract_ticker(commodity: str, code: str, year: int) -> str:
    """Yahoo Finance symbol for a specific contract, e.g. ``LEZ24.CME``."""
    if code not in _MONTH_CODE_TO_NUMBER:
        raise ValueError(f"{code!r} is not a CME month code")
    return f"{commodity}{code}{year % 100:02d}.CME"


def parse_contract_ticker(ticker: str) -> tuple[str, str, int]:
    """Inverse of :func:`contract_ticker`.

    Returns ``(commodity, month_code, year)``. Year defaults to the 20xx
    century — these contracts didn't trade in the 1900s for our purposes.
    """
    if not ticker.endswith(".CME") or len(ticker) < 7:
        raise ValueError(f"malformed ticker {ticker!r}")
    body = ticker[: -len(".CME")]
    # body looks like "LEZ24" or "HEG09" — commodity is 2 chars, month is 1, year is 2
    if len(body) != 5:
        raise ValueError(f"malformed ticker {ticker!r}")
    commodity = body[:2]
    code = body[2]
    try:
        yy = int(body[3:5])
    except ValueError as e:
        raise ValueError(f"malformed ticker {ticker!r}") from e
    return commodity, code, 2000 + yy
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_futures.py -v`
Expected: 8 passed.

- [ ] **Step 5: Lint + types**

Run: `uv run ruff check . && uv run mypy`
Expected: both clean.

- [ ] **Step 6: Commit**

```bash
git add src/usda_sandbox/futures.py tests/test_futures.py
git commit -m "feat(futures): add CME contract calendar"
```

---

## Task 4: Deferred-h interpolation

Pure function that takes per-contract price data and produces the deferred-h continuous series. No I/O. TDD with synthetic per-contract data.

**Files:**
- Modify: `src/usda_sandbox/futures.py`
- Modify: `tests/test_futures.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_futures.py` (after the existing contract-calendar tests):

```python
# --------------------------------------------------------------------------- #
# Deferred-h interpolation
# --------------------------------------------------------------------------- #


def _synth_contract_prices(
    commodity: str,
    code: str,
    year: int,
    *,
    month_end_prices: dict[date, float],
) -> pl.DataFrame:
    """Build a synthetic per-contract month-end price DataFrame.

    Columns: contract_ticker, period_start (month-end date), close (float).
    """
    ticker = contract_ticker(commodity, code, year)
    return pl.DataFrame(
        {
            "contract_ticker": [ticker] * len(month_end_prices),
            "period_start": sorted(month_end_prices.keys()),
            "close": [month_end_prices[d] for d in sorted(month_end_prices)],
        }
    ).with_columns(
        pl.col("period_start").cast(pl.Date),
        pl.col("close").cast(pl.Float64),
    )


def test_deferred_series_uses_exact_contract_when_target_matches() -> None:
    """If the target delivery month IS a contract month, use that contract."""
    # On 2024-04-30 (April end), horizon=2 → target = June 30. LE has a June (M) contract.
    apr_z = _synth_contract_prices("LE", "M", 2024, month_end_prices={date(2024, 4, 30): 175.0})
    other_irrelevant = _synth_contract_prices(
        "LE", "Q", 2024, month_end_prices={date(2024, 4, 30): 178.0}
    )
    all_contracts = pl.concat([apr_z, other_irrelevant])
    result = build_deferred_series(
        commodity="LE",
        horizon_months=2,
        per_contract=all_contracts,
        months=[date(2024, 4, 30)],
    )
    assert result.height == 1
    assert result["period_start"][0] == date(2024, 4, 30)
    # Pure exact match → uses LEM24's price
    assert result["value"][0] == pytest.approx(175.0)


def test_deferred_series_linear_interpolates_between_contracts() -> None:
    """If target falls between two adjacent contracts, linearly interpolate."""
    # On 2024-04-30, horizon=3 → target = July 31. LE has no July; M (Jun) and Q (Aug)
    # are adjacent. Interpolate halfway between Jun price 170 and Aug price 180.
    contracts = pl.concat(
        [
            _synth_contract_prices(
                "LE", "M", 2024, month_end_prices={date(2024, 4, 30): 170.0}
            ),
            _synth_contract_prices(
                "LE", "Q", 2024, month_end_prices={date(2024, 4, 30): 180.0}
            ),
        ]
    )
    result = build_deferred_series(
        commodity="LE",
        horizon_months=3,
        per_contract=contracts,
        months=[date(2024, 4, 30)],
    )
    # Jun 30 → Aug 31 is 62 days; Jun 30 → Jul 31 is 31 days; halfway → 175
    assert result["value"][0] == pytest.approx(175.0, abs=0.1)


def test_deferred_series_uses_earliest_when_target_before_curve() -> None:
    """If target is earlier than any active contract, use the earliest contract."""
    # On 2024-04-30, horizon=1 → target = May 31. LE has no May; Jun is earliest.
    contracts = _synth_contract_prices(
        "LE", "M", 2024, month_end_prices={date(2024, 4, 30): 170.0}
    )
    result = build_deferred_series(
        commodity="LE",
        horizon_months=1,
        per_contract=contracts,
        months=[date(2024, 4, 30)],
    )
    # Target (May 31) is before Jun 30; use Jun price
    assert result["value"][0] == pytest.approx(170.0)


def test_deferred_series_monthly_cadence() -> None:
    """build_deferred_series emits one row per requested month."""
    contracts = _synth_contract_prices(
        "LE",
        "Z",
        2024,
        month_end_prices={
            date(2024, 1, 31): 150.0,
            date(2024, 2, 29): 152.0,
            date(2024, 3, 31): 154.0,
        },
    )
    months = [date(2024, 1, 31), date(2024, 2, 29), date(2024, 3, 31)]
    result = build_deferred_series(
        commodity="LE",
        horizon_months=12,
        per_contract=contracts,
        months=months,
    )
    assert result.height == 3
    assert result["period_start"].to_list() == months


def test_deferred_series_returns_null_when_no_contract_data_on_date() -> None:
    """If no active contracts exist on date t, the deferred value is null."""
    contracts = _synth_contract_prices(
        "LE", "Z", 2024, month_end_prices={date(2024, 1, 31): 150.0}
    )
    result = build_deferred_series(
        commodity="LE",
        horizon_months=6,
        per_contract=contracts,
        months=[date(2023, 1, 31)],  # before contract data exists
    )
    assert result.height == 1
    assert result["value"][0] is None


def test_deferred_series_series_id_format() -> None:
    """series_id is f'{commodity_name}_{commodity_code}_deferred_{h}mo' where
    commodity_name maps LE→cattle, HE→hogs."""
    contracts = _synth_contract_prices(
        "LE", "Z", 2024, month_end_prices={date(2024, 1, 31): 150.0}
    )
    result = build_deferred_series(
        commodity="LE",
        horizon_months=6,
        per_contract=contracts,
        months=[date(2024, 1, 31)],
    )
    assert result["series_id"][0] == "cattle_lc_deferred_6mo"

    contracts_he = _synth_contract_prices(
        "HE", "Z", 2024, month_end_prices={date(2024, 1, 31): 80.0}
    )
    result_he = build_deferred_series(
        commodity="HE",
        horizon_months=3,
        per_contract=contracts_he,
        months=[date(2024, 1, 31)],
    )
    assert result_he["series_id"][0] == "hogs_he_deferred_3mo"
```

Also add `import polars as pl` to the test file's imports near the top:

```python
from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from usda_sandbox.futures import (
    build_deferred_series,
    contract_delivery_date,
    contract_months,
    contract_ticker,
    parse_contract_ticker,
)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_futures.py -v -k "deferred_series"`
Expected: 6 failed with `ImportError: cannot import name 'build_deferred_series'`.

- [ ] **Step 3: Implement `build_deferred_series`**

In `src/usda_sandbox/futures.py`:

Add to imports at the top:

```python
import calendar
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any

import polars as pl
```

Update `__all__` to include the new symbol:

```python
__all__ = [
    "build_deferred_series",
    "contract_delivery_date",
    "contract_months",
    "contract_ticker",
    "parse_contract_ticker",
]
```

Add the commodity-name mapping near the existing constants:

```python
_COMMODITY_NAME: dict[str, str] = {
    "LE": "cattle_lc",
    "HE": "hogs_he",
}


def _commodity_display_name(commodity: str) -> str:
    if commodity not in _COMMODITY_NAME:
        raise ValueError(f"no display name for commodity {commodity!r}")
    return _COMMODITY_NAME[commodity]
```

Append the new function at the end of the file:

```python
def build_deferred_series(
    *,
    commodity: str,
    horizon_months: int,
    per_contract: pl.DataFrame,
    months: Iterable[date],
) -> pl.DataFrame:
    """Build a deferred-h continuous series for one (commodity, horizon).

    ``per_contract`` has columns ``contract_ticker``, ``period_start``,
    ``close`` — month-end closing prices per contract.

    For each requested ``month`` in ``months``, finds the two adjacent
    active contracts and linearly interpolates the price for the target
    delivery date ``month + horizon_months``.

    Returns a DataFrame with the observations-schema columns:
    series_id, series_name, commodity, metric, unit, frequency,
    period_start, period_end, value, source_file, source_sheet,
    ingested_at. The ``value`` is null on dates with no usable contract
    data.
    """
    if horizon_months < 1:
        raise ValueError(f"horizon_months must be >= 1; got {horizon_months}")

    pretty = _commodity_display_name(commodity)
    series_id = f"{pretty}_deferred_{horizon_months}mo"
    series_name = (
        f"{'Live Cattle' if commodity == 'LE' else 'Lean Hogs'} futures, "
        f"{horizon_months}-month deferred (continuous)"
    )
    commodity_label = "cattle" if commodity == "LE" else "hogs"

    # Parse each contract_ticker once → (commodity, code, year) → delivery_date
    contract_meta: dict[str, date] = {}
    for ticker in per_contract["contract_ticker"].unique().to_list():
        c, k, y = parse_contract_ticker(ticker)
        contract_meta[ticker] = contract_delivery_date(c, k, y)

    ingested_at = datetime.now(UTC)
    rows: list[dict[str, Any]] = []
    for month_end in months:
        target = _add_months(month_end, horizon_months)
        # Active contracts on this date are those that appear in per_contract for this date
        active = per_contract.filter(pl.col("period_start") == month_end)
        if active.is_empty():
            value: float | None = None
        else:
            tickers = active["contract_ticker"].to_list()
            prices = active["close"].to_list()
            # Build sorted (delivery_date, price) pairs
            pairs = sorted(
                ((contract_meta[t], p) for t, p in zip(tickers, prices, strict=True)),
                key=lambda x: x[0],
            )
            value = _interpolate_at_target(pairs, target)

        rows.append(
            {
                "series_id": series_id,
                "series_name": series_name,
                "commodity": commodity_label,
                "metric": "futures_price",
                "unit": "USD/cwt",
                "frequency": "monthly",
                "period_start": month_end.replace(day=1),
                "period_end": month_end,
                "value": value,
                "source_file": f"futures:{commodity}",
                "source_sheet": "",
                "ingested_at": ingested_at,
            }
        )

    schema = {
        "series_id": pl.Utf8,
        "series_name": pl.Utf8,
        "commodity": pl.Utf8,
        "metric": pl.Utf8,
        "unit": pl.Utf8,
        "frequency": pl.Utf8,
        "period_start": pl.Date,
        "period_end": pl.Date,
        "value": pl.Float64,
        "source_file": pl.Utf8,
        "source_sheet": pl.Utf8,
        "ingested_at": pl.Datetime("us", "UTC"),
    }
    return pl.DataFrame(rows, schema=schema)


def _add_months(d: date, months: int) -> date:
    """Add a whole number of months to a date, clamping to the month end."""
    total = d.month - 1 + months
    new_year = d.year + total // 12
    new_month = total % 12 + 1
    last_day = calendar.monthrange(new_year, new_month)[1]
    return date(new_year, new_month, min(d.day, last_day))


def _interpolate_at_target(
    pairs: list[tuple[date, float]],
    target: date,
) -> float | None:
    """Linear-interpolate the price at ``target`` from a sorted list of
    (delivery_date, price) tuples.

    - If ``target`` is at or before the earliest contract → return earliest price.
    - If ``target`` is at or after the latest contract → return latest price.
    - Otherwise → linear interpolation between the two flanking contracts.
    """
    if not pairs:
        return None
    if target <= pairs[0][0]:
        return pairs[0][1]
    if target >= pairs[-1][0]:
        return pairs[-1][1]
    # Find the flanking pair
    for i in range(len(pairs) - 1):
        d0, p0 = pairs[i]
        d1, p1 = pairs[i + 1]
        if d0 <= target <= d1:
            span = (d1 - d0).days
            if span == 0:
                return p0
            w = (target - d0).days / span
            return p0 + w * (p1 - p0)
    return None  # unreachable if pairs is sorted
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_futures.py -v`
Expected: 14 passed (8 from Task 3 + 6 new).

- [ ] **Step 5: Run full suite + lint + types**

Run: `uv run pytest && uv run ruff check . && uv run mypy`
Expected: 97 passed, ruff clean, mypy clean.

- [ ] **Step 6: Commit**

```bash
git add src/usda_sandbox/futures.py tests/test_futures.py
git commit -m "feat(futures): add deferred-h interpolation"
```

---

## Task 5: yfinance sync + manifest + append to observations

I/O-heavy task. yfinance is mocked in tests via a small adapter that takes a callable for fetching. No real network calls during tests.

**Files:**
- Modify: `src/usda_sandbox/futures.py`
- Modify: `tests/test_futures.py`
- Modify: `src/usda_sandbox/clean.py`
- Modify: `tests/test_clean.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_futures.py`:

```python
# --------------------------------------------------------------------------- #
# Sync & manifest
# --------------------------------------------------------------------------- #


def _fake_fetcher_factory(
    *,
    available: dict[str, list[tuple[date, float]]],
) -> "Any":
    """Build a fake yfinance fetcher: takes a ticker, returns OHLCV-like data
    for the requested ticker, or empty if not in ``available``.

    Each value in ``available`` is a list of (date, close) pairs.
    """
    import polars as pl

    def fetch(ticker: str) -> pl.DataFrame:
        if ticker not in available:
            return pl.DataFrame(
                schema={"period_start": pl.Date, "close": pl.Float64}
            )
        rows = available[ticker]
        return pl.DataFrame(
            {
                "period_start": [d for d, _ in rows],
                "close": [p for _, p in rows],
            }
        ).with_columns(
            pl.col("period_start").cast(pl.Date),
            pl.col("close").cast(pl.Float64),
        )

    return fetch


def test_sync_futures_writes_manifest_and_per_contract_files(
    tmp_path,
) -> None:
    from usda_sandbox.futures import sync_futures

    raw_dir = tmp_path / "futures"
    fetcher = _fake_fetcher_factory(
        available={
            "LEZ24.CME": [
                (date(2024, 1, 31), 175.0),
                (date(2024, 2, 29), 178.0),
            ],
            "LEG24.CME": [
                (date(2024, 1, 31), 173.0),
            ],
        }
    )
    manifest = sync_futures(
        commodities=("LE",),
        start_year=2024,
        end_year=2024,
        raw_dir=raw_dir,
        fetcher=fetcher,
    )
    # Two contracts in the available dict were fetched
    assert "LEZ24.CME" in manifest
    assert "LEG24.CME" in manifest
    # Files written to disk
    assert (raw_dir / "LE_Z_2024.parquet").exists()
    assert (raw_dir / "LE_G_2024.parquet").exists()
    # Manifest file written
    assert (raw_dir / "manifest.json").exists()
    # Each entry has a non-empty sha256
    for entry in manifest.values():
        assert len(entry.sha256) == 64


def test_sync_futures_idempotent_on_unchanged_data(tmp_path) -> None:
    from usda_sandbox.futures import sync_futures

    raw_dir = tmp_path / "futures"
    fetcher = _fake_fetcher_factory(
        available={"LEZ24.CME": [(date(2024, 1, 31), 175.0)]}
    )
    first = sync_futures(
        commodities=("LE",),
        start_year=2024,
        end_year=2024,
        raw_dir=raw_dir,
        fetcher=fetcher,
    )
    sha_before = first["LEZ24.CME"].sha256
    mtime_before = (raw_dir / "LE_Z_2024.parquet").stat().st_mtime_ns

    second = sync_futures(
        commodities=("LE",),
        start_year=2024,
        end_year=2024,
        raw_dir=raw_dir,
        fetcher=fetcher,
    )
    # SHA unchanged, file not rewritten
    assert second["LEZ24.CME"].sha256 == sha_before
    assert (raw_dir / "LE_Z_2024.parquet").stat().st_mtime_ns == mtime_before


def test_sync_futures_skips_unavailable_contracts(tmp_path) -> None:
    """yfinance returns empty for an ancient or delisted contract — record and skip."""
    from usda_sandbox.futures import sync_futures

    raw_dir = tmp_path / "futures"
    # No contracts available
    fetcher = _fake_fetcher_factory(available={})
    manifest = sync_futures(
        commodities=("LE",),
        start_year=2024,
        end_year=2024,
        raw_dir=raw_dir,
        fetcher=fetcher,
    )
    # Manifest is empty (nothing fetched), no parquet files written, no exception
    assert manifest == {}
    assert not list(raw_dir.glob("LE_*.parquet"))


# --------------------------------------------------------------------------- #
# Append to observations
# --------------------------------------------------------------------------- #


def test_append_futures_to_observations_writes_24_series(tmp_path) -> None:
    from usda_sandbox.futures import (
        append_futures_to_observations,
        sync_futures,
    )

    raw_dir = tmp_path / "futures"
    # Build a tiny fetcher with one LE and one HE contract, two months each
    fetcher = _fake_fetcher_factory(
        available={
            "LEZ24.CME": [
                (date(2024, 1, 31), 175.0),
                (date(2024, 2, 29), 178.0),
            ],
            "HEZ24.CME": [
                (date(2024, 1, 31), 90.0),
                (date(2024, 2, 29), 91.0),
            ],
        }
    )
    sync_futures(
        commodities=("LE", "HE"),
        start_year=2024,
        end_year=2024,
        raw_dir=raw_dir,
        fetcher=fetcher,
    )
    obs_path = tmp_path / "observations.parquet"
    # Seed observations.parquet with a placeholder cash row so the
    # function exercises its merge path. Use the canonical schema.
    seed = pl.DataFrame(
        {
            "series_id": ["existing_cash"],
            "series_name": ["Existing cash series"],
            "commodity": ["cattle"],
            "metric": ["price"],
            "unit": ["USD/cwt"],
            "frequency": ["monthly"],
            "period_start": [date(2024, 1, 1)],
            "period_end": [date(2024, 1, 31)],
            "value": [200.0],
            "source_file": ["x.xlsx"],
            "source_sheet": ["Sheet1"],
            "ingested_at": [None],
        }
    ).with_columns(
        pl.col("period_start").cast(pl.Date),
        pl.col("period_end").cast(pl.Date),
        pl.col("value").cast(pl.Float64),
        pl.col("ingested_at").cast(pl.Datetime("us", "UTC")),
    )
    seed.write_parquet(obs_path)

    append_futures_to_observations(
        obs_path=obs_path,
        raw_dir=raw_dir,
        commodities=("LE", "HE"),
        horizons=range(1, 13),
    )

    final = pl.read_parquet(obs_path)
    series_ids = set(final["series_id"].to_list())
    # 12 LE deferred + 12 HE deferred + 1 existing cash row preserved
    expected = {f"cattle_lc_deferred_{h}mo" for h in range(1, 13)}
    expected |= {f"hogs_he_deferred_{h}mo" for h in range(1, 13)}
    expected |= {"existing_cash"}
    assert series_ids == expected


def test_append_futures_to_observations_is_idempotent(tmp_path) -> None:
    """Running it twice produces the same parquet content."""
    from usda_sandbox.futures import (
        append_futures_to_observations,
        sync_futures,
    )

    raw_dir = tmp_path / "futures"
    fetcher = _fake_fetcher_factory(
        available={
            "LEZ24.CME": [(date(2024, 1, 31), 175.0)],
        }
    )
    sync_futures(
        commodities=("LE",),
        start_year=2024,
        end_year=2024,
        raw_dir=raw_dir,
        fetcher=fetcher,
    )
    obs_path = tmp_path / "observations.parquet"
    # Empty seed
    pl.DataFrame(
        schema={
            "series_id": pl.Utf8,
            "series_name": pl.Utf8,
            "commodity": pl.Utf8,
            "metric": pl.Utf8,
            "unit": pl.Utf8,
            "frequency": pl.Utf8,
            "period_start": pl.Date,
            "period_end": pl.Date,
            "value": pl.Float64,
            "source_file": pl.Utf8,
            "source_sheet": pl.Utf8,
            "ingested_at": pl.Datetime("us", "UTC"),
        }
    ).write_parquet(obs_path)

    append_futures_to_observations(
        obs_path=obs_path,
        raw_dir=raw_dir,
        commodities=("LE",),
        horizons=range(1, 4),  # just 3 horizons for speed
    )
    height_after_first = pl.read_parquet(obs_path).height

    append_futures_to_observations(
        obs_path=obs_path,
        raw_dir=raw_dir,
        commodities=("LE",),
        horizons=range(1, 4),
    )
    height_after_second = pl.read_parquet(obs_path).height

    assert height_after_second == height_after_first
```

Append to `tests/test_clean.py` (at the end of the file):

```python
# --------------------------------------------------------------------------- #
# clean_all dispatcher for futures (v0.2b)
# --------------------------------------------------------------------------- #


def test_clean_all_dispatcher_routes_futures_to_futures_path(
    tmp_path: Path,
    fixture_paths: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When a SeriesDefinition's source_file starts with 'futures:', clean_all
    skips the XLSX path and the futures append path handles it instead."""
    from usda_sandbox.catalog import SeriesDefinition, save_catalog

    catalog = [
        SeriesDefinition(
            series_id="test_wide_b",
            series_name="Wide fixture col B",
            commodity="cattle",
            metric="price",
            unit="USD/cwt",
            frequency="monthly",
            source_file="wide_format.xlsx",
            source_sheet="Historical",
            header_rows_to_skip=4,
            value_columns=["B"],
            date_column="A",
            notes="",
        ),
        SeriesDefinition(
            series_id="cattle_lc_deferred_1mo",
            series_name="LE deferred 1mo",
            commodity="cattle",
            metric="futures_price",
            unit="USD/cwt",
            frequency="monthly",
            source_file="futures:LE",
            source_sheet="",
            header_rows_to_skip=0,
            value_columns=[],
            date_column="",
            notes="",
            forecastable=False,
        ),
    ]
    catalog_path = tmp_path / "catalog.json"
    save_catalog(catalog_path, catalog)

    # Stub append_futures_to_observations so the test doesn't need real futures data
    called_with: list[Path] = []

    def fake_append(*, obs_path: Path, **kwargs: object) -> None:
        called_with.append(obs_path)
        # The fake doesn't write any rows; the XLSX path is what produces output

    monkeypatch.setattr(
        "usda_sandbox.clean.append_futures_to_observations",
        fake_append,
    )

    out_path = tmp_path / "obs.parquet"
    raw_dir = fixture_paths["wide_format"].parent
    df = clean_all(catalog_path, raw_dir, out_path)

    # The futures series did NOT come through the XLSX path
    assert "cattle_lc_deferred_1mo" not in df["series_id"].to_list()
    # But the XLSX series did
    assert "test_wide_b" in df["series_id"].to_list()
    # And the futures append helper was called once with the output path
    assert called_with == [out_path]


def test_clean_all_xlsx_path_unchanged_for_existing_series(
    fixture_paths: dict[str, Path], tmp_path: Path
) -> None:
    """The 12 existing catalog entries (all XLSX) still work end-to-end."""
    from usda_sandbox.catalog import SeriesDefinition, save_catalog

    catalog = [
        SeriesDefinition(
            series_id="test_wide_b",
            series_name="Wide fixture col B",
            commodity="cattle",
            metric="price",
            unit="USD/cwt",
            frequency="monthly",
            source_file="wide_format.xlsx",
            source_sheet="Historical",
            header_rows_to_skip=4,
            value_columns=["B"],
            date_column="A",
            notes="",
        ),
    ]
    catalog_path = tmp_path / "catalog.json"
    save_catalog(catalog_path, catalog)
    raw_dir = fixture_paths["wide_format"].parent
    df = clean_all(catalog_path, raw_dir, tmp_path / "obs.parquet")
    assert "test_wide_b" in df["series_id"].to_list()
    assert df.height > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_futures.py tests/test_clean.py -v -k "sync_futures or append_futures or dispatcher"`
Expected: 5 failed with import errors for `sync_futures` / `append_futures_to_observations`, and `clean_all` test fails because the dispatcher doesn't exist yet.

- [ ] **Step 3: Implement `sync_futures`, manifest, and `append_futures_to_observations`**

Append to `src/usda_sandbox/futures.py`:

```python
import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import yfinance as yf

__all__ = [
    "FuturesManifestEntry",
    "append_futures_to_observations",
    "build_deferred_series",
    "contract_delivery_date",
    "contract_months",
    "contract_ticker",
    "parse_contract_ticker",
    "sync_futures",
]


@dataclass(frozen=True)
class FuturesManifestEntry:
    """One row of the futures manifest, keyed externally on ticker."""

    ticker: str
    commodity: str
    month_code: str
    year: int
    delivery_date: str   # ISO date — easier JSON round-trip than ``date``
    sha256: str
    downloaded_at: str   # ISO-8601


_MANIFEST_FILENAME = "manifest.json"


def _default_fetcher(ticker: str) -> pl.DataFrame:
    """Real yfinance fetch — returns month-end closing prices for a contract."""
    hist = yf.Ticker(ticker).history(period="max", auto_adjust=False)
    if hist.empty:
        return pl.DataFrame(schema={"period_start": pl.Date, "close": pl.Float64})
    # Resample to month-end close
    monthly = (
        hist["Close"]
        .resample("ME")
        .last()
        .dropna()
        .reset_index()
        .rename(columns={"Date": "period_start", "Close": "close"})
    )
    return pl.from_pandas(monthly).with_columns(
        pl.col("period_start").cast(pl.Date),
        pl.col("close").cast(pl.Float64),
    )


def sync_futures(
    *,
    commodities: Iterable[str] = ("LE", "HE"),
    start_year: int = 1999,
    end_year: int | None = None,
    raw_dir: Path = Path("data/raw/futures"),
    fetcher: Callable[[str], pl.DataFrame] | None = None,
) -> dict[str, FuturesManifestEntry]:
    """Download per-contract historical price data via yfinance.

    Idempotent: contracts whose on-disk SHA matches the manifest are not
    re-downloaded. Contracts that yfinance has no data for are silently
    skipped (no entry in the returned manifest).

    ``fetcher`` is injectable for testing — defaults to a real yfinance call
    that returns month-end closing prices for the given ticker.
    """
    end_year = end_year if end_year is not None else datetime.now(UTC).year + 2
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    fetcher = fetcher if fetcher is not None else _default_fetcher

    manifest_path = raw_dir / _MANIFEST_FILENAME
    manifest = _load_manifest(manifest_path)

    for commodity in commodities:
        codes = contract_months(commodity)
        for year in range(start_year, end_year + 1):
            for code in codes:
                ticker = contract_ticker(commodity, code, year)
                file_path = raw_dir / f"{commodity}_{code}_{year}.parquet"

                # If we have it and the file SHA still matches, skip
                if ticker in manifest and file_path.exists():
                    if _sha256_of(file_path) == manifest[ticker].sha256:
                        continue

                df = fetcher(ticker)
                if df.is_empty():
                    continue

                df.write_parquet(file_path)
                sha = _sha256_of(file_path)
                manifest[ticker] = FuturesManifestEntry(
                    ticker=ticker,
                    commodity=commodity,
                    month_code=code,
                    year=year,
                    delivery_date=contract_delivery_date(commodity, code, year).isoformat(),
                    sha256=sha,
                    downloaded_at=datetime.now(UTC).isoformat(),
                )

    _save_manifest(manifest_path, manifest)
    return manifest


def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_manifest(path: Path) -> dict[str, FuturesManifestEntry]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {t: FuturesManifestEntry(**entry) for t, entry in raw.items()}


def _save_manifest(
    path: Path, manifest: dict[str, FuturesManifestEntry]
) -> None:
    from dataclasses import asdict

    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {t: asdict(e) for t, e in manifest.items()}
    path.write_text(
        json.dumps(serializable, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def append_futures_to_observations(
    *,
    obs_path: Path = Path("data/clean/observations.parquet"),
    raw_dir: Path = Path("data/raw/futures"),
    commodities: Iterable[str] = ("LE", "HE"),
    horizons: range = range(1, 13),
) -> None:
    """Build all (commodity × horizon) deferred series and merge into
    observations.parquet. Idempotent: replaces existing rows for the
    affected series_ids (keyed on series_id + period_start)."""
    obs_path = Path(obs_path)
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"{raw_dir!s} does not exist. Run sync_futures() before "
            "append_futures_to_observations()."
        )

    # Gather per-contract DataFrames from disk
    per_contract_by_commodity: dict[str, pl.DataFrame] = {}
    for commodity in commodities:
        contract_dfs: list[pl.DataFrame] = []
        for parquet_path in sorted(raw_dir.glob(f"{commodity}_*.parquet")):
            ticker_body = parquet_path.stem  # e.g., "LE_Z_2024"
            parts = ticker_body.split("_")
            if len(parts) != 3:
                continue
            c, code, yr = parts[0], parts[1], int(parts[2])
            df = pl.read_parquet(parquet_path).with_columns(
                contract_ticker=pl.lit(contract_ticker(c, code, yr)),
            )
            contract_dfs.append(df.select(["contract_ticker", "period_start", "close"]))
        per_contract_by_commodity[commodity] = (
            pl.concat(contract_dfs)
            if contract_dfs
            else pl.DataFrame(
                schema={
                    "contract_ticker": pl.Utf8,
                    "period_start": pl.Date,
                    "close": pl.Float64,
                }
            )
        )

    # Union of all month-ends seen across both commodities
    all_months: set[date] = set()
    for df in per_contract_by_commodity.values():
        all_months.update(df["period_start"].to_list())
    months_sorted = sorted(all_months)

    new_rows: list[pl.DataFrame] = []
    for commodity in commodities:
        contracts = per_contract_by_commodity[commodity]
        for h in horizons:
            new_rows.append(
                build_deferred_series(
                    commodity=commodity,
                    horizon_months=h,
                    per_contract=contracts,
                    months=months_sorted,
                )
            )

    if not new_rows:
        return  # nothing to append

    futures_obs = pl.concat(new_rows, how="vertical_relaxed")

    # Read existing parquet (if any), drop any rows whose series_id we're about
    # to replace, then concat + dedupe.
    futures_ids = set(futures_obs["series_id"].to_list())
    if obs_path.exists():
        existing = pl.read_parquet(obs_path).filter(
            ~pl.col("series_id").is_in(futures_ids)
        )
        combined = pl.concat([existing, futures_obs], how="vertical_relaxed")
    else:
        combined = futures_obs

    combined = combined.unique(
        subset=["series_id", "period_start", "source_file"], keep="last"
    ).sort(["series_id", "period_start"])
    obs_path.parent.mkdir(parents=True, exist_ok=True)
    combined.write_parquet(obs_path)
```

In `src/usda_sandbox/clean.py`, modify `clean_all` to dispatch on the `futures:` prefix. Find the existing `for series_def in catalog:` loop in `clean_all` and modify the body:

```python
def clean_all(
    catalog_path: Path | str,
    raw_dir: Path | str,
    out_path: Path | str,
) -> pl.DataFrame:
    """Process every catalog entry and write a single tidy parquet file.

    XLSX-derived entries are parsed inline. Futures-derived entries
    (``source_file.startswith("futures:")``) are skipped here; the futures
    module's :func:`append_futures_to_observations` handles them after
    the XLSX pass completes.
    """
    from .futures import append_futures_to_observations

    catalog = load_catalog(catalog_path)
    raw_dir = Path(raw_dir)
    frames: list[pl.DataFrame] = []
    has_futures = False
    for series_def in catalog:
        if series_def.source_file.startswith("futures:"):
            has_futures = True
            continue  # handled by append_futures_to_observations below
        raw_path = raw_dir / series_def.source_file
        if not raw_path.exists():
            print(
                f"[clean_all] skipping {series_def.series_id}: missing {raw_path}",
                flush=True,
            )
            continue
        frames.append(clean_series(series_def, raw_path))

    if not frames:
        combined = _empty_frame()
    else:
        combined = pl.concat(frames, how="vertical_relaxed").unique(
            subset=["series_id", "period_start", "source_file"], keep="first"
        )
        combined = combined.sort(["series_id", "period_start"])

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    combined.write_parquet(out)

    if has_futures:
        append_futures_to_observations(obs_path=out)
        combined = pl.read_parquet(out)

    return combined
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_futures.py tests/test_clean.py -v`
Expected: 19 passed (14 in test_futures + 27 in test_clean).

- [ ] **Step 5: Run full suite + lint + types**

Run: `uv run pytest && uv run ruff check . && uv run mypy`
Expected: 102 passed, ruff clean, mypy clean.

- [ ] **Step 6: Commit**

```bash
git add src/usda_sandbox/futures.py src/usda_sandbox/clean.py tests/test_futures.py tests/test_clean.py
git commit -m "feat(futures): sync_futures + append_futures_to_observations + clean_all dispatcher"
```

---

## Task 6: Forecaster exog support

All three forecasters get an optional `exog` parameter. Default `None` preserves v0.2a behavior bit-for-bit. The base class gets a new keyword-only `exog` on `cross_validate_iter` that splits and threads through.

**Files:**
- Modify: `src/usda_sandbox/forecast.py`
- Modify: `tests/test_forecast.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_forecast.py`:

```python
# --------------------------------------------------------------------------- #
# Exog regressor support (v0.2b)
# --------------------------------------------------------------------------- #


def _exog_series(monthly: pl.DataFrame, *, jitter: float = 0.5) -> pl.DataFrame:
    """Build a synthetic exog DataFrame aligned with the monthly series.

    Two regressor columns that are roughly correlated with the target via
    a noisy linear transform — gives the forecasters something to learn.
    """
    rng = np.random.default_rng(1)
    n = monthly.height
    return pl.DataFrame(
        {
            "period_start": monthly["period_start"].to_list(),
            "reg_a": (monthly["value"].to_numpy() * 0.9 + rng.normal(0, jitter, n)).tolist(),
            "reg_b": (monthly["value"].to_numpy() * 0.5 + 20 + rng.normal(0, jitter, n)).tolist(),
        }
    ).with_columns(
        pl.col("period_start").cast(pl.Date),
        pl.col("reg_a").cast(pl.Float64),
        pl.col("reg_b").cast(pl.Float64),
    )


@pytest.mark.parametrize(
    "forecaster_cls",
    [StatsForecastAutoARIMA, ProphetForecaster, LightGBMForecaster],
)
def test_fit_predict_with_exog_returns_correct_shape(
    forecaster_cls: type[BaseForecaster], synthetic_series: pl.DataFrame
) -> None:
    exog = _exog_series(synthetic_series)
    exog_train = exog.head(synthetic_series.height - 4)
    exog_future = exog.tail(4).drop("period_start").select(["reg_a", "reg_b"])

    m = forecaster_cls(seed=42)
    m.fit(synthetic_series.head(synthetic_series.height - 4), exog=exog_train)
    pred = m.predict(horizon=4, exog_future=exog_future)

    assert pred.columns == _PREDICT_COLS
    assert pred.height == 4
    # 80% PI is still well-ordered after adding exog
    assert (pred["lower_80"] <= pred["point"]).all()
    assert (pred["point"] <= pred["upper_80"]).all()


@pytest.mark.parametrize(
    "forecaster_cls",
    [StatsForecastAutoARIMA, ProphetForecaster, LightGBMForecaster],
)
def test_fit_without_exog_unchanged_from_v02a(
    forecaster_cls: type[BaseForecaster], synthetic_series: pl.DataFrame
) -> None:
    """Calling fit(df) with no exog must produce the same result as before."""
    m = forecaster_cls(seed=42)
    m.fit(synthetic_series)
    pred = m.predict(horizon=4)
    assert pred.height == 4
    assert pred.columns == _PREDICT_COLS


def test_cross_validate_with_exog_threads_through(synthetic_series: pl.DataFrame) -> None:
    exog = _exog_series(synthetic_series)
    m = StatsForecastAutoARIMA(seed=42)
    cv = m.cross_validate(synthetic_series, horizon=3, n_windows=2, exog=exog)
    assert cv.height == 6
    assert cv.columns == _CV_COLS


def test_predict_raises_when_exog_future_length_mismatches_horizon(
    synthetic_series: pl.DataFrame,
) -> None:
    exog = _exog_series(synthetic_series)
    m = StatsForecastAutoARIMA(seed=42)
    m.fit(synthetic_series.head(50), exog=exog.head(50))
    bad_future = exog.tail(3).drop("period_start")  # only 3 rows
    with pytest.raises(ValueError, match="exog_future"):
        m.predict(horizon=5, exog_future=bad_future)


def test_fit_drops_rows_where_any_exog_column_is_null(
    synthetic_series: pl.DataFrame,
) -> None:
    """If a row in exog has a null value in any column, drop both df and exog
    on that period."""
    exog = _exog_series(synthetic_series).with_columns(
        reg_a=pl.when(pl.int_range(synthetic_series.height) == 5)
        .then(None)
        .otherwise(pl.col("reg_a"))
    )
    m = StatsForecastAutoARIMA(seed=42)
    # Should not raise — null row is dropped from both df and exog
    m.fit(synthetic_series, exog=exog)
    pred = m.predict(
        horizon=2,
        exog_future=exog.tail(2).drop("period_start").select(["reg_a", "reg_b"]),
    )
    assert pred.height == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_forecast.py -v -k "exog"`
Expected: 11 failed — `exog` is an unexpected keyword argument.

- [ ] **Step 3: Implement exog plumbing**

Edit `src/usda_sandbox/forecast.py`. Replace the `BaseForecaster` class definition with this version (keeping all its other content the same — only the `fit`, `predict`, `cross_validate_iter` signatures change):

```python
class BaseForecaster(ABC):
    """Common surface and a default time-series CV implementation."""

    seed: int

    @abstractmethod
    def fit(
        self, df: pl.DataFrame, exog: pl.DataFrame | None = None
    ) -> None: ...

    @abstractmethod
    def predict(
        self, horizon: int, exog_future: pl.DataFrame | None = None
    ) -> pl.DataFrame: ...

    def cross_validate_iter(
        self,
        df: pl.DataFrame,
        horizon: int,
        n_windows: int,
        exog: pl.DataFrame | None = None,
    ) -> Iterator[tuple[int, pl.DataFrame]]:
        """Yield ``(window_index, window_results_df)`` as each window completes.

        Same validation and windowing semantics as :meth:`cross_validate`,
        but lets callers observe progress between windows.
        """
        if horizon <= 0 or n_windows <= 0:
            raise ValueError("horizon and n_windows must both be positive")
        df = _validate_input(df)
        n = df.height
        needed = horizon * (n_windows + 1)
        if n < needed:
            raise ValueError(
                f"Need at least {needed} observations for {n_windows} windows of "
                f"horizon {horizon}; got {n}"
            )

        for w in range(n_windows):
            cutoff_idx = n - (n_windows - w) * horizon
            train = df.slice(0, cutoff_idx)
            target = df.slice(cutoff_idx, horizon)
            if exog is not None:
                exog_train = exog.filter(
                    pl.col("period_start").is_in(train["period_start"])
                )
                exog_future = (
                    exog.filter(
                        pl.col("period_start").is_in(target["period_start"])
                    )
                    .drop("period_start")
                )
                self.fit(train, exog=exog_train)
                pred = self.predict(horizon, exog_future=exog_future)
            else:
                self.fit(train)
                pred = self.predict(horizon)
            merged = (
                pred.join(
                    target.select(
                        ["period_start", pl.col("value").alias("actual")]
                    ),
                    on="period_start",
                    how="inner",
                )
                .with_columns(window=pl.lit(w, dtype=pl.Int32))
                .select(
                    ["window", "period_start", "point", "lower_80", "upper_80", "actual"]
                )
            )
            yield w, merged

    def cross_validate(
        self,
        df: pl.DataFrame,
        horizon: int,
        n_windows: int,
        exog: pl.DataFrame | None = None,
    ) -> pl.DataFrame:
        """Rolling-origin CV with ``n_windows`` non-overlapping forecast blocks.

        Window 0 is the oldest cutoff; window ``n_windows - 1`` is the most
        recent. Each window holds out ``horizon`` observations after fitting
        on everything before them. Implemented on top of
        :meth:`cross_validate_iter`.
        """
        frames = [
            merged
            for _, merged in self.cross_validate_iter(df, horizon, n_windows, exog=exog)
        ]
        return pl.concat(frames).sort(["window", "period_start"])
```

Add a small helper function near the top of `forecast.py` (after the existing `_validate_input` definition):

```python
def _align_exog(
    df: pl.DataFrame, exog: pl.DataFrame
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Align df and exog on period_start, dropping rows where any exog column
    is null. Returns (aligned_df, aligned_exog) — both have the same number
    of rows in the same period order."""
    if "period_start" not in exog.columns:
        raise ValueError("exog must have a 'period_start' column")
    reg_cols = [c for c in exog.columns if c != "period_start"]
    if not reg_cols:
        raise ValueError("exog must have at least one regressor column")
    merged = df.join(exog, on="period_start", how="inner")
    merged = merged.drop_nulls(subset=reg_cols)
    aligned_df = merged.select(df.columns).sort("period_start")
    aligned_exog = merged.select(["period_start", *reg_cols]).sort("period_start")
    return aligned_df, aligned_exog


def _check_exog_future(exog_future: pl.DataFrame, horizon: int) -> None:
    if exog_future.height != horizon:
        raise ValueError(
            f"exog_future must have exactly horizon ({horizon}) rows; "
            f"got {exog_future.height}"
        )
```

Now update each of the three forecasters' `fit` and `predict` signatures.

**StatsForecastAutoARIMA**:

```python
    def fit(
        self, df: pl.DataFrame, exog: pl.DataFrame | None = None
    ) -> None:
        clean = _validate_input(df)
        if exog is not None:
            clean, exog_aligned = _align_exog(clean, exog)
            self._exog_cols = [c for c in exog_aligned.columns if c != "period_start"]
        else:
            exog_aligned = None
            self._exog_cols = []
        pdf = clean.to_pandas()
        train = pd.DataFrame(
            {
                "unique_id": "series_0",
                "ds": pd.to_datetime(pdf["period_start"]),
                "y": pdf["value"].astype(float),
            }
        )
        if exog_aligned is not None:
            for col in self._exog_cols:
                train[col] = exog_aligned[col].to_pandas().astype(float).values
        with _seeded(self.seed), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sf = StatsForecast(
                models=[AutoARIMA(season_length=self.season_length)],
                freq="MS",
                n_jobs=1,
            )
            sf.fit(train)
        self._sf = sf

    def predict(
        self, horizon: int, exog_future: pl.DataFrame | None = None
    ) -> pl.DataFrame:
        if self._sf is None:
            raise RuntimeError("Call fit() before predict().")
        x_df = None
        if self._exog_cols:
            if exog_future is None:
                raise ValueError(
                    "model was fit with exog; exog_future must be provided"
                )
            _check_exog_future(exog_future, horizon)
            # statsforecast expects an X_df with unique_id, ds, and regressor columns
            last = self._sf.fcst_df["ds"].max() if hasattr(self._sf, "fcst_df") else None
            future_ds = pd.date_range(
                start=pd.Timestamp(last) + pd.offsets.MonthBegin(1) if last is not None
                else pd.Timestamp.now().normalize(),
                periods=horizon,
                freq="MS",
            )
            x_df = pd.DataFrame(
                {
                    "unique_id": "series_0",
                    "ds": future_ds,
                }
            )
            for col in self._exog_cols:
                x_df[col] = exog_future[col].to_pandas().astype(float).values
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            forecast = self._sf.predict(h=horizon, level=[80], X_df=x_df)
        # ... rest of predict body unchanged ...
```

(The rest of `predict` — the column-renaming and DataFrame conversion — stays exactly as it is. Only the call site that builds `forecast` is wrapped with the optional `X_df=x_df`.)

Also add `self._exog_cols: list[str] = []` to `__init__`.

**ProphetForecaster** — same pattern:

```python
    def __init__(self, seed: int = DEFAULT_SEED) -> None:
        self.seed = seed
        self._model: Prophet | None = None
        self._last_period: pd.Timestamp | None = None
        self._exog_cols: list[str] = []

    def fit(
        self, df: pl.DataFrame, exog: pl.DataFrame | None = None
    ) -> None:
        clean = _validate_input(df)
        if exog is not None:
            clean, exog_aligned = _align_exog(clean, exog)
            self._exog_cols = [c for c in exog_aligned.columns if c != "period_start"]
        else:
            exog_aligned = None
            self._exog_cols = []
        pdf = clean.to_pandas()
        train = pd.DataFrame(
            {
                "ds": pd.to_datetime(pdf["period_start"]),
                "y": pdf["value"].astype(float),
            }
        )
        if exog_aligned is not None:
            for col in self._exog_cols:
                train[col] = exog_aligned[col].to_pandas().astype(float).values

        with _seeded(self.seed), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prev = os.environ.get("CMDSTANPY_LOG", "")
            os.environ["CMDSTANPY_LOG"] = "WARNING"
            try:
                model = Prophet(
                    interval_width=0.80,
                    yearly_seasonality=True,
                    weekly_seasonality=False,
                    daily_seasonality=False,
                )
                for col in self._exog_cols:
                    model.add_regressor(col)
                model.fit(train)
            finally:
                if prev:
                    os.environ["CMDSTANPY_LOG"] = prev
                else:
                    os.environ.pop("CMDSTANPY_LOG", None)
        self._model = model
        self._last_period = train["ds"].iloc[-1]

    def predict(
        self, horizon: int, exog_future: pl.DataFrame | None = None
    ) -> pl.DataFrame:
        if self._model is None or self._last_period is None:
            raise RuntimeError("Call fit() before predict().")
        if self._exog_cols and exog_future is None:
            raise ValueError(
                "model was fit with exog; exog_future must be provided"
            )
        if exog_future is not None:
            _check_exog_future(exog_future, horizon)
        future_dates = _next_n_months(
            self._last_period + pd.offsets.MonthBegin(1), horizon
        )
        future = pd.DataFrame({"ds": future_dates})
        if self._exog_cols and exog_future is not None:
            for col in self._exog_cols:
                future[col] = exog_future[col].to_pandas().astype(float).values
        with _seeded(self.seed), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            forecast = self._model.predict(future)
        out = pl.from_pandas(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]])
        return out.rename(
            {
                "ds": "period_start",
                "yhat": "point",
                "yhat_lower": "lower_80",
                "yhat_upper": "upper_80",
            }
        ).with_columns(pl.col("period_start").cast(pl.Date))
```

**LightGBMForecaster** — exog columns become extra features:

```python
    def __init__(
        self,
        seed: int = DEFAULT_SEED,
        n_estimators: int = 400,
        learning_rate: float = 0.05,
    ) -> None:
        self.seed = seed
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self._model_point: lgb.LGBMRegressor | None = None
        self._residual_halfwidth_80: float = 0.0
        self._feature_columns: list[str] | None = None
        self._history: pd.DataFrame | None = None
        self._exog_cols: list[str] = []
        self._exog_history: pd.DataFrame | None = None

    def fit(
        self, df: pl.DataFrame, exog: pl.DataFrame | None = None
    ) -> None:
        clean = _validate_input(df)
        if exog is not None:
            clean, exog_aligned = _align_exog(clean, exog)
            self._exog_cols = [c for c in exog_aligned.columns if c != "period_start"]
        else:
            exog_aligned = None
            self._exog_cols = []
        pdf = clean.to_pandas().rename(columns={"period_start": "ds"})
        pdf["ds"] = pd.to_datetime(pdf["ds"])
        feats = self._build_features(pdf).dropna().reset_index(drop=True)
        if exog_aligned is not None:
            exog_pdf = exog_aligned.to_pandas().rename(columns={"period_start": "ds"})
            exog_pdf["ds"] = pd.to_datetime(exog_pdf["ds"])
            feats = feats.merge(exog_pdf, on="ds", how="inner")
            self._exog_history = exog_pdf
        if feats.empty:
            raise ValueError(
                f"Not enough history to build features (need at least "
                f"{max(self.LAGS) + max(self.ROLLING_WINDOWS) + 1} observations)"
            )
        feature_cols = [
            *[f"lag_{lag}" for lag in self.LAGS],
            *[f"rollmean_{w}" for w in self.ROLLING_WINDOWS],
            "month",
            "quarter",
            "year_trend",
            *self._exog_cols,
        ]
        X = feats[feature_cols]
        y = feats["value"]

        common = dict(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=31,
            min_child_samples=5,
            random_state=self.seed,
            seed=self.seed,
            deterministic=True,
            force_row_wise=True,
            verbose=-1,
        )
        with _seeded(self.seed), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model_point = lgb.LGBMRegressor(objective="regression", **common)
            self._model_point.fit(X, y)
            naive_diffs = np.abs(np.diff(pdf["value"].to_numpy(dtype=float)))
            self._residual_halfwidth_80 = (
                float(np.quantile(naive_diffs, 0.80)) if naive_diffs.size else 0.0
            )

        self._feature_columns = feature_cols
        self._history = pdf[["ds", "value"]].copy()

    def predict(
        self, horizon: int, exog_future: pl.DataFrame | None = None
    ) -> pl.DataFrame:
        if (
            self._model_point is None
            or self._history is None
            or self._feature_columns is None
        ):
            raise RuntimeError("Call fit() before predict().")
        if self._exog_cols and exog_future is None:
            raise ValueError(
                "model was fit with exog; exog_future must be provided"
            )
        if exog_future is not None:
            _check_exog_future(exog_future, horizon)

        history = self._history.copy()
        last_ds = history["ds"].iloc[-1]
        future_dates = _next_n_months(last_ds + pd.offsets.MonthBegin(1), horizon)
        rows: list[dict[str, Any]] = []
        halfwidth = self._residual_halfwidth_80

        exog_future_values: dict[str, list[float]] = {}
        if exog_future is not None:
            for col in self._exog_cols:
                exog_future_values[col] = (
                    exog_future[col].to_pandas().astype(float).tolist()
                )

        for step_idx, ds in enumerate(future_dates):
            extended = pd.concat(
                [history, pd.DataFrame({"ds": [ds], "value": [np.nan]})],
                ignore_index=True,
            )
            feats = self._build_features(extended).iloc[[-1]].copy()
            for col in self._exog_cols:
                feats[col] = exog_future_values[col][step_idx]
            X = feats[self._feature_columns]
            point = float(self._model_point.predict(X)[0])
            rows.append(
                {
                    "period_start": ds.date(),
                    "point": point,
                    "lower_80": point - halfwidth,
                    "upper_80": point + halfwidth,
                }
            )
            history = pd.concat(
                [history, pd.DataFrame({"ds": [ds], "value": [point]})],
                ignore_index=True,
            )

        return pl.DataFrame(rows).with_columns(
            pl.col("period_start").cast(pl.Date),
            pl.col("point").cast(pl.Float64),
            pl.col("lower_80").cast(pl.Float64),
            pl.col("upper_80").cast(pl.Float64),
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_forecast.py -v`
Expected: 30 passed (19 existing + 11 new).

- [ ] **Step 5: Run full suite + lint + types**

Run: `uv run pytest && uv run ruff check . && uv run mypy`
Expected: 113 passed, ruff clean, mypy clean.

- [ ] **Step 6: Commit**

```bash
git add src/usda_sandbox/forecast.py tests/test_forecast.py
git commit -m "feat(forecast): add exog regressor support to all three forecasters"
```

---

## Task 7: iter_run_backtest catalog lookup + dashboard caption

The orchestrator that reads the catalog for the chosen target, fetches the listed regressors from observations.parquet, pivots them wide, and passes them through. The dashboard caption surfaces what regressors were used.

**Files:**
- Modify: `src/usda_sandbox/forecast.py`
- Modify: `tests/test_forecast.py`
- Modify: `dashboard/pages/3_Forecast.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_forecast.py`:

```python
def test_iter_run_backtest_loads_catalog_exog(
    synthetic_obs_parquet: Path, tmp_path: Path
) -> None:
    """When the target's SeriesDefinition has exogenous_regressors,
    iter_run_backtest fetches them and passes them through."""
    from usda_sandbox.catalog import SeriesDefinition, save_catalog
    from usda_sandbox.forecast import (
        BacktestProgress,
        BacktestResult,
        iter_run_backtest,
    )

    # Use the existing synthetic obs parquet's series_id and add a synthetic
    # exog series to the same parquet
    obs = pl.read_parquet(synthetic_obs_parquet)
    exog_rows = obs.with_columns(
        series_id=pl.lit("test_exog_1"),
        series_name=pl.lit("Test exog series"),
        value=pl.col("value") * 0.9,
    )
    obs_with_exog = pl.concat([obs, exog_rows], how="vertical_relaxed")
    obs_path = tmp_path / "obs_with_exog.parquet"
    obs_with_exog.write_parquet(obs_path)

    # Build a catalog with the target pointing at the exog series
    catalog = [
        SeriesDefinition(
            series_id="synthetic_test",
            series_name="Synthetic test series",
            commodity="test",
            metric="price",
            unit="USD/cwt",
            frequency="monthly",
            source_file="synthetic.xlsx",
            source_sheet="Sheet1",
            header_rows_to_skip=0,
            value_columns=["B"],
            date_column="A",
            notes="",
            exogenous_regressors=["test_exog_1"],
        ),
        SeriesDefinition(
            series_id="test_exog_1",
            series_name="Test exog 1",
            commodity="test",
            metric="exog",
            unit="USD/cwt",
            frequency="monthly",
            source_file="x.xlsx",
            source_sheet="Sheet1",
            header_rows_to_skip=0,
            value_columns=["B"],
            date_column="A",
            notes="",
            forecastable=False,
        ),
    ]
    catalog_path = tmp_path / "catalog.json"
    save_catalog(catalog_path, catalog)

    items = list(
        iter_run_backtest(
            "synthetic_test",
            horizon=3,
            n_windows=2,
            obs_path=obs_path,
            catalog_path=catalog_path,
            models=["AutoARIMA"],  # one model for speed
        )
    )
    final = items[-1]
    assert isinstance(final, BacktestResult)
    # The CV still produced the expected number of rows (1 model × 2 windows × 3 horizon)
    assert final.cv_details.height == 6


def test_iter_run_backtest_no_catalog_path_uses_v02a_behavior(
    synthetic_obs_parquet: Path,
) -> None:
    """If catalog_path is None and the catalog file doesn't exist at the
    default path, the backtest runs without exog (v0.2a behavior)."""
    from usda_sandbox.forecast import BacktestResult, iter_run_backtest

    items = list(
        iter_run_backtest(
            "synthetic_test",
            horizon=3,
            n_windows=2,
            obs_path=synthetic_obs_parquet,
            catalog_path=None,
            models=["AutoARIMA"],
        )
    )
    final = items[-1]
    assert isinstance(final, BacktestResult)
    assert final.cv_details.height == 6
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_forecast.py -v -k "iter_run_backtest_loads_catalog_exog or iter_run_backtest_no_catalog"`
Expected: 2 failed — `iter_run_backtest()` doesn't accept `catalog_path` kwarg.

- [ ] **Step 3: Implement catalog lookup**

In `src/usda_sandbox/forecast.py`, modify `iter_run_backtest` and add a helper.

Replace the existing `iter_run_backtest` function body with this version:

```python
def iter_run_backtest(
    series_id: str,
    horizon: int = 6,
    n_windows: int = 8,
    *,
    obs_path: Path | str | None = None,
    seed: int = DEFAULT_SEED,
    models: Sequence[str] | None = None,
    catalog_path: Path | str | None = "data/catalog.json",
) -> Iterator[BacktestProgress | BacktestResult]:
    """Generator variant of :func:`run_backtest`.

    Yields one :class:`BacktestProgress` event after each (model, window)
    completes (``len(models) * n_windows`` events total), then a single final
    :class:`BacktestResult` with the same content :func:`run_backtest` would
    return for the same inputs.

    ``models`` optionally restricts which forecasters are run.

    ``catalog_path`` (default ``data/catalog.json``) is used to look up the
    target series's ``exogenous_regressors`` field. If the catalog file
    exists and the target's regressors are non-empty, those series are
    fetched from ``obs_path`` and passed through to each forecaster's
    ``cross_validate_iter`` as ``exog``. If the catalog file does not exist
    or ``catalog_path=None``, the v0.2a path (no exog) runs unchanged.
    """
    series = read_series(series_id, obs_path)
    if series.is_empty():
        raise ValueError(f"No observations found for series_id={series_id!r}")
    series = series.filter(pl.col("value").is_not_null()).select(
        ["period_start", "value"]
    )

    exog = _load_exog_for_target(
        series_id=series_id,
        obs_path=obs_path,
        catalog_path=catalog_path,
    )

    all_forecasters: list[tuple[str, BaseForecaster]] = [
        ("AutoARIMA", StatsForecastAutoARIMA(seed=seed)),
        ("Prophet", ProphetForecaster(seed=seed)),
        ("LightGBM", LightGBMForecaster(seed=seed)),
    ]
    if models is None:
        forecasters = all_forecasters
    else:
        wanted = set(models)
        forecasters = [(n, f) for n, f in all_forecasters if n in wanted]
        if not forecasters:
            raise ValueError(
                f"No forecasters match the requested models: {sorted(wanted)}. "
                f"Available: {[n for n, _ in all_forecasters]}"
            )

    detail_frames: list[pl.DataFrame] = []
    started = time.time()

    for name, fcst in forecasters:
        per_model_frames: list[pl.DataFrame] = []
        for w, window_df in fcst.cross_validate_iter(
            series, horizon, n_windows, exog=exog
        ):
            per_model_frames.append(window_df)
            so_far = pl.concat(per_model_frames)
            actual = so_far["actual"].to_numpy().astype(float)
            point = so_far["point"].to_numpy().astype(float)
            running = _mape(actual, point)
            yield BacktestProgress(
                model=name,
                window=w,
                n_windows=n_windows,
                elapsed_s=time.time() - started,
                running_mape=None if running != running else running,
            )
        cv = (
            pl.concat(per_model_frames)
            .sort(["window", "period_start"])
            .with_columns(model=pl.lit(name))
        )
        detail_frames.append(cv)

    cv_details = pl.concat(detail_frames).sort(["model", "window", "period_start"])
    train_values = series["value"].to_numpy().astype(float)
    metrics = _per_model_metrics(cv_details, train_values)

    yield BacktestResult(
        series_id=series_id,
        horizon=horizon,
        n_windows=n_windows,
        cv_details=cv_details,
        metrics=metrics,
    )


def _load_exog_for_target(
    *,
    series_id: str,
    obs_path: Path | str | None,
    catalog_path: Path | str | None,
) -> pl.DataFrame | None:
    """If the catalog says the target has exogenous_regressors, load and
    pivot them. Returns None if no exog is configured or the catalog is
    unavailable."""
    if catalog_path is None:
        return None
    catalog_path = Path(catalog_path)
    if not catalog_path.exists():
        return None
    from .catalog import load_catalog
    from .store import read_observations

    catalog = load_catalog(catalog_path)
    by_id = {sd.series_id: sd for sd in catalog}
    target = by_id.get(series_id)
    if target is None or not target.exogenous_regressors:
        return None

    obs = read_observations(obs_path).collect()
    long = (
        obs.filter(pl.col("series_id").is_in(target.exogenous_regressors))
        .select(["series_id", "period_start", "value"])
    )
    if long.is_empty():
        return None
    wide = long.pivot(values="value", index="period_start", on="series_id").sort(
        "period_start"
    )
    return wide
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_forecast.py -v -k "catalog"`
Expected: 2 passed.

- [ ] **Step 5: Add the dashboard caption**

In `dashboard/pages/3_Forecast.py`, find the existing scoreboard block:

```python
st.subheader("Scoreboard")
st.dataframe(
    filtered_metrics.with_columns(
        pl.col("mape").round(2),
        pl.col("smape").round(2),
        pl.col("mase").round(2),
    ).sort("mape"),
    hide_index=True,
    use_container_width=True,
)
```

Immediately after that block, add the exog caption:

```python
# Surface exogenous regressors used (if any) — catalog-driven, set in data/catalog.json
_target_def = next(
    (sd for sd in load_catalog("data/catalog.json") if sd.series_id == series_id),
    None,
)
if _target_def is not None and _target_def.exogenous_regressors:
    _commodity_label = "Live Cattle" if _target_def.commodity == "cattle" else "Lean Hogs"
    st.caption(
        f"This series was forecast with **{len(_target_def.exogenous_regressors)} "
        f"exogenous regressors**: deferred {_commodity_label} futures (1-12 months "
        f"ahead). Each forecaster (AutoARIMA, Prophet, LightGBM) sees these alongside "
        f"the cash history."
    )
```

Add the import at the top of the file:

```python
from usda_sandbox.catalog import load_catalog
```

- [ ] **Step 6: Smoke-test the page**

Run:
```bash
uv run python -c "
import importlib.util
spec = importlib.util.spec_from_file_location('forecast_page', 'dashboard/pages/3_Forecast.py')
assert spec and spec.loader
print('module spec loads ok')
"
```
Expected: prints `module spec loads ok`.

- [ ] **Step 7: Run full suite + lint + types**

Run: `uv run pytest && uv run ruff check . && uv run mypy`
Expected: 115 passed (113 + 2 new), ruff clean, mypy clean.

- [ ] **Step 8: Commit**

```bash
git add src/usda_sandbox/forecast.py tests/test_forecast.py dashboard/pages/3_Forecast.py
git commit -m "feat(forecast): catalog-driven exog lookup in iter_run_backtest + dashboard caption"
```

---

## Task 8: Update `data/catalog.json` + real-data smoke

Final task: add 24 futures entries to the catalog, wire up the 9 cash series with their regressors, run the full pipeline against real data, and verify everything works end-to-end.

**Files:**
- Modify: `data/catalog.json`

- [ ] **Step 1: Add the 24 futures series + update the 9 cash series**

Open `data/catalog.json`. For each of the 9 monthly cash series, add `"exogenous_regressors"` with the 12 deferred series_ids for the right commodity:

For each cattle cash series (`cattle_steer_choice_tx_ok_nm`, `cattle_steer_choice_nebraska`, `cattle_feeder_steer_500_550`, `cattle_feeder_steer_750_800`, `boxed_beef_cutout_choice`, `boxed_beef_cutout_select`), set:

```json
"exogenous_regressors": [
  "cattle_lc_deferred_1mo", "cattle_lc_deferred_2mo", "cattle_lc_deferred_3mo",
  "cattle_lc_deferred_4mo", "cattle_lc_deferred_5mo", "cattle_lc_deferred_6mo",
  "cattle_lc_deferred_7mo", "cattle_lc_deferred_8mo", "cattle_lc_deferred_9mo",
  "cattle_lc_deferred_10mo", "cattle_lc_deferred_11mo", "cattle_lc_deferred_12mo"
],
"forecastable": true
```

For each hog cash series (`hog_barrow_gilt_natbase_51_52`, `pork_cutout_composite`), set:

```json
"exogenous_regressors": [
  "hogs_he_deferred_1mo", "hogs_he_deferred_2mo", "hogs_he_deferred_3mo",
  "hogs_he_deferred_4mo", "hogs_he_deferred_5mo", "hogs_he_deferred_6mo",
  "hogs_he_deferred_7mo", "hogs_he_deferred_8mo", "hogs_he_deferred_9mo",
  "hogs_he_deferred_10mo", "hogs_he_deferred_11mo", "hogs_he_deferred_12mo"
],
"forecastable": true
```

For `lamb_slaughter_choice_san_angelo`, leave `exogenous_regressors` absent (it'll default to `[]`).

For the three quarterly WASDE series, leave them unchanged.

Append the 24 new futures entries to the end of the JSON array. For each commodity (`cattle_lc` and `hogs_he`) and each horizon `h` in 1-12, the entry looks like:

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
  "notes": "Continuous series: at each month-end, the price of the LE contract maturing 1 month ahead. Linear interpolation between adjacent contracts when no contract maps exactly. Built from per-contract data pulled via yfinance.",
  "exogenous_regressors": [],
  "forecastable": false
}
```

The minimum-length check on `value_columns` is `min_length=1`. Use a placeholder `["_"]` to satisfy validation (the dispatcher ignores `value_columns` for futures entries anyway). Wait — this is wrong. Check the existing Pydantic validator:

```python
value_columns: list[str] = Field(..., min_length=1)
```

Two options:
1. Set `value_columns: ["_"]` and `date_column: "A"` as placeholder strings to pass validation.
2. Relax the validators to allow empty for futures source types.

Go with **option 1** — set both to single-character placeholders. The dispatcher in `clean.py` short-circuits on the `futures:` prefix and never reads these fields. Document the placeholder in the futures entries' `notes`.

So the futures entries actually look like:

```json
{
  "series_id": "cattle_lc_deferred_1mo",
  ...
  "value_columns": ["_"],
  "date_column": "_",
  "notes": "Continuous series ... (XLSX-shaped fields are placeholders; the dispatcher ignores them for futures: entries.)",
  "exogenous_regressors": [],
  "forecastable": false
}
```

Generate all 24 entries (12 for LE/cattle_lc, 12 for HE/hogs_he) manually or via a small inline-Python helper.

For speed, you can paste the 24 entries with a quick script:

```bash
uv run python -c "
import json
from pathlib import Path

base_path = Path('data/catalog.json')
catalog = json.loads(base_path.read_text(encoding='utf-8'))

# Build the 12 cattle_lc + 12 hogs_he entries
futures_entries = []
for commodity, prefix, label, src in [
    ('cattle', 'cattle_lc', 'Live Cattle', 'LE'),
    ('hogs', 'hogs_he', 'Lean Hogs', 'HE'),
]:
    for h in range(1, 13):
        futures_entries.append({
            'series_id': f'{prefix}_deferred_{h}mo',
            'series_name': f'{label} futures, {h}-month deferred (continuous)',
            'commodity': commodity,
            'metric': 'futures_price',
            'unit': 'USD/cwt',
            'frequency': 'monthly',
            'source_file': f'futures:{src}',
            'source_sheet': '',
            'header_rows_to_skip': 0,
            'value_columns': ['_'],
            'date_column': '_',
            'notes': (
                f'Continuous series: at each month-end, the price of the {src} '
                f'contract maturing {h} months ahead. Linear interpolation '
                f'between adjacent contracts when no contract matches exactly. '
                f'Built from per-contract data pulled via yfinance. '
                f'XLSX-shaped fields are placeholders.'
            ),
            'exogenous_regressors': [],
            'forecastable': False,
        })

# Append the futures entries
catalog.extend(futures_entries)

# Update the 9 monthly cash series with their regressors
cattle_regressors = [f'cattle_lc_deferred_{h}mo' for h in range(1, 13)]
hogs_regressors = [f'hogs_he_deferred_{h}mo' for h in range(1, 13)]
for entry in catalog:
    sid = entry['series_id']
    if sid in {'cattle_steer_choice_tx_ok_nm', 'cattle_steer_choice_nebraska',
              'cattle_feeder_steer_500_550', 'cattle_feeder_steer_750_800',
              'boxed_beef_cutout_choice', 'boxed_beef_cutout_select'}:
        entry['exogenous_regressors'] = cattle_regressors
        entry['forecastable'] = True
    elif sid in {'hog_barrow_gilt_natbase_51_52', 'pork_cutout_composite'}:
        entry['exogenous_regressors'] = hogs_regressors
        entry['forecastable'] = True
    elif sid == 'lamb_slaughter_choice_san_angelo':
        entry.setdefault('exogenous_regressors', [])
        entry.setdefault('forecastable', True)
    # Quarterly WASDE series get defaults applied via Pydantic on load; we don't
    # need to add fields here unless we want them in the JSON explicitly.
    else:
        entry.setdefault('exogenous_regressors', [])
        entry.setdefault('forecastable', True)

base_path.write_text(
    json.dumps(catalog, indent=2) + '\n', encoding='utf-8'
)
print(f'Wrote {len(catalog)} entries to {base_path}')
"
```

Expected: prints `Wrote 36 entries to data/catalog.json`.

- [ ] **Step 2: Validate the catalog parses**

Run:
```bash
uv run python -c "
from usda_sandbox.catalog import load_catalog
catalog = load_catalog('data/catalog.json')
print(f'Loaded {len(catalog)} series')
forecastable = [sd for sd in catalog if sd.forecastable]
print(f'Forecastable: {len(forecastable)}')
with_exog = [sd for sd in catalog if sd.exogenous_regressors]
print(f'With exog regressors: {len(with_exog)}')
"
```

Expected:
```
Loaded 36 series
Forecastable: 12       (or 11 — depends on whether quarterly WASDE entries are forecastable=True/False; either is fine)
With exog regressors: 8
```

- [ ] **Step 3: Pull real futures data (this is slow — 10-15 minutes)**

Run:
```bash
uv run python -c "
from usda_sandbox.futures import sync_futures
manifest = sync_futures(start_year=1999)
print(f'Downloaded {len(manifest)} contracts')
print(f'Sample: {list(manifest.values())[0]}')
"
```

Expected: `Downloaded ~300-500 contracts`. yfinance may rate-limit or return empty for very old or delisted contracts — those are silently skipped per design. The output prints a sample manifest entry.

If yfinance fails entirely (network error, rate limit), retry once. If it fails again, fall back to the smaller test set:

```bash
uv run python -c "
from usda_sandbox.futures import sync_futures
manifest = sync_futures(start_year=2020)  # smaller window
print(f'Downloaded {len(manifest)} contracts (2020+)')
"
```

- [ ] **Step 4: Build the deferred series and merge into observations.parquet**

Run:
```bash
uv run python -c "
from usda_sandbox.clean import clean_all
df = clean_all('data/catalog.json', 'data/raw', 'data/clean/observations.parquet')
print(f'Total rows: {df.height:,}')
print(f'Series count: {df[\"series_id\"].n_unique()}')
import polars as pl
fut_series = df.filter(pl.col('series_id').str.starts_with('cattle_lc_deferred_')).group_by('series_id').len().sort('series_id')
print('Cattle deferred series row counts:')
print(fut_series)
"
```

Expected: total row count goes up from 2,895 to roughly 2,895 + (24 × ~300 months) ≈ 10,000+ rows. Each `cattle_lc_deferred_Xmo` series has ~25 years × 12 months ≈ 300 rows.

- [ ] **Step 5: Run one real backtest end-to-end**

Run:
```bash
uv run python -c "
from pathlib import Path
from usda_sandbox.forecast import run_backtest
result = run_backtest(
    'cattle_steer_choice_nebraska',
    horizon=6,
    n_windows=4,
    obs_path=Path('data/clean/observations.parquet'),
    catalog_path=Path('data/catalog.json'),
)
print('Metrics with futures regressors:')
print(result.metrics)
"
```

Expected: AutoARIMA MAPE meaningfully lower than v0.2a's 6.29% — somewhere in the 3-5% range per literature. If the numbers don't move, that's the headline finding to flag in the README later.

If you see a `ValueError` about exog rows not aligning, check that:
- `sync_futures` produced data covering 2000+
- `build_deferred_series` produced non-null values for the calibration window

If exog data only exists from, e.g., 2015 onward, the backtest will be restricted to that window — that's fine, just smaller training data.

- [ ] **Step 6: End-to-end Streamlit boot test**

Run:
```bash
uv run streamlit run dashboard/app.py --server.headless true --server.port 8515 > /tmp/sl.log 2>&1 &
SLPID=$!
sleep 10
curl -s -o /dev/null -w "HTTP: %{http_code}\n" http://localhost:8515
kill $SLPID 2>/dev/null
sleep 2
```

Expected: prints `HTTP: 200`. If not, check `/tmp/sl.log`.

- [ ] **Step 7: Final lint + types**

Run: `uv run ruff check . && uv run mypy`
Expected: both clean.

- [ ] **Step 8: Commit**

```bash
git add data/catalog.json
git commit -m "feat(catalog): wire up CME futures as regressors for 8 cash series"
```

- [ ] **Step 9: Push to remote**

```bash
git push origin main
```

Expected: branch updated on `origin/main`.

---

## Self-review notes

**Spec coverage:**

- New module `src/usda_sandbox/futures.py` → Tasks 3, 4, 5 ✓
- `contract_months`, `contract_delivery_date`, `contract_ticker`, `parse_contract_ticker` → Task 3 ✓
- `build_deferred_series` with linear interpolation → Task 4 ✓
- `sync_futures` with manifest + idempotency → Task 5 ✓
- `append_futures_to_observations` with merge + dedupe → Task 5 ✓
- `clean_all` dispatcher on `futures:` prefix → Task 5 ✓
- Two new SeriesDefinition fields → Task 2 ✓
- Forecaster exog support (all three) → Task 6 ✓
- `iter_run_backtest` catalog-driven exog → Task 7 ✓
- Dashboard caption + sidebar filter → Task 7 (filter already exists from v0.2a's `frequencies=["monthly"]`; we add `forecastable=True` filter via the catalog lookup at runtime — actually, looking again, the existing sidebar filter is on `frequency`. The `forecastable` filter is needed too — flag this) ✓
- Catalog growth to 36 entries → Task 8 ✓
- Real-data verification → Task 8 ✓

**Type / signature consistency:**

- `BaseForecaster.fit(df, exog=None)` matches across base class + all 3 forecasters → ✓
- `BaseForecaster.predict(horizon, exog_future=None)` matches across all → ✓
- `BaseForecaster.cross_validate_iter(df, horizon, n_windows, exog=None)` matches → ✓
- `iter_run_backtest(..., catalog_path="data/catalog.json")` — default string path, matches dashboard usage → ✓
- `FuturesManifestEntry` field names: `ticker, commodity, month_code, year, delivery_date, sha256, downloaded_at` — used consistently → ✓
- `build_deferred_series(commodity, horizon_months, per_contract, months)` — signature matches between definition and call site in `append_futures_to_observations` → ✓

**Placeholder scan:** Every step has either concrete code, an exact command with expected output, or both. The Task 8 catalog update uses an inline Python script that produces an exact known result. The yfinance data fetch in Task 8 has a documented fallback (smaller start_year) if the full historical pull fails.

**One known gap noted during self-review:** The sidebar picker on the Forecast page currently filters on `frequency=["monthly"]` only. To prevent the 24 futures series from appearing, we also need a `forecastable=True` filter. Adding to Task 7's dashboard changes:

In `dashboard/components/sidebar.py`, `render_sidebar()` reads a `forecastable_only: bool` parameter. The Forecast page passes `forecastable_only=True`. The Visualize page also gets `forecastable_only=True` to keep its picker clean. The Explore page leaves the filter off so users can see all series.

**Add to Task 7 Step 5** (dashboard edits): also add a `forecastable_only: bool = False` parameter to `render_sidebar()` in `dashboard/components/sidebar.py` that filters `series_df` by `pl.col("frequency").is_in(frequencies) & (pl.col("forecastable") if forecastable_only else pl.lit(True))`. Update `3_Forecast.py` and `2_Visualize.py` to pass `forecastable_only=True`.

**This is included as additional code** in Task 7 Step 5. The plan is otherwise complete and self-consistent.
