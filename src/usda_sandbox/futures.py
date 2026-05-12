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
4. **Append** — builds the 24 derived deferred series (12 horizons x 2
   commodities) and merges them into ``observations.parquet``.
"""

from __future__ import annotations

import calendar
from collections.abc import Iterable
from datetime import UTC, date, datetime
from typing import Any

import polars as pl

__all__ = [
    "build_deferred_series",
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

_COMMODITY_NAME: dict[str, str] = {
    "LE": "cattle_lc",
    "HE": "hogs_he",
}


def _commodity_display_name(commodity: str) -> str:
    if commodity not in _COMMODITY_NAME:
        raise ValueError(f"no display name for commodity {commodity!r}")
    return _COMMODITY_NAME[commodity]


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

    # Parse each contract_ticker once -> (commodity, code, year) -> delivery_date
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

    schema: dict[str, Any] = {
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
    """Add a whole number of months to a date, returning the last day of the
    resulting month.

    Inputs are always month-end dates; the output is the last calendar day of
    the target month.
    """
    total = d.month - 1 + months
    new_year = d.year + total // 12
    new_month = total % 12 + 1
    last_day = calendar.monthrange(new_year, new_month)[1]
    return date(new_year, new_month, last_day)


def _interpolate_at_target(
    pairs: list[tuple[date, float]],
    target: date,
) -> float | None:
    """Linear-interpolate the price at ``target`` from a sorted list of
    (delivery_date, price) tuples.

    - If ``target`` is at or before the earliest contract -> return earliest price.
    - If ``target`` is at or after the latest contract -> return latest price.
    - Otherwise -> linear interpolation between the two flanking contracts.
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
