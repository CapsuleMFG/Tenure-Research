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
