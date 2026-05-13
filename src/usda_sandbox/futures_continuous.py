"""Continuous front-month CME futures ingest from Stooq.

This module is the v0.2c replacement for the per-contract / deferred-h
exog regressor design in ``futures.py``. Stooq's monthly CSV endpoint
returns ~30 years of back-adjusted continuous front-month closes per
commodity, which gives enough cash-overlap to actually run CV with the
forecasters' rank requirements satisfied.

The design and rationale live in
``docs/superpowers/specs/2026-05-13-continuous-futures-regressor-design.md``.
"""

from __future__ import annotations

import hashlib
import io
import json
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

__all__ = [
    "ContinuousManifestEntry",
    "_default_stooq_fetcher",
    "append_continuous_to_observations",
    "sync_continuous_futures",
]


@dataclass(frozen=True)
class ContinuousManifestEntry:
    """One row of the continuous-futures manifest, keyed externally on symbol.

    ``missing=True`` (with ``sha256=""``) records that we attempted to fetch
    this symbol but Stooq returned no data — useful for an audit trail of
    which symbols are unavailable (e.g., ``gf.c`` if Stooq doesn't carry it).
    """

    symbol: str
    sha256: str
    downloaded_at: str
    missing: bool = False


_STOOQ_CSV_URL = "https://stooq.com/q/d/l/?s={symbol}&i=m"


_EMPTY_CONTINUOUS_SCHEMA: dict[str, Any] = {
    "period_start": pl.Date,
    "close": pl.Float64,
}


def _parse_stooq_csv(text: str) -> pl.DataFrame:
    """Parse Stooq's monthly OHLCV CSV into (period_start, close) rows.

    Stooq returns the literal string ``"No data"`` (no header) when it has
    nothing for a symbol; that and the empty string both yield an empty
    DataFrame with the canonical schema.
    """
    text = text.strip()
    if not text or text.lower().startswith("no data"):
        return pl.DataFrame(schema=_EMPTY_CONTINUOUS_SCHEMA)
    raw = pl.read_csv(io.StringIO(text), try_parse_dates=True)
    if "Date" not in raw.columns or "Close" not in raw.columns:
        return pl.DataFrame(schema=_EMPTY_CONTINUOUS_SCHEMA)
    return (
        raw.select(
            period_start=pl.col("Date").cast(pl.Date),
            close=pl.col("Close").cast(pl.Float64),
        )
        .drop_nulls()
    )


def _default_stooq_fetcher(symbol: str) -> pl.DataFrame:
    """Real Stooq fetch — month-end closes for one continuous symbol.

    Returns an empty DataFrame (canonical schema) on HTTP error or
    "No data" response. Never raises for network issues.
    """
    url = _STOOQ_CSV_URL.format(symbol=symbol)
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            body = resp.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError, OSError):
        return pl.DataFrame(schema=_EMPTY_CONTINUOUS_SCHEMA)
    return _parse_stooq_csv(body)


def sync_continuous_futures(
    *,
    symbols: Iterable[str] = ("le.c", "he.c", "gf.c"),
    raw_dir: Path = Path("data/raw/futures_continuous"),
    fetcher: Callable[[str], pl.DataFrame] | None = None,
) -> dict[str, ContinuousManifestEntry]:
    raise NotImplementedError("filled in Task 3")


def append_continuous_to_observations(
    *,
    obs_path: Path = Path("data/clean/observations.parquet"),
    raw_dir: Path = Path("data/raw/futures_continuous"),
    symbols: Iterable[str] = ("le.c", "he.c", "gf.c"),
) -> None:
    raise NotImplementedError("filled in Task 5")
