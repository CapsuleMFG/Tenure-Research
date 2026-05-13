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


def _default_stooq_fetcher(symbol: str) -> pl.DataFrame:
    raise NotImplementedError("filled in Task 2")


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
