"""Continuous front-month CME futures ingest via yfinance.

This module is the v0.2c replacement for the per-contract / deferred-h
exog regressor design in ``futures.py``. yfinance's continuous front-month
symbols (``LE=F``, ``HE=F``, ``GF=F``) return ~25 years of monthly closes
each, which gives enough cash-overlap to actually run CV with the
forecasters' rank requirements satisfied. (We originally planned to use
Stooq's free CSV endpoint, but Stooq has since moved CSV downloads behind
a captcha-gated API key.)

The design and rationale live in
``docs/superpowers/specs/2026-05-13-continuous-futures-regressor-design.md``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl
import yfinance as yf

__all__ = [
    "ContinuousManifestEntry",
    "_default_fetcher",
    "append_continuous_to_observations",
    "sync_continuous_futures",
]


@dataclass(frozen=True)
class ContinuousManifestEntry:
    """One row of the continuous-futures manifest, keyed externally on symbol.

    ``missing=True`` (with ``sha256=""``) records that we attempted to fetch
    this symbol but yfinance returned no data — useful for an audit trail
    of which symbols are unavailable.
    """

    symbol: str
    sha256: str
    downloaded_at: str
    missing: bool = False


_MANIFEST_FILENAME = "manifest.json"


# Symbol → (series_id, series_name, commodity) — single source of truth.
# Keys are yfinance continuous-front-month symbols.
_SYMBOL_META: dict[str, tuple[str, str, str]] = {
    "LE=F": (
        "cattle_lc_front",
        "Live Cattle front-month futures (continuous)",
        "cattle",
    ),
    "HE=F": (
        "hogs_he_front",
        "Lean Hogs front-month futures (continuous)",
        "hogs",
    ),
    "GF=F": (
        "cattle_feeder_front",
        "Feeder Cattle front-month futures (continuous)",
        "cattle",
    ),
}


_EMPTY_CONTINUOUS_SCHEMA: dict[str, Any] = {
    "period_start": pl.Date,
    "close": pl.Float64,
}


def _safe_filename(symbol: str) -> str:
    """yfinance continuous symbols contain ``=`` (e.g. ``LE=F``). Replace it
    with ``_`` for filesystem safety; the original symbol is preserved in
    the manifest and used everywhere else."""
    return symbol.replace("=", "_")


def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_manifest(path: Path) -> dict[str, ContinuousManifestEntry]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {s: ContinuousManifestEntry(**entry) for s, entry in raw.items()}


def _save_manifest(
    path: Path, manifest: dict[str, ContinuousManifestEntry]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {s: asdict(e) for s, e in manifest.items()}
    path.write_text(
        json.dumps(serializable, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _fill_monthly_gaps(df: pl.DataFrame) -> pl.DataFrame:
    """Reindex to a contiguous month-start range and forward-fill close.

    yfinance's ``interval="1mo"`` resampling occasionally drops months (we've
    seen it drop e.g., 2001-04, 2026-02, 2026-03 for ``LE=F`` despite those
    months having full trading activity). Futures trade continuously, so the
    most defensible reconstruction is to carry the previous month's close
    forward — that's also what an analyst would do glancing at the raw
    series. The cache keeps yfinance's raw output; we only forward-fill at
    append time so the parquet remains a faithful record of what was
    served.
    """
    if df.is_empty():
        return df
    df = df.with_columns(pl.col("period_start").dt.month_start()).sort("period_start")
    earliest = df["period_start"].min()
    latest = df["period_start"].max()
    if earliest is None or latest is None:
        return df
    full_range = pl.date_range(earliest, latest, interval="1mo", eager=True)
    grid = pl.DataFrame({"period_start": full_range})
    return (
        grid.join(df, on="period_start", how="left")
        .with_columns(pl.col("close").forward_fill())
        .drop_nulls("close")
    )


def _default_fetcher(symbol: str) -> pl.DataFrame:
    """Real yfinance fetch — month-end closes for one continuous symbol.

    Returns an empty DataFrame (canonical schema ``period_start: Date``,
    ``close: Float64``) on any network / API error so the rest of the sync
    proceeds. Never raises for fetch failures.
    """
    try:
        hist = yf.Ticker(symbol).history(
            period="max", auto_adjust=False, interval="1mo"
        )
    except Exception:  # yfinance raises various exception types; catch all.
        return pl.DataFrame(schema=_EMPTY_CONTINUOUS_SCHEMA)
    if hist is None or hist.empty:
        return pl.DataFrame(schema=_EMPTY_CONTINUOUS_SCHEMA)
    # yfinance returns a pandas DataFrame indexed by Timestamp. Build a
    # polars frame with our canonical schema.
    monthly = (
        hist["Close"]
        .dropna()
        .reset_index()
        .rename(columns={"Date": "period_start", "Close": "close"})
    )
    return pl.from_pandas(monthly).select(
        period_start=pl.col("period_start").cast(pl.Date),
        close=pl.col("close").cast(pl.Float64),
    )


def sync_continuous_futures(
    *,
    symbols: Iterable[str] = ("LE=F", "HE=F", "GF=F"),
    raw_dir: Path = Path("data/raw/futures_continuous"),
    fetcher: Callable[[str], pl.DataFrame] | None = None,
) -> dict[str, ContinuousManifestEntry]:
    """Download monthly continuous front-month closes via yfinance.

    Idempotent: SHA-keyed manifest. Symbols that yfinance returns no data
    for are recorded as ``missing=True`` and not re-attempted on subsequent
    runs (delete the manifest entry to force a retry).

    ``fetcher`` is injectable for tests — defaults to :func:`_default_fetcher`.
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    fetcher = fetcher if fetcher is not None else _default_fetcher

    manifest_path = raw_dir / _MANIFEST_FILENAME
    manifest = _load_manifest(manifest_path)

    for symbol in symbols:
        file_path = raw_dir / f"{_safe_filename(symbol)}.parquet"

        # Already-known-missing symbols: don't re-attempt every run.
        # Delete the manifest entry manually to force a retry.
        if symbol in manifest and manifest[symbol].missing:
            continue

        # Already-cached symbols whose on-disk SHA still matches: skip the
        # fetch. This is a per-byte check so any upstream change forces a
        # refresh.
        if (
            symbol in manifest
            and file_path.exists()
            and _sha256_of(file_path) == manifest[symbol].sha256
        ):
            continue

        df = fetcher(symbol)
        if df.is_empty():
            manifest[symbol] = ContinuousManifestEntry(
                symbol=symbol,
                sha256="",
                downloaded_at=datetime.now(UTC).isoformat(),
                missing=True,
            )
            continue

        df.write_parquet(file_path)
        manifest[symbol] = ContinuousManifestEntry(
            symbol=symbol,
            sha256=_sha256_of(file_path),
            downloaded_at=datetime.now(UTC).isoformat(),
            missing=False,
        )

    _save_manifest(manifest_path, manifest)
    return manifest


def append_continuous_to_observations(
    *,
    obs_path: Path = Path("data/clean/observations.parquet"),
    raw_dir: Path = Path("data/raw/futures_continuous"),
    symbols: Iterable[str] = ("LE=F", "HE=F", "GF=F"),
) -> None:
    """Build observations rows from cached continuous-futures parquets
    and merge them into ``observations.parquet``. Idempotent on
    ``(series_id, period_start, source_file)``."""
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"{raw_dir!s} does not exist. Run sync_continuous_futures() "
            "before append_continuous_to_observations()."
        )

    obs_path = Path(obs_path)
    ingested_at = datetime.now(UTC)

    new_frames: list[pl.DataFrame] = []
    for symbol in symbols:
        if symbol not in _SYMBOL_META:
            continue
        file_path = raw_dir / f"{_safe_filename(symbol)}.parquet"
        if not file_path.exists():
            continue  # missing symbol — sync recorded it; nothing to append
        series_id, series_name, commodity = _SYMBOL_META[symbol]
        cached = pl.read_parquet(file_path)
        if cached.is_empty():
            continue
        cached = _fill_monthly_gaps(cached)
        frame = cached.select(
            series_id=pl.lit(series_id),
            series_name=pl.lit(series_name),
            commodity=pl.lit(commodity),
            metric=pl.lit("futures_price"),
            unit=pl.lit("USD/cwt"),
            frequency=pl.lit("monthly"),
            period_start=pl.col("period_start").dt.month_start(),
            period_end=pl.col("period_start"),
            value=pl.col("close"),
            source_file=pl.lit(f"futures_continuous:{symbol}"),
            source_sheet=pl.lit(""),
            ingested_at=pl.lit(ingested_at).cast(pl.Datetime("us", "UTC")),
        )
        new_frames.append(frame)

    if not new_frames:
        return

    new_obs = pl.concat(new_frames, how="vertical_relaxed")
    new_ids = set(new_obs["series_id"].to_list())

    if obs_path.exists():
        existing = pl.read_parquet(obs_path).filter(
            ~pl.col("series_id").is_in(new_ids)
        )
        combined = pl.concat([existing, new_obs], how="vertical_relaxed")
    else:
        combined = new_obs

    combined = combined.unique(
        subset=["series_id", "period_start", "source_file"], keep="last"
    ).sort(["series_id", "period_start"])
    obs_path.parent.mkdir(parents=True, exist_ok=True)
    combined.write_parquet(obs_path)
