"""Daily front-month futures ingest via yfinance.

Sibling to :mod:`usda_sandbox.futures_continuous` but at *daily* cadence.
The monthly series feed forecasting (which runs on monthly closes); the
daily series feed the v2.0 Brief, Basis, and Decide pages where producers
expect to see today's number.

Series ids:

    cattle_lc_front_daily       LE=F daily close
    cattle_feeder_front_daily   GF=F daily close
    hogs_he_front_daily         HE=F daily close

These are tagged ``frequency="daily"`` in ``observations.parquet`` and use
``source_file="futures_daily:<symbol>"`` so the cleaner pipeline keeps the
monthly and daily streams cleanly separated.
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
    "DailyManifestEntry",
    "_default_fetcher_daily",
    "append_daily_to_observations",
    "sync_daily_futures",
]


@dataclass(frozen=True)
class DailyManifestEntry:
    symbol: str
    sha256: str
    downloaded_at: str
    n_rows: int = 0
    missing: bool = False


_MANIFEST_FILENAME = "manifest.json"


# Symbol -> (series_id, series_name, commodity, unit).
# Livestock futures use USD/cwt; grain futures use US cents per bushel
# (yfinance returns the raw cent figure, e.g. 455.25 = $4.5525/bu). We
# preserve that and document the unit so the UI can render it correctly.
_SYMBOL_META: dict[str, tuple[str, str, str, str]] = {
    "LE=F": (
        "cattle_lc_front_daily",
        "Live Cattle front-month futures (daily continuous)",
        "cattle",
        "USD/cwt",
    ),
    "HE=F": (
        "hogs_he_front_daily",
        "Lean Hogs front-month futures (daily continuous)",
        "hogs",
        "USD/cwt",
    ),
    "GF=F": (
        "cattle_feeder_front_daily",
        "Feeder Cattle front-month futures (daily continuous)",
        "cattle",
        "USD/cwt",
    ),
    "ZC=F": (
        "corn_front_daily",
        "Corn front-month futures (daily continuous)",
        "grain",
        "cents/bushel",
    ),
    "ZM=F": (
        "soybean_meal_front_daily",
        "Soybean Meal front-month futures (daily continuous)",
        "grain",
        "USD/short_ton",
    ),
    "ZO=F": (
        "oats_front_daily",
        "Oats front-month futures (daily continuous)",
        "grain",
        "cents/bushel",
    ),
}


_EMPTY_DAILY_SCHEMA: dict[str, Any] = {
    "period_start": pl.Date,
    "close": pl.Float64,
}


def _safe_filename(symbol: str) -> str:
    return symbol.replace("=", "_")


def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_manifest(path: Path) -> dict[str, DailyManifestEntry]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {s: DailyManifestEntry(**entry) for s, entry in raw.items()}


def _save_manifest(path: Path, manifest: dict[str, DailyManifestEntry]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {s: asdict(e) for s, e in manifest.items()}
    path.write_text(
        json.dumps(serializable, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _default_fetcher_daily(symbol: str) -> pl.DataFrame:
    """Real yfinance fetch — daily closes for one continuous symbol.

    Pulls ~5 years of daily history (yfinance's ``period="max"`` for
    continuous symbols is capped well shorter than the monthly series).
    Returns an empty DataFrame on any fetch error so a single bad symbol
    doesn't sink the rest of the sync.
    """
    try:
        hist = yf.Ticker(symbol).history(
            period="5y", auto_adjust=False, interval="1d"
        )
    except Exception:
        return pl.DataFrame(schema=_EMPTY_DAILY_SCHEMA)
    if hist is None or hist.empty:
        return pl.DataFrame(schema=_EMPTY_DAILY_SCHEMA)
    daily = (
        hist["Close"]
        .dropna()
        .reset_index()
        .rename(columns={"Date": "period_start", "Close": "close"})
    )
    return pl.from_pandas(daily).select(
        period_start=pl.col("period_start").cast(pl.Date),
        close=pl.col("close").cast(pl.Float64),
    )


def sync_daily_futures(
    *,
    symbols: Iterable[str] = ("LE=F", "HE=F", "GF=F", "ZC=F", "ZM=F", "ZO=F"),
    raw_dir: Path = Path("data/raw/futures_daily"),
    fetcher: Callable[[str], pl.DataFrame] | None = None,
) -> dict[str, DailyManifestEntry]:
    """Download daily front-month closes via yfinance.

    Idempotent on SHA. Daily refresh is the intended cadence — the manifest
    forces a fresh pull every time the hash on disk differs from what
    yfinance returns. ``fetcher`` is injectable for tests.
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    fetcher = fetcher if fetcher is not None else _default_fetcher_daily

    manifest_path = raw_dir / _MANIFEST_FILENAME
    manifest = _load_manifest(manifest_path)

    for symbol in symbols:
        file_path = raw_dir / f"{_safe_filename(symbol)}.parquet"

        if symbol in manifest and manifest[symbol].missing:
            # Don't auto-retry known-missing symbols; remove the manifest
            # entry manually to force a retry.
            continue

        df = fetcher(symbol)
        if df.is_empty():
            manifest[symbol] = DailyManifestEntry(
                symbol=symbol,
                sha256="",
                downloaded_at=datetime.now(UTC).isoformat(),
                n_rows=0,
                missing=True,
            )
            continue

        df.write_parquet(file_path)
        manifest[symbol] = DailyManifestEntry(
            symbol=symbol,
            sha256=_sha256_of(file_path),
            downloaded_at=datetime.now(UTC).isoformat(),
            n_rows=df.height,
            missing=False,
        )

    _save_manifest(manifest_path, manifest)
    return manifest


def append_daily_to_observations(
    *,
    obs_path: Path = Path("data/clean/observations.parquet"),
    raw_dir: Path = Path("data/raw/futures_daily"),
    symbols: Iterable[str] = ("LE=F", "HE=F", "GF=F", "ZC=F", "ZM=F", "ZO=F"),
) -> None:
    """Merge cached daily futures parquets into ``observations.parquet``.

    Idempotent on ``(series_id, period_start, source_file)``. Rows tagged
    ``frequency="daily"``; ``period_start = period_end`` for daily.
    """
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"{raw_dir!s} does not exist. Run sync_daily_futures() first."
        )

    obs_path = Path(obs_path)
    ingested_at = datetime.now(UTC)

    new_frames: list[pl.DataFrame] = []
    for symbol in symbols:
        if symbol not in _SYMBOL_META:
            continue
        file_path = raw_dir / f"{_safe_filename(symbol)}.parquet"
        if not file_path.exists():
            continue
        series_id, series_name, commodity, unit = _SYMBOL_META[symbol]
        cached = pl.read_parquet(file_path)
        if cached.is_empty():
            continue
        frame = cached.select(
            series_id=pl.lit(series_id),
            series_name=pl.lit(series_name),
            commodity=pl.lit(commodity),
            metric=pl.lit("futures_price"),
            unit=pl.lit(unit),
            frequency=pl.lit("daily"),
            period_start=pl.col("period_start"),
            period_end=pl.col("period_start"),
            value=pl.col("close"),
            source_file=pl.lit(f"futures_daily:{symbol}"),
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
