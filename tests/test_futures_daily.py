"""Tests for usda_sandbox.futures_daily — daily yfinance fetch + ingest."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import polars as pl

from usda_sandbox.futures_daily import (
    DailyManifestEntry,
    append_daily_to_observations,
    sync_daily_futures,
)


def _fake_fetcher(symbol: str) -> pl.DataFrame:
    """Deterministic stub for the yfinance call — same shape, no network."""
    if symbol == "LE=F":
        return pl.DataFrame(
            {
                "period_start": [date(2026, 5, 13), date(2026, 5, 14), date(2026, 5, 15)],
                "close": [245.0, 246.5, 247.7],
            }
        ).with_columns(pl.col("close").cast(pl.Float64))
    if symbol == "HE=F":
        return pl.DataFrame(
            {
                "period_start": [date(2026, 5, 13), date(2026, 5, 14), date(2026, 5, 15)],
                "close": [102.0, 102.8, 103.4],
            }
        ).with_columns(pl.col("close").cast(pl.Float64))
    if symbol == "GF=F":
        return pl.DataFrame(
            {
                "period_start": [date(2026, 5, 13), date(2026, 5, 14), date(2026, 5, 15)],
                "close": [358.0, 360.0, 361.3],
            }
        ).with_columns(pl.col("close").cast(pl.Float64))
    return pl.DataFrame()


def test_sync_daily_futures_writes_manifest_and_parquets(tmp_path: Path) -> None:
    manifest = sync_daily_futures(raw_dir=tmp_path, fetcher=_fake_fetcher)
    assert set(manifest.keys()) == {"LE=F", "HE=F", "GF=F"}
    for sym, entry in manifest.items():
        assert isinstance(entry, DailyManifestEntry)
        assert entry.symbol == sym
        assert entry.n_rows == 3
        assert not entry.missing
        # Parquet on disk
        parquet_path = tmp_path / f"{sym.replace('=', '_')}.parquet"
        assert parquet_path.exists()
        df = pl.read_parquet(parquet_path)
        assert df.height == 3
        assert set(df.columns) == {"period_start", "close"}

    # Manifest JSON written
    m_json = json.loads((tmp_path / "manifest.json").read_text())
    assert set(m_json.keys()) == {"LE=F", "HE=F", "GF=F"}


def test_append_daily_to_observations_emits_daily_rows(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    sync_daily_futures(raw_dir=raw_dir, fetcher=_fake_fetcher)

    obs_path = tmp_path / "observations.parquet"
    append_daily_to_observations(obs_path=obs_path, raw_dir=raw_dir)

    obs = pl.read_parquet(obs_path)
    daily = obs.filter(pl.col("frequency") == "daily")
    # 3 symbols × 3 days = 9 rows
    assert daily.height == 9
    assert set(daily["series_id"].unique()) == {
        "cattle_lc_front_daily", "cattle_feeder_front_daily", "hogs_he_front_daily"
    }
    assert (daily["unit"] == "USD/cwt").all()
    # source_file prefix
    assert daily["source_file"].str.starts_with("futures_daily:").all()


def test_append_is_idempotent_on_rerun(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    sync_daily_futures(raw_dir=raw_dir, fetcher=_fake_fetcher)
    obs_path = tmp_path / "observations.parquet"

    append_daily_to_observations(obs_path=obs_path, raw_dir=raw_dir)
    first = pl.read_parquet(obs_path).height
    append_daily_to_observations(obs_path=obs_path, raw_dir=raw_dir)
    second = pl.read_parquet(obs_path).height
    assert first == second
