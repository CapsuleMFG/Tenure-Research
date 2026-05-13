"""Tests for the continuous front-month futures ingest module."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl
import pytest

from usda_sandbox.futures_continuous import (
    ContinuousManifestEntry,
    _default_stooq_fetcher,
    append_continuous_to_observations,
    sync_continuous_futures,
)


def test_manifest_entry_round_trips_through_dict() -> None:
    entry = ContinuousManifestEntry(
        symbol="le.c",
        sha256="abc",
        downloaded_at="2026-05-13T10:00:00+00:00",
        missing=False,
    )
    assert ContinuousManifestEntry(**asdict(entry)) == entry


def test_parse_stooq_csv_returns_period_start_and_close() -> None:
    from usda_sandbox.futures_continuous import _parse_stooq_csv
    csv_text = (
        "Date,Open,High,Low,Close,Volume\n"
        "2024-01-31,170.0,175.0,168.0,172.5,1000\n"
        "2024-02-29,172.0,178.0,170.0,176.0,1200\n"
    )
    df = _parse_stooq_csv(csv_text)
    assert df.columns == ["period_start", "close"]
    assert df["period_start"].to_list() == [date(2024, 1, 31), date(2024, 2, 29)]
    assert df["close"].to_list() == [172.5, 176.0]
    assert df["period_start"].dtype == pl.Date
    assert df["close"].dtype == pl.Float64


def test_parse_stooq_csv_returns_empty_on_no_data_marker() -> None:
    from usda_sandbox.futures_continuous import _parse_stooq_csv
    df = _parse_stooq_csv("No data")
    assert df.is_empty()
    assert df.columns == ["period_start", "close"]
    assert df["period_start"].dtype == pl.Date
    assert df["close"].dtype == pl.Float64


def test_parse_stooq_csv_returns_empty_on_empty_input() -> None:
    from usda_sandbox.futures_continuous import _parse_stooq_csv
    df = _parse_stooq_csv("")
    assert df.is_empty()
    assert df.columns == ["period_start", "close"]


def _synthetic_continuous_frame(
    *, start: date, n: int, base: float = 170.0
) -> pl.DataFrame:
    """Build a monthly continuous frame: month-ends from ``start`` over ``n`` months."""
    rows = []
    y, m = start.year, start.month
    for i in range(n):
        # Use day=28 to dodge calendar edge cases in tests.
        rows.append({"period_start": date(y, m, 28), "close": base + i})
        m += 1
        if m == 13:
            m = 1
            y += 1
    return pl.DataFrame(rows).with_columns(
        pl.col("period_start").cast(pl.Date),
        pl.col("close").cast(pl.Float64),
    )


def test_sync_writes_one_parquet_per_symbol(tmp_path: Path) -> None:
    raw_dir = tmp_path / "futures_continuous"
    df = _synthetic_continuous_frame(start=date(2020, 1, 1), n=12)

    def fake_fetcher(symbol: str) -> pl.DataFrame:
        return df

    manifest = sync_continuous_futures(
        symbols=("le.c",),
        raw_dir=raw_dir,
        fetcher=fake_fetcher,
    )
    assert "le.c" in manifest
    assert manifest["le.c"].missing is False
    assert manifest["le.c"].sha256
    assert (raw_dir / "le.c.parquet").exists()
    on_disk = pl.read_parquet(raw_dir / "le.c.parquet")
    assert on_disk.equals(df)


def test_sync_writes_manifest_json(tmp_path: Path) -> None:
    raw_dir = tmp_path / "futures_continuous"
    df = _synthetic_continuous_frame(start=date(2020, 1, 1), n=6)
    sync_continuous_futures(
        symbols=("le.c",),
        raw_dir=raw_dir,
        fetcher=lambda s: df,
    )
    payload = json.loads((raw_dir / "manifest.json").read_text(encoding="utf-8"))
    assert "le.c" in payload
    assert payload["le.c"]["missing"] is False
    assert payload["le.c"]["sha256"]


def test_sync_idempotent_when_data_unchanged(tmp_path: Path) -> None:
    raw_dir = tmp_path / "futures_continuous"
    df = _synthetic_continuous_frame(start=date(2020, 1, 1), n=6)
    calls = []

    def counting_fetcher(symbol: str) -> pl.DataFrame:
        calls.append(symbol)
        return df

    sync_continuous_futures(
        symbols=("le.c",), raw_dir=raw_dir, fetcher=counting_fetcher
    )
    sync_continuous_futures(
        symbols=("le.c",), raw_dir=raw_dir, fetcher=counting_fetcher
    )
    # First sync: 1 fetcher call. Second sync: skipped because SHA matches.
    assert calls == ["le.c"]
    on_disk = pl.read_parquet(raw_dir / "le.c.parquet")
    assert on_disk.equals(df)


def test_sync_records_missing_symbol_without_raising(tmp_path: Path) -> None:
    raw_dir = tmp_path / "futures_continuous"

    def empty_fetcher(symbol: str) -> pl.DataFrame:
        return pl.DataFrame(
            schema={"period_start": pl.Date, "close": pl.Float64}
        )

    manifest = sync_continuous_futures(
        symbols=("gf.c",), raw_dir=raw_dir, fetcher=empty_fetcher
    )
    assert manifest["gf.c"].missing is True
    assert manifest["gf.c"].sha256 == ""
    assert not (raw_dir / "gf.c.parquet").exists()


def test_sync_skips_known_missing_symbols_on_rerun(tmp_path: Path) -> None:
    raw_dir = tmp_path / "futures_continuous"
    calls = []

    def empty_fetcher(symbol: str) -> pl.DataFrame:
        calls.append(symbol)
        return pl.DataFrame(
            schema={"period_start": pl.Date, "close": pl.Float64}
        )

    sync_continuous_futures(
        symbols=("gf.c",), raw_dir=raw_dir, fetcher=empty_fetcher
    )
    sync_continuous_futures(
        symbols=("gf.c",), raw_dir=raw_dir, fetcher=empty_fetcher
    )
    # First call: 1 attempt. Second call: skip because known-missing.
    assert calls == ["gf.c"]


def test_sync_replaces_changed_data(tmp_path: Path) -> None:
    """When the cached parquet is deleted, a re-sync repopulates from the fetcher."""
    raw_dir = tmp_path / "futures_continuous"
    df_v1 = _synthetic_continuous_frame(start=date(2020, 1, 1), n=4, base=100.0)
    df_v2 = _synthetic_continuous_frame(start=date(2020, 1, 1), n=4, base=200.0)

    sync_continuous_futures(
        symbols=("le.c",), raw_dir=raw_dir, fetcher=lambda s: df_v1
    )
    # Simulate the user forcing a refresh (delete cached parquet; manifest
    # entry stays, but Guard 2 won't fire because file_path.exists() is False).
    (raw_dir / "le.c.parquet").unlink()
    sync_continuous_futures(
        symbols=("le.c",), raw_dir=raw_dir, fetcher=lambda s: df_v2
    )
    on_disk = pl.read_parquet(raw_dir / "le.c.parquet")
    assert on_disk["close"].to_list() == df_v2["close"].to_list()


def _empty_observations() -> pl.DataFrame:
    """Minimal observations.parquet schema for round-trip tests."""
    return pl.DataFrame(
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
    )


def test_append_writes_one_row_per_symbol_per_month(tmp_path: Path) -> None:
    raw_dir = tmp_path / "futures_continuous"
    obs_path = tmp_path / "observations.parquet"
    _empty_observations().write_parquet(obs_path)

    df_le = _synthetic_continuous_frame(start=date(2020, 1, 1), n=3, base=170.0)
    sync_continuous_futures(
        symbols=("le.c",), raw_dir=raw_dir, fetcher=lambda s: df_le
    )

    append_continuous_to_observations(
        obs_path=obs_path, raw_dir=raw_dir, symbols=("le.c",)
    )

    obs = pl.read_parquet(obs_path)
    le_rows = obs.filter(pl.col("series_id") == "cattle_lc_front")
    assert le_rows.height == 3
    first = le_rows.sort("period_start").row(0, named=True)
    assert first["series_name"] == "Live Cattle front-month futures (continuous)"
    assert first["commodity"] == "cattle"
    assert first["metric"] == "futures_price"
    assert first["unit"] == "USD/cwt"
    assert first["frequency"] == "monthly"
    assert first["value"] == pytest.approx(170.0)
    assert first["source_file"] == "futures_continuous:le.c"


def test_append_preserves_existing_observations(tmp_path: Path) -> None:
    raw_dir = tmp_path / "futures_continuous"
    obs_path = tmp_path / "observations.parquet"

    # Seed observations with one unrelated cash row
    seed = pl.DataFrame(
        {
            "series_id": ["cattle_steer_choice_nebraska"],
            "series_name": ["seed"],
            "commodity": ["cattle"],
            "metric": ["price"],
            "unit": ["USD/cwt"],
            "frequency": ["monthly"],
            "period_start": [date(2024, 1, 1)],
            "period_end": [date(2024, 1, 31)],
            "value": [180.0],
            "source_file": ["livestock-prices.xlsx"],
            "source_sheet": ["Historical"],
            "ingested_at": [datetime(2024, 1, 1, tzinfo=UTC)],
        }
    ).with_columns(
        pl.col("period_start").cast(pl.Date),
        pl.col("period_end").cast(pl.Date),
        pl.col("value").cast(pl.Float64),
        pl.col("ingested_at").cast(pl.Datetime("us", "UTC")),
    )
    seed.write_parquet(obs_path)

    df = _synthetic_continuous_frame(start=date(2020, 1, 1), n=2)
    sync_continuous_futures(
        symbols=("le.c",), raw_dir=raw_dir, fetcher=lambda s: df
    )
    append_continuous_to_observations(
        obs_path=obs_path, raw_dir=raw_dir, symbols=("le.c",)
    )

    obs = pl.read_parquet(obs_path)
    cash = obs.filter(pl.col("series_id") == "cattle_steer_choice_nebraska")
    assert cash.height == 1
    assert obs.filter(pl.col("series_id") == "cattle_lc_front").height == 2


def test_append_idempotent_on_rerun(tmp_path: Path) -> None:
    raw_dir = tmp_path / "futures_continuous"
    obs_path = tmp_path / "observations.parquet"
    _empty_observations().write_parquet(obs_path)
    df = _synthetic_continuous_frame(start=date(2020, 1, 1), n=4)
    sync_continuous_futures(
        symbols=("le.c",), raw_dir=raw_dir, fetcher=lambda s: df
    )
    append_continuous_to_observations(
        obs_path=obs_path, raw_dir=raw_dir, symbols=("le.c",)
    )
    append_continuous_to_observations(
        obs_path=obs_path, raw_dir=raw_dir, symbols=("le.c",)
    )
    obs = pl.read_parquet(obs_path)
    le_rows = obs.filter(pl.col("series_id") == "cattle_lc_front")
    assert le_rows.height == 4  # unchanged on second run


def test_append_raises_if_raw_dir_missing(tmp_path: Path) -> None:
    obs_path = tmp_path / "observations.parquet"
    _empty_observations().write_parquet(obs_path)
    missing_dir = tmp_path / "nope"
    with pytest.raises(FileNotFoundError, match="does not exist"):
        append_continuous_to_observations(
            obs_path=obs_path, raw_dir=missing_dir
        )
