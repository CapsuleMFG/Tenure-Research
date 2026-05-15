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
    _safe_filename,
    append_continuous_to_observations,
    sync_continuous_futures,
)


def test_manifest_entry_round_trips_through_dict() -> None:
    entry = ContinuousManifestEntry(
        symbol="LE=F",
        sha256="abc",
        downloaded_at="2026-05-13T10:00:00+00:00",
        missing=False,
    )
    assert ContinuousManifestEntry(**asdict(entry)) == entry


def test_safe_filename_replaces_equals() -> None:
    """yfinance symbols contain ``=`` which we sanitize for on-disk filenames."""
    assert _safe_filename("LE=F") == "LE_F"
    assert _safe_filename("HE=F") == "HE_F"
    assert _safe_filename("plain") == "plain"


def test_fill_monthly_gaps_forward_fills_missing_months() -> None:
    """yfinance's monthly resampling occasionally drops months; the append
    path forward-fills so the regressor has one row per calendar month."""
    from usda_sandbox.futures_continuous import _fill_monthly_gaps

    sparse = pl.DataFrame(
        {
            "period_start": [date(2024, 1, 1), date(2024, 3, 1), date(2024, 5, 1)],
            "close": [100.0, 110.0, 120.0],
        }
    ).with_columns(
        pl.col("period_start").cast(pl.Date),
        pl.col("close").cast(pl.Float64),
    )
    filled = _fill_monthly_gaps(sparse)
    assert filled.height == 5
    assert filled["period_start"].to_list() == [
        date(2024, 1, 1),
        date(2024, 2, 1),
        date(2024, 3, 1),
        date(2024, 4, 1),
        date(2024, 5, 1),
    ]
    # Feb gets January's close; April gets March's.
    assert filled["close"].to_list() == [100.0, 100.0, 110.0, 110.0, 120.0]


def test_fill_monthly_gaps_returns_empty_on_empty_input() -> None:
    from usda_sandbox.futures_continuous import _fill_monthly_gaps

    empty = pl.DataFrame(
        schema={"period_start": pl.Date, "close": pl.Float64}
    )
    out = _fill_monthly_gaps(empty)
    assert out.is_empty()


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
        symbols=("LE=F",),
        raw_dir=raw_dir,
        fetcher=fake_fetcher,
    )
    assert "LE=F" in manifest
    assert manifest["LE=F"].missing is False
    assert manifest["LE=F"].sha256
    assert (raw_dir / "LE_F.parquet").exists()
    on_disk = pl.read_parquet(raw_dir / "LE_F.parquet")
    assert on_disk.equals(df)


def test_sync_writes_manifest_json(tmp_path: Path) -> None:
    raw_dir = tmp_path / "futures_continuous"
    df = _synthetic_continuous_frame(start=date(2020, 1, 1), n=6)
    sync_continuous_futures(
        symbols=("LE=F",),
        raw_dir=raw_dir,
        fetcher=lambda s: df,
    )
    payload = json.loads((raw_dir / "manifest.json").read_text(encoding="utf-8"))
    assert "LE=F" in payload
    assert payload["LE=F"]["missing"] is False
    assert payload["LE=F"]["sha256"]


def test_sync_idempotent_when_data_unchanged(tmp_path: Path) -> None:
    raw_dir = tmp_path / "futures_continuous"
    df = _synthetic_continuous_frame(start=date(2020, 1, 1), n=6)
    calls = []

    def counting_fetcher(symbol: str) -> pl.DataFrame:
        calls.append(symbol)
        return df

    sync_continuous_futures(
        symbols=("LE=F",), raw_dir=raw_dir, fetcher=counting_fetcher
    )
    sync_continuous_futures(
        symbols=("LE=F",), raw_dir=raw_dir, fetcher=counting_fetcher
    )
    # First sync: 1 fetcher call. Second sync: skipped because SHA matches.
    assert calls == ["LE=F"]
    on_disk = pl.read_parquet(raw_dir / "LE_F.parquet")
    assert on_disk.equals(df)


def test_sync_records_missing_symbol_without_raising(tmp_path: Path) -> None:
    raw_dir = tmp_path / "futures_continuous"

    def empty_fetcher(symbol: str) -> pl.DataFrame:
        return pl.DataFrame(
            schema={"period_start": pl.Date, "close": pl.Float64}
        )

    manifest = sync_continuous_futures(
        symbols=("GF=F",), raw_dir=raw_dir, fetcher=empty_fetcher
    )
    assert manifest["GF=F"].missing is True
    assert manifest["GF=F"].sha256 == ""
    assert not (raw_dir / "GF_F.parquet").exists()


def test_sync_skips_known_missing_symbols_on_rerun(tmp_path: Path) -> None:
    raw_dir = tmp_path / "futures_continuous"
    calls = []

    def empty_fetcher(symbol: str) -> pl.DataFrame:
        calls.append(symbol)
        return pl.DataFrame(
            schema={"period_start": pl.Date, "close": pl.Float64}
        )

    sync_continuous_futures(
        symbols=("GF=F",), raw_dir=raw_dir, fetcher=empty_fetcher
    )
    sync_continuous_futures(
        symbols=("GF=F",), raw_dir=raw_dir, fetcher=empty_fetcher
    )
    # First call: 1 attempt. Second call: skip because known-missing.
    assert calls == ["GF=F"]


def test_sync_replaces_changed_data(tmp_path: Path) -> None:
    """When the cached parquet is deleted, a re-sync repopulates from the fetcher."""
    raw_dir = tmp_path / "futures_continuous"
    df_v1 = _synthetic_continuous_frame(start=date(2020, 1, 1), n=4, base=100.0)
    df_v2 = _synthetic_continuous_frame(start=date(2020, 1, 1), n=4, base=200.0)

    sync_continuous_futures(
        symbols=("LE=F",), raw_dir=raw_dir, fetcher=lambda s: df_v1
    )
    # Simulate the user forcing a refresh (delete cached parquet; manifest
    # entry stays, but Guard 2 won't fire because file_path.exists() is False).
    (raw_dir / "LE_F.parquet").unlink()
    sync_continuous_futures(
        symbols=("LE=F",), raw_dir=raw_dir, fetcher=lambda s: df_v2
    )
    on_disk = pl.read_parquet(raw_dir / "LE_F.parquet")
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
        symbols=("LE=F",), raw_dir=raw_dir, fetcher=lambda s: df_le
    )

    append_continuous_to_observations(
        obs_path=obs_path, raw_dir=raw_dir, symbols=("LE=F",)
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
    assert first["source_file"] == "futures_continuous:LE=F"


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
        symbols=("LE=F",), raw_dir=raw_dir, fetcher=lambda s: df
    )
    append_continuous_to_observations(
        obs_path=obs_path, raw_dir=raw_dir, symbols=("LE=F",)
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
        symbols=("LE=F",), raw_dir=raw_dir, fetcher=lambda s: df
    )
    append_continuous_to_observations(
        obs_path=obs_path, raw_dir=raw_dir, symbols=("LE=F",)
    )
    append_continuous_to_observations(
        obs_path=obs_path, raw_dir=raw_dir, symbols=("LE=F",)
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


def test_clean_all_dispatcher_routes_futures_continuous_prefix(
    tmp_path: Path,
) -> None:
    """clean_all skips futures_continuous: series in the XLSX loop and
    calls append_continuous_to_observations after."""
    from usda_sandbox.clean import clean_all

    # Minimal catalog with one futures_continuous entry only
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(
        json.dumps(
            [
                {
                    "series_id": "cattle_lc_front",
                    "series_name": "Live Cattle front-month futures (continuous)",
                    "commodity": "cattle",
                    "metric": "futures_price",
                    "unit": "USD/cwt",
                    "frequency": "monthly",
                    "source_file": "futures_continuous:LE=F",
                    "source_sheet": "",
                    "header_rows_to_skip": 0,
                    "value_columns": ["X"],
                    "date_column": "X",
                    "notes": "test",
                    "exogenous_regressors": [],
                    "forecastable": False,
                }
            ],
            indent=2,
        ),
        encoding="utf-8",
    )

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    fc_dir = raw_dir / "futures_continuous"
    df = _synthetic_continuous_frame(start=date(2020, 1, 1), n=3)
    sync_continuous_futures(
        symbols=("LE=F",), raw_dir=fc_dir, fetcher=lambda s: df
    )

    out_path = tmp_path / "observations.parquet"
    clean_all(catalog_path, raw_dir, out_path)

    obs = pl.read_parquet(out_path)
    assert obs.filter(pl.col("series_id") == "cattle_lc_front").height == 3
