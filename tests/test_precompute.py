"""Tests for usda_sandbox.precompute — forecast cache builder."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import polars as pl
import pytest

from usda_sandbox.precompute import (
    build_forecast_cache,
    load_forecast_cache,
)


def _synthetic_history(start_year: int = 2014, months: int = 144) -> pl.DataFrame:
    """Plausibly-priced monthly series with a mild trend + seasonality."""
    import math

    rows = []
    yr, mo = start_year, 1
    for i in range(months):
        seasonal = 8.0 * math.sin(2 * math.pi * mo / 12.0)
        trend = 100.0 + 0.4 * i
        noise = ((i * 7919) % 11 - 5) * 0.3  # small deterministic noise
        rows.append((date(yr, mo, 1), trend + seasonal + noise))
        mo += 1
        if mo == 13:
            mo = 1
            yr += 1
    return pl.DataFrame(rows, schema=["period_start", "value"], orient="row").with_columns(
        pl.col("period_start").cast(pl.Date), pl.col("value").cast(pl.Float64)
    )


def _write_obs_parquet(tmp_path: Path, series_id: str) -> Path:
    df = _synthetic_history()
    obs = df.with_columns(
        series_id=pl.lit(series_id),
        series_name=pl.lit("Test Series"),
        commodity=pl.lit("cattle"),
        metric=pl.lit("price"),
        unit=pl.lit("USD/cwt"),
        frequency=pl.lit("monthly"),
        period_end=pl.col("period_start"),
        source_file=pl.lit("synthetic"),
        source_sheet=pl.lit("synthetic"),
        ingested_at=pl.lit("2026-05-15T00:00:00Z"),
    ).select(
        [
            "series_id",
            "series_name",
            "commodity",
            "metric",
            "unit",
            "frequency",
            "period_start",
            "period_end",
            "value",
            "source_file",
            "source_sheet",
            "ingested_at",
        ]
    )
    out = tmp_path / "observations.parquet"
    obs.write_parquet(out)
    return out


def _write_catalog(tmp_path: Path, series_id: str) -> Path:
    catalog = [
        {
            "series_id": series_id,
            "series_name": "Test Series",
            "commodity": "cattle",
            "metric": "price",
            "unit": "USD/cwt",
            "frequency": "monthly",
            "source_file": "synthetic.xlsx",
            "source_sheet": "Sheet1",
            "header_rows_to_skip": 0,
            "value_columns": ["B"],
            "date_column": "A",
            "notes": "Synthetic test series.",
            "exogenous_regressors": [],
            "forecastable": True,
        }
    ]
    out = tmp_path / "catalog.json"
    out.write_text(json.dumps(catalog), encoding="utf-8")
    return out


def test_build_forecast_cache_writes_expected_schema(tmp_path: Path) -> None:
    series_id = "synthetic_test_series"
    obs_path = _write_obs_parquet(tmp_path, series_id)
    catalog_path = _write_catalog(tmp_path, series_id)
    out_path = tmp_path / "forecasts.json"

    build_forecast_cache(
        obs_path=obs_path,
        catalog_path=catalog_path,
        out_path=out_path,
        cv_horizon=3,
        n_windows=4,
        forward_horizon=6,
    )

    assert out_path.exists()
    cache = json.loads(out_path.read_text())

    # Top-level shape
    assert {"generated_at", "horizon", "n_windows", "forward_horizon",
            "by_series", "by_series_errors"} <= set(cache.keys())
    assert cache["horizon"] == 3
    assert cache["n_windows"] == 4
    assert cache["forward_horizon"] == 6

    # Per-series shape
    assert series_id in cache["by_series"]
    entry = cache["by_series"][series_id]
    for key in (
        "series_name", "commodity", "unit", "frequency", "winner_model",
        "winner_metrics", "scoreboard", "latest_actual", "prior_month_actual",
        "prior_year_actual", "forward", "sparkline",
    ):
        assert key in entry, f"missing key {key} in cache entry"

    # Winner metrics
    assert entry["winner_metrics"]["mape"] >= 0
    assert entry["winner_model"] in {"AutoARIMA", "Prophet", "LightGBM"}

    # Forward horizon size matches what we asked for
    assert len(entry["forward"]) == 6
    for record in entry["forward"]:
        assert record["lower_80"] <= record["point"] <= record["upper_80"]


def test_load_forecast_cache_returns_stub_when_missing(tmp_path: Path) -> None:
    cache = load_forecast_cache(tmp_path / "no_such_file.json")
    assert cache == {"generated_at": None, "by_series": {}, "by_series_errors": {}}


@pytest.mark.parametrize("short_obs", [10, 30])
def test_build_forecast_cache_records_error_when_too_short(tmp_path: Path, short_obs: int) -> None:
    """If a series has fewer obs than CV requires, the error should be captured
    in by_series_errors without sinking the cache."""
    series_id = "too_short"
    df = _synthetic_history(months=short_obs)
    obs = df.with_columns(
        series_id=pl.lit(series_id),
        series_name=pl.lit("Too short"),
        commodity=pl.lit("cattle"),
        metric=pl.lit("price"),
        unit=pl.lit("USD/cwt"),
        frequency=pl.lit("monthly"),
        period_end=pl.col("period_start"),
        source_file=pl.lit("synthetic"),
        source_sheet=pl.lit("synthetic"),
        ingested_at=pl.lit("2026-05-15T00:00:00Z"),
    ).select(
        [
            "series_id", "series_name", "commodity", "metric", "unit",
            "frequency", "period_start", "period_end", "value",
            "source_file", "source_sheet", "ingested_at",
        ]
    )
    obs_path = tmp_path / "observations.parquet"
    obs.write_parquet(obs_path)

    catalog_path = _write_catalog(tmp_path, series_id)
    out_path = tmp_path / "forecasts.json"
    build_forecast_cache(
        obs_path=obs_path,
        catalog_path=catalog_path,
        out_path=out_path,
        cv_horizon=6,
        n_windows=12,
    )
    cache = json.loads(out_path.read_text())
    assert series_id in cache["by_series_errors"]
    assert series_id not in cache["by_series"]
