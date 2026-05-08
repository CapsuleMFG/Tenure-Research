"""Tests for the storage helpers.

Covers all four exposed accessors (``read_observations``, ``read_series``,
``list_series``, ``duckdb_connection``) against a small parquet built from
the cleaner fixtures.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from usda_sandbox.store import (
    OBS_VIEW_NAME,
    duckdb_connection,
    list_series,
    read_observations,
    read_series,
)


def test_read_observations_returns_lazyframe(fixtures_obs_parquet: Path) -> None:
    lf = read_observations(fixtures_obs_parquet)
    assert isinstance(lf, pl.LazyFrame)
    df = lf.collect()
    assert df.height > 0
    # Schema contract — every catalog column must be present
    assert {"series_id", "period_start", "value", "frequency"} <= set(df.columns)


def test_read_series_returns_one_series_sorted(fixtures_obs_parquet: Path) -> None:
    df = read_series("store_test_wide_b", fixtures_obs_parquet)
    assert df["series_id"].unique().to_list() == ["store_test_wide_b"]
    starts = df["period_start"].to_list()
    assert starts == sorted(starts)


def test_read_series_unknown_id_returns_empty_frame(
    fixtures_obs_parquet: Path,
) -> None:
    df = read_series("does_not_exist", fixtures_obs_parquet)
    assert df.height == 0
    # Column shape is preserved even on empty result
    assert "value" in df.columns


def test_list_series_one_row_per_series_with_metadata(
    fixtures_obs_parquet: Path,
) -> None:
    summary = list_series(fixtures_obs_parquet)

    assert summary["series_id"].to_list() == [
        "store_test_wasde_c",
        "store_test_wide_b",
    ]
    expected_cols = {
        "series_id",
        "series_name",
        "commodity",
        "metric",
        "unit",
        "frequency",
        "n_obs",
        "n_nulls",
        "first_period",
        "last_period",
    }
    assert expected_cols <= set(summary.columns)

    # Values match what the fixture cleaner produced
    wide_row = summary.filter(pl.col("series_id") == "store_test_wide_b").row(0, named=True)
    assert wide_row["frequency"] == "monthly"
    assert wide_row["unit"] == "USD/cwt"
    assert wide_row["n_obs"] == 7  # matches test_clean wide-format expectation
    assert wide_row["n_nulls"] >= 1  # NA / blank tokens preserved as nulls


def test_duckdb_connection_exposes_obs_view(fixtures_obs_parquet: Path) -> None:
    con = duckdb_connection(fixtures_obs_parquet)
    try:
        # The view name is the documented contract
        rows = con.execute(f"SELECT COUNT(*) FROM {OBS_VIEW_NAME}").fetchone()
        assert rows is not None
        assert rows[0] > 0

        # SQL-shaped query: per-series stats
        result = con.execute(
            f"""
            SELECT series_id, COUNT(*) AS n_obs
            FROM {OBS_VIEW_NAME}
            GROUP BY series_id
            ORDER BY series_id
            """
        ).fetchall()
        ids = [r[0] for r in result]
        assert ids == ["store_test_wasde_c", "store_test_wide_b"]
    finally:
        con.close()


def test_duckdb_connection_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        duckdb_connection(tmp_path / "nope.parquet")
