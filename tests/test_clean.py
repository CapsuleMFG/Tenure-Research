"""Tests for the cleaner.

These exercise the messy bits found in real USDA spreadsheets: multi-row
headers, sparse year columns with quarter labels, "NA"/"--" null tokens,
blank separator rows, and "Yr Jan-Dec" annual rollups inside otherwise
quarterly tables.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pytest

from usda_sandbox.catalog import (
    SeriesDefinition,
    load_catalog,
    save_catalog,
)
from usda_sandbox.clean import (
    OBSERVATIONS_SCHEMA,
    _coerce_float,
    _coerce_year,
    _quarter_bounds,
    clean_all,
    clean_series,
)

# --------------------------------------------------------------------------- #
# Coercion helpers
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (123.45, 123.45),
        (0, 0.0),
        ("123.45", 123.45),
        ("1,234.56", 1234.56),
        ("NA", None),
        ("N/A", None),
        ("--", None),
        ("", None),
        ("   ", None),
        (None, None),
        ("not-a-number", None),
        (True, None),
    ],
)
def test_coerce_float(value: object, expected: float | None) -> None:
    assert _coerce_float(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (2024, 2024),
        (2024.0, 2024),
        ("2024", 2024),
        (None, None),
        ("", None),
        (" ", None),
        ("Yr Jan-Dec", None),
        (1850, None),  # out of plausible range
    ],
)
def test_coerce_year(value: object, expected: int | None) -> None:
    assert _coerce_year(value) == expected


def test_quarter_bounds_maps_q_labels_to_calendar_quarters() -> None:
    assert _quarter_bounds(2024, "Q1 Jan-Mar") == (date(2024, 1, 1), date(2024, 3, 31))
    assert _quarter_bounds(2024, "Q2 Apr-Jun") == (date(2024, 4, 1), date(2024, 6, 30))
    assert _quarter_bounds(2024, "Q3 Jul-Sep") == (date(2024, 7, 1), date(2024, 9, 30))
    assert _quarter_bounds(2024, "Q4 Oct-Dec") == (date(2024, 10, 1), date(2024, 12, 31))
    assert _quarter_bounds(2024, "Annual") is None


# --------------------------------------------------------------------------- #
# Wide-format cleaning
# --------------------------------------------------------------------------- #


def _wide_def(value_col: str = "B") -> SeriesDefinition:
    return SeriesDefinition(
        series_id=f"test_wide_{value_col.lower()}",
        series_name=f"Test wide series in column {value_col}",
        commodity="cattle",
        metric="price",
        unit="USD/cwt",
        frequency="monthly",
        source_file="wide_format.xlsx",
        source_sheet="Historical",
        header_rows_to_skip=4,
        value_columns=[value_col],
        date_column="A",
        notes="",
    )


def test_clean_series_wide_format_skips_headers_and_handles_nulls(
    fixture_paths: dict[str, Path],
) -> None:
    df = clean_series(_wide_def("B"), fixture_paths["wide_format"])

    # 7 data rows in the fixture excluding the blank separator
    assert df.height == 7
    assert dict(df.schema) == OBSERVATIONS_SCHEMA

    # Period bounds for monthly: first-of-month → last-of-month
    starts = df["period_start"].to_list()
    ends = df["period_end"].to_list()
    assert starts[0] == date(2024, 1, 1)
    assert ends[0] == date(2024, 1, 31)
    assert starts[-1] == date(2024, 7, 1)
    assert ends[-1] == date(2024, 7, 31)

    # Null tokens (None, blank, "  ", "--") all surface as null values
    null_starts = df.filter(pl.col("value").is_null())["period_start"].to_list()
    assert date(2024, 4, 1) in null_starts  # truly empty cell
    assert date(2024, 6, 1) in null_starts  # whitespace + "--" in row separator role

    # Numeric values pass through unchanged
    march_value = df.filter(pl.col("period_start") == date(2024, 3, 1))["value"].item()
    assert march_value == pytest.approx(185.20)


def test_clean_series_wide_format_picks_correct_column(
    fixture_paths: dict[str, Path],
) -> None:
    df_b = clean_series(_wide_def("B"), fixture_paths["wide_format"])
    df_e = clean_series(_wide_def("E"), fixture_paths["wide_format"])

    # Column B (cattle) and E (hogs) are independent — different value patterns
    march_b = df_b.filter(pl.col("period_start") == date(2024, 3, 1))["value"].item()
    march_e = df_e.filter(pl.col("period_start") == date(2024, 3, 1))["value"].item()
    assert march_b == pytest.approx(185.20)
    assert march_e == pytest.approx(75.55)

    # Column E has "NA" for the first two rows
    jan_feb_e = df_e.filter(pl.col("period_start").is_in([date(2024, 1, 1), date(2024, 2, 1)]))
    assert jan_feb_e["value"].null_count() == 2


# --------------------------------------------------------------------------- #
# WASDE quarterly cleaning
# --------------------------------------------------------------------------- #


def _wasde_def(value_col: str = "C") -> SeriesDefinition:
    return SeriesDefinition(
        series_id=f"test_wasde_{value_col.lower()}",
        series_name="Test WASDE quarterly series",
        commodity="cattle",
        metric="production",
        unit="million_lbs",
        frequency="quarterly",
        source_file="wasde_format.xlsx",
        source_sheet="WASDE_Beef",
        header_rows_to_skip=3,
        value_columns=[value_col],
        date_column="A+B",
        notes="",
    )


def test_clean_series_wasde_forward_fills_year_and_drops_annual(
    fixture_paths: dict[str, Path],
) -> None:
    df = clean_series(_wasde_def("C"), fixture_paths["wasde_format"])

    # Fixture: 4 quarters of 2023 + 4 quarters of 2024 = 8. "Yr Jan-Dec" is dropped.
    assert df.height == 7  # last fixture row Q3-2024 is included; Yr-2023 is dropped
    assert df.filter(pl.col("period_start") == date(2023, 4, 1))["value"].item() == 6700.0
    # Q4 2023 — year forward-fill must work for the 2nd-4th quarter rows
    assert df.filter(pl.col("period_start") == date(2023, 10, 1))["value"].item() == 6800.0

    # No "Yr Jan-Dec" rollup made it through
    rollup_year_starts = df.filter(pl.col("period_start") == date(2023, 1, 1))
    assert rollup_year_starts.height == 1  # only the Q1 row, not the rollup

    # Quarter bounds correct
    q3_2024 = df.filter(pl.col("period_start") == date(2024, 7, 1))
    assert q3_2024["period_end"].item() == date(2024, 9, 30)
    # Q3 2024 had production="NA" → null
    assert q3_2024["value"].null_count() == 1


def test_clean_series_unknown_sheet_raises(fixture_paths: dict[str, Path]) -> None:
    bad = _wide_def("B").model_copy(update={"source_sheet": "DoesNotExist"})
    with pytest.raises(KeyError):
        clean_series(bad, fixture_paths["wide_format"])


# --------------------------------------------------------------------------- #
# clean_all
# --------------------------------------------------------------------------- #


def test_clean_all_combines_and_dedupes(
    fixture_paths: dict[str, Path], tmp_path: Path
) -> None:
    # Two series referencing the same fixture file — one wide, one WASDE
    catalog = [_wide_def("B"), _wasde_def("C")]
    catalog_path = tmp_path / "catalog.json"
    save_catalog(catalog_path, catalog)
    # Sanity: load_catalog round-trips
    assert load_catalog(catalog_path) == catalog

    raw_dir = fixture_paths["wide_format"].parent
    out_path = tmp_path / "obs.parquet"
    df = clean_all(catalog_path, raw_dir, out_path)

    assert out_path.exists()
    on_disk = pl.read_parquet(out_path)
    assert on_disk.height == df.height

    # Both series present
    series_ids = set(df["series_id"].to_list())
    assert series_ids == {"test_wide_b", "test_wasde_c"}

    # Schema matches the documented contract
    assert dict(df.schema) == OBSERVATIONS_SCHEMA

    # Idempotency: a second run replaces, doesn't append
    df2 = clean_all(catalog_path, raw_dir, out_path)
    assert df2.height == df.height


def test_clean_all_skips_missing_source_files(
    fixture_paths: dict[str, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    catalog = [
        _wide_def("B"),
        _wide_def("B").model_copy(
            update={
                "series_id": "missing_file_series",
                "source_file": "does-not-exist.xlsx",
            }
        ),
    ]
    catalog_path = tmp_path / "catalog.json"
    save_catalog(catalog_path, catalog)

    raw_dir = fixture_paths["wide_format"].parent
    df = clean_all(catalog_path, raw_dir, tmp_path / "obs.parquet")

    assert "missing_file_series" not in df["series_id"].to_list()
    assert "missing_file_series" in capsys.readouterr().out
