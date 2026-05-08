"""Test setup — regenerate the synthetic XLSX fixtures before any test runs."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.fixtures import build_all
from usda_sandbox.catalog import SeriesDefinition, save_catalog
from usda_sandbox.clean import clean_all


@pytest.fixture(scope="session")
def fixture_paths() -> dict[str, Path]:
    """Build all synthetic XLSX fixtures once per session and return their paths."""
    return build_all()


@pytest.fixture(scope="session")
def fixtures_obs_parquet(
    fixture_paths: dict[str, Path],
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    """Build a small obs parquet from the synthetic XLSX fixtures.

    Wires ``clean_all`` end-to-end so the storage helpers are exercised against
    real cleaner output, not a hand-rolled DataFrame.
    """
    tmp = tmp_path_factory.mktemp("store_fixtures")
    catalog = [
        SeriesDefinition(
            series_id="store_test_wide_b",
            series_name="Wide fixture col B",
            commodity="cattle",
            metric="price",
            unit="USD/cwt",
            frequency="monthly",
            source_file="wide_format.xlsx",
            source_sheet="Historical",
            header_rows_to_skip=4,
            value_columns=["B"],
            date_column="A",
            notes="",
        ),
        SeriesDefinition(
            series_id="store_test_wasde_c",
            series_name="WASDE fixture col C",
            commodity="cattle",
            metric="production",
            unit="million_lbs",
            frequency="quarterly",
            source_file="wasde_format.xlsx",
            source_sheet="WASDE_Beef",
            header_rows_to_skip=3,
            value_columns=["C"],
            date_column="A+B",
            notes="",
        ),
    ]
    catalog_path = tmp / "catalog.json"
    save_catalog(catalog_path, catalog)

    raw_dir = fixture_paths["wide_format"].parent
    out_path = tmp / "observations.parquet"
    clean_all(catalog_path, raw_dir, out_path)
    return out_path
