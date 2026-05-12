"""Tests for the conformal calibration module.

All tests use small synthetic DataFrames so they run in milliseconds.
The 69 existing tests stay green; this file adds the new coverage.
"""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from usda_sandbox.calibration import apply_conformal_scaling


def _sample_forecast() -> pl.DataFrame:
    """Forecast DataFrame matching the schema produced by forecaster.predict()."""
    return pl.DataFrame(
        {
            "period_start": [date(2026, 4, 1), date(2026, 5, 1), date(2026, 6, 1)],
            "point": [100.0, 110.0, 120.0],
            "lower_80": [90.0, 95.0, 100.0],
            "upper_80": [115.0, 130.0, 145.0],
        }
    ).with_columns(
        pl.col("period_start").cast(pl.Date),
        pl.col("point").cast(pl.Float64),
        pl.col("lower_80").cast(pl.Float64),
        pl.col("upper_80").cast(pl.Float64),
    )


def test_apply_scaling_identity_at_one() -> None:
    fc = _sample_forecast()
    out = apply_conformal_scaling(fc, scale=1.0)
    assert out.equals(fc)


def test_apply_scaling_doubles_half_widths_preserving_asymmetry() -> None:
    fc = pl.DataFrame(
        {
            "period_start": [date(2026, 4, 1)],
            "point": [100.0],
            "lower_80": [90.0],   # 10 below point
            "upper_80": [115.0],  # 15 above point
        }
    ).with_columns(pl.col("period_start").cast(pl.Date))
    out = apply_conformal_scaling(fc, scale=2.0)
    # Lower side: 10 → 20 below point → 80
    assert out["lower_80"][0] == pytest.approx(80.0)
    # Upper side: 15 → 30 above point → 130
    assert out["upper_80"][0] == pytest.approx(130.0)
    # Point unchanged
    assert out["point"][0] == 100.0
    # Date unchanged
    assert out["period_start"][0] == date(2026, 4, 1)


def test_apply_scaling_zero_scale_collapses_to_point() -> None:
    fc = _sample_forecast()
    out = apply_conformal_scaling(fc, scale=0.0)
    for row in out.iter_rows(named=True):
        assert row["lower_80"] == pytest.approx(row["point"])
        assert row["upper_80"] == pytest.approx(row["point"])


def test_apply_scaling_preserves_schema_and_column_order() -> None:
    fc = _sample_forecast()
    out = apply_conformal_scaling(fc, scale=1.5)
    assert out.columns == fc.columns
    assert dict(out.schema) == dict(fc.schema)
    assert out.height == fc.height


def test_apply_scaling_empty_forecast_returns_empty() -> None:
    fc = pl.DataFrame(
        schema={
            "period_start": pl.Date,
            "point": pl.Float64,
            "lower_80": pl.Float64,
            "upper_80": pl.Float64,
        }
    )
    out = apply_conformal_scaling(fc, scale=1.5)
    assert out.height == 0
    assert dict(out.schema) == dict(fc.schema)
